import argparse
from itertools import count

import gym
import time
import scipy.optimize
import random
import tkinter as tk
import fileinput
import pickle

import torch
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--load', type=bool, default=False)
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor (default: 0.995)')
parser.add_argument('--env-name', default="Hopper-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                    help='gae (default: 0.97)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--max-kl', type=float, default=2e-2, metavar='G',
                    help='max kl value (default: 2e-2)')
parser.add_argument('--damping', type=float, default=1e-1, metavar='G',
                    help='damping (default: 1e-1)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=15, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true', default=True,
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

env = gym.make(args.env_name)

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

env.seed(args.seed)
torch.manual_seed(args.seed)

policy_net = Policy(num_inputs, num_actions)
value_net = Value(num_inputs)
reward_net = Reward(num_inputs+num_actions)
reward_optim = torch.optim.Adam(reward_net.parameters(), lr=3e-3)
reward_loss_func = torch.nn.MSELoss()
choice = None
done = False
human_choices = []
recordings = []


def render_video(steps, seed):
    env.seed(seed)
    env.reset()
    env.render()
    time.sleep(0.2)
    for step in steps:
        env.step(step)
        env.render()
        time.sleep(0.0025)


def reward_loss(probability_1_over_2, probability_2_over_1, weights):
    return Variable(-torch.sum(probability_1_over_2*weights + probability_2_over_1*(1-weights)), requires_grad=True)

def update_reward_net(steps_one, states_one, steps_two, states_two, seed_one, seed_two, env, i_episode):
    global human_choices
    global recordings
    
    while(True):
        render_video(steps_one, seed_one)
        render_video(steps_two, seed_two)
        print("Press 1 if first, 2 if second, 3 if equally good, 4 if neither, anything else to replay, episode " + str(i_episode))
        for line in fileinput.input():
            input = None
            try:
                input = int(line.strip())
            except:
                input = None
            
            if input == 4:
                break
            elif input == 1:
                human_choices.append((Variable(torch.cat((torch.DoubleTensor(steps_one),torch.DoubleTensor(states_one)),1), requires_grad=True),
                                      Variable(torch.cat((torch.DoubleTensor(steps_two),torch.DoubleTensor(states_two)),1), requires_grad=True),
                                      1.))
            elif input == 2:
                human_choices.append((Variable(torch.cat((torch.DoubleTensor(steps_one),torch.DoubleTensor(states_one)),1), requires_grad=True),
                                      Variable(torch.cat((torch.DoubleTensor(steps_two),torch.DoubleTensor(states_two)),1), requires_grad=True),
                                      0.))
            elif input == 3:
                human_choices.append((Variable(torch.cat((torch.DoubleTensor(steps_one),torch.DoubleTensor(states_one)),1), requires_grad=True),
                                      Variable(torch.cat((torch.DoubleTensor(steps_two),torch.DoubleTensor(states_two)),1), requires_grad=True),
                                      0.5))
            break
        fileinput.close()
        
        if input == None:
            continue
        
        recordings.append((steps_one, states_one, steps_two, states_two, seed_one, seed_two))
        exp_sums_ones = [torch.exp(reward_net(x)).sum() for x,_,_ in human_choices]
        exp_sums_twos = [torch.exp(reward_net(y)).sum() for _,y,_ in human_choices]
        probability_1_over_2 = torch.DoubleTensor([x/(x+y)*0.9+0.05 for x, y in zip(exp_sums_ones, exp_sums_twos)])
        probability_2_over_1 = torch.DoubleTensor([y/(x+y)*0.9+0.05 for x, y in zip(exp_sums_ones, exp_sums_twos)])
        weights = torch.DoubleTensor([w for _,_,w in human_choices])
        loss = reward_loss(probability_1_over_2, probability_2_over_1, weights)
        reward_optim.zero_grad()
        loss.backward()
        reward_optim.step()
        if input != None:
            break
                
    for _ in range(1):

        exp_sums_ones = [torch.exp(reward_net(x)).sum() for x,_,_ in human_choices]
        exp_sums_twos = [torch.exp(reward_net(y)).sum() for _,y,_ in human_choices]
        probability_1_over_2 = torch.DoubleTensor([x/(x+y)*0.9+0.05 for x, y in zip(exp_sums_ones, exp_sums_twos)])
        probability_2_over_1 = torch.DoubleTensor([y/(x+y)*0.9+0.05 for x, y in zip(exp_sums_ones, exp_sums_twos)])
        weights = torch.DoubleTensor([w for _,_,w in human_choices])
        loss = reward_loss(probability_1_over_2, probability_2_over_1, weights)
        reward_optim.zero_grad()
        loss.backward()
        reward_optim.step()


def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action

def update_params(batch):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(batch.state)
    values = value_net(Variable(states))

    returns = torch.Tensor(actions.size(0),1)
    deltas = torch.Tensor(actions.size(0),1)
    advantages = torch.Tensor(actions.size(0),1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = Variable(returns)

    # Original code uses the same LBFGS to optimize the value loss
    def get_value_loss(flat_params):
        set_flat_params_to(value_net, torch.Tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        values_ = value_net(Variable(states))

        value_loss = (values_ - targets).pow(2).mean()

        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * args.l2_reg
        value_loss.backward()
        return (value_loss.data.double().numpy(), get_flat_grad_from(value_net).data.double().numpy())

    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(value_net).double().numpy(), maxiter=25)
    set_flat_params_to(value_net, torch.Tensor(flat_params))

    advantages = (advantages - advantages.mean()) / advantages.std()

    action_means, action_log_stds, action_stds = policy_net(Variable(states))
    fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

    def get_loss(volatile=False):
        if volatile:
            with torch.no_grad():
                action_means, action_log_stds, action_stds = policy_net(Variable(states))
        else:
            action_means, action_log_stds, action_stds = policy_net(Variable(states))
                
        log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
        action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
        return action_loss.mean()


    def get_kl():
        mean1, log_std1, std1 = policy_net(Variable(states))

        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    trpo_step(policy_net, get_loss, get_kl, args.max_kl, args.damping)


def save_models():
    torch.save(policy_net, './nets/policy_net')
    torch.save(value_net, './nets/value_net')
    torch.save(reward_net, './nets/reward_net')
    with open('./nets/net.data', 'wb') as filehandle:
        pickle.dump(recordings, filehandle)
        
def load_models():
    global recordings
    policy_net = torch.load('./nets/policy_net')
    value_net = torch.load('./nets/value_net')
    reward_net = torch.load('./nets/reward_net')
    with open('./nets/net.data', 'rb') as filehandle:
        recordings = pickle.load(filehandle)

if __name__ == "__main__":
    # if args.load:
        # print('\n\n\n loading \n\n\n')
    policy_net = torch.load('./nets/policy_net')
    value_net = torch.load('./nets/value_net')
    reward_net = torch.load('./nets/reward_net')
    with open('./nets/net.data', 'rb') as filehandle:
        recordings = pickle.load(filehandle)
    
    print('\n\n\n loaded \n\n\n')
    
    running_state = ZFilter((num_inputs,), clip=5)
    running_reward = ZFilter((1,), demean=False, clip=10)

    for i_episode in range(0,201):
        memory = Memory()

        num_steps = 0
        reward_batch = 0
        num_episodes = 0
        one = random.randint(0,args.batch_size-2)
        two = one + 1
        steps_one = []
        steps_two = []
        states_one = []
        states_two = []
        seed_one = None
        seed_two = None
        
        # while num_steps < args.batch_size:
        for idx in range(args.batch_size):
            seed = random.randint(0,1000000)
            env.seed(seed)
            state = env.reset()
            
            if idx == one:
                seed_one = seed
            if idx == two:
                seed_two = seed
            
            state = running_state(state)

            reward_sum = 0
            for t in range(400): # Don't infinite loop while learning
                action = select_action(state)
                action = action.data[0].numpy()
                next_state, _, _, _ = env.step(action)
                reward = reward_net(torch.cat((torch.from_numpy(next_state),torch.from_numpy(action)), 0))
                reward_sum += reward
                if idx == one:
                    states_one.append(state)
                    steps_one.append(action)
                if idx == two:
                    states_two.append(state)
                    steps_two.append(action)

                next_state = running_state(next_state)

                mask = int(t!=399)
                # if done:
                #     mask = 0

                memory.push(state, np.array([action]), mask, next_state, reward)

                state = next_state
            # num_steps += (t-1)
            num_episodes += 1
            reward_batch += reward_sum

        reward_batch /= num_episodes
        batch = memory.sample()
        update_params(batch)
        update_reward_net(steps_one, states_one, steps_two, states_two, seed_one, seed_two, env, i_episode)

        if i_episode % args.log_interval == 0:
            print(i_episode, reward_batch)
        
        if i_episode % 20 == 0 and i_episode > 0:
            save_models()
