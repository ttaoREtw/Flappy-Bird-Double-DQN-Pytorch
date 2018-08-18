TARGET_UPDATE_CYCLE = 10
SAVE_MODEL_CYCLE = 500
LOGGING_CYCLE = 1
BATCH = 32
DISCOUNT = 0.99
LEARNING_RATE = 1e-4
OBSERV = 50000
CAPACITY = 50000


import time
import os
import random
from collections import OrderedDict
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from replay_buffer import ReplayBuffer

class Trainer(object):
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.seed = random.randint(0, 20180818)
        self.optimizer = optim.Adam(agent.parameters, lr=LEARNING_RATE)
        self.buffer = ReplayBuffer(capacity=CAPACITY)
        self.total_step = 0

    def run(self, device='cpu', buffer=False, explore=False):
        """Run an episode and buffer"""
        self.env.reset()
        self.env.env.seed(self.seed)
        state = self.env.get_screen()
        states = np.asarray([state for _ in range(4)]) # shape (4, 84, 84)
        step = 0
        accumulated_reward = 0
        while True:
            action = self.agent.make_action(torch.Tensor([states]).to(device), explore=explore)
            state_next, reward, done = self.env.step(action)
            states_next = np.concatenate([states[1:, :, :], [state_next]], axis=0)
            step += 1
            accumulated_reward += reward
            if buffer:
                self.buffer.append(states, action, reward, states_next, done)
            states = states_next
            if done:
                break    
        return accumulated_reward, step

    def _fill_buffer(self, num, device='cpu'):
        start = time.time()
        while self.buffer.size < num:
            self.run(device, buffer=True, explore=True)
            print('Fill buffer: {}/{}'.format(self.buffer.size, self.buffer.capacity))
        print('Filling buffer takes {:.3f} seconds'.format(time.time() - start))

    def train(self, device='cpu'):
        self.env.change_record_every_episode(100000000)
        self._fill_buffer(OBSERV, device)
        if self.env.record_every_episode:
            self.env.change_record_every_episode(self.env.record_every_episode)

        episode = 0
        while 'training' != 'converge':
            self.env.reset()
            state = self.env.get_screen()
            states = np.asarray([state for _ in range(4)]) # shape (4, 84, 84)
            step_prev = self.total_step
            accumulated_reward = 0
            done = False
            n_flap = 0
            n_none = 0
            while not done:
                #### --------------------
                #### Add a new transition
                action = self.agent.make_action(torch.Tensor([states]).to(device), explore=True)
                state_next, reward, done = self.env.step(action)
                states_next = np.concatenate([states[1:, :, :], [state_next]], axis=0)
                self.total_step += 1
                accumulated_reward += reward
                self.buffer.append(states, action, reward, states_next, done)
                states = states_next
                #### --------------------

                #### --------------------
                #### Training step
                start = time.time()
                # prepare training data
                minibatch = self.buffer.sample(n_sample=BATCH)
                _states = [b[0] for b in minibatch]
                _actions = [b[1] for b in minibatch]
                _rewards = [b[2] for b in minibatch]
                _states_next = [b[3] for b in minibatch]
                _dones = [b[4] for b in minibatch]

                ys = []
                for i in range(len(minibatch)):
                    terminal = _dones[i]
                    r = _rewards[i]
                    if terminal:
                        y = r
                    else:
                        # Double DQN
                        s_t_next = torch.Tensor([_states_next[i]]).to(device)
                        online_act = self.agent.make_action(s_t_next)
                        y = r + DISCOUNT * self.agent.Q(s_t_next, online_act, target=True)
                    ys.append(y)
                ys = torch.Tensor(ys).to(device)

                # Apply gradient
                self.optimizer.zero_grad()
                input = torch.Tensor(_states).to(device)            
                output = self.agent.net(input) # shape (BATCH, 2)
                actions_one_hot = np.zeros([BATCH, 2])
                actions_one_hot[np.arange(BATCH), _actions] = 1.0
                actions_one_hot = torch.Tensor(actions_one_hot).to(device)
                ys_hat = (output * actions_one_hot).sum(dim=1)
                loss = F.smooth_l1_loss(ys_hat, ys)
                loss.backward()
                self.optimizer.step()
                #### --------------------

                # logging
                if action == 0:
                    n_flap += 1
                else:
                    n_none += 1

                if done and self.total_step % LOGGING_CYCLE == 0:
                    log = '[{}, {}] alive: {}, reward: {}, F/N: {}/{}, loss: {:.4f}, epsilon: {:.4f}, time: {:.3f}'.format(
                        episode, 
                        self.total_step, 
                        self.total_step - step_prev, 
                        accumulated_reward, 
                        n_flap, 
                        n_none, 
                        loss.item(), 
                        self.agent.epsilon, 
                        time.time() - start)
                    print(log)

                self.agent.update_epsilon()
                if self.total_step % TARGET_UPDATE_CYCLE == 0:
                    #print('[Update target network]')
                    self.agent.update_target()

                if self.total_step % SAVE_MODEL_CYCLE == 0:
                    print('[Save model]')
                    self.save(id=self.total_step)
            episode += 1


    def save(self, id):
        filename = 'tmp/models/model_{}.pth.tar'.format(id)
        dirpath = os.path.dirname(filename)
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        checkpoint = {
            'net': self.agent.net.state_dict(),
            'target': self.agent.target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_step': self.total_step
        }
        torch.save(checkpoint, filename)

    def load(self, filename, device='cpu'):
        ckpt = torch.load(filename, map_location=lambda storage, loc: storage)
        ## Deal with the missing of bn.num_batches_tracked
        net_new = OrderedDict()
        tar_new = OrderedDict()

        for k, v in ckpt['net'].items():
            for _k, _v in self.agent.net.state_dict().items():
                if k == _k:
                    net_new[k] = v

        for k, v in ckpt['target'].items():
            for _k, _v in self.agent.target.state_dict().items():
                if k == _k:
                    tar_new[k] = v
        
        self.agent.net.load_state_dict(net_new)
        self.agent.target.load_state_dict(tar_new)
        ## -----------------------------------------------
        
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.total_step = ckpt['total_step']








