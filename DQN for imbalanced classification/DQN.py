import torch
import random
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import namedtuple

Transition = namedtuple('transition', ('state', 'action', 'next_state', 'reward'))

gamma = 0.8
batch_size = 7
capacity = 28

class ReplayMemory :

    def __init__(self, capacity) :
        self.capacity = capacity
        self.memory = []
        self.index = 0
        
        return

    def push(self, state, action, next_state, reward) :

        if len(self.memory) < self.capacity :
            self.memory.append(None)

        self.memory[self.index] = Transition(state, action, next_state, reward)
        self.index = (self.index + 1) % self.capacity

        return

    def sample(self, batch_size) :
        return random.sample(self.memory, batch_size)

    def __len__(self) :
        return len(self.memory)

class Brain :

    def __init__(self, num_observations, num_actions) :

        self.num_observations = num_observations
        self.num_actions = num_actions
        self.memory = ReplayMemory(capacity)

        # Setting Neural network structure.
        self.model = nn.Sequential()
        self.model.add_module('fc1', nn.Linear(self.num_observations, 512))
        self.model.add_module('relu1', nn.ReLU())

        self.model.add_module('fc2', nn.Linear(512, 512))
        self.model.add_module('relu2', nn.ReLU())

        self.model.add_module('fc3', nn.Linear(512, self.num_actions))

        # GPU setting.
        self.model.cuda()

        # self.model.parameters() : calling all of parameters consisting of model network.
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.05)

        return

    def replay(self) :

        if len(self.memory) < batch_size :
            return

        else :
            transitions = self.memory.sample(batch_size)

            # zip : pairing each elements of iterable in order.
            batch = Transition(*zip(*transitions))

            # transformation data-type to tensor (default dim = 0).
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

            # get the model ready for evaluation from input to output (feed-forward)
            self.model.eval()

            # due to self.model.eval(), self.model(state_batch) evaluates output.
            # the input is 'state_batch'
            # self.model(state_batch).gather(1, action_batch) means that
            # in dim=1, pick the elements corresponding to action_batch structure.
            # q(s, a)
            state_action_values = self.model(state_batch).gather(1, action_batch)

            # tuple(map(lambda s : s is not None, (2, 3, 1, None))) ==> (True, True, True, False)
            # non_final_mask is binary elemented(0, 1) tensor(vector).
            non_final_mask = torch.ByteTensor(tuple(map(lambda s : s is not None, batch.next_state))).cuda()

            # 'torch.ByteTensor' makes assigning elements according to location of tensor.
            # using 'detach()', store max of next_q value of next_state_values tensor.
            next_state_values = torch.zeros(batch_size).cuda()
            next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach()

            # reward + gamma * max q(s', a') in tensor form.
            expected_state_action_values = reward_batch + gamma * next_state_values

            # get the model ready for train
            self.model.train()

            # setting a loss function
            # F.smooth_l1_loss means Huber function.
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1)).cuda()

            # initialize the gradient.
            self.optimizer.zero_grad()

            # calculate back-propagation.
            loss.backward()

            # apply calculated back-propagation to optimizer.
            self.optimizer.step()

            return

    def decide_action(self, state, episode, test_flag) :

        epsilon = 0.2 / (episode + 1)
        
        if not test_flag :
            
            if epsilon <= np.random.uniform(0, 1) :
                self.model.eval()
                with torch.no_grad() :
                    action = self.model(state).max(1)[1].view(1, 1) # .view(1, 1) transform a tensor to size 1*1
            else :
                action = torch.LongTensor([[random.randrange(self.num_actions)]])

            return action.cuda()
                
        elif test_flag :
            self.model.eval()
            return self.model(state).max(1)[1].view(1)

# once the class Agent created, the rest of classes are created.
class Agent :

    def __init__(self, num_observations, num_actions) :
        self.brain = Brain(num_observations, num_actions)
        return

    def update_q_function(self) :
        self.brain.replay()
        return

    def get_action(self, state, episode) :
        return self.brain.decide_action(state, episode, False)

    def memorize(self, state, action, next_state, reward) :
        self.brain.memory.push(state, action, next_state, reward)
        return

    def test(self, state, episode) :
        return self.brain.decide_action(state, episode, True)
