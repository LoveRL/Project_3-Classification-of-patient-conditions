import os
import torch
import numpy as np

data_route = 'C:\\Users\\seo\\Desktop\\Lab\\Projects\\정부) 재활로봇\\임상 data\\lightgbm'
NumTrainData = 502

class environment :
    
    def __init__(self) :
        os.chdir(data_route)
        self.data = np.loadtxt('train_1.csv', delimiter=',', dtype=np.float32)
        self.done = False
        self.step_cnt = 0
        return

    def reset(self) :
        self.step_cnt = 0
        np.random.shuffle(self.data)
        temp = self.data[self.step_cnt][:-1]
        
        return torch.unsqueeze(torch.from_numpy(temp).type(torch.FloatTensor).cuda(), 0)
    
    def step(self, action) :

        self.answer_action = self.data[self.step_cnt][-1]
        
        # in case of minority class
        if self.answer_action in [2, 3, 4] :
            if action == self.answer_action :
                reward = 10
            else :
                reward = -10
                self.done = True
        
        # in case of majority class
        else :
            if action == self.answer_action :
                reward = 0.083
            else :
                reward = -0.083

        if self.step_cnt + 1 <= NumTrainData-1 :
            self.done = False
            next_state = self.data[self.step_cnt+1][:-1]
            next_state = torch.unsqueeze(torch.from_numpy(next_state).type(torch.FloatTensor).cuda(), 0)
            self.step_cnt += 1
        else :
            self.done = True
            next_state = None

        reward = torch.FloatTensor([reward]).cuda()
        return next_state, reward, self.done
