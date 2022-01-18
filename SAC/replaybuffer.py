import numpy as np
import torch
import torch.nn as nn
import sys
import kornia



class ReplayBufer():
    def __init__(self, max_size, act_dim, state_dim, device, image_pad=4, enc_type='Linear', aug_flag=False):
            
        self.max_size = max_size
        self.aug_flag = aug_flag
        self.enc_type = enc_type
        
        # self.aug = nn.Sequential(
        #     nn.ReplicationPad2d(image_pad),
        #     kornia.augmentation.RandomCrop((state_dim[-1], state_dim[-1]))
        # )

        if enc_type == 'Conv' or enc_type == 'PointCloud':
            self.state_buf = np.zeros((max_size, *state_dim), dtype=np.float32)
            self.next_state_buf = np.zeros((max_size, *state_dim), dtype=np.float32)
        else:
            self.state_buf = np.zeros((max_size, state_dim), dtype=np.float32)
            self.next_state_buf = np.zeros((max_size, state_dim), dtype=np.float32)
            
        self.act_buf = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(max_size, dtype=np.float32)
        self.done_buf = np.zeros(max_size, dtype=np.float32)
        self.ptr = 0
        self.device = device
        

    def store(self, state, act, rew, next_state, done):
        index = self.ptr % self.max_size
        self.state_buf[index] = state
        self.act_buf[index] = act
        self.rew_buf[index] = rew
        self.next_state_buf[index] = next_state
        self.done_buf[index] = done
        self.ptr += 1
        

    def sample_batch(self, batch_size):
        max_buf = min(self.ptr, self.max_size)
        idxs = np.random.choice(max_buf, batch_size)

        states = torch.as_tensor(self.state_buf[idxs], dtype=torch.float32, device=self.device)
        # states = self.aug(states)
        actions = torch.as_tensor(self.act_buf[idxs], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(self.rew_buf[idxs], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(self.next_state_buf[idxs], dtype=torch.float32, device=self.device)
        # next_states = self.aug(next_states)
        dones = torch.as_tensor(self.done_buf[idxs], dtype=torch.float32, device=self.device)

        return states, actions, rewards, next_states, dones
