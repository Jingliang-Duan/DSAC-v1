from __future__ import print_function
import torch
import numpy as np
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue
import time
from Model import QNet, ValueNet, PolicyNet
from utils import *
import torch.nn as nn
from torch.distributions import Normal

from gym.utils import seeding


class Replay_buffer():
    def __init__(self, args, shared_queue,shared_value,i):
        self.agent_id = i
        self.storage = []
        self.priority_buffer = []
        self.args = args
        seed = self.args.seed
        np.random.seed(seed)

        self.experience_in_queue=shared_queue[0][self.agent_id]
        self.experience_out_queue = shared_queue[1][self.agent_id]

        self.stop_sign = shared_value[1]
        self.ptr = 0


    def push(self, data):
        if len(self.storage) == self.args.buffer_size_max:
            self.storage[int(self.ptr)] = data[0:-1]
            self.priority_buffer[int(self.ptr)] = data[-1]
            self.ptr = (self.ptr + 1) % self.args.buffer_size_max
        else:
            self.storage.append(data[0:-1])
            self.priority_buffer.append(data[-1])

    #self.experience_queue.put((self.counter.value, last_state, u, reward, state, micro_step, done))
    def sample(self, batch_size, epsilon = 1e-6):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        state, a, r, state_next,  done = [],[],[], [],[]
        for i in ind:
            S, A, R, S_N,  D = self.storage[i]
            state.append(np.array(S, copy=False))
            a.append(np.array(A, copy=False))
            r.append(np.array(R, copy=False))
            state_next.append(np.array(S_N, copy=False))
            done.append(np.array(D, copy=False))
        return np.array(state),  np.array(a),  np.array(r), np.array(state_next),  np.array(done)

    def numpy_to_tensor(self,s, a, r, s_next, done):
        s = torch.FloatTensor(s)
        a = torch.FloatTensor(a)
        r = torch.FloatTensor(r)
        s_next = torch.FloatTensor(s_next)
        done = torch.FloatTensor(done)
        return s, a, r, s_next, done

    def run(self):
        while not self.stop_sign.value:
            if not self.experience_in_queue.empty():
                self.push(self.experience_in_queue.get())
            if len(self.storage) <= self.args.initial_buffer_size:
                pass
            else:
                s, a, r, s_next, done = self.sample(self.args.batch_size)
                s, a, r, s_next, done = self.numpy_to_tensor(s, a, r, s_next, done)
                if self.args.NN_type == "CNN":
                    s = s.permute(0,3,1,2)
                    s_next = s_next.permute(0,3,1,2)
                if not self.experience_out_queue.full():
                    self.experience_out_queue.put((s, a, r, s_next, done))

        time.sleep(5)
        while not self.experience_out_queue.empty():
            self.experience_out_queue.get()
        while not self.experience_in_queue.empty():
            self.experience_in_queue.get()

def test():
    def fff(x):
        return x,x+1,x+2
    print(fff(1)[0:2])


if __name__ == "__main__":
    test()
