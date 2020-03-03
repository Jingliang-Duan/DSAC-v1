from __future__ import print_function
import torch
import numpy as np
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue
import time
from Model import PolicyNet
import gym
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_online(env_name, last_method_idx, Method_Name):
    # make a total dataframe
    df_list = []
    init_method = 0
    for method_idx in range(init_method,last_method_idx + 1,1):
        iteration = np.load('./' + env_name + '/method_' + str(method_idx) + '/result/iteration.npy')
        time = np.load('./' + env_name + '/method_' + str(method_idx) + '/result/time.npy')
        average_return_with_diff_base = np.load('./' + env_name + '/method_'
                                                + str(method_idx) + '/result/average_return_with_diff_base.npy')
        average_return_max_1 = list(map(lambda x: x[0], average_return_with_diff_base))
        average_return_max_3 = list(map(lambda x: x[1], average_return_with_diff_base))
        average_return_max_5 = list(map(lambda x: x[2], average_return_with_diff_base))
        alpha = np.load('./' + env_name + '/method_' + str(method_idx) + '/result/alpha.npy')
        method_name = Method_Name[method_idx]
        method_name = [method_name] * iteration.shape[0]

        df_for_this_method = pd.DataFrame(dict(method_name=method_name,
                                               iteration=iteration,
                                               time=time,
                                               average_return=average_return_max_3,
                                               alpha=alpha))
        df_list.append(df_for_this_method)
    total_dataframe = df_list[0].append(df_list[1:], ignore_index=True) if last_method_idx > init_method else df_list[0]
    f1 = plt.figure(1)
    plt.subplot(211)
    sns.lineplot(x="iteration", y="average_return", hue="method_name", data=total_dataframe)
    plt.title(env_name)

    plt.subplot(212)
    sns.lineplot(x="iteration", y="alpha", hue="method_name", data=total_dataframe)

    plt.pause(10)
    f1.clf()

class Test():
    def __init__(self, args, shared_value, share_net):
        super(Test, self).__init__()
        seed = args.seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.stop_sign = shared_value[1]
        self.iteration_counter = shared_value[2]
        self.iteration = self.iteration_counter.value
        self.args = args
        self.env = gym.make(args.env_name)
        self.device = torch.device("cpu")
        self.actor = PolicyNet(args).to(self.device)
        self.actor_share = share_net[0]
        self.log_alpha = share_net[1]

        self.test_step = 0
        self.episode_num = 5
        self.test_interval = 20000
        self.start_time = time.time()
        self.list_of_n_episode_rewards_history = []
        self.time_history = []
        self.alpha_history = []
        self.average_return_with_diff_base_history = []
        self.average_reward_history = []
        self.iteration_history = []



    def run_an_episode(self):
        reward_list = []
        done = 0
        state = self.env.reset()
        while not done and len(reward_list) < self.args.max_step:
            state_tensor = torch.FloatTensor(state.copy()).float().to(self.device)
            u, log_prob = self.actor.get_action(state_tensor.unsqueeze(0), True)
            u = u.squeeze(0)
            state, reward, done, load_action = self.env.step(u)
            #self.env.render(mode='human')
            reward_list.append(reward)
        episode_return = sum(reward_list)
        episode_len = len(reward_list)
        return np.array(reward_list), episode_return, episode_len

    def average_max_n(self, list_for_average, n):
        sorted_list = sorted(list_for_average, reverse=True)
        return sum(sorted_list[:n]) / n

    def run_n_episodes(self, n):
        assert n >= 5, "n must be at least 5"
        list_of_n_episode_rewards = []
        list_of_return = []
        list_of_len = []
        for _ in range(n):
            reward_list, episode_return, episode_len = self.run_an_episode()
            list_of_n_episode_rewards.append(self.run_an_episode())
            list_of_return.append(episode_return)
            list_of_len.append(episode_len)
        average_return_with_diff_base = np.array([self.average_max_n(list_of_return, x) for x in [1, 3, 5]])
        average_reward = sum(list_of_return)/sum(list_of_len)
        return np.array(list_of_n_episode_rewards), average_return_with_diff_base, average_reward

    def run(self):
        while not self.stop_sign.value:
            if self.iteration_counter.value % self.test_interval == 0:
                self.iteration = self.iteration_counter.value
                self.actor.load_state_dict(self.actor_share.state_dict())
                delta_time = time.time() - self.start_time
                list_of_n_episode_rewards, average_return_with_diff_base, average_reward = self.run_n_episodes(self.episode_num)
                self.iteration_history.append(self.iteration)
                self.time_history.append(delta_time)
                self.list_of_n_episode_rewards_history.append(list_of_n_episode_rewards)
                self.average_return_with_diff_base_history.append(average_return_with_diff_base)
                self.average_reward_history.append(average_reward)
                self.alpha_history.append(self.log_alpha.detach().exp().item())
                print('Saving test data of the {} iteration.'.format(self.iteration))
                np.save('./' + self.args.env_name + '/method_' + str(self.args.method) + '/result/iteration',
                        np.array(self.iteration_history))
                np.save('./' + self.args.env_name + '/method_' + str(self.args.method) + '/result/time',
                        np.array(self.time_history))
                np.save('./' + self.args.env_name + '/method_' + str(self.args.method) + '/result/list_of_n_episode_rewards',
                        np.array(self.list_of_n_episode_rewards_history))
                np.save('./' + self.args.env_name + '/method_' + str(self.args.method) + '/result/average_return_with_diff_base',
                        np.array(self.average_return_with_diff_base_history))
                np.save('./' + self.args.env_name + '/method_' + str(self.args.method) + '/result/average_reward',
                        np.array(self.average_reward_history))
                np.save('./' + self.args.env_name + '/method_' + str(self.args.method) + '/result/alpha',
                        np.array(self.alpha_history))

                plot_online(self.args.env_name, self.args.method, self.args.method_name)

                if self.iteration >= self.args.max_train:
                    self.stop_sign.value = 1
                    break


def test():
    a = torch.tensor([1,-1,1.])
    print(torch.abs(a))



if __name__ == "__main__":
    test()



