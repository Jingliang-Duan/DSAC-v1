import torch
import numpy as np
import gym
from Model import PolicyNet, QNet
import seaborn as sns
import pandas as pd
import copy
from functools import reduce
import matplotlib.pyplot as plt


def plot_online(env_name, last_method_idx, Method_Name, max_state):
    # make a total dataframe
    df_list = []
    init_method = 0
    for method_idx in range(init_method, last_method_idx + 1, 1):
        df_list_for_this_method = []
        iteration = np.load('./' + env_name + '/method_' + str(method_idx) + '/result/iteration_evaluation.npy')
        evaluated_Q_mean = np.load('./' + env_name + '/method_' + str(method_idx)
                                      + '/result/evaluated_Q_mean.npy', allow_pickle=True)
        true_gamma_return_mean = np.load('./' + env_name + '/method_' + str(method_idx)
                                            + '/result/true_gamma_return_mean.npy', allow_pickle=True)

        method_name = Method_Name[method_idx]
        df_for_this_method_1 = pd.DataFrame(dict(method_name=method_name,
                                                 iteration=np.array(iteration),
                                                 Q=np.array(evaluated_Q_mean),
                                                 is_true='estimation'))
        df_for_this_method_2 = pd.DataFrame(dict(method_name=method_name,
                                                 iteration=np.array(iteration),
                                                 Q=np.array(true_gamma_return_mean),
                                                 is_true='ground truth'))

        df_for_this_method = df_for_this_method_1.append(df_for_this_method_2, ignore_index=True)
        df_list.append(df_for_this_method)
    total_dataframe = df_list[0].append(df_list[1:], ignore_index=True) if last_method_idx > init_method else df_list[0]
    f1 = plt.figure(1)
    sns.lineplot(x="iteration", y="Q", hue="method_name", style="is_true", data=total_dataframe)
    plt.title(env_name)

    plt.pause(5)
    f1.clf()


def cal_gamma_return_of_an_episode(reward_list, entropy_list, gamma):
    n = len(reward_list)
    gamma_list = np.array([np.power(gamma, i) for i in range(n)])
    reward_list = np.array(reward_list)
    entropy_list = np.array(entropy_list)
    gamma_return = np.array([sum(reward_list[i:] * gamma_list[:(n - i)]) for i in range(n)])
    gamma_list_for_entropy = np.array([0 if i == 0 else np.power(gamma, i) for i in range(n)])
    gamma_return_for_entropy = np.array([sum(entropy_list[i:] * gamma_list_for_entropy[:(n - i)]) for i in range(n)])
    return gamma_return + gamma_return_for_entropy


class Evaluator(object):
    def __init__(self, args, shared_value, share_net):
        seed = args.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.stop_sign = shared_value[1]
        self.iteration_counter = shared_value[2]
        self.iteration = self.iteration_counter.value
        self.share_net = share_net
        self.args = args
        self.env = gym.make(args.env_name)
        self.device = torch.device("cpu")
        self.actor = PolicyNet(args).to(self.device)
        self.Q_net1 = QNet(args).to(self.device)
        self.Q_net2 = QNet(args).to(self.device)
        self.actor_share = share_net[4]
        self.Q_net1_share = share_net[0]
        self.Q_net2_share = share_net[2]
        self.log_alpha_share = share_net[-1]
        self.alpha = np.exp(self.log_alpha_share.detach().item()) if args.alpha == 'auto' else 0

        self.evaluation_interval = 50000
        self.max_state_num_evaluated_in_an_episode = 500
        self.episode_num_to_run = 10
        self.iteration_history = []
        self.evaluated_Q_mean_history=[]
        self.true_gamma_return_mean_history=[]
        # self.n_episodes_info_history = []
        self.evaluated_Q_history = []
        self.true_gamma_return_history = []

    def run_an_episode(self):
        state_list = []
        action_list = []
        log_prob_list = []
        reward_list = []
        evaluated_Q_list = []
        done = 0
        state = self.env.reset()
        while not done and len(reward_list) < self.args.max_step:
            state_tensor = torch.FloatTensor(state.copy()).float().to(self.device)
            u, log_prob = self.actor.get_action(state_tensor.unsqueeze(0), self.args.stochastic_actor)
            state_list.append(state.copy())
            action_list.append(u.copy())
            log_prob_list.append(log_prob)
            if self.args.double_Q and not self.args.double_actor:
                q = torch.min(
                    self.Q_net1(state_tensor.unsqueeze(0), torch.FloatTensor(u.copy()).to(self.device))[0],
                    self.Q_net2(state_tensor.unsqueeze(0), torch.FloatTensor(u.copy()).to(self.device))[0])
            else:
                q = self.Q_net1(state_tensor.unsqueeze(0), torch.FloatTensor(u.copy()).to(self.device))[0]
            evaluated_Q_list.append(q.detach().item())
            u = u.squeeze(0)
            state, reward, done, load_action = self.env.step(u)
            # self.env.render(mode='human')
            reward_list.append(reward * self.args.reward_scale)
        entropy_list = list(-self.alpha * np.array(log_prob_list))
        true_gamma_return_list = cal_gamma_return_of_an_episode(reward_list, entropy_list, self.args.gamma)
        episode_return = sum(reward_list)
        episode_len = len(reward_list)

        return dict(state_list=np.array(state_list),
                    action_list=np.array(action_list),
                    log_prob_list=np.array(log_prob_list),
                    reward_list=np.array(reward_list),
                    evaluated_Q_list=np.array(evaluated_Q_list),
                    true_gamma_return_list=true_gamma_return_list,
                    episode_return=episode_return,
                    episode_len=episode_len)

    def run_n_episodes(self, n, max_state):
        n_episode_state_list = []
        n_episode_action_list = []
        n_episode_log_prob_list = []
        n_episode_reward_list = []
        n_episode_evaluated_Q_list = []
        n_episode_true_gamma_return_list = []
        n_episode_return_list = []
        n_episode_len_list = []
        for _ in range(n):
            episode_info = self.run_an_episode()
            n_episode_state_list.append(episode_info['state_list'])
            n_episode_action_list.append(episode_info['action_list'])
            n_episode_log_prob_list.append(episode_info['log_prob_list'])
            n_episode_reward_list.append(episode_info['reward_list'])
            n_episode_evaluated_Q_list.append(episode_info['evaluated_Q_list'])
            n_episode_true_gamma_return_list.append(episode_info['true_gamma_return_list'])
            n_episode_return_list.append(episode_info['episode_return'])
            n_episode_len_list.append(episode_info['episode_len'])

        #n_episode_evaluated_Q_list_history = list(map(lambda x: x['n_episode_evaluated_Q_list'], n_episodes_info_history))
        #n_episode_true_gamma_return_list_history = list(map(lambda x: x['n_episode_true_gamma_return_list'], n_episodes_info_history))

        def concat_interest_epi_part_of_one_ite_and_mean(list_of_n_epi):
            tmp = list(copy.deepcopy(list_of_n_epi))
            tmp[0] = tmp[0] if len(tmp[0]) <= max_state else tmp[0][:max_state]

            def reduce_fuc(a, b):
                return np.concatenate([a, b]) if len(b) < max_state else np.concatenate([a, b[:max_state]])

            interest_epi_part_of_one_ite = reduce(reduce_fuc, tmp)
            return sum(interest_epi_part_of_one_ite) / len(interest_epi_part_of_one_ite)

        evaluated_Q_mean = concat_interest_epi_part_of_one_ite_and_mean(np.array(n_episode_evaluated_Q_list))

        true_gamma_return_mean = concat_interest_epi_part_of_one_ite_and_mean(
            np.array(n_episode_true_gamma_return_list))
        return evaluated_Q_mean, true_gamma_return_mean
        # return dict(n_episode_state_list=np.array(n_episode_state_list),
        #             n_episode_action_list=np.array(n_episode_action_list),
        #             n_episode_log_prob_list=np.array(n_episode_log_prob_list),
        #             n_episode_reward_list=np.array(n_episode_reward_list),
        #             n_episode_evaluated_Q_list=np.array(n_episode_evaluated_Q_list),
        #             n_episode_true_gamma_return_list=np.array(n_episode_true_gamma_return_list),
        #             n_episode_return_list=np.array(n_episode_return_list),
        #             n_episode_len_list=np.array(n_episode_len_list))

    def run(self):
        while not self.stop_sign.value:
            if self.iteration_counter.value % self.evaluation_interval == 0:
                self.alpha = np.exp(self.log_alpha_share.detach().item()) if self.args.alpha == 'auto' else 0
                self.iteration = self.iteration_counter.value
                self.actor.load_state_dict(self.actor_share.state_dict())
                self.Q_net1.load_state_dict(self.Q_net1_share.state_dict())
                self.Q_net2.load_state_dict(self.Q_net2_share.state_dict())
                evaluated_Q_mean, true_gamma_return_mean = self.run_n_episodes(self.episode_num_to_run,self.max_state_num_evaluated_in_an_episode)
                self.iteration_history.append(self.iteration)
                self.evaluated_Q_mean_history.append(evaluated_Q_mean)
                self.true_gamma_return_mean_history.append(true_gamma_return_mean)
                print('Saving evaluation results of the {} iteration.'.format(self.iteration))
                np.save('./' + self.args.env_name + '/method_' + str(self.args.method) + '/result/iteration_evaluation',
                        np.array(self.iteration_history))
                np.save('./' + self.args.env_name + '/method_' + str(self.args.method) + '/result/evaluated_Q_mean',
                        np.array(self.evaluated_Q_mean_history))
                np.save('./' + self.args.env_name + '/method_' + str(
                    self.args.method) + '/result/true_gamma_return_mean',
                        np.array(self.true_gamma_return_mean_history))

                plot_online(self.args.env_name, self.args.method, self.args.method_name,
                            self.max_state_num_evaluated_in_an_episode)


def test():
    print(cal_gamma_return_of_an_episode([1, 2, 4], [2, 3, 4], 0.9))


def test_plot_online():
    for i in range(4):
        iteration = np.array(range(60))

        n_episodes_info_history = []
        for _ in iteration:
            estimated_q = []
            true_q = []
            for _ in range(10):
                len = np.random.randint(100, 700)
                estimated_q.append(np.random.random(len))
                true_q.append(np.random.random(len))
            n_episodes_info_history.append(dict(n_episode_evaluated_Q_list=np.array(estimated_q),
                                                n_episode_true_gamma_return_list=np.array(true_q)))

        np.save('./' + 'test_data' + '/method_' + str(i) + '/result/iteration_evaluation',
                np.array(iteration))
        np.save('./' + 'test_data' + '/method_' + str(i) + '/result/n_episodes_info_history',
                np.array(n_episodes_info_history))

    method_name = {0: 'DSAC-10', 1: 'DSAC-20', 2: 'SAC', 3: 'Double-Q SAC',
                   4: 'TD3', 5: 'DDPG', 6: 'TD4', 7: 'DSAC-50'}

    for _ in range(3):
        env_name = 'test_data'
        plot_online(env_name, _, method_name, 500)


if __name__ == '__main__':
    test_plot_online()
