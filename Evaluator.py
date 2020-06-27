import torch
import numpy as np
import gym
from Model import PolicyNet, QNet
import seaborn as sns
import pandas as pd
import copy
from functools import reduce
import matplotlib.pyplot as plt
import time


def plot_online(env_name, last_method_idx, Method_Name, max_state):
    # make a total dataframe
    df_list_evaluation = []
    df_list_performance = []
    init_method = 0
    for method_idx in range(init_method, last_method_idx + 1, 1):
        evaluated_Q_mean = np.load('./' + env_name + '/method_' + str(method_idx)
                                   + '/result/evaluated_Q_mean.npy', allow_pickle=True)
        evaluated_Q_std = np.load('./' + env_name + '/method_' + str(method_idx)
                                  + '/result/evaluated_Q_std.npy', allow_pickle=True)
        true_gamma_return_mean = np.load('./' + env_name + '/method_' + str(method_idx)
                                         + '/result/true_gamma_return_mean.npy', allow_pickle=True)
        iteration = np.load('./' + env_name + '/method_' + str(method_idx) + '/result/iteration.npy')/1000000
        time = np.load('./' + env_name + '/method_' + str(method_idx) + '/result/time.npy')
        average_return_with_diff_base = np.load('./' + env_name + '/method_'
                                                + str(method_idx) + '/result/average_return_with_diff_base.npy')

        average_return_max_best = list(map(lambda x: x[0], average_return_with_diff_base))
        average_return_max_better = list(map(lambda x: x[1], average_return_with_diff_base))
        average_return_max_all = list(map(lambda x: x[2], average_return_with_diff_base))
        alpha = np.load('./' + env_name + '/method_' + str(method_idx) + '/result/alpha.npy')
        a_std = np.load('./' + env_name + '/method_' + str(method_idx) + '/result/a_std.npy')
        a_std=np.mean(a_std,axis=1)
        a_abs = np.load('./' + env_name + '/method_' + str(method_idx) + '/result/a_abs.npy')
        a_abs=np.mean(a_abs,axis=1)
        policy_entropy=np.load('./' + env_name + '/method_' + str(method_idx) + '/result/policy_entropy.npy')

        method_name = Method_Name[method_idx]




        df_for_this_method_performance = pd.DataFrame(dict(method_name=method_name,
                                                           iteration=iteration,
                                                           time=time,
                                                           average_return=average_return_max_better,
                                                           Q_std=evaluated_Q_std,
                                                           alpha=alpha*100,
                                                           policy_entropy=policy_entropy,))

        df_for_this_method_1 = pd.DataFrame(dict(method_name=method_name,
                                                 iteration=iteration,
                                                 Q=evaluated_Q_mean,
                                                 is_true='estimation',
                                                 action=a_abs,
                                                 a_type='abs',))
        df_for_this_method_2 = pd.DataFrame(dict(method_name=method_name,
                                                 iteration=iteration,
                                                 Q=true_gamma_return_mean,
                                                 is_true='ground truth',
                                                 action = a_std,
                                                 a_type='std',))

        df_for_this_method = df_for_this_method_1.append(df_for_this_method_2, ignore_index=True)
        df_list_evaluation.append(df_for_this_method)
        df_list_performance.append(df_for_this_method_performance)
    total_dataframe_evaluation = df_list_evaluation[0].append(df_list_evaluation[1:],
                                                              ignore_index=True) if last_method_idx > init_method else \
    df_list_evaluation[0]
    total_dataframe_performance = df_list_performance[0].append(df_list_performance[1:],
                                                                ignore_index=True) if last_method_idx > init_method else \
    df_list_performance[0]

    f1 = plt.figure(1, figsize=(12, 8))

    plt.subplot(321)
    sns.lineplot(x="iteration", y="average_return", hue="method_name", data=total_dataframe_performance, legend=False)
    plt.title(env_name)

    plt.subplot(322)
    sns.lineplot(x="iteration", y="Q", hue="method_name", style="is_true", data=total_dataframe_evaluation,
                 legend=False)

    plt.subplot(323)
    sns.lineplot(x="iteration", y="Q_std", hue="method_name", data=total_dataframe_performance, legend=False)

    plt.subplot(324)
    sns.lineplot(x="iteration", y="action", hue="method_name", style="a_type", data=total_dataframe_evaluation, legend=False)

    plt.subplot(325)
    sns.lineplot(x="iteration", y="policy_entropy", hue="method_name", data=total_dataframe_performance, legend=False)

    plt.subplot(326)
    sns.lineplot(x="iteration", y="time", hue="method_name", data=total_dataframe_performance, legend='brief')
    sns.lineplot(x="iteration", y="alpha", hue="method_name", data=total_dataframe_performance, legend=False)
    plt.ylabel('100*alpha & time')

    plt.pause(10)
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

        self.evaluation_interval = 20000
        self.max_state_num_evaluated_in_an_episode = 200
        self.episode_num_evaluation = 10
        self.episode_num_test = 5
        self.time = time.time()
        self.list_of_n_episode_rewards_history = []
        self.time_history = []
        self.alpha_history = []
        self.average_return_with_diff_base_history = []
        self.average_reward_history = []
        self.iteration_history = []
        self.evaluated_Q_mean_history = []
        self.evaluated_Q_std_history = []
        self.true_gamma_return_mean_history = []
        self.policy_entropy_history =[]
        self.a_std_history = []
        self.a_abs_history = []


        # self.n_episodes_info_history = []

    def average_max_n(self, list_for_average, n):
        sorted_list = sorted(list_for_average, reverse=True)
        return sum(sorted_list[:n]) / n

    def run_an_episode(self,deterministic,mode):
        #state_list = []
        action_list = []
        log_prob_list = []
        reward_list = []
        evaluated_Q_list = []
        Q_std_list = []
        a_std_list =[]
        done = 0
        state = self.env.reset()
        while not done and len(reward_list) < self.args.max_step:
            state_tensor = torch.FloatTensor(state.copy()).float().to(self.device)
            if self.args.NN_type == "CNN":
                state_tensor = state_tensor.permute(2, 0, 1)
            u, log_prob, a_std = self.actor.get_action_test(state_tensor.unsqueeze(0), deterministic)
            #state_list.append(state.copy())
            log_prob_list.append(log_prob)
            a_std_list.append(a_std)
            if self.args.double_Q and not self.args.double_actor:
                q = torch.min(
                    self.Q_net1.evaluate(state_tensor.unsqueeze(0), torch.FloatTensor(u.copy()).to(self.device))[0],
                    self.Q_net2.evaluate(state_tensor.unsqueeze(0), torch.FloatTensor(u.copy()).to(self.device))[0])
            else:
                q, q_std, _ = self.Q_net1.evaluate(state_tensor.unsqueeze(0),
                                                   torch.FloatTensor(u.copy()).to(self.device))
            evaluated_Q_list.append(q.detach().item())
            if self.args.distributional_Q:
                Q_std_list.append(q_std.detach().item())
            else:
                Q_std_list.append(0)
            u = u.squeeze(0)
            state, reward, done, load_action = self.env.step(u)
            # self.env.render(mode='human')
            action_list.append(u)
            reward_list.append(reward * self.args.reward_scale)
        if mode=="Evaluation":
            entropy_list = list(-self.alpha * np.array(log_prob_list))
            true_gamma_return_list = cal_gamma_return_of_an_episode(reward_list, entropy_list, self.args.gamma)
            policy_entropy = -sum(log_prob_list) / len(log_prob_list)
            a_std_mean=np.mean(np.array(a_std_list),axis=0)
            a_abs_mean = np.mean(np.abs(np.array(action_list)),axis=0)
            return dict(#state_list=np.array(state_list),
                        #action_list=np.array(action_list),
                        log_prob_list=np.array(log_prob_list),
                        policy_entropy = policy_entropy,
                        #reward_list=np.array(reward_list),
                        a_std_mean=a_std_mean,
                        a_abs_mean=a_abs_mean,
                        evaluated_Q_list=np.array(evaluated_Q_list),
                        Q_std_list=np.array(Q_std_list),
                        true_gamma_return_list=true_gamma_return_list,)
        elif mode=="Performance":
            episode_return = sum(reward_list) / self.args.reward_scale
            episode_len = len(reward_list)
            return dict(episode_return=episode_return,
                        episode_len=episode_len)

    def run_n_episodes(self, n, max_state, deterministic, mode):
        n_episode_state_list = []
        n_episode_action_list = []
        n_episode_log_prob_list = []
        n_episode_reward_list = []
        n_episode_evaluated_Q_list = []
        n_episode_Q_std_list = []
        n_episode_true_gamma_return_list = []
        n_episode_return_list = []
        n_episode_len_list = []
        n_episode_policyentropy_list=[]
        n_episode_a_std_list=[]
        for _ in range(n):
            episode_info = self.run_an_episode(deterministic,mode)
            # n_episode_state_list.append(episode_info['state_list'])
            # n_episode_action_list.append(episode_info['action_list'])
            # n_episode_log_prob_list.append(episode_info['log_prob_list'])
            #n_episode_reward_list.append(episode_info['reward_list'])
            if mode == "Evaluation":
                n_episode_evaluated_Q_list.append(episode_info['evaluated_Q_list'])
                n_episode_Q_std_list.append(episode_info['Q_std_list'])
                n_episode_true_gamma_return_list.append(episode_info['true_gamma_return_list'])
                n_episode_policyentropy_list.append(episode_info['policy_entropy'])
                n_episode_a_std_list.append(episode_info['a_std_mean'])
                n_episode_action_list.append(episode_info['a_abs_mean'])
            elif mode == "Performance":
                n_episode_return_list.append(episode_info['episode_return'])
                n_episode_len_list.append(episode_info['episode_len'])
        if mode == "Evaluation":
            average_policy_entropy= sum(n_episode_policyentropy_list) / len(n_episode_policyentropy_list)
            average_a_std=np.mean(np.array(n_episode_a_std_list), axis=0)
            average_a_abs = np.mean(np.array(n_episode_action_list), axis=0)
            # n_episode_evaluated_Q_list_history = list(map(lambda x: x['n_episode_evaluated_Q_list'], n_episodes_info_history))
            # n_episode_true_gamma_return_list_history = list(map(lambda x: x['n_episode_true_gamma_return_list'], n_episodes_info_history))

            def concat_interest_epi_part_of_one_ite_and_mean(list_of_n_epi):
                tmp = list(copy.deepcopy(list_of_n_epi))
                tmp[0] = tmp[0] if len(tmp[0]) <= max_state else tmp[0][:max_state]

                def reduce_fuc(a, b):
                    return np.concatenate([a, b]) if len(b) < max_state else np.concatenate([a, b[:max_state]])

                interest_epi_part_of_one_ite = reduce(reduce_fuc, tmp)
                return sum(interest_epi_part_of_one_ite) / len(interest_epi_part_of_one_ite)

            evaluated_Q_mean = concat_interest_epi_part_of_one_ite_and_mean(np.array(n_episode_evaluated_Q_list))
            evaluated_Q_std = concat_interest_epi_part_of_one_ite_and_mean(np.array(n_episode_Q_std_list))
            true_gamma_return_mean = concat_interest_epi_part_of_one_ite_and_mean(
                np.array(n_episode_true_gamma_return_list))

            return dict(evaluated_Q_mean=evaluated_Q_mean,
                        true_gamma_return_mean=true_gamma_return_mean,
                        evaluated_Q_std=evaluated_Q_std,
                        n_episode_reward_list=np.array(n_episode_reward_list),
                        policy_entropy=average_policy_entropy,
                        a_std=average_a_std,
                        a_abs=average_a_abs)
        elif mode=="Performance":
            average_return_with_diff_base = np.array([self.average_max_n(n_episode_return_list, x) for x in
                                                      [1, self.episode_num_test - 2, self.episode_num_test]])
            average_reward = sum(n_episode_return_list) / sum(n_episode_len_list)
            return dict(n_episode_reward_list=np.array(n_episode_reward_list),
                        average_return_with_diff_base=average_return_with_diff_base,
                        average_reward=average_reward,)

    def run(self):
        while not self.stop_sign.value:
            if self.iteration_counter.value % self.evaluation_interval == 0:
                self.alpha = np.exp(self.log_alpha_share.detach().item()) if self.args.alpha == 'auto' else 0
                self.iteration = self.iteration_counter.value
                self.actor.load_state_dict(self.actor_share.state_dict())
                self.Q_net1.load_state_dict(self.Q_net1_share.state_dict())
                self.Q_net2.load_state_dict(self.Q_net2_share.state_dict())

                delta_time = time.time() - self.time
                self.time = time.time()
                if self.args.stochastic_actor:
                    n_episode_info = self.run_n_episodes(self.episode_num_evaluation,self.max_state_num_evaluated_in_an_episode, False, mode="Evaluation")
                else:
                    n_episode_info = self.run_n_episodes(self.episode_num_evaluation,self.max_state_num_evaluated_in_an_episode, True, mode="Evaluation")
                self.iteration_history.append(self.iteration)
                self.evaluated_Q_mean_history.append(n_episode_info['evaluated_Q_mean'])
                self.evaluated_Q_std_history.append(n_episode_info['evaluated_Q_std'])
                self.true_gamma_return_mean_history.append(n_episode_info['true_gamma_return_mean'])
                self.time_history.append(delta_time)
                # self.list_of_n_episode_rewards_history.append(list_of_n_episode_rewards)
                self.alpha_history.append(self.alpha)
                self.policy_entropy_history.append(n_episode_info['policy_entropy'])
                self.a_std_history.append(n_episode_info['a_std'])
                self.a_abs_history.append(n_episode_info['a_abs'])
                n_episode_info_test = self.run_n_episodes(self.episode_num_test, self.max_state_num_evaluated_in_an_episode, True, mode="Performance")
                self.average_return_with_diff_base_history.append(n_episode_info_test['average_return_with_diff_base'])
                self.average_reward_history.append(n_episode_info_test['average_reward'])

                print('Saving evaluation results of the {} iteration.'.format(self.iteration))
                np.save('./' + self.args.env_name + '/method_' + str(self.args.method) + '/result/iteration',
                        np.array(self.iteration_history))
                np.save('./' + self.args.env_name + '/method_' + str(self.args.method) + '/result/evaluated_Q_mean',
                        np.array(self.evaluated_Q_mean_history))
                np.save('./' + self.args.env_name + '/method_' + str(self.args.method) + '/result/evaluated_Q_std',
                        np.array(self.evaluated_Q_std_history))
                np.save('./' + self.args.env_name + '/method_' + str(
                    self.args.method) + '/result/true_gamma_return_mean',
                        np.array(self.true_gamma_return_mean_history))
                np.save('./' + self.args.env_name + '/method_' + str(self.args.method) + '/result/time',
                        np.array(self.time_history))
                # np.save('./' + self.args.env_name + '/method_' + str(self.args.method) + '/result/list_of_n_episode_rewards',
                #         np.array(self.list_of_n_episode_rewards_history))
                np.save('./' + self.args.env_name + '/method_' + str(
                    self.args.method) + '/result/average_return_with_diff_base',
                        np.array(self.average_return_with_diff_base_history))
                np.save('./' + self.args.env_name + '/method_' + str(self.args.method) + '/result/average_reward',
                        np.array(self.average_reward_history))
                np.save('./' + self.args.env_name + '/method_' + str(self.args.method) + '/result/alpha',
                        np.array(self.alpha_history))
                np.save('./' + self.args.env_name + '/method_' + str(self.args.method) + '/result/policy_entropy',
                        np.array(self.policy_entropy_history))
                np.save('./' + self.args.env_name + '/method_' + str(self.args.method) + '/result/a_std',
                        np.array(self.a_std_history))
                np.save('./' + self.args.env_name + '/method_' + str(self.args.method) + '/result/a_abs',
                        np.array(self.a_abs_history))

                plot_online(self.args.env_name, self.args.method, self.args.method_name,
                            self.max_state_num_evaluated_in_an_episode)

                if self.iteration >= self.args.max_train:
                    self.stop_sign.value = 1
                    break


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
