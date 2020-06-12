from __future__ import print_function
import numpy as np
import time
import matplotlib as mpl
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from functools import reduce
import copy

sns.set(style="darkgrid")
METHOD_IDX_TO_METHOD_NAME = {0: 'DSAC', 1: 'SAC', 2: 'Double-Q SAC', 3: 'TD4',
                             4: 'TD3', 5: 'DDPG', 6: 'Single-Q SAC'}


def make_a_figure_of_n_runs_for_average_performance(env_name, run_numbers, method_numbers, init_run=0, init_method=0):
    # make a total dataframe
    df_list = []
    init_run = init_run
    init_method = init_method
    last_number_of_each_run = []
    for run_idx_ in range(init_run, init_run + run_numbers, 1):
        last_number_of_each_method = []
        for method_idx in range(init_method, init_method + method_numbers, 1):
            iteration = np.load(
                './' + env_name + '-run' + str(run_idx_) + '/method_' + str(method_idx) + '/result/iteration.npy')
            time = np.load('./' + env_name + '-run' + str(run_idx_) + '/method_' + str(method_idx) + '/result/time.npy')
            average_return_with_diff_base = np.load('./' + env_name + '-run' + str(run_idx_) + '/method_'
                                                    + str(method_idx) + '/result/average_return_with_diff_base.npy')
            average_return_max_best = list(map(lambda x: x[0], average_return_with_diff_base))
            average_return_max_better = list(map(lambda x: x[1], average_return_with_diff_base))
            average_return_max_all = list(map(lambda x: x[2], average_return_with_diff_base))
            last_number_of_each_method.append(average_return_max_better[-1])

            alpha = np.load(
                './' + env_name + '-run' + str(run_idx_) + '/method_' + str(method_idx) + '/result/alpha.npy')

            run_idx = np.ones(shape=iteration.shape, dtype=np.int32) * run_idx_
            method_name = METHOD_IDX_TO_METHOD_NAME[method_idx]
            method_name = [method_name] * iteration.shape[0]

            df_for_this_run_and_method = pd.DataFrame(dict(run_idx=run_idx,
                                                           Algorithms=method_name,
                                                           iteration=(iteration / 10).astype(np.int32) / 100000,
                                                           time=time,
                                                           average_return=average_return_max_better,
                                                           alpha=alpha))
            df_list.append(df_for_this_run_and_method)
        last_number_of_each_run.append(last_number_of_each_method)
    last_for_print = []
    std_for_print = []
    for i in range(method_numbers):
        last_for_print.append(np.mean(list(map(lambda x: x[i], last_number_of_each_run))))
        std_for_print.append(np.std(list(map(lambda x: x[i], last_number_of_each_run))))
    total_dataframe = df_list[0].append(df_list[1:], ignore_index=True) \
        if run_numbers > 1 or method_numbers > 1 else df_list[0]
    f1 = plt.figure(1)
    ax1 = f1.add_axes([0.17, 0.12, 0.8, 0.86])
    sns.lineplot(x="iteration", y="average_return", hue="Algorithms", data=total_dataframe, linewidth=2,
                 palette="bright",
                 hue_order=[METHOD_IDX_TO_METHOD_NAME[0],
                            METHOD_IDX_TO_METHOD_NAME[1],
                            METHOD_IDX_TO_METHOD_NAME[2],
                            METHOD_IDX_TO_METHOD_NAME[6],
                            METHOD_IDX_TO_METHOD_NAME[3],
                            METHOD_IDX_TO_METHOD_NAME[4],
                            METHOD_IDX_TO_METHOD_NAME[5]])

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles[1:], labels=labels[1:], loc='upper left', frameon=False, fontsize=15)

    #ax1.get_legend().remove()

    ax1.set_ylabel('Average Return', fontsize=15)
    ax1.set_xlabel("Million iterations", fontsize=15)
    #plt.xlim(0, 3)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    print(last_for_print)
    print(std_for_print)
    # ax1.set_title(env_name, fontsize=15)

    # f3 = plt.figure(2)
    # sns.lineplot(x="iteration", y="alpha", hue="method_name", MountainCarContinuous-v0=total_dataframe)
    # plt.title(env_name + '_alpha')

    # plt.show()


def make_a_figure_of_n_runs_for_value_estimation(env_name, run_numbers, method_numbers, max_state=500, init_run=0,
                                                 init_method=0):
    # make a total dataframe
    df_list = []
    init_run = init_run
    init_method = init_method
    for run_idx_ in range(init_run, init_run + run_numbers, 1):
        for method_idx in range(init_method, init_method + method_numbers, 1):
            df_list_for_this_method = []
            iteration = np.load('./' + env_name + '-run' + str(run_idx_) + '/method_' + str(
                method_idx) + '/result/iteration.npy')
            evaluated_Q_mean = np.load('./' + env_name + '-run' + str(run_idx_) + '/method_' + str(method_idx)
                                  + '/result/evaluated_Q_mean.npy', allow_pickle=True)
            true_gamma_return_mean = np.load('./' + env_name + '-run' + str(run_idx_) + '/method_' + str(method_idx)
                                        + '/result/true_gamma_return_mean.npy', allow_pickle=True)

            assert len(iteration) == len(evaluated_Q_mean) == len(true_gamma_return_mean)
            method_name = METHOD_IDX_TO_METHOD_NAME[method_idx]

            run_idx = np.ones(shape=np.array(iteration).shape, dtype=np.int32) * run_idx_
            df_for_this_run_and_method_1 = pd.DataFrame({'Algorithms': method_name,
                                                         'iteration': (np.array(iteration) / 10).astype(
                                                             np.int32) / 100000,
                                                         'Average Q-value': np.array(evaluated_Q_mean),
                                                         'difference': np.array(evaluated_Q_mean) - np.array(
                                                             true_gamma_return_mean),
                                                         'Value Type': 'estimated',
                                                         'run_idx': run_idx})
            df_for_this_run_and_method_2 = pd.DataFrame({'Algorithms': method_name,
                                                         'iteration': (np.array(iteration) / 10).astype(
                                                             np.int32) / 100000,
                                                         'Average Q-value': np.array(true_gamma_return_mean),
                                                         'difference': np.array(evaluated_Q_mean) - np.array(
                                                             true_gamma_return_mean),
                                                         'Value Type': 'true',
                                                         'run_idx': run_idx})

            df_for_this_method = df_for_this_run_and_method_1.append(df_for_this_run_and_method_2, ignore_index=True)
            df_list.append(df_for_this_method)
    total_dataframe = df_list[0].append(df_list[1:], ignore_index=True) \
        if run_numbers > 1 or method_numbers > 1 else df_list[0]
    f1 = plt.figure(3)
    ax1 = f1.add_axes([0.155, 0.12, 0.82, 0.86])
    sns.lineplot(x="iteration", y="Average Q-value", hue="Algorithms", style="Value Type", data=total_dataframe,
                 linewidth=2, palette="bright",
                 hue_order=[METHOD_IDX_TO_METHOD_NAME[0],
                            METHOD_IDX_TO_METHOD_NAME[1],
                            METHOD_IDX_TO_METHOD_NAME[2],
                            METHOD_IDX_TO_METHOD_NAME[6],
                            METHOD_IDX_TO_METHOD_NAME[3],
                            METHOD_IDX_TO_METHOD_NAME[4],
                            METHOD_IDX_TO_METHOD_NAME[5]]
                 )

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles[1:], labels=labels[1:], loc='upper left', frameon=False, fontsize=15)
    #ax1.get_legend().remove()
    ax1.set_ylabel('Average Q-value', fontsize=15)
    ax1.set_xlabel("Million iterations", fontsize=15)
    #plt.xlim(0, 3)
    #plt.ylim(-20, 180)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    # ax1.set_title(env_name, fontsize=15)
    f2 = plt.figure(4)
    ax2 = f2.add_axes([0.155, 0.12, 0.82, 0.86])
    sns.lineplot(x="iteration", y="difference", hue="Algorithms", data=total_dataframe, linewidth=2, palette="bright",
                 hue_order=[METHOD_IDX_TO_METHOD_NAME[0],
                            METHOD_IDX_TO_METHOD_NAME[1],
                            METHOD_IDX_TO_METHOD_NAME[2],
                            METHOD_IDX_TO_METHOD_NAME[6],
                            METHOD_IDX_TO_METHOD_NAME[3],
                            METHOD_IDX_TO_METHOD_NAME[4],
                            METHOD_IDX_TO_METHOD_NAME[5]]
                 )

    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles=handles[1:], labels=labels[1:], loc='upper left', frameon=False, fontsize=15)
    #ax2.get_legend().remove()
    ax2.set_ylabel('Average Q-value Estimation Bias', fontsize=15)
    ax2.set_xlabel("Million iterations", fontsize=15)
    #plt.xlim(0, 3)
    #plt.ylim(-40, 80)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    # ax2.set_title(env_name, fontsize=15)


def test_value_plot():
    for run in range(4):
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

            np.save('./' + 'test_data' + '-run' + str(run) + '/method_' + str(i) + '/result/iteration_evaluation.npy',
                    np.array(iteration))
            np.save(
                './' + 'test_data' + '-run' + str(run) + '/method_' + str(i) + '/result/n_episodes_info_history.npy',
                np.array(n_episodes_info_history))


    env_name = 'test_data'
    make_a_figure_of_n_runs_for_value_estimation(env_name, 4, 4)


if __name__ == '__main__':
    env_name = "Ant-v2"

    run_numbers = 1
    method_numbers = 1
    init_run = 0
    init_method = 0
    make_a_figure_of_n_runs_for_value_estimation(env_name, run_numbers, method_numbers, init_run=init_run,
                                                 init_method=init_method)
    make_a_figure_of_n_runs_for_average_performance(env_name, run_numbers, method_numbers, init_run=init_run,
                                                    init_method=init_method)
    plt.show()
