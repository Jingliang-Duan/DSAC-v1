from __future__ import print_function
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue
import argparse
import random
import os
import time
from Actor import Actor
from Learner import Learner
from Evaluator import Evaluator
from Simulation import Simulation
from Buffer import Replay_buffer
from Model import QNet, PolicyNet
import my_optim
import gym

def built_parser(method):
    parser = argparse.ArgumentParser()


    '''Task'''
    parser.add_argument("--env_name", default="Ant-v2")
    #Humanoid-v2 Ant-v2 HalfCheetah-v2 Walker2d-v2 InvertedDoublePendulum-v2
    parser.add_argument('--state_dim', dest='list', type=int, default=[])
    parser.add_argument('--action_dim', type=int, default=[])
    parser.add_argument('--action_high', dest='list', type=float, default=[],action="append")
    parser.add_argument('--action_low', dest='list', type=float, default=[],action="append")
    parser.add_argument("--NN_type", default="mlp", help='mlp or CNN')
    parser.add_argument("--code_model", default="train", help='train or simu')

    '''general hyper-parameters'''
    parser.add_argument('--critic_lr' , type=float, default=0.00008,help='critic learning rate')
    parser.add_argument('--actor_lr', type=float, default=0.00005, help='actor learning rate')
    parser.add_argument('--end_lr', type=float, default=0.000001, help='learning rate at the end point of annealing')
    parser.add_argument('--tau', type=float, default=0.001, help='learning rate for target')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--delay_update', type=int, default=2, help='update interval for policy, target')
    parser.add_argument('--reward_scale', type=float, default=0.2, help='reward = reward_scale * environmental reward ')

    '''hyper-parameters for soft-Q based algorithm'''
    parser.add_argument('--alpha_lr', type=float, default=0.00005,help='learning rate for temperature')
    parser.add_argument('--target_entropy',  default="auto",help="auto or some value such as -2")

    '''hyper-parameters for soft-Q based algorithm'''
    parser.add_argument('--max_step', type=int, default=1000, help='maximum length of an episode')
    parser.add_argument('--buffer_size_max', type=int, default=500000, help='replay memory size')
    parser.add_argument('--initial_buffer_size', type=int, default=2000, help='Learner waits until replay memory stores this number of transition')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_hidden_cell', type=int, default=256)

    '''other setting'''
    parser.add_argument("--max_train", type=int, default=3000000)
    parser.add_argument("--decay_T_max", type=int, default=parser.parse_args().max_train, help='for learning rate annealing')
    parser.add_argument('--load_param_period', type=int, default=20)
    parser.add_argument('--save_model_period', type=int, default=20000)
    parser.add_argument('--init_time', type=float, default=0.00)
    parser.add_argument('--seed', type=int, default=1, help='initial seed (default: 1)')

    '''parallel architecture'''
    parser.add_argument("--num_buffers", type=int, default=3)
    parser.add_argument("--num_learners", type=int, default=4) #note that too many learners may cause bad update for shared network
    parser.add_argument("--num_actors", type=int, default=6)

    '''method list'''
    parser.add_argument("--method", type=int, default=method)


    parser.add_argument('--method_name', type=dict,
                        default={0: 'DSAC', 1: 'SAC', 2: 'Double-Q SAC',
                                 3: 'TD4', 4: 'TD3',5: 'DDPG', 6:'Single-Q SAC'})
    if parser.parse_args().method_name[method] == "DSAC":
        parser.add_argument("--distributional_Q", default=True)
        parser.add_argument("--stochastic_actor", default=True)
        parser.add_argument("--double_Q", default=False)
        parser.add_argument("--double_actor", default=False)
        parser.add_argument("--adaptive_bound", default=False)
        parser.add_argument('--alpha', default="auto", help="auto or some value such as 1")
        parser.add_argument('--TD_bound', type=float, default=10)
        parser.add_argument('--bound',  default=True)
    elif parser.parse_args().method_name[method] == "SAC":
        parser.add_argument("--distributional_Q", default=False)
        parser.add_argument("--stochastic_actor", default=True)
        parser.add_argument("--double_Q", default=True)
        parser.add_argument("--double_actor", default=False)
        parser.add_argument('--alpha', default="auto", help="auto or some value such as 1")
    elif parser.parse_args().method_name[method] == "Double-Q SAC":
        parser.add_argument("--distributional_Q", default=False)
        parser.add_argument("--stochastic_actor", default=True)
        parser.add_argument("--double_Q", default=True)
        parser.add_argument("--double_actor", default=True)
        parser.add_argument('--alpha', default="auto", help="auto or some value such as 1")
    elif parser.parse_args().method_name[method] == "TD4":
        parser.add_argument("--distributional_Q", default=True)
        parser.add_argument("--stochastic_actor", default=False)
        parser.add_argument("--double_Q", default=False)
        parser.add_argument("--double_actor", default=False)
        parser.add_argument('--alpha', default=0, help="auto or some value such as 1")
        parser.add_argument("--adaptive_bound", default=False)
        parser.add_argument("--policy_smooth", default=True)
        parser.add_argument('--TD_bound', type=float, default=10)
        parser.add_argument('--bound',  default=True)
    elif parser.parse_args().method_name[method] == "TD3":
        parser.add_argument("--distributional_Q", default=False)
        parser.add_argument("--stochastic_actor", default=False)
        parser.add_argument("--double_Q", default=True)
        parser.add_argument("--double_actor", default=False)
        parser.add_argument("--policy_smooth", default=True)
        parser.add_argument('--alpha', default=0, help="auto or some value such as 1")
    elif parser.parse_args().method_name[method] == "DDPG":
        parser.add_argument("--distributional_Q", default=False)
        parser.add_argument("--stochastic_actor", default=False)
        parser.add_argument("--double_Q", default=False)
        parser.add_argument("--double_actor", default=False)
        parser.add_argument("--policy_smooth", default=False)
        parser.add_argument('--alpha', default=0, help="auto or some value such as 1")
    if parser.parse_args().method_name[method] == "Single-Q SAC":
        parser.add_argument("--distributional_Q", default=False)
        parser.add_argument("--stochastic_actor", default=True)
        parser.add_argument("--double_Q", default=False)
        parser.add_argument("--double_actor", default=False)
        parser.add_argument('--alpha', default="auto", help="auto or some value such as 1")
    return parser.parse_args()


def actor_agent(args, shared_queue, shared_value,share_net, lock, i):
    actor = Actor(args, shared_queue, shared_value,share_net, lock, i)
    actor.run()

def leaner_agent(args, shared_queue,shared_value,share_net,share_optimizer,device,lock,i):

    leaner = Learner(args, shared_queue,shared_value,share_net,share_optimizer,device,lock,i)
    leaner.run()


def evaluate_agent(args, shared_value, share_net):

    evaluator = Evaluator(args, shared_value, share_net)
    evaluator.run()

def buffer(args, shared_queue, shared_value,i):
    buffer = Replay_buffer(args, shared_queue, shared_value,i)
    buffer.run()

def simu_agent(args, shared_value):
    simu = Simulation(args, shared_value)
    simu.run()

def main(method):
    args = built_parser(method=method)
    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    args.state_dim = state_dim
    args.action_dim = action_dim
    action_high = env.action_space.high
    action_low = env.action_space.low
    args.action_high = action_high.tolist()
    args.action_low = action_low.tolist()
    args.seed = np.random.randint(0,30)
    args.init_time = time.time()

    if args.alpha == 'auto' and args.target_entropy == 'auto' :
        delta_a = np.array(args.action_high, dtype=np.float32)-np.array(args.action_low, dtype=np.float32)
        args.target_entropy = -1*args.action_dim #+ sum(np.log(delta_a/2))

    Q_net1 = QNet(args)
    Q_net1.train()
    Q_net1.share_memory()
    Q_net1_target = QNet(args)
    Q_net1_target.train()
    Q_net1_target.share_memory()
    Q_net2 = QNet(args)
    Q_net2.train()
    Q_net2.share_memory()
    Q_net2_target = QNet(args)
    Q_net2_target.train()
    Q_net2_target.share_memory()
    actor1 = PolicyNet(args)

    actor1.train()
    actor1.share_memory()
    actor1_target = PolicyNet(args)
    actor1_target.train()
    actor1_target.share_memory()
    actor2 = PolicyNet(args)
    actor2.train()
    actor2.share_memory()
    actor2_target = PolicyNet(args)
    actor2_target.train()
    actor2_target.share_memory()


    Q_net1_target.load_state_dict(Q_net1.state_dict())
    Q_net2_target.load_state_dict(Q_net2.state_dict())
    actor1_target.load_state_dict(actor1.state_dict())
    actor2_target.load_state_dict(actor2.state_dict())



    Q_net1_optimizer = my_optim.SharedAdam(Q_net1.parameters(), lr=args.critic_lr)
    Q_net1_optimizer.share_memory()
    Q_net2_optimizer = my_optim.SharedAdam(Q_net2.parameters(), lr=args.critic_lr)
    Q_net2_optimizer.share_memory()
    actor1_optimizer = my_optim.SharedAdam(actor1.parameters(), lr=args.actor_lr)
    actor1_optimizer.share_memory()
    actor2_optimizer = my_optim.SharedAdam(actor2.parameters(), lr=args.actor_lr)
    actor2_optimizer.share_memory()
    log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True)
    log_alpha.share_memory_()
    alpha_optimizer = my_optim.SharedAdam([log_alpha], lr=args.alpha_lr)
    alpha_optimizer.share_memory()

    share_net = [Q_net1,Q_net1_target,Q_net2,Q_net2_target,actor1,actor1_target,actor2,actor2_target,log_alpha]
    share_optimizer=[Q_net1_optimizer,Q_net2_optimizer,actor1_optimizer,actor2_optimizer,alpha_optimizer]

    experience_in_queue = []
    experience_out_queue = []
    for i in range(args.num_buffers):
        experience_in_queue.append(Queue(maxsize=10))
        experience_out_queue.append(Queue(maxsize=10))
    shared_queue = [experience_in_queue, experience_out_queue]
    step_counter = mp.Value('i', 0)
    stop_sign = mp.Value('i', 0)
    iteration_counter = mp.Value('i', 0)
    shared_value = [step_counter, stop_sign,iteration_counter]
    lock = mp.Lock()
    procs=[]
    if args.code_model=="train":
        for i in range(args.num_actors):
            procs.append(Process(target=actor_agent, args=(args, shared_queue, shared_value,[actor1,Q_net1], lock, i)))
        for i in range(args.num_buffers):
            procs.append(Process(target=buffer, args=(args, shared_queue, shared_value,i)))
        procs.append(Process(target=evaluate_agent, args=(args, shared_value, share_net)))
        for i in range(args.num_learners):
            #device = torch.device("cuda")
            device = torch.device("cpu")
            procs.append(Process(target=leaner_agent, args=(args, shared_queue, shared_value,share_net,share_optimizer,device,lock,i)))
    elif args.code_model=="simu":
        procs.append(Process(target=simu_agent, args=(args, shared_value)))

    for p in procs:
        p.start()
    for p in procs:
        p.join()

if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["OMP_NUM_THREADS"] = "1"
    #os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    for i in range(0,7,1):
        main(i)





