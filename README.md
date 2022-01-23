# Distributional Soft Actor Critic Algorithm

This repository contains the DSAC algorithm using pytorch, see details in our paper

1) Jingliang Duan, Ynag Guan, Shengbo Eben Li, et al. Distributional Soft Actor-critic: Off-policy Reinforcement Learning for Addressing Value Estimation Errors. IEEE Transactions on Neural Networks and Learning Systems, 2021. 

## Supplementary Materials
see https://arxiv.org/abs/2001.02811 for preprint version
Video: https://youtu.be/TTmYAup79N0 ; https://www.bilibili.com/video/BV1fa4y1h7Mo

##Notes
Our scripts work for Ubuntu, Windows, and MacOS. However, the scripts may fail to work in some ubuntu systems due to different environment configurations. In this case, the code stop at forward() of the NN module due to the utilization of Multiprocessing. We have not identified a specific reason for the weird problem. I guess ubuntu requires some special configuration for running multiprocess within torch. We are actively seeking to resolve this problem.
