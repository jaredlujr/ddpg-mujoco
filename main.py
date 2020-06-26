"""Reinforment learning DDPG-mujoco
CS489 project
Author: Lu Jiarui 
Date: 2020/06/11
"""
from __future__ import print_function

import argparse
import torch

from train import train, test 

parser = argparse.ArgumentParser(description='DDPG-Mujoco')
parser.add_argument('--env-name', type=str, default='Ant-v2',
                    help='environment to train on or test (default: Ant-v2')
parser.add_argument('--do-test', type=bool, default=True,
                    help='do test and load local checkpoint')
parser.add_argument('--actor-checkpoint', type=str, default='ddpg-Actor-Ant-v2-500.ckpt', 
                    help='initial checkpoint of DDPG model, pytorch chenckpoint')
parser.add_argument('--critic-checkpoint', type=str, default='ddpg-Critic-Ant-v2-500.ckpt', 
                    help='initial checkpoint of DDPG model, pytorch chenckpoint')

if __name__ == "__main__":
    args = parser.parse_args()
    if not args.do_test:
        print('[INFO] Entering training subroutine')
        if args.actor_checkpoint:
            train(args.env_name, args.actor_checkpoint, args.critic_checkpoint)
        else:
            train(args.env_name)
    if args.do_test:
        print('[INFO] Entering testing subroutine')
        test('Ant-v2', args.actor_checkpoint, args.critic_checkpoint)
        