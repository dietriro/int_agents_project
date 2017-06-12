#!/usr/bin/env python

import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
# from keras.engine.training import collect_trainable_weights
import json
import sys
from time import sleep
import scipy.io as sio

from SimulationEnvironment import SimulationEnvironment
from ReplayBuffer import ReplayBuffer
from DDPG_Networks import ActorNetwork, CriticNetwork
from OU import OU
import timeit


OU = OU()  # Ornstein-Uhlenbeck Process


def teach_robot(goal, train_indicator=1):  # 1 means Train, 0 means simply Run
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.5
    TAU = 0.001  # Target Network HyperParameters
    LRA = 0.0005  # Learning rate for Actor
    LRC = 0.005  # Learning rate for Critic
    
    action_dim = 2  # cmd_vel in linear.x and angular.z
    state_dim = 365  # Map
    
    np.random.seed(1337)

    vision = False
    
    EXPLORE = 100000.
    episode_count = 100
    max_steps = 800
    reward = 0
    done = False
    step = 0
    epsilon = 1
    indicator = 0

    loss_all = np.zeros([episode_count, 1])
    reward_all = np.zeros([episode_count, 1])
	
    # Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)
    
    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)  # Create replay buffer
    
    # Generate a new stage environment
    # ToDo: add env
    goals = np.array([[5.0, 5.0],
                      [5.0, 0.5],
                      [1.5, 2.5],
                      [2.5, 5.0],
                      [3.0, 2.0]])
    # goals = np.array([[5.0, 5.0]])
    env = SimulationEnvironment(goals[0])

    # Now load the weight
    # print('Now we load the weight')
    # try:
    #     actor.model.load_weights('actormodel.h5')
    #     critic.model.load_weights('criticmodel.h5')
    #     actor.target_model.load_weights('actormodel.h5')
    #     critic.target_model.load_weights('criticmodel.h5')
    #     print('Weight load successfully')
    # except:
    #     print('Cannot find the weight')

    print('RL of Robot begins...')
    for i in range(episode_count):
    
        print('Episode : ' + str(i) + ' Replay Buffer ' + str(buff.count()))
    
        env.reset()
    
        sleep(0.5)
    
        env.step([0, 0])
        env.set_goal(goals[i*goals.shape[0]/episode_count])
        (s_t, r_t, done) = env.get_state_reward_1d()

        total_loss = 0.
        total_reward = 0.
        for j in range(max_steps):
            loss = 0
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros(action_dim)
            noise_t = np.zeros(action_dim)

            a_t_original = actor.model.predict(s_t)[0]
            
            noise_t[0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0], 0.0, 0.60, 0.30)
            noise_t[1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[1], 0.0, 0.60, 0.30)

            # The following code do the stochastic brake
            # if random.random() <= 0.1:
            #    print('********Now we apply the brake***********')
            #    noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2],  0.2 , 1.00, 0.10)
            
            # print(a_t_original[1], noise_t[1])
            
            a_t[0] = a_t_original[0] + noise_t[0]
            a_t[1] = a_t_original[1] + noise_t[1]
            
            # r = random.random()
            # if r < 0.3-i/episode_count:
            #     a_t[0] = 0
            # elif r < 0.6- 2*(i/episode_count):
            #     a_t[1] = 0
            
            env.step(a_t)

            (s_t1, r_t, done) = env.get_state_reward_1d()

            buff.add(s_t, a_t, r_t, s_t1, done)  # Add replay buffer

            # Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            states = states.reshape((states.shape[0], states.shape[2]))
            new_states = new_states.reshape((new_states.shape[0], new_states.shape[2]))

            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])
            
            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA * target_q_values[k]
            
            if (train_indicator):
                loss += critic.model.train_on_batch([states, actions], y_t)
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()
            
            total_loss += loss
            total_reward += r_t
            s_t = s_t1.copy()
            
            print('Episode [%i]  Step [%i]  Goal [%.2f, %.2f]  Pose [%+.2f, %+.2f, %+.2f]  Action [%+.2f, %+.2f]  Reward [%+.2f]  Loss [%+.2f]\r' %
                  (i, step, env.goal[0], env.goal[1], env.pose[0], env.pose[1], env.pose[2], a_t[0], a_t[1], r_t, loss))
            
            step += 1
            if done:
                print('########### GOAL REACHED SUCCESSFULLY ############')
                break
        
        if np.mod(i, 3) == 0:
            if (train_indicator):
                print('Now we save model')
                actor.model.save_weights('/home/robin/catkin_ws/src/int_agents_project/data/actormodel_obst_02.h5', overwrite=True)
                with open('/home/robin/catkin_ws/src/int_agents_project/data/actormodel_obst_02.json', 'w') as outfile:
                    json.dump(actor.model.to_json(), outfile)
                
                critic.model.save_weights('/home/robin/catkin_ws/src/int_agents_project/data/criticmodel_obst_02.h5', overwrite=True)
                with open('/home/robin/catkin_ws/src/int_agents_project/data/criticmodel_obst_02.json', 'w') as outfile:
                    json.dump(critic.model.to_json(), outfile)

        loss_all[i] = total_loss
        reward_all[i] = total_reward

        print('DATA @ ' + str(i) + '-th Episode  : Reward ' + str(total_reward))
        print('Total Reward:     ' + str(total_reward))
        print('Total Loss:       ' + str(total_loss))
        print('Total Step:       ' + str(step))
        print('')
    
    sio.savemat('/home/robin/catkin_ws/src/int_agents_project/data/eval_obst_02.mat' , {'loss': loss_all, 'reward': reward_all})

    # env.end()  # This is for shutting down TORCS
    print('Finished learning!')


if __name__ == '__main__':
    if sys.argv < 3:
        print('Goal was not properly specified through argument!')
    else:
        goal = np.array(sys.argv[1:3], float)
        teach_robot(goal)
