import numpy as np
import math
from keras.initializations import normal, identity
from keras.models import Sequential, Model, model_from_json
# from keras.engine.training import collect_trainable_weights
from keras.layers import Dense, Flatten, Input, merge, Lambda, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600


class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)
        K.set_learning_phase(1)  # set learning phase

        #Now create the model
        self.model , self.weights, self.state = self.create_actor_network(state_size, action_size)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size)
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.initialize_all_variables())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={self.state: states, self.action_gradient: action_grads})

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_size, action_dim=6):
        print('Building Actor Network')
        
        S = Input(shape=state_size)
        # Convolutional Layers
        c0 = Conv2D(16, 3, 3, activation='relu')(S)
        p0 = MaxPooling2D(pool_size=(2, 2))(c0)
        c1 = Conv2D(32, 3, 3, activation='relu')(p0)
        p1 = MaxPooling2D(pool_size=(2, 2))(c1)
        c2 = Conv2D(64, 3, 3, activation='relu')(p1)
        p2 = MaxPooling2D(pool_size=(2, 2))(c2)
        d0 = Dropout(0.25)(p2)
        f0 = Flatten()(d0)
        # Fully Connected Layers
        h0 = Dense(HIDDEN1_UNITS, activation='relu')(f0)
        h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
        
        V = Dense(action_dim,activation='tanh',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
        
        model = Model(input=S,output=V)
        
        return model, model.trainable_weights, S


class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size
        
        K.set_session(sess)

        # Now create the model
        self.model, self.action, self.state = self.create_critic_network(state_size, action_size)
        self.target_model, self.target_action, self.target_state = self.create_critic_network(state_size, action_size)
        self.action_grads = tf.gradients(self.model.output, self.action)  # GRADIENTS for policy update
        self.sess.run(tf.initialize_all_variables())
    
    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]
    
    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in xrange(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU) * critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)
    
    def create_critic_network(self, state_size, action_dim):
        print('Building Critic Network')

        S = Input(shape=state_size)
        # Convolutional Layers for
        c0 = Conv2D(16, 3, 3, activation='relu')(S)
        p0 = MaxPooling2D(pool_size=(2, 2))(c0)
        c1 = Conv2D(32, 3, 3, activation='relu')(p0)
        p1 = MaxPooling2D(pool_size=(2, 2))(c1)
        c2 = Conv2D(64, 3, 3, activation='relu')(p1)
        p2 = MaxPooling2D(pool_size=(2, 2))(c2)
        d0 = Dropout(0.25)(p2)
        f0 = Flatten()(d0)
        # Fully Connected Layers
        h0 = Dense(HIDDEN1_UNITS, activation='relu')(f0)
        h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
        
        A = Input(shape=[action_dim], name='action2')
        a1 = Dense(HIDDEN2_UNITS, activation='linear')(A)
        
        h2 = merge([h1, a1], mode='sum')
        h3 = Dense(HIDDEN2_UNITS, activation='relu')(h2)
        V = Dense(action_dim, activation='linear')(h3)
        
        model = Model(input=[S, A], output=V)
        
        adam = Adam(lr=self.LEARNING_RATE)
        
        model.compile(loss='mse', optimizer=adam)
        
        return model, A, S