## -------- ALGORITHM: Deep Q-Learning with experience replay.-----------------
## ----------------------------------------------------------------------------
## Bibliography: Mnih et al 2015 (Deep Mind). LETTER Nature.    
       
## ----------------- AGENT : Deep Q-Learning Algorithm  -----------------------
## ----------------------------------------------------------------------------
import numpy as np
import random
from collections import deque

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import clone_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import LeakyReLU

class DQNAgent:
    def __init__(self, dim_state, num_action, agent_conf):
        self.state_size = dim_state
        self.action_size = num_action
        self.memory = deque(maxlen=agent_conf['max_memory'])
        self.gamma = agent_conf['gamma']    
        self.epsilon = agent_conf['epsilon']
        self.epsilon_min = agent_conf['epsilon_min']
        self.epsilon_decay = agent_conf['epsilon_decay']
        self.learning_rate = agent_conf['learning_rate']
        self.critic = self._build_model()
        self.actor = self.copy_to_actor()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        critic = Sequential()
        critic.add(Dense(200, input_dim = self.state_size, activation = 'relu', 
                         kernel_initializer = 'he_normal',
                         bias_initializer = initializers.Constant(0.1)))
        #critic.add(Dense(50, activation = 'relu', 
        #                 kernel_initializer = 'he_normal',
        #                 bias_initializer = initializers.Constant(0.1)))
        critic.add(Dense(self.action_size, activation = 'linear',
                         kernel_initializer = 'he_normal',
                         bias_initializer = initializers.Constant(0.1)))

        critic.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate)) 
        return critic

    def copy_to_actor(self):
        actor = clone_model(self.critic)
        actor.set_weights(self.critic.get_weights())

        return actor

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            act_values = self.critic.predict(np.array([state]))
            return np.argmax(act_values[0]) 

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        x_value = np.zeros((batch_size,self.state_size))
        y_value = np.zeros((batch_size,self.action_size))
        i=0

        for state, action, reward, next_state, done in minibatch:
            
            Q_valuesPred = self.actor.predict(np.array([next_state]))
            target = reward + self.gamma * done * np.amax(Q_valuesPred[0])

            target_f = self.critic.predict(np.array([state]))    
            target_f[0][action] = target

            x_value[i] = state
            y_value[i] = target_f 
            i=i+1

        history=self.critic.fit(x_value, y_value, batch_size = batch_size, epochs=1, verbose=0)

        q_history = history.history['loss'][0]
        q_evaluate = self.critic.evaluate(x_value, y_value,verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return q_history, q_evaluate
           
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)