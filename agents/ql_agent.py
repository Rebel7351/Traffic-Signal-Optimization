# """Q-learning Agent class."""
# from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy


# class QLAgent:
#     """Q-learning Agent class."""

#     def __init__(self, starting_state, state_space, action_space, alpha=0.5, gamma=0.95, exploration_strategy=EpsilonGreedy()):
#         """Initialize Q-learning agent."""
#         self.state = starting_state
#         self.state_space = state_space
#         self.action_space = action_space
#         self.action = None
#         self.alpha = alpha
#         self.gamma = gamma
#         self.q_table = {self.state: [0 for _ in range(action_space.n)]}
#         self.exploration = exploration_strategy
#         self.acc_reward = 0

#     def act(self):
#         """Choose action based on Q-table."""
#         self.action = self.exploration.choose(self.q_table, self.state, self.action_space)
#         return self.action

#     def learn(self, next_state, reward, done=False):
#         """Update Q-table with new experience."""
#         if next_state not in self.q_table:
#             self.q_table[next_state] = [0 for _ in range(self.action_space.n)]

#         s = self.state
#         s1 = next_state
#         a = self.action
#         self.q_table[s][a] = self.q_table[s][a] + self.alpha * (
#             reward + self.gamma * max(self.q_table[s1]) - self.q_table[s][a]
#         )
#         self.state = s1
#         self.acc_reward += reward




# # """Q-learning Agent class(sarsa)."""
# # # uses and learns same policy
# # from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy
# #
# #
# # class QLAgent:
# #     """Q-learning Agent class."""
# #
# #     def __init__(self, starting_state, state_space, action_space, alpha=0.5, gamma=0.95, exploration_strategy=EpsilonGreedy()):
# #         """Initialize Q-learning agent."""
# #         self.state = starting_state
# #         self.state_space = state_space
# #         self.action_space = action_space
# #         self.action = None
# #         self.alpha = alpha
# #         self.gamma = gamma
# #         self.q_table = {self.state: [0 for _ in range(action_space.n)]}
# #         self.exploration = exploration_strategy
# #         self.acc_reward = 0
# #
# #     def act(self):
# #         """Choose action based on Q-table."""
# #         self.action = self.exploration.choose(self.q_table, self.state, self.action_space)
# #
# #         return self.action
# #
# #     def learn(self, next_state, reward, done=False):
# #         """Update Q-table with new experience."""
# #         if next_state not in self.q_table:
# #             self.q_table[next_state] = [0 for _ in range(self.action_space.n)]
# #         print("learning ql/sarsa")
# #
# #         s = self.state
# #         s1 = next_state
# #         a = self.action
# #         a1=self.exploration.choose(self.q_table, next_state, self.action_space)
# #         self.q_table[s][a] = self.q_table[s][a] + self.alpha * (
# #             reward + self.gamma * self.q_table[s1][a1] - self.q_table[s][a]
# #         )
# #         self.state = s1
# #         self.acc_reward += reward





# """  DQN Agent class."""
# # target is calculated using target network(off policy based) and learning is done in batches
# from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy
# import tensorflow as tf
# from tensorflow.keras.layers import Dense
# from keras import Model ,Input
# from keras.models import clone_model
# from keras.losses import Huber



# class QLAgent:
#     """Q-learning Agent class."""

#     def __init__(self, starting_state, state_space, action_space, alpha=0.5, gamma=0.95, exploration_strategy=EpsilonGreedy()):
#         """Initialize Q-learning agent."""
#         self.state = starting_state
#         self.state_space = state_space
#         self.action_space = action_space
#         self.action = None
#         self.alpha = alpha
#         self.gamma = gamma
#         self.step_counter=0
#         self.TARGET_UPDATE_AFTER = 4
#         self.batch_size=70

#         # model to update policy
#         net_input=Input(shape=(len(self.state),))
#         x=Dense(64,activation="relu")(net_input)
#         x=Dense(32,activation="relu")(x)
#         x = Dense(25, activation="relu")(x)
#         # action_space is descrete
#         net_output=Dense(self.action_space.n)(x)
#         self.q_net = Model(inputs=net_input,outputs=net_output)
#         self.q_net.compile(optimizer="Adam")
#         # calculating loss using predefined function
#         self.loss_fn=Huber()
#         # model to get target value and overcome drawbacks of dqn
#         self.target_net=clone_model(self.q_net)

#         self.exploration = exploration_strategy
#         self.acc_reward = 0

#         # replay buffer to store the responce of model
#         self.replay_buffer=[]
#         self.max_transitions=1_00_000

#         print('init DQN')

#     def act(self):
#         """Choose action based on Q-table."""
#         self.action = self.exploration.choose(self.q_net, self.state, self.action_space)
#         return self.action

#     def learn(self, next_state, reward, done=False):
#         # print("learning dqn")
#         # inserting response to replay_buffer
#         def insert_transition(transition):
#             if len(self.replay_buffer) >= self.max_transitions:
#                 self.replay_buffer.pop(0)
#             self.replay_buffer.append(transition)

#         # Samples a batch of transitions from Replay Buffer randomly
#         def sample_transitions(batch_size=16):
#             random_indices = tf.random.uniform(shape=(batch_size,), minval=0, maxval=len(self.replay_buffer),
#                                                dtype=tf.int32)
#             sampled_current_states = []
#             sampled_actions = []
#             sampled_rewards = []
#             sampled_next_states = []
#             sampled_terminals = []

#             for index in random_indices:
#                 sampled_current_states.append(self.replay_buffer[index][0])
#                 sampled_actions.append(self.replay_buffer[index][1])
#                 sampled_rewards.append(self.replay_buffer[index][2])
#                 sampled_next_states.append(self.replay_buffer[index][3])
#                 sampled_terminals.append(self.replay_buffer[index][4])

#             return tf.convert_to_tensor(sampled_current_states), tf.convert_to_tensor(
#                 sampled_actions), tf.convert_to_tensor(
#                 sampled_rewards), tf.convert_to_tensor(sampled_next_states), tf.convert_to_tensor(sampled_terminals)

#         s = self.state
#         s1 = next_state
#         a = self.action
#         transitions=[s, a, reward, s1, done]
#         insert_transition(transitions)
#         self.state = next_state
#         self.acc_reward += reward
#         self.step_counter+=1

#         current_states, actions, rewards, next_states, terminals = sample_transitions(self.batch_size)
#         # target value using offpolicy
#         next_action_values = tf.reduce_max(self.target_net(next_states), axis=1)
#         targets = tf.where(terminals, rewards, rewards + self.gamma * next_action_values)
#         with tf.GradientTape() as tape:
#             preds = self.q_net(current_states)
#             batch_nums = tf.range(0, limit=self.batch_size)
#             indices = tf.stack((tf.cast(batch_nums, tf.int32), tf.cast(actions, tf.int32)), axis=1)
#             current_values = tf.gather_nd(preds, indices)
#             loss = self.loss_fn(targets, current_values)
#         grads = tape.gradient(loss, self.q_net.trainable_weights)
#         self.q_net.optimizer.apply_gradients(zip(grads, self.q_net.trainable_weights))


#         if self.step_counter % self.TARGET_UPDATE_AFTER == 0:
#             self.target_net.set_weights(self.q_net.get_weights())



# # includes decoupling of choosing action and training network
# """ Double DQN Agent class."""
# # target is calculated using target network(off policy based) and learning is done in batches
# from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy
# import tensorflow as tf
# from tensorflow.keras.layers import Dense
# from keras import Model ,Input
# from keras.models import clone_model
# from keras.losses import Huber



# class QLAgent:
#     """Q-learning Agent class."""

#     def __init__(self, starting_state, state_space, action_space, alpha=0.5, gamma=0.95, exploration_strategy=EpsilonGreedy()):
#         """Initialize Q-learning agent."""
#         self.state = starting_state
#         self.state_space = state_space
#         self.action_space = action_space
#         self.action = None
#         self.alpha = alpha
#         self.gamma = gamma
#         self.step_counter=0
#         self.TARGET_UPDATE_AFTER = 4
#         self.batch_size=70

#         # model to update policy
#         net_input=Input(shape=(len(self.state),))
#         x=Dense(64,activation="relu")(net_input)
#         x=Dense(32,activation="relu")(x)
#         x = Dense(25, activation="relu")(x)
#         # action_space is descrete
#         net_output=Dense(self.action_space.n)(x)
#         self.q_net = Model(inputs=net_input,outputs=net_output)
#         self.q_net.compile(optimizer="Adam")
#         # calculating loss using predefined function
#         self.loss_fn=Huber()
#         # model to get target value and overcome drawbacks of dqn
#         self.target_net=clone_model(self.q_net)

#         self.exploration = exploration_strategy
#         self.acc_reward = 0

#         # replay buffer to store the responce of model
#         self.replay_buffer=[]
#         self.max_transitions=1_00_000

#         print('init DoubleDQN')



#     def act(self):
#         """Choose action based on Q-table."""
#         self.action = self.exploration.choose(self.q_net, self.state, self.action_space)
#         return self.action

#     def learn(self, next_state, reward, done=False):
#         print("learning double dqn")
#         # inserting response to replay_buffer
#         def insert_transition(transition):
#             if len(self.replay_buffer) >= self.max_transitions:
#                 self.replay_buffer.pop(0)
#             self.replay_buffer.append(transition)

#         # Samples a batch of transitions from Replay Buffer randomly
#         def sample_transitions(batch_size=16):
#             random_indices = tf.random.uniform(shape=(batch_size,), minval=0, maxval=len(self.replay_buffer),
#                                                dtype=tf.int32)
#             sampled_current_states = []
#             sampled_actions = []
#             sampled_rewards = []
#             sampled_next_states = []
#             sampled_terminals = []

#             for index in random_indices:
#                 sampled_current_states.append(self.replay_buffer[index][0])
#                 sampled_actions.append(self.replay_buffer[index][1])
#                 sampled_rewards.append(self.replay_buffer[index][2])
#                 sampled_next_states.append(self.replay_buffer[index][3])
#                 sampled_terminals.append(self.replay_buffer[index][4])

#             return tf.convert_to_tensor(sampled_current_states), tf.convert_to_tensor(
#                 sampled_actions), tf.convert_to_tensor(
#                 sampled_rewards), tf.convert_to_tensor(sampled_next_states), tf.convert_to_tensor(sampled_terminals)

#         s = self.state
#         s1 = next_state
#         a = self.action
#         transitions=[s, a, reward, s1, done]
#         insert_transition(transitions)
#         self.state = next_state
#         self.acc_reward += reward
#         self.step_counter+=1

#         current_states, actions, rewards, next_states, terminals = sample_transitions(self.batch_size)
#         # target value using offpolicy
#         next_action = tf.argmax(self.q_net(next_states), axis=1)
#         batch_nums = tf.range(0, limit=self.batch_size)
#         indices = tf.stack((tf.cast(batch_nums, tf.int32), tf.cast(next_action, tf.int32)), axis=1)
#         preds=self.target_net(next_states)
#         next_action_values=tf.gather_nd(preds, indices)
#         targets = tf.where(terminals, rewards, rewards + self.gamma * next_action_values)
#         with tf.GradientTape() as tape:
#             preds = self.q_net(current_states)
#             batch_nums = tf.range(0, limit=self.batch_size)
#             indices = tf.stack((tf.cast(batch_nums, tf.int32), tf.cast(actions, tf.int32)), axis=1)
#             current_values = tf.gather_nd(preds, indices)
#             loss = self.loss_fn(targets, current_values)
#         grads = tape.gradient(loss, self.q_net.trainable_weights)
#         self.q_net.optimizer.apply_gradients(zip(grads, self.q_net.trainable_weights))


#         if self.step_counter % self.TARGET_UPDATE_AFTER == 0:
#             self.target_net.set_weights(self.q_net.get_weights())






""" dueling DQN Agent class."""
# target is calculated using target network(off policy based) and learning is done in batches
from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy
import tensorflow as tf
from tensorflow.keras.layers import Dense
from keras import Model ,Input
from keras.models import clone_model
from keras.losses import Huber
class QLAgent:


    def __init__(self, starting_state, state_space, action_space, alpha=0.5, gamma=0.95, exploration_strategy=EpsilonGreedy()):

        self.state = starting_state
        self.state_space = state_space
        self.action_space = action_space
        self.action = None
        self.alpha = alpha
        self.gamma = gamma
        self.step_counter = 0
        self.TARGET_UPDATE_AFTER = 4
        self.batch_size = 70

        # model to update policy
        net_input = Input(shape=(len(self.state),))
        x = Dense(64, activation="relu")(net_input)
        x = Dense(32, activation="relu")(x)
        x = Dense(25, activation="relu")(x)

        # Dueling architecture
        value_stream = Dense(1)(x)
        advantage_stream = Dense(self.action_space.n)(x)
        self.q_net = Model(inputs=net_input, outputs=[value_stream, advantage_stream])
        self.q_net.compile(optimizer="Adam")

        # calculating loss using predefined function
        self.loss_fn = Huber()

        # model to get target value and overcome drawbacks of dqn
        self.target_net = clone_model(self.q_net)

        self.exploration = exploration_strategy
        self.acc_reward = 0

        # replay buffer to store the response of model
        self.replay_buffer = []
        self.max_transitions = 1_00_000

        print('init DuelingDQNAgent')

    def act(self):
        """Choose action based on Q-table."""
        self.action = self.exploration.choose(self.q_net, self.state, self.action_space)
        return self.action

    def learn(self, next_state, reward, done=False):
        # print("learning duel double dqn")
        # inserting response to replay_buffer
        def insert_transition(transition):
            if len(self.replay_buffer) >= self.max_transitions:
                self.replay_buffer.pop(0)
            self.replay_buffer.append(transition)

        # Samples a batch of transitions from Replay Buffer randomly
        def sample_transitions(batch_size=16):
            random_indices = tf.random.uniform(shape=(batch_size,), minval=0, maxval=len(self.replay_buffer),
                                               dtype=tf.int32)

            sampled_current_states = []
            sampled_actions = []
            sampled_rewards = []
            sampled_next_states = []
            sampled_terminals = []

            for index in random_indices:
                sampled_current_states.append(self.replay_buffer[index][0])
                sampled_actions.append(self.replay_buffer[index][1])
                sampled_rewards.append(self.replay_buffer[index][2])
                sampled_next_states.append(self.replay_buffer[index][3])
                sampled_terminals.append(self.replay_buffer[index][4])

            # print(tf.convert_to_tensor(sampled_current_states)," ",sampled_actions," ",sampled_rewards," ",sampled_next_states," ",sampled_terminals)
            return tf.convert_to_tensor(sampled_current_states), tf.convert_to_tensor(
                sampled_actions), tf.convert_to_tensor(
                sampled_rewards), tf.convert_to_tensor(sampled_next_states), tf.convert_to_tensor(sampled_terminals)

        s = self.state
        s1 = next_state
        a = self.action
        transitions = [s, a, reward, s1, done]

        insert_transition(transitions)
        self.state = next_state
        self.acc_reward += reward
        self.step_counter += 1

        current_states, actions, rewards, next_states, terminals = sample_transitions(self.batch_size)

        # target value using offpolicy
        next_value, next_advantage = self.target_net(next_states)
        next_q_values = next_value + (next_advantage - tf.reduce_mean(next_advantage, axis=1, keepdims=True))

        next_action_values = tf.reduce_max(next_q_values, axis=1)

        targets = tf.where(terminals, rewards, rewards + self.gamma * next_action_values)


        with tf.GradientTape() as tape:
            value, advantage = self.q_net(current_states)
            q_values = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
            batch_nums = tf.range(0, limit=self.batch_size)

            actions = tf.reshape(actions, (-1,))


            indices = tf.stack((tf.cast(batch_nums, tf.int32), tf.cast(actions, tf.int32)), axis=1)

            q_values = tf.gather_nd(q_values, indices)

            loss = self.loss_fn(targets, q_values)

        grads = tape.gradient(loss, self.q_net.trainable_weights)
        self.q_net.optimizer.apply_gradients(zip(grads, self.q_net.trainable_weights))

        if self.step_counter % self.TARGET_UPDATE_AFTER == 0:
            self.target_net.set_weights(self.q_net.get_weights())



