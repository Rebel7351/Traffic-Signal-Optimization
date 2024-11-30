# """Epsilon Greedy Exploration Strategy."""
# import numpy as np


# class EpsilonGreedy:
#     """Epsilon Greedy Exploration Strategy."""

#     def __init__(self, initial_epsilon=1.0, min_epsilon=0.0, decay=0.99):
#         """Initialize Epsilon Greedy Exploration Strategy."""
#         self.initial_epsilon = initial_epsilon
#         self.epsilon = initial_epsilon
#         self.min_epsilon = min_epsilon
#         self.decay = decay

#     def choose(self, q_table, state, action_space):
#         """Choose action based on epsilon greedy strategy."""
#         if np.random.rand() < self.epsilon:
#             action = int(action_space.sample())
#         else:
#             action = np.argmax(q_table[state])

#         self.epsilon = max(self.epsilon * self.decay, self.min_epsilon)
#         # print(self.epsilon)
#         return action

#     def reset(self):
#         """Reset epsilon to initial value."""
#         self.epsilon = self.initial_epsilon

"""Epsilon Greedy Exploration Strategy for duel dqn and duel double dqn."""
import numpy as np
import tensorflow as tf


class EpsilonGreedy:
    """Epsilon Greedy Exploration Strategy."""

    def __init__(self, initial_epsilon=1.0, min_epsilon=0.0, decay=0.99):
        """Initialize Epsilon Greedy Exploration Strategy."""
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.step=0

    def choose(self, q_net, state, action_space):
        """Choose action based on epsilon greedy strategy."""

        if np.random.rand() < self.epsilon:
            # print(action_space.n)
            action = int(action_space.sample())
        else:
            value, advantage = q_net(tf.expand_dims(state,axis=0))
            q_values = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))

            action = tf.argmax(q_values[0])
        # print(self.epsilon)
        self.epsilon = max(self.epsilon * self.decay, self.min_epsilon)

        if(self.epsilon==self.min_epsilon):
            self.step+=1
            # if(self.step>1000):
            #     self.epsilon=self.initial_epsilon

        # print(self.epsilon)
        return action

    def reset(self):
        """Reset epsilon to initial value."""
        self.epsilon = self.initial_epsilon

