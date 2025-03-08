import tensorflow as tf
import numpy as np
import gym
import logging

logger = logging.getLogger('rl')
logger.setLevel(logging.DEBUG)


class PolicyGradientAgent:

    def __init__(self, num_actions, state_dim, num_hidden):
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.num_hidden = num_hidden
        self.build_policy_graph()
        self.build_value_graph()

    def build_policy_graph(self):
        '''
        Supervised learning of what actions to take
        '''
        params = tf.get_variable('policy_params', [self.state_dim, self.num_actions])  # [4, 2]
        self.pl_state = tf.placeholder(tf.float32, [None, self.state_dim], name='state')  # [bs, 4]
        self.pl_actions = tf.placeholder(tf.float32, [None, self.num_actions], name='actions')  # 1-hot vector
        self.pl_advantages = tf.placeholder(tf.float32, [None, 1], name='advantages')
        # inference
        linear = tf.matmul(self.pl_state, params)  # [bs, 2]
        self.pl_probabilities = tf.nn.softmax(linear)  # [bs, 2]
        # loss calculation
        good_probs = tf.reduce_sum(tf.multiply(self.pl_probabilities, self.pl_actions), axis=1)
        loss = -tf.reduce_sum(tf.log(good_probs) * self.pl_advantages)
        # # setup the optimizer
        self.pl_optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

    def build_value_graph(self):
        # sess.run(calculated) to calculate the value of the state
        self.vl_state = tf.placeholder(tf.float32, [None, self.state_dim])  # [batch_size, states]
        w1 = tf.get_variable("w1", [self.state_dim, self.num_hidden])  # [states, hidden]
        b1 = tf.get_variable("b1", [self.num_hidden])
        h1 = tf.nn.relu(tf.matmul(self.vl_state, w1) + b1)  # [batch_size, hidden]
        w2 = tf.get_variable("w2", [self.num_hidden, 1])  # [hidden, 1]
        b2 = tf.get_variable("b2", [1])
        self.vl_calculated = tf.matmul(h1, w2) + b2  # [batch_size, 1]

        # sess.run(optimizer) to update the value of a state
        self.vl_newvals = tf.placeholder(tf.float32, [None, 1])
        diffs = self.vl_calculated - self.vl_newvals
        loss = tf.nn.l2_loss(diffs)
        self.vl_optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)

    def next_action(self, sess, observation):
        probs = sess.run(self.pl_probabilities, {self.pl_state: [observation]})
        return 0 if np.random.uniform(0, 1) < probs[0][0] else 1

    def train_policy(self, sess, states, actions, advantages):
        sess.run(self.pl_optimizer, {
            self.pl_state: states,
            self.pl_advantages: advantages,
            self.pl_actions: actions
        })

    def train_value(self, sess, states, update_vals):
        sess.run(self.vl_optimizer, {
            self.vl_state: states,
            self.vl_newvals: update_vals
        })

    def get_state_value(self, sess, state):
        result = sess.run(self.vl_calculated, {self.vl_state: [state]})
        return result[0][0]


def policy_gradient_optimization(env, sess, agent):
    observation = env.reset()

    actions = []
    states = []
    transitions = []
    advantages = []
    total_rewards = 0

    for _ in range(200):
        # compute next action
        action = agent.next_action(sess, observation)
        states.append(observation)  # record state
        actions.append(np.eye(2)[action])  # record action
        old_obs = observation
        observation, reward, done, _ = env.step(action)  # move world forward
        transitions.append((old_obs, action, reward))  # record transition
        total_rewards += reward
        if done:
            break

    # calculate the return of each transition
    update_vals = []
    transitions = np.array(transitions)
    for idx, trans in enumerate(transitions):
        observation, action, reward = trans
        # calculate discounted monte-carlo return
        future_reward = np.sum(0.97 ** np.arange(len(transitions) - idx) * transitions[idx:, 2])
        update_vals.append([future_reward])
        advantages.append([future_reward - agent.get_state_value(sess, observation)])

    agent.train_policy(sess, states, actions, advantages)
    agent.train_value(sess, states, update_vals)

    return total_rewards

env = gym.make('CartPole-v0')
agent = PolicyGradientAgent(2, 4, 10)
num_episodes = 500

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(num_episodes):
        reward = policy_gradient_optimization(env, sess, agent)
        if i % 50 == 0:
            logger.info('Loop {} gets reward: {}'.format(i, reward))

    # visualize!
    for _ in range(5):  # run it a few times so you can see the fun properly.
        obs = env.reset()
        while True:
            env.render()
            nextstate, reward, done, _ = env.step(agent.next_action(sess, obs))
            if done:
                break
            obs = nextstate
