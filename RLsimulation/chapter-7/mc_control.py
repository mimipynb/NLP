import gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict


SCORE = 0  # index for the score from the state tuple

STATE = 0  # index of state in the generate_episode() result
ACTION = 1  # index of state in the generate_episode() result
REWARD = 2  # index of state in the generate_episode() result

EPSILON = 0.1  # for our epsilon-greedy policy


def generate_episode(policy, env):
    result = []
    cur_state = env.reset()
    while True:
        P = policy(cur_state)
        policy_action = np.random.choice(np.arange(len(P)), p=P)
        nextstate, reward, done, _ = env.step(policy_action)
        result.append([cur_state, policy_action, reward])
        if done:
            return result
        cur_state = nextstate


def epsilon_greedy_mc_control(env, num_episodes):
    '''
    On-policy first-visit MC control for epsilon-soft policies
    '''
    num_actions = env.action_space.n
    Q = defaultdict(lambda: np.zeros(num_actions))
    returns = defaultdict(list)

    def epsilon_soft_policy(state):
        # A* denotes the optimal action
        A_star = np.argmax(Q[state])
        # handle all non-optimal-action values first
        policy_for_state = np.ones(num_actions, dtype=float) * EPSILON / num_actions
        # optimal-action value
        policy_for_state[A_star] = 1 - EPSILON + EPSILON / num_actions
        return policy_for_state

    for _ in range(num_episodes):
        episode = generate_episode(epsilon_soft_policy, env)
        appeared_s_a = set([(step[STATE], step[ACTION]) for step in episode])
        for s, a in appeared_s_a:
            first_occurance = next(idx
                                   for idx, step in enumerate(episode)
                                   if step[STATE] == s and step[ACTION] == a)
            G = sum([step[REWARD] for step in episode[first_occurance:]])
            returns[(s, a)].append(G)
            Q[s][a] = np.mean(returns[(s, a)])

    return Q


env = gym.make('Blackjack-v0')
Q = epsilon_greedy_mc_control(env, num_episodes=100000)


V = defaultdict(float)
for state, actions in Q.items():
    action_value = np.max(actions)
    V[state] = action_value


X, Y = np.meshgrid(
    np.arange(12, 22),  # player's hand
    np.arange(1, 11)  # dealer showing
    )

no_usable_ace = np.apply_along_axis(lambda idx: V[(idx[0], idx[1], False)], 2, np.dstack([X, Y]))
usable_ace = np.apply_along_axis(lambda idx: V[(idx[0], idx[1], True)], 2, np.dstack([X, Y]))

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 4), subplot_kw={'projection': '3d'})
ax0.plot_surface(X, Y, no_usable_ace, cmap=plt.cm.YlGnBu_r)
ax0.set_xlabel('Hole Cards')
ax0.set_ylabel('Dealer')
ax0.set_zlabel('MC Estimated Value')
ax0.set_title('No Useable Ace')

ax1.plot_surface(X, Y, usable_ace, cmap=plt.cm.YlGnBu_r)
ax1.set_xlabel('Hole Cards')
ax1.set_ylabel('Dealer')
ax1.set_zlabel('MC Estimated Value')
ax1.set_title('Useable Ace')

plt.show()