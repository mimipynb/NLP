import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict


SCORE = 0  # index for the score from the state tuple

STATE = 0  # index of state in the generate_episode() result
ACTION = 1  # index of state in the generate_episode() result
REWARD = 2  # index of state in the generate_episode() result


def generate_episode(policy, env):
    result = []
    cur_state = env.reset()
    while True:
        policy_action = policy(cur_state)
        nextstate, reward, done, _ = env.step(policy_action)
        result.append([cur_state, policy_action, reward])
        if done:
            return result
        cur_state = nextstate


def first_visit_mc_policy_evaluation(policy, env, num_episodes):
    V = defaultdict(float)
    returns = defaultdict(list)

    for _ in range(num_episodes):
        episode = generate_episode(policy, env)
        appeared_states = set([tuple(step[STATE]) for step in episode])
        for s in appeared_states:
            first_occurance_of_s = next(idx
                                        for idx, step in enumerate(episode)
                                        if step[STATE] == s)
            G = sum([step[REWARD] for step in episode[first_occurance_of_s:]])
            returns[s].append(G)
            V[s] = np.mean(returns[s])

    return V


def policy(state):
    return 0 if state[SCORE] >= 20 else 1


env = gym.make('Blackjack-v0')
V = first_visit_mc_policy_evaluation(policy, env, num_episodes=100000)

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