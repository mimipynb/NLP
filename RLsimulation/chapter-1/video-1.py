# Environments, not agents
from gym import envs
print(envs.registry.all())

# the most typical environment
import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())

# another fun environment
import gym
env = gym.make('MountainCar-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())