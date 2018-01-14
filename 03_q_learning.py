"""
initalize Q for all s and a = 0
choose s
Repeat:
    select action a and execute it
    get reward r and new start s'
    update table
    Q(s, a) = r + maxQ(s',a')
    s = s'
"""
import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4',
            'is_slippery': False}
)
env = gym.make('FrozenLake-v3')

Q = np.zeros([env.observation_space.n, env.action_space.n])
num_episodes = 2000

rewards = []

def rargmax(vector):
    m = np.amax(vector)
    maxs = np.nonzero(vector == m)[0]
    return random.choice(maxs)

for i in range(num_episodes):
    s = env.reset()
    done = False
    cum_reward = 0
    while not done:
        a = rargmax(Q[s,:])
        new_state, reward, done, info = env.step(a)
        Q[s, a] = reward + np.max(Q[new_state, :])
        cum_reward += reward
        s = new_state
    rewards.append(cum_reward)
print('success rate: {}'.format(np.sum(rewards)/num_episodes))
print('Final Q-Table Values')
print('LEFT DOWN RIGHT UP')
print(Q)
plt.bar(range(len(rewards)), rewards, color='blue')
plt.show()
