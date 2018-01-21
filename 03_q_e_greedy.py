import gym 
import numpy as np
from gym.envs.registration import register
import random
import matplotlib.pyplot as plt

register(
        id="FrozenLake-v3",
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name': '4x4',
                'is_slippery': False}
        )

env = gym.make('FrozenLake-v3')
"""
input env
output optimal path

make q table
upate q table to optimal table
get policy from q table

Initalize for all s and a as 0
Repeat number of episodes
set stats
set action a
    Repeat number of step in episode
    task action a and get reward and next_state
    select action a from policy using derived strategy
    update q table
    q(s,a) = r + lamda* max(next_state, next_action)
    s = s'
    a = a'
"""

Q = np.zeros([env.observation_space.n, env.action_space.n])

def get_action_from(Q, i, state):
    """e-greedy
    input state, i 
    output action

    toss coin if it is less than e select random action
    else select greed action

    e should be decay
    """
    e = 1./((i//100) + 1)
    a = None
    if np.random.rand(1) < e:
        a = env.action_space.sample()
    else:
        a = np.argmax(Q[state, :])
    return a

dis = .99
rewards = []
cum_steps = []
num_episodes = 2000

for i in range(num_episodes):
    s = env.reset()
    done = False
    cum_step = 0
    while not done:
        a = get_action_from(Q, i, s)
        next_state, reward, done, _ = env.step(a)
        Q[s, a] = reward + dis * np.max(Q[next_state, :])
        s = next_state
        cum_step += 1
    cum_steps.append(cum_step)

# print('success rate: {}'.format(np.sum(rewards)/num_episodes))
print('Final Q-Table Values')
print('LEFT DOWN RIGHT UP')
print(Q)
plt.plot(np.cumsum(cum_steps), range(num_episodes))
# plt.bar(range(len(rewards)), rewards, color='blue')
plt.show()


