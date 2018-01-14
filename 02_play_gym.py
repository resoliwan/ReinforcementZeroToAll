"""
Make frozn lake game.
init env
while
    get input
    if input not arrow:
        print('Game end')
        break;
    
    if arrow:
        state, reward, done, info = env.step(action)

    if done:
        print('Game end', reward)
"""
import gym
from gym.envs.registration import register
import readchar

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

arrow_keys = {
    '\x1b[A': UP,
    '\x1b[B': DOWN,
    '\x1b[C': RIGHT,
    '\x1b[D': LEFT}

register(
        id = 'FrozenLake-v3',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name': '4x4', 'is_slippery': False}
        )

env = gym.make('FrozenLake-v3')
env.render()

while True:
    key = readchar.readkey()
    if key not in arrow_keys.keys():
        print('Game end')
        break;
    else:   
        state, reward, done, info = env.step(arrow_keys[key])
        print('state {}, reward {}, done {}, info {}'.format(state, reward, done, info))

        env.render()

    if done:
        print('Game end reward :', reward)




