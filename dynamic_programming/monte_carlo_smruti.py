import numpy as np
from grid_world import standard_grid
from utils import print_values, print_policy
from pprint import pprint



GAMMA = 0.9
EPSILON = 0.2
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')
N_EPISODES = 100

# epsilon greedy action selection
def epsilon_action(a, eps):
    p = np.random.random()
    if p < 1-eps:
        return a
    else:
        posible_actions = tuple([i for i in ALL_POSSIBLE_ACTIONS if i is not a])
        return np.random.choice(posible_actions)


def play_game(grid, policy):
    states_actions_rewards = []
    starting_state = (2, 0)
    grid.set_state(starting_state)
    while True:
        state = grid.current_state()
        action = epsilon_action(P[state], EPSILON)
        reward = grid.move(action)
        states_actions_rewards.append((state, action, reward))
        if grid.game_over():
            break
    states_actions_gains = []
    gain = 0
    for (state, action, reward) in reversed(states_actions_rewards):
        gain = reward + GAMMA * gain
        states_actions_gains.append((state, action, gain))
        
    states_actions_gains.reverse()
    return states_actions_gains


#if __name__ == '__main__':
grid = standard_grid(obey_prob=1, step_cost=None)

G = dict()
Q = dict()
for state in grid.all_states():
    G[state] = dict()
    Q[state] = dict()
    for action in ALL_POSSIBLE_ACTIONS:
        G[state][action] = []
        Q[state][action] = 0

V = dict()
for state in grid.all_states():
    V[state] = 0

P = dict()
for state in grid.non_terminal_states():
    P[state] = np.random.choice(ALL_POSSIBLE_ACTIONS)

for episode in range(N_EPISODES):
    states_actions_gains=play_game(grid, P)

    for (state, action, gain) in states_actions_gains:
        G[state][action].append(gain)

    for state in grid.all_states():
        for action in ALL_POSSIBLE_ACTIONS:
            gain_list = G[state][action]
            Q[state][action] = np.mean(gain_list) if len(gain_list) > 0 else 0





