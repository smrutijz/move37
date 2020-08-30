import numpy as np
from grid_world import standard_grid
from utils import print_values, print_policy
from pprint import pprint


SMALL_ENOUGH = 1e-3
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

def best_action_value(grid, V, state):
    best_a = None
    best_value = float("-inf")
    grid.set_state(state)
    for action in ALL_POSSIBLE_ACTIONS:
        transititions = grid.get_transition_probs(action)
        expected_reward = 0
        expected_value = 0
        for (prob, r, state_prime) in transititions:
            expected_reward += prob * r
            expected_value += prob * V[state_prime]
        value = expected_reward + GAMMA * expected_value
                    
        if value > best_value:
            best_action = action
            best_value = value
    return best_action, best_value




def calculate_values(grid):
    V = dict()
    for state in grid.all_states():
        V[state] = 0
    episode = 0
    while True:
        biggest_change = 0
        for state in grid.non_terminal_states():
            v_old = V[state]
            _, v_new = best_action_value(grid, V, state)
            V[state] = v_new
            biggest_change = max(biggest_change, np.abs(v_old - v_new))
        episode += 1
        #print("episode:", episode, "max vlaue change:", biggest_change)
        if biggest_change <= SMALL_ENOUGH:
            break
    return V

def calculate_greedy_policy(grid, V):
    P = dict()
    for state in grid.non_terminal_states():
        P[state] = np.random.choice(ALL_POSSIBLE_ACTIONS)
    for state in grid.non_terminal_states():
        best_action, _ = best_action_value(grid, V, state)
        P[state] = best_action
    return P


if __name__ == '__main__':
  grid = standard_grid(obey_prob=0.8, step_cost=None)

  # print rewards
  print("rewards:")
  print_values(grid.rewards, grid)

  # calculate accurate values for each square
  V = calculate_values(grid)

  # calculate the optimum policy based on our values
  P = calculate_greedy_policy(grid, V)

  # our goal here is to verify that we get the same answer as with policy iteration
  print("values:")
  print_values(V, grid)
  print("policy:")
  print_policy(P, grid)


