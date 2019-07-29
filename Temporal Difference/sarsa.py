import sys
import gym
import numpy as np
import random
import math
from collections import defaultdict, deque
env = gym.make('CliffWalking-v0')

def update_Q_sarsa(alpha, gamma, Q, state, action, reward, next_state=None, next_action=None):
    current = Q[state][action]
    Qsa_next = Q[next_state][next_action] if next_state is not None else 0
    target = reward + (gamma * Qsa_next)
    new_value = current + (alpha * (target - current))
    return new_value

def epsilon_greedy(Q, state, nA, eps):
    if random.random() > eps:
        return np.argmax(Q[state])
    else:
        return random.choice(np.arange(env.action_space.n))

def sarsa(env, num_episodes, alpha, gamma=1.0, plot_every=100):
    nA = env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))

    tmp_scores = deque(maxlen=plot_every)
    avg_scores = deque(maxlen=num_episodes)

    for i_episode in range(1, num_episodes + 1):
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
#             print('\r Episode {}/{}'.format(i_episode,num_episodes), end="")
            sys.stdout.flush()
        score = 0
        state = env.reset()

        eps = 1.0/i_episode
        action = epsilon_greedy(Q, state, nA, eps)
        while True:
            next_state, reward, done, info = env.step(action)
            score += reward
            if not done:
                next_action = epsilon_greedy(Q, next_state, nA, eps)
                Q[state][action] = update_Q_sarsa(alpha, gamma, Q, \
                                                  state, action, reward, next_state, next_action)
                state = next_state
                action = next_action
            if done:
                Q[state][action] = update_Q_sarsa(alpha, gamma, Q, state, action, reward)
                tmp_scores.append(score)    # append score
                break
        if(i_episode % plot_every == 0):
            avg_scores.append(np.mean(tmp_scores))
    print("Average score {:.2f}".format(np.max(avg_scores)))
    return Q


Q_sarsa = sarsa(env, 5000, .01)

# print the estimated optimal policy
policy_sarsa = np.array([np.argmax(Q_sarsa[key]) if key in Q_sarsa else -1 for key in np.arange(48)]).reshape(4,12)
# check_test.run_check('td_control_check', policy_sarsa)
print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
print(policy_sarsa)

# plot the estimated optimal state-value function
V_sarsa = ([np.max(Q_sarsa[key]) if key in Q_sarsa else 0 for key in np.arange(48)])
# plot_values(V_sarsa)
