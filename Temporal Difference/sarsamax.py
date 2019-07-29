def epsilon_greedy(Q, state, nA, eps):
    if random.random() > eps:
        return np.argmax(Q[state])
    else:
        return random.choice(np.arange(nA))

def update_Q_sarsamax(alpha, gamma, Q, state, action, reward, next_state=None):
    current = Q[state][action]
    Qsa_next = np.max(Q[next_state]) if next_state is not None else 0
    target = reward + (gamma*Qsa_next)
    new_value = current +(alpha*(target - current))
    return new_value

def q_learning(env, num_episodes, alpha, gamma=1.0, plot_every=100):
    nA = env.action_space.n
    Q = defaultdict(lambda:np.zeros(nA))
    tmp_scores = deque(maxlen=plot_every)
    avg_scores = deque(maxlen=num_episodes)

    for i_episode in range(1, num_episodes+1):
        if i_episode % 100 == 0:
            print('\r Episode {}/{}'.format(i_episode,num_episodes),end="")
            sys.stdout.flush()
        score = 100
        state = env.reset()
        eps = 1.0/i_episode
        while True:
            action = epsilon_greedy(Q, state, nA, eps)
            next_state, reward, done, info = env.step(action)
            score += reward
            Q[state][action] = update_Q_sarsamax(alpha, gamma, Q, state, action, reward, next_state)
            state = next_state
            if done:
                tmp_scores.append(score)
                break
        if (i_episode % plot_every == 0):
            avg_scores.append(np.mean(tmp_scores))
    print("Average Score {:.2f}".format(np.mean(avg_scores)))
    return Q

Q_sarsa = q_learning(env, 5000, .01)

# print the estimated optimal policy
policy_sarsa = np.array([np.argmax(Q_sarsa[key]) if key in Q_sarsa else -1 for key in np.arange(48)]).reshape(4,12)
# check_test.run_check('td_control_check', policy_sarsa)
print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
print(policy_sarsa)

# plot the estimated optimal state-value function
V_sarsa = ([np.max(Q_sarsa[key]) if key in Q_sarsa else 0 for key in np.arange(48)])
# plot_values(V_sarsa)
