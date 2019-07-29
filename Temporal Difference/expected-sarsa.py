def epsilon_greedy(Q, state, nA, eps):
    if random.random() > eps:
        return np.argmax(Q[state])
    else:
        return random.choice(np.arange(nA))

def update_Q_expsarsa(alpha, gamma, nA, eps, Q, state, action, reward, next_state=None):
    current = Q[state][action]
    policy_s = np.ones(nA)*eps/nA
    policy_s[np.argmax(Q[next_state])] = 1 - eps + eps/nA
    Qsa_next = np.dot(Q[next_state], policy_s)
    target = reward + (gamma * Qsa_next)
    new_value = current + alpha*(target -current)
    return new_value

def expected_sarsa(env, num_episodes, alpha, gamma=1.0, print_every=100):
    nA = env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))
    tmp_scores = deque(maxlen=print_every)
    avg_scores = deque(maxlen=num_episodes)

    for i_episode in range(1, 1 + num_episodes):
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes),end="")
            sys.stdout.flush()
        score = 0
        state = env.reset()
        eps = 0.005

        while True:
            action = epsilon_greedy(Q, state, nA, eps)
            next_state, reward, done, info = env.step(action)
            score += reward
            Q[state][action] = update_Q_expsarsa(alpha, gamma, nA, eps, Q, state, action, reward, next_state)
            state = next_state
            if done:
                tmp_scores.append(score)
                break
            if i_episode % print_every == 0:
                avg_scores.append(np.mean(tmp_scores))

    print("best score {:.2f}".format(max(avg_scores)))
    return Q

Q_sarsa = expected_sarsa(env, 5000, .01)

# print the estimated optimal policy
policy_sarsa = np.array([np.argmax(Q_sarsa[key]) if key in Q_sarsa else -1 for key in np.arange(48)]).reshape(4,12)
# check_test.run_check('td_control_check', policy_sarsa)
print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
print(policy_sarsa)

# plot the estimated optimal state-value function
V_sarsa = ([np.max(Q_sarsa[key]) if key in Q_sarsa else 0 for key in np.arange(48)])
# plot_values(V_sarsa)    
