## Temporal Difference Methods:

* Whereas Monte Carlo (MC) prediction methods must wait until the end of an episode to update the value function estimate, temporal-difference (TD) methods update the value function after every time step.


### Sarsa
![img](assets/sarsa.png)

* Sarsa(0) (or Sarsa) is an on-policy TD control method. It is guaranteed to converge to the optimal action-value function $q_∗$​, as long as the step-size parameter $\alpha$ is sufficiently small and $\epsilon$ is chosen to satisfy the **Greedy in the Limit with Infinite Exploration (GLIE)** conditions.


### Expected Sarsa
![img](assets/expected-sarsa.png)

* Sarsamax (or Q-Learning) is an off-policy TD control method. It is guaranteed to converge to the optimal action value function $q_∗$​​, under the same conditions that guarantee convergence of the Sarsa control algorithm.


### Sarsamax
![img](assets/sarsamax.png)

* Expected Sarsa is an on-policy TD control method. It is guaranteed to converge to the optimal action value function $q_*$​, under the same conditions that guarantee convergence of Sarsa and Sarsamax.
