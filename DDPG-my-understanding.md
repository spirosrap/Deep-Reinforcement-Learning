## Explain to me what the RL algorithm DDPG is and how it works

![img](https://cdn-images-1.medium.com/max/1600/1*qV8STzz6mEYIKjOXyibtrQ.png)

### Background

DDPG is a policy gradient (PG) method. In PG methods we use non linear function approximators (Deep Learning networks) to learn value functions. In the past it was considered that Function Approximators were unstable and difficult.

When action spaces are continuous it's difficult to use DQN-like networks, the problem becomes intractable. We can solve this problem if we use function approximators (PG methods) to estimate the action value function.

Consider a standard RL environment $E$ in which the agents interacts with in discrete timesteps. At each time step the agent receives and observation $s_t$ (Environment assumed fully observed).

The agent behaves according to the policy $\pi : \mathcal{S} \rightarrow \mathcal{P}(\mathcal{A})$ (Maps states $\mathcal{S}$ to probability distributions over actions $\mathcal{A}$)

The $E$ environment may be stochastic and we model it as an MDP (Markov Decision Process):

State Space $\mathcal{S}$ and action space over real numbers $\mathcal{A}=\mathbb{R}^{N}$

Initial distribution $p(s_1)$

Transition dynamics $p(s_{t+1} | s_t,a_t)$

Reward function $r(s_t,a_t)$

Return from a state: $R_{t}=\sum_{i=t}^{T} \gamma^{(i-t)} r\left(s_{i}, a_{i}\right)$ with discounting factor $\gamma\in[0,1]$

The return may be stochastic because the policy may be stochastic.

We denote the state visitation distribution as $\rho^{\pi}$

The action value function describes the expected return after taking an action $a_t$ in state $s_t$ following policy $\pi$.

$Q^{\pi}\left(s_{t}, a_{t}\right)=\mathbb{E}_{r_{i \geq t}, s_{i>t} \sim E, a_{i>t} \sim \pi}\left[R_{t} | s_{t}, a_{t}\right]$

The above is written also as follows using the **Bellman equation**

$Q^{\pi}\left(s_{t}, a_{t}\right)=\mathbb{E}_{r_{t}, s_{t+1} \sim E}\left[r\left(s_{t}, a_{t}\right)+\gamma \mathbb{E}_{a_{t+1} \sim \pi}\left[Q^{\pi}\left(s_{t+1}, a_{t+1}\right)\right]\right]$

When the target policy is deterministic we can say it's a fucntion $\mu : \mathcal{S} \leftarrow \mathcal{A}$ so we no longer need to use an expectation inside the braket:

$Q^{\mu}\left(s_{t}, a_{t}\right)=\mathbb{E}_{r_{t}, s_{t+1} \sim E}\left[r\left(s_{t}, a_{t}\right)+\gamma Q^{\mu}\left(s_{t+1}, \mu\left(s_{t+1}\right)\right)\right]$

Because the above expectation depends only on the Environment we can calculate it off-policy using different transitions from another stochastic behavior policy $\beta$ (Different distribution of trajectories)

Q-learning , an off policy method, uses the greedy policy

$\mu(s)=\arg \max _{a} Q(s, a)$


We use function approximator to compute $Q(s_t,a_t)$ parameterized by $\theta^{Q}$

We find the right parameters by minimizing the loss:

$L\left(\theta^{Q}\right)=\mathbb{E}_{s_{t} \sim \rho^{\beta}, a_{t} \sim \beta, r_{t} \sim E}\left[\left(Q\left(s_{t}, a_{t} | \theta^{Q}\right)-y_{t}\right)^{2}\right]$

$y_{t}=r\left(s_{t}, a_{t}\right)+\gamma Q\left(s_{t+1}, \mu\left(s_{t+1}\right) | \theta^{Q}\right)$

It is obvious that $y_t$ depends on $\theta^{Q}$ but this is ignored.

In DDPG, they have adapted Q-Learning with use of neural networks as function approximators. To scale it they used a Replay buffer and a target network to calculate $y_t$

### The algorithm

An actor-critic approach based on the DPG algorithm is used.

DPG maintains an actor function $\mu\left(s | \theta^{\mu}\right)$ (specifies the policy) for mapping (deterministically) states to a specific action .

The critic $Q(s,a)$ is learned using the Bellman equation (Q-learning)

$\nabla_{\theta^{\mu}} J \approx \mathbb{E}_{s_{t} \sim \rho^{\beta}}\left[\nabla_{\theta^{\mu}} Q\left.\left(s, a | \theta^{Q}\right)\right|_{s=s_{t}, a=\mu\left(s_{t} | \theta^{\mu}\right)}\right]$

and with the chain rule:

$\nabla_{\theta^{\mu}} J \approx \mathbb{E}_{s_{t} \sim \rho^{\beta}}\left[\nabla_{a} Q\left.\left(s, a | \theta^{Q}\right)\right|_{s=s_{t}, a=\mu\left(s_{t}\right)} \nabla_{\theta_{\mu}} \mu\left.\left(s | \theta^{\mu}\right)\right|_{s=s_{t}}\right]$

##### Replay Buffer

The challenge when using Deep NNs for RL is that optimization algorithms assume that the samples are independently and identically distributed. However, in RL the states highly depend on each other when they're generated sequentially. Also we have to use mini-batches to take advantage of hardware optimizations.

To address these issues DDPG uses a replay buffer. The replay buffer is a cache of a finite size $\mathcal{R}$ from which the algorithm samples actions randomly. So, with we get to train using an uncorrelated transitions, thus improving training.
