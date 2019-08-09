# Proximal Policy Optimization Algorithm - Notes from the paper
https://arxiv.org/abs/1707.06347

## Introduction
### Basics
PPO alternates between sampling data through interaction with the environment, and optimizing a **surrogate objective function** using stochastic gradient ascent.

Whereas standard policy gradient methods perform one gradient update per data sample, we propose a novel objective function that enables multiple epochs of minibatch updates.

### PPO Advantages

PPO has advantages over:
* Q-Learning (with a neural network function approximation) *fails on many simple problems and is poorly understood*
* VPG methods (REINFORCE) *have poor data efficiency  and robustness*
* TRPO is complicated and not compatible with architectures that include noise (dropout) or parameter sharing (between the policy and  value function)

### PPO Characteristics
* clipped probability ratios
* To optimize the policies we alternate between sampling data from the policy and performing updates on the policy.

Most popular form of the gradient estimator:

$\hat{g}=\hat{\mathbb{E}}_{t}\left[\nabla_{\theta} \log \pi_{\theta}\left(a_{t} | s_{t}\right) \hat{A}_{t}\right]$

the expectation $\hat{\mathbb{E}}_{t}[...]$ indicates the empirical average over a finite batch of samples, in an algorithm that alternates between sampling and optimization.

Implementations that use automatic differentiation software work by constructing an **objective function** whose gradient is the policy gradient estimator.

Estimator $\hat{g}$ is obtained by differentiating the objective function: $L^{P G}(\theta)=\hat{\mathbb{E}}_{t}\left[\log \pi_{\theta}\left(a_{t} | s_{t}\right) \hat{A}_{t}\right]$

### Trust region Methods

In TRPO the "surrogate objective" is maximized subject to a constraint of the size of the policy update.

$\underset{\theta}{\operatorname{maximize}} \quad \hat{\mathbb{E}}_{t}\left[\frac{\pi_{\theta}\left(a_{t} | s_{t}\right)}{\pi_{\theta_{\text { old }}}\left(a_{t} | s_{t}\right)} \hat{A}_{t}\right]$

Subject to $\hat{\mathbb{E}}_{t}\left[\mathrm{KL}\left[\pi_{\theta \mathrm{old}}\left(\cdot | s_{t}\right), \pi_{\theta}\left(\cdot | s_{t}\right)\right]\right] \leq \delta$

**But** actually the theory justifying TRPO suggests using a penalty instead of a constraint:

$\underset{\theta}{\operatorname{maximize}} \hat{\mathbb{E}}_{t}\left[\frac{\pi_{\theta}\left(a_{t} | s_{t}\right)}{\pi_{\theta_{\text { old }}\left(a_{t} | s_{t}\right)}} \hat{A}_{t}-\beta \mathrm{KL}\left[\pi_{\theta_{\text { old }}}\left(\cdot | s_{t}\right), \pi_{\theta}\left(\cdot | s_{t}\right)\right]\right]$
***
TRPO maximizes a surrogate objective:

$L^{C P I}(\theta)=\hat{\mathbb{E}}_{t}\left[\frac{\pi_{\theta}\left(a_{t} | s_{t}\right)}{\pi_{\theta_{\text { old }}}\left(a_{t} | s_{t}\right)} \hat{A}_{t}\right]=\hat{\mathbb{E}}_{t}\left[r_{t}(\theta) \hat{A}_{t}\right]$

CPI refers to a specific (conservative) policy iteration.

We need a constraint because without a constraint the maximization will lead to an excessively large policy update:

Tge main objective that is propposed is

$L^{C L I P}(\theta)=\hat{\mathbb{E}}_{t}\left[\min \left(r_{t}(\theta) \hat{A}_{t}, \operatorname{clip}\left(r_{t}(\theta), 1-\epsilon, 1+\epsilon\right) \hat{A}_{t}\right)\right]$

The "$\operatorname{clip}\left(r_{t}(\theta), 1-\epsilon, 1+\epsilon\right) \hat{A}_{t}$" keeps the ratio between $[1-\epsilon, 1+\epsilon]$

Finally we take the minimum between the surrogate objective and the clipped surrogate objective.

Note  that $L^{C L I P}(\theta)=L^{C P I}(\theta)$ while $\theta$  around $\theta_{old}$ ($r=1$) , $r_{t}(\theta)=\frac{\pi_{\theta}\left(a_{t} | s_{t}\right)}{\pi_{\theta_{\mathrm{old}}}\left(a_{t} | s_{t}\right)}$

![img](https://cdn.mathpix.com/snip/images/rOE0Qk0X01GdUDxYH2S7m0WQ4dzMcWeJS4D9ZMEi8nU.original.fullsize.png)

### Another approach


### General outline of the Algorithm

1. First, collect some trajectories based on some policy $\pi_\theta$​, and initialize theta prime $\theta'=\theta$
2. Next, compute the gradient of the clipped surrogate function using the trajectories
3. Update $\theta'$ using gradient ascent $\theta'\leftarrow\theta' +\alpha \nabla_{\theta'}L_{\rm sur}^{\rm clip}(\theta', \theta)$
4. Then we repeat step 2-3 without generating new trajectories. Typically, step 2-3 are only repeated a few times
5. Set $\theta=\theta'$, go back to step 1, repeat.
