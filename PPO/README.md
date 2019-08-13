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

![Grahps on the clipped onbjective](https://cdn.mathpix.com/snip/images/rOE0Qk0X01GdUDxYH2S7m0WQ4dzMcWeJS4D9ZMEi8nU.original.fullsize.png)

### Another approach

* Use a penalty on KL divergence
* adapt the penalty coefficient so that we achieve some target value of the KL divergence $d_{targ}$ each policy update.

In each policy update the two following steps are performed:

* Optimize KL-Penalized objective: $L^{K L P E N}(\theta)=\hat{\mathbb{E}}_{t}\left[\frac{\pi_{\theta}\left(a_{t} | s_{t}\right)}{\pi_{\theta_{\text { old }}}\left(a_{t} | s_{t}\right)} \hat{A}_{t}-\beta \mathrm{KL}\left[\pi_{\theta_{\text { old }}}\left(\cdot | s_{t}\right), \pi_{\theta}\left(\cdot | s_{t}\right)\right]\right]$
* Compute $d=\hat{\mathbb{E}}_{t}\left[\mathrm{KL}\left[\pi_{\theta \mathrm{old}}\left(\cdot | s_{t}\right), \pi_{\theta}\left(\cdot | s_{t}\right)\right]\right]$
  * $\text { If } d<d_{\operatorname{targ}} / 1.5, \beta \leftarrow \beta / 2$
  * $\text { If } d>d_{\text { targ }} \times 1.5, \beta \leftarrow \beta \times 2$
* The updated $\beta$ is used in the next policy update
* The parameters 1.5 and 2 above are chosen heuristically, but the algorithm is not very sensitive to them.
* The initial value of $\beta$ is a hyperparameter but the algorithm quickly adjust it. Thus, it is not important.

## The PPO Algorithm

With automatic differentiation, we optimize $L^{CLIP}$ or $L^{KLPEN}$ (by performing stochastic gradient ascent on this objective)

We must use a loss function that combines the policy surrogate and a value function error term. Augmenting by adding an entropy bonus for ensuring sufficient exploration.

We maximize in each iteration (Optimize using stochastic gradient ascent):

$L_{t}^{C L I P+V F+S}(\theta)=\hat{\mathbb{E}}_{t}\left[L_{t}^{C L I P}(\theta)-c_{1} L_{t}^{V F}(\theta)+c_{2} S\left[\pi_{\theta}\right]\left(s_{t}\right)\right]$

$c_{1}$ and $c_{2}$ are coefficients and $S$ is the entropy bonus and $L_{t}^{V F} = \left(V_{\theta}\left(s_{t}\right)-V_{t}^{\mathrm{targ}}\right)^{2}$ (Squared error loss)

One style of policy gradient implementation runs the policy for $T$ timesteps (less than the length of an episode) and uses the samples for the updates. The advantage estimator should not look beyond T also:

$\hat{A}_{t}=-V\left(s_{t}\right)+r_{t}+\gamma r_{t+1}+\cdots+\gamma^{T-t+1} r_{T-1}+\gamma^{T-t} V\left(s_{T}\right)$

$t{\large\in} [0, T]$

We can use a truncated version of generalized advantage estimation:

$\begin{array}{l}{\hat{A}_{t}=\delta_{t}+(\gamma \lambda) \delta_{t+1}+\cdots+\cdots+(\gamma \lambda)^{T-t+1} \delta_{T-1}} \\ {\text { where } \quad \delta_{t}=r_{t}+\gamma V\left(s_{t+1}\right)-V\left(s_{t}\right)}\end{array}$

In the following implementation fixed trajectory length segements are used.

* N parallel actors
* In each iteration
  * Each actor collects T timesteps of data
  * Construct surrogate loss on $N*T$ timesteps of data.
  * Optimize using SGD (Adam or Minibatch)

![Algorithm 1 PPO, Actor-Critic Style](https://cdn.mathpix.com/snip/images/mDUtjnTlKORxpuxx8oWa91o9nwwYNhEkhNNNaDskG1I.original.fullsize.png)


### General outline of the Algorithm

1. First, collect some trajectories based on some policy $\pi_\theta$​, and initialize theta prime $\theta'=\theta$
2. Next, compute the gradient of the clipped surrogate function using the trajectories
3. Update $\theta'$ using gradient ascent $\theta'\leftarrow\theta' +\alpha \nabla_{\theta'}L_{\rm sur}^{\rm clip}(\theta', \theta)$
4. Then we repeat step 2-3 without generating new trajectories. Typically, step 2-3 are only repeated a few times
5. Set $\theta=\theta'$, go back to step 1, repeat.
