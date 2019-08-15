## REINFORCE: $\hat{g} :=\frac{1}{m} \sum_{i=1}^{m} \sum_{t=0}^{H} \nabla_{\theta} \log \pi_{\theta}\left(a_{t}^{(i)} | s_{t}^{(i)}\right) R\left(\tau^{(i)}\right)$

### Basics
* The REINFORCE algorithm is the main and initial example of policy gradient methods.
* The main idea is to push up actions that lead to increased rewards and push down actions that lead to decreased rewards. Repeat until arriving to the optimal policy.
* On policy method
* Exploration vs Exploitation: Randomness usually decreases as the training goes on. Usually determined with a greedy parameter.

The general form of the gradient (the gradient we look to maximize):
![JPtheta](https://i.imgur.com/YfWbNnB.png)


Notice the advantage function $A^{\pi_{\theta}}(s_t,a_t)$ instead of $R(\tau^{(i)})$

Algorithm pseudo-code:

![pseudo-code](https://spinningup.openai.com/en/latest/_images/math/47a7bd5139a29bc2d2dc85cef12bba4b07b1e831.svg)

Algorithm pseudo-code second version less math-heavy:

![Imgur](https://i.imgur.com/xggCX2h.png)



1. Use policy $\pi_{\theta}$ to collect `m` trajectories $\tau^{(1)}, \tau^{(2)}, \ldots, \tau^{(m)}$ with horizon $H$. The `i-th` trajectory: $\tau^{(i)}=\left(s_{0}^{(i)}, a_{0}^{(i)}, \ldots, s_{H}^{(i)}, a_{H}^{(i)}, s_{H+1}^{(i)}\right)$
2. Use trajectories to estimate gradient $\nabla_{\theta} U(\theta)$. $\nabla_{\theta} U(\theta) \approx \hat{g} :=\frac{1}{m} \sum_{i=1}^{m} \sum_{t=0}^{H} \nabla_{\theta} \log \pi_{\theta}\left(a_{t}^{(i)} | s_{t}^{(i)}\right) R\left(\tau^{(i)}\right)$
3. Update the weights of the policy $\theta \leftarrow \theta+\alpha \hat{g}$
4. `loop`

$m$: episodes/trajectories

$H$: how long is the episode/trajectory


**Code**:

$\hat{g} :=\frac{1}{m} \sum_{i=1}^{m} \nabla_{\theta} R\left(\tau^{(i)}\right) \sum_{t=0}^{H} \log \pi_{\theta}\left(a_{t}^{(i)} | s_{t}^{(i)}\right)$

```
for i,log_prob in enumerate(saved_log_probs):
    policy_loss.append(-log_prob * R_sum)
policy_loss = torch.cat(policy_loss).sum()

optimizer.zero_grad()
policy_loss.backward()
optimizer.step()
```


## Derivation of $\nabla_{\theta} U(\theta) \approx \hat{g}=\frac{1}{m} \sum_{i=1}^{m} \sum_{t=0}^{H} \nabla_{\theta} \log \pi_{\theta}\left(a_{t}^{(i)} | s_{t}^{(i)}\right) R\left(\tau^{(i)}\right)$

#### Calculate $\nabla_{\theta} U(\theta)$

$$
\begin{aligned} \nabla_{\theta} U(\theta) &=\nabla_{\theta} \sum_{\tau} P(\tau ; \theta) R(\tau) &(1)\\ &=\sum_{\tau} \nabla_{\theta} P(\tau ; \theta) R(\tau) &(2)\\ &=\sum_{\tau} \frac{P(\tau ; \theta)}{P(\tau ; \theta)} \nabla_{\theta} P(\tau ; \theta) R(\tau) &(2)\\ &=\sum_{\tau} P(\tau ; \theta) \frac{\nabla_{\theta} P(\tau ; \theta)}{P(\tau ; \theta)} R(\tau) &(4) \\ &=\sum_{\tau} P(\tau ; \theta) \nabla_{\theta} \log P(\tau ; \theta) R(\tau) &(5) \end{aligned}
$$

#### Simplify $\nabla_{\theta} \log \mathbb{P}\left(\tau^{(i)} ; \theta\right)$

$$
\begin{aligned} \nabla_{\theta} \log \mathbb{P}\left(\tau^{(i)} ; \theta\right) &=\nabla_{\theta} \log \left[\prod_{t=0}^{H} \mathbb{P}\left(s_{t+1}^{(i)} | s_{t}^{(i)}, a_{t}^{(i)}\right) \pi_{\theta}\left(a_{t}^{(i)} | s_{t}^{(i)}\right)\right] &(1) \\ &=\nabla_{\theta}\left[\sum_{t=0}^{H} \log \mathbb{P}\left(s_{t+1}^{(i)} | s_{t}^{(i)}, a_{t}^{(i)}\right)+\sum_{t=0}^{H} \log \pi_{\theta}\left(a_{t}^{(i)} | s_{t}^{(i)}\right]\right] &(2) \\ &=\nabla_{\theta} \sum_{t=0}^{H} \log \mathbb{P}\left(s_{t+1}^{(i)} | s_{t}^{(i)}, a_{t}^{(i)}\right)+\nabla_{\theta} \sum_{t=0}^{H} \log \pi_{\theta}\left(a_{t}^{(i)} | s_{t}^{(i)}\right) &(3) \\
&=\nabla_{\theta} \sum_{t=0}^{H} \log \pi_{\theta}\left(a_{t}^{(i)} | s_{t}^{(i)}\right) &(4) \\
&=\sum_{t=0}^{H} \nabla_{\theta} \log \pi_{\theta}\left(a_{t}^{(i)} | s_{t}^{(i)}\right) &(5)

 \end{aligned}
$$

$\nabla_{\theta} \sum_{t=0}^{H} \log \mathbb{P}\left(s_{t+1}^{(i)} | s_{t}^{(i)}, a_{t}^{(i)}\right)=0$ because $\sum_{t=0}^{H} \log \mathbb{P}\left(s_{t+1}^{(i)} | s_{t}^{(i)}, a_{t}^{(i)}\right)$ has no dependence on $\theta$


### About the Pytorch implementation included in the current folder (`reinforce.py`)

Changing the full connected layer's (`fc_units`) units to 16,32,64 or more doesn't improve
how quick the model converges - if it ever converges. Also, adding an additional layer
doesn't improve the effectiveness of the algorithm.

### Sources

* https://spinningup.openai.com/en/latest/algorithms/vpg.html
* https://github.com/udacity/deep-reinforcement-learning
* http://joschu.net/docs/thesis.pdf (second chapter), Optimizing Expectations: From Deep Reinforcement Learning to Stochastic Computation Graphs, Schulman 2016(a)
