## Deep QRLN
*For a more like the paper implementation: https://github.com/8Gitbrix/DQN*

***
**Uses Convolutions**

**Convolutions are inspired by Hube's and Wiesels' work (1960)**

(*The receptive field is a portion of sensory space that can elicit neuronal responses when stimulated.*)

*Hubel and Wiesel advanced the theory that receptive fields of cells at one level of the visual system are formed from input by cells at a lower level of the visual system. In this way, small, simple receptive fields could be combined to form large, complex receptive fields.*
***
We use convolutional filters (a Deep learning network) to approximate the action value function:

$Q^{*}(s, a)=\max _{\pi} \mathbb{E}\left[r_{t}+\gamma r_{t+1}+\gamma^{2} r_{t+2}+\ldots | s_{t}=s, a_{t}=a, \pi\right]$

Function approximators have been unstable due to correlations present in the sequence of observations. Small Q updates may significantly change the policy and therefor the data distribution. There are also correlations between Q and target values.

Uses experience replay that randomizes over data.

To perform experience replay we store the agent’s experiences $e_{t}=\left(s_{t}, a_{t}, r_{t}, s_{t+1}\right)$ at each time-step $t$ in a data set $D_{t}=\left\{e_{1}, \ldots, e_{t}\right\}$. During learning, we apply Q-learning updates, on samples (or minibatches) of experience $\left(s, a, r, s^{\prime}\right) \sim U(D)$ , drawn uniformly at random from the pool of stored samples.

Uses Loss Function:

$L_{i}\left(\theta_{i}\right)=\mathbb{E}_{\left(s, a, r, s^{\prime}\right) \sim \mathrm{U}(D)}\left[\left(r+\gamma \max _{a^{\prime}} Q\left(s^{\prime}, a^{\prime} ; \theta_{i}^{-}\right)-Q\left(s, a ; \theta_{i}\right)\right)^{2}\right]$

$\theta_i$ are the parameters of the Q-network at iteration $i$ and $\theta_i^-$ are the network parameters used to compute the target at iteration i. The target network parameters $\theta_i^-$ are only updated with the Q-network parameters ($\theta_i$) every C steps and are held fixed between individual updates.

![DQN Pseudocode](https://miro.medium.com/max/1400/1*nb61CxDTTAWR1EJnbCl1cA.png)


In the paper:

>We used the same 1) network architecture, 2)hyperparameter values and 3) learning procedure throughout—taking high-dimensional data (210|160 colour video at 60 Hz) as input—to demonstrate that our approach robustly learns **successful policies over a variety of games based solely on sensory inputs with only very minimal prior knowledge** (that is, merely the input data were visual images, and the number of actions available in each game, but not their correspondences)

Important the individual core components of the DQN agent:
* the replay memory
* separate target Q-network
* deep convolutional network architecture

without these the performance deteriorates

**The representations learned by DQN are able to generalize to data generated from policies other than its own**

***
We want to approximate the action value function $\hat{q}(S, A, w)$ corresponding to state $S$ and action $A$

$$
\Delta w=\alpha \cdot \underbrace{\overbrace{\left(R+\gamma \max _{a} \hat{q}\left(S^{\prime}, a, w^{-}\right)\right.}_{\text { TD target }}-\hat{q}(\underbrace{S, A, w )}_{\text { old value }})} \nabla_{w} \hat{q}(S, A, w)
$$

Implemented in python ``` loss = F.mse_loss(Q_expected, Q_targets) ```

$w^{-}$ : weights of a separate network (`qnetwork_local` from Udacity exercise) that are not changed during learning

$\left(S, A, R, S^{\prime}\right)$ : Experience tuple
