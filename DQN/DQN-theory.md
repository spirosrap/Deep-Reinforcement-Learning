## Deep QRLN

We want to approximate the action value function $\hat{q}(S, A, w)$ corresponding to state $S$ and action $A$

$$
\Delta w=\alpha \cdot \underbrace{\overbrace{\left(R+\gamma \max _{a} \hat{q}\left(S^{\prime}, a, w^{-}\right)\right.}_{\text { TD target }}-\hat{q}(\underbrace{S, A, w )}_{\text { old value }})} \nabla_{w} \hat{q}(S, A, w)
$$

Implemented in python ``` loss = F.mse_loss(Q_expected, Q_targets) ```

$w^{-}$ : weights of a separate network (`qnetwork_local` from Udacity exercise) that are not changed during learning

$\left(S, A, R, S^{\prime}\right)$ : Experience tuple
