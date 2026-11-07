# Hand verification -- Assignment 3

This file backs up Section III-D of the report with the full arithmetic.
Two separate things live here and should not be confused with each other:

1. A small, hand-chosen **illustrative** network (2 inputs, 2 hidden ReLU
   units, 2 outputs), used only to show the forward pass / Bellman target /
   TD error / loss / gradient-update mechanics on matrices small enough to
   write out completely. These weights were never trained. They are not a
   result.
2. The **real seed-4 trained network** (loaded from
   `results/seed4_qnetwork_weights.pt`, the actual 8-32-32-7 architecture),
   evaluated on a real transition sampled from the real environment. These
   numbers are genuine experimental output, produced by
   `code/hand_verification.py`, not computed by hand.

Reproduce everything here with:

```
cd code
python hand_verification.py
```

which also writes `results/hand_verification_numbers.json`.

## Part 1 -- Illustrative small network (not the trained model)

Toy setup, in the same style as Lecture 11's worked example, adapted to
cost minimization (`min` instead of `max`). Two hidden ReLU units, two
outputs standing in for two actions ("Left" $\approx A_b$, "Right"
$\approx A_c$), with a 2-dimensional toy state encoding -- not our real
8-dim one-hot, just small enough to write by hand.

$$W_1=\begin{bmatrix}0.5&0.2\\0.3&0.4\end{bmatrix},\quad b_1=\begin{bmatrix}0\\0\end{bmatrix},\quad
W_2=\begin{bmatrix}0.6&0.5\\0.4&0.7\end{bmatrix},\quad b_2=\begin{bmatrix}0\\0\end{bmatrix}$$

**Forward pass**, state $s=[1,0]$, action taken = Left:

$$z_1=W_1 s+b_1=[0.5,\ 0.3],\qquad h_1=\mathrm{ReLU}(z_1)=[0.5,\ 0.3]$$
$$Q_\theta(s,\cdot)=W_2 h_1+b_2=[0.45,\ 0.41]\ \Rightarrow\ Q_\theta(s,\text{Left})=0.45$$

**Sampled transition**: cost $c=2$, next state $s'=[0,1]$, not terminal
($d=0$). With $\theta^-=\theta$ (target net not yet out of sync):

$$h_1'=\mathrm{ReLU}(W_1 s')=[0.2,\ 0.4],\qquad Q_{\theta^-}(s',\cdot)=W_2 h_1'+b_2=[0.32,\ 0.36]$$
$$\min_{a'}Q_{\theta^-}(s',a')=0.32$$

**Bellman target, TD error, loss** ($\gamma=1$):

$$y=c+\gamma(1-d)\min_{a'}Q_{\theta^-}(s',a')=2+1(1-0)(0.32)=2.32$$
$$\delta=y-Q_\theta(s,\text{Left})=2.32-0.45=1.87$$
$$L=\delta^2=1.87^2=3.4969$$

**Backprop** (single sample, $\alpha=0.1$). Because $L=\delta^2$ with $y$
treated as a constant, $\partial L/\partial Q_\theta(s,a)=-2\delta=-3.74$.
Only the "Left" output head was used, so the output-layer gradient vector
is $g^{(o)}=[-3.74,\ 0]$ (the chain rule in the loss code above uses
$L=\delta^2$ directly rather than the $\tfrac{1}{2B}\delta^2$ convention
of Lecture 11, which only rescales $g^{(o)}$ by a constant factor and does
not change the argument). Propagating through $Q_\theta=W_2h_1+b_2$ and
$h_1=\mathrm{ReLU}(W_1 s+b_1)$:

$$\frac{\partial L}{\partial W_2}=g^{(o)}h_1^\top=\begin{bmatrix}-1.87&-1.122\\0&0\end{bmatrix},\qquad
\frac{\partial L}{\partial b_2}=\begin{bmatrix}-3.74\\0\end{bmatrix}$$
$$\frac{\partial L}{\partial h_1}=W_2^\top g^{(o)}=\begin{bmatrix}-2.244\\-1.87\end{bmatrix},\qquad
\frac{\partial L}{\partial z_1}=\frac{\partial L}{\partial h_1}\odot\mathbb{1}[z_1>0]=\begin{bmatrix}-2.244\\-1.87\end{bmatrix}$$
$$\frac{\partial L}{\partial W_1}=\frac{\partial L}{\partial z_1}s^\top=\begin{bmatrix}-2.244&0\\-1.87&0\end{bmatrix},\qquad
\frac{\partial L}{\partial b_1}=\begin{bmatrix}-2.244\\-1.87\end{bmatrix}$$

**Gradient-descent update**, $\theta\leftarrow\theta-\alpha\nabla_\theta L$:

$$W_1^{\text{new}}=\begin{bmatrix}0.7244&0.2\\0.487&0.4\end{bmatrix},\qquad
W_2^{\text{new}}=\begin{bmatrix}0.787&0.6122\\0.4&0.7\end{bmatrix}$$
$$b_1^{\text{new}}=[0.2244,\ 0.187],\qquad b_2^{\text{new}}=[0.374,\ 0]$$

The point of this toy example is the mechanics: the target $y$ gets
computed first and is then held fixed, the TD error measures how far the
online prediction is from that target, and backprop pushes every weight
that contributed to the prediction in the direction that would have
shrunk $\delta$. None of these numbers appear anywhere in the trained
seed-4 network below.

## Part 2 -- Real seed-4 network, one real transition

Loaded from `results/seed4_qnetwork_weights.pt` (the actual trained
8-32-32-7 network), the exact architecture used for all five experiments.
A transition was sampled with a fresh RNG stream (`seed=999999`, not
reused from training or evaluation) using the same `step()` function
used throughout training.

Sampled transition: $s=b$, $a=A_e$, giving $c=2$, $s'=e$, $d=0$.

First 5 of 32 hidden-layer-1 pre/post-activations:

```
z1[:5] = [ 0.011556,  0.611428, -0.167326, -0.025436,  0.787745]
h1[:5] = [ 0.011556,  0.611428,  0.000000,  0.000000,  0.787745]
```

(entries 3 and 4 were clipped to 0 by ReLU, as expected for negative
pre-activations). First 5 of 32 hidden-layer-2 values:

```
z2[:5] = [ 1.299297,  0.934944, -0.014675,  1.694506,  1.931410]
h2[:5] = [ 1.299297,  0.934944,  0.000000,  1.694506,  1.931410]
```

Full 7-action output of the online network at $s=b$ (order $A_b,A_c,A_d,A_e,A_f,A_g,A_h$):

```
Q_theta(b,.) = [5.9462, 6.5005, 6.9486, 6.1291, 5.0477, 3.8309, 0.8800]
```

Only $A_d,A_e$ are feasible at $b$; the network has not been told this at
inference time, so the other 5 entries are meaningless and must be masked
before any argmin/min, exactly as in training. The action actually taken
was $A_e$:

$$Q_\theta(b,A_e)=6.1291$$

Target network at $s'=e$ (same weights as the online network here, since
no sync has happened since the last training step):

```
Q_theta-(e,.) = [5.7188, 6.1834, 6.2149, 5.7786, 4.9172, 3.9649, 0.9855]
```

State $e$ has exactly one feasible action, $A_g$ (index 5), so after
masking, $\min_{a'}Q_{\theta^-}(e,a')=Q_{\theta^-}(e,A_g)=3.9649$ -- this
matches the unmasked value exactly, confirming the mask picked the right
entry.

**Bellman target, TD error, loss**:

$$y=2+1.0(1-0)(3.9649)=5.9649$$
$$\delta=5.9649-6.1291=-0.1642$$
$$L=\delta^2=0.0270$$

The TD error is small and negative: the network's current estimate of
$Q(b,A_e)$ is slightly above the one-step bootstrap target, which is the
kind of residual error expected from a converged-but-not-exact estimator
on a stochastic environment, not a sign of a bug.

## Part 3 -- real minibatch loss (sanity check against the training code)

To check the real training loss function end to end (not just one
transition), 8 fresh transitions were rolled out from `a` using the
trained seed-4 network greedily ($\epsilon=0$), with predictions and
targets computed exactly as `train_one_seed()` does:

```
y_b    = [8.1291, 5.9649, 4.0005, 1.0000, 8.1291, 5.9649, 4.0005, 1.0000]
pred_q = [8.4433, 6.1291, 3.9649, 1.0005, 8.4433, 6.1291, 3.9649, 1.0005]
batch MSE loss (torch.nn.functional.mse_loss) = 0.031736
```

The repeated pattern (two identical 4-step rollouts back to back) happens
because the greedy policy is deterministic once $\epsilon=0$ and the
environment resampled the same intended branch both times; this is
expected, not a bug. `pred_q` for the first transition, $Q_\theta(a,A_b)
\approx 8.443$, is close to the Assignment-2 optimal value $V^*(a)=8.44$,
and the later entries ($\approx6.13$, $\approx3.96$, $\approx1.00$) line
up with the Assignment-2 optimal values $V^*(b)=6.20$ and the per-step
cost-to-go along $b\to e\to g\to h$, which is exactly the cross-check the
report's discussion section relies on.

## Limitations found

- The training loss (Fig. 2 in the report) plateaus around 0.1-0.2 rather
  than approaching 0; this is environment stochasticity and the 0.05
  exploration floor, not a training bug, and is explicitly not claimed as
  convergence in the report.
- The 1000-episode evaluation used $\epsilon=0$ but the environment's own
  $0.8/0.2$ randomness remains, so individual evaluation episodes still
  range from the best case (cost 8) to a worse case (cost 11); this matches
  the discrete edge-cost structure of the graph and is reported as-is in
  Table II of the report.
- No seed failed to reach $h$ or failed to recover the optimal policy; we
  did not need to diagnose a training failure for this submission.
