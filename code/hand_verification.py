"""
Hand-verification numbers for Assignment 3, Section VIII / verification.md.

Produces TWO separate things, never mixed:

1. An ILLUSTRATIVE small 2-hidden-unit network with hand-chosen weights
   (same style as Lecture 11's worked example, adapted to cost-minimization
   with min instead of max). Used only to show the forward pass / Bellman
   target / TD error / loss / gradient-update mechanics on paper-sized
   matrices. These weights were NEVER trained and must not be presented as
   experimentally learned.

2. The REAL seed-4 trained network (loaded from
   results/seed4_qnetwork_weights.pt) evaluated on one ACTUAL transition
   sampled from the real environment with the real training-time step()
   function. These are genuine experimental numbers.

Run from the code/ directory: python hand_verification.py
"""
import os
import json
import numpy as np
import torch

import neural_q_learning as m

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

print("=" * 70)
print("PART 1: ILLUSTRATIVE SMALL NETWORK (hand-chosen, NOT trained)")
print("=" * 70)

# Small illustrative state encoding: 2-dim feature (NOT the real 8-dim
# one-hot) just to keep every matrix hand-computable, exactly in the style
# of Lecture 11. Pretend "x1" flags "at state a", "x2" flags "action toward
# b already available"; this is a toy stand-in, not our real state space.
W1 = np.array([[0.50, 0.20],
               [0.30, 0.40]])   # 2x2, hidden x input
b1 = np.array([0.0, 0.0])
W2 = np.array([[0.60, 0.50],
               [0.40, 0.70]])   # 2x2, output x hidden  (2 "actions": Left=Ab, Right=Ac)
b2 = np.array([0.0, 0.0])

s = np.array([1.0, 0.0])        # illustrative "state a" encoding
a_taken = 0                     # Left / Ab
gamma = 1.0
alpha = 0.1

z1 = W1 @ s + b1
h1 = np.maximum(z1, 0)
Q_s = W2 @ h1 + b2
print("z1 =", z1)
print("h1 = ReLU(z1) =", h1)
print("Q_theta(s,.) = W2 h1 + b2 =", Q_s)
print(f"Q_theta(s, a_taken=Left) = {Q_s[a_taken]:.4f}")

# one illustrative sampled transition: cost c=2, next state s', not terminal
c = 2.0
d = 0
s_next = np.array([0.0, 1.0])   # illustrative "state b" encoding
z1p = W1 @ s_next + b1
h1p = np.maximum(z1p, 0)
Q_sp = W2 @ h1p + b2            # target network == online network here (theta- = theta)
min_next = Q_sp.min()
y = c + gamma * (1 - d) * min_next
delta = y - Q_s[a_taken]
L = delta ** 2
print(f"s' = {s_next}, Q_theta-(s',.) = {Q_sp}, min_a' = {min_next:.4f}")
print(f"y = c + gamma(1-d)min_a' Q = {c} + {gamma}*(1-{d})*{min_next:.4f} = {y:.4f}")
print(f"TD error delta = y - Q(s,a) = {y:.4f} - {Q_s[a_taken]:.4f} = {delta:.4f}")
print(f"Single-sample loss L = delta^2 = {L:.4f}")

# gradient of L wrt output layer (single sample, cost-min convention: same
# chain rule as Lecture 11, only the target construction uses min not max)
dL_dQsa = -2 * delta            # dL/dQ(s,a_taken), since L=(y-Q)^2, y treated as constant
g_out = np.zeros(2)
g_out[a_taken] = dL_dQsa
dW2 = np.outer(g_out, h1)
db2 = g_out
dh1 = W2.T @ g_out
dz1 = dh1 * (z1 > 0)            # ReLU derivative
dW1 = np.outer(dz1, s)
db1 = dz1

print("dL/dW2 =\n", dW2)
print("dL/db2 =", db2)
print("dL/dW1 =\n", dW1)
print("dL/db1 =", db1)

W1_new = W1 - alpha * dW1
W2_new = W2 - alpha * dW2
b1_new = b1 - alpha * db1
b2_new = b2 - alpha * db2
print("W1_new =\n", W1_new)
print("W2_new =\n", W2_new)
print("b1_new =", b1_new, " b2_new =", b2_new)

illustrative = {
    "W1": W1.tolist(), "b1": b1.tolist(), "W2": W2.tolist(), "b2": b2.tolist(),
    "s": s.tolist(), "a_taken": "Left (illustrative Ab)", "c": c, "s_next": s_next.tolist(), "d": d,
    "z1": z1.tolist(), "h1": h1.tolist(), "Q_s": Q_s.tolist(),
    "Q_sp": Q_sp.tolist(), "min_next": float(min_next), "y": float(y),
    "delta": float(delta), "loss": float(L),
    "dW1": dW1.tolist(), "dW2": dW2.tolist(), "db1": db1.tolist(), "db2": db2.tolist(),
    "W1_new": W1_new.tolist(), "W2_new": W2_new.tolist(),
    "b1_new": b1_new.tolist(), "b2_new": b2_new.tolist(),
    "alpha": alpha, "gamma": gamma,
}

print()
print("=" * 70)
print("PART 2: REAL seed-4 TRAINED NETWORK, ONE REAL SAMPLED TRANSITION")
print("=" * 70)

online = m.QNetwork()
online.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "seed4_qnetwork_weights.pt")))
online.eval()

# Draw one real transition with a fresh, documented RNG seed (not reused
# from training or evaluation streams) using the SAME step() function used
# throughout training.
rng = np.random.default_rng(999_999)
s_state = "b"
a_action = "Ae"
s_next_state, cost = m.step(rng, s_state, a_action)
done = 1 if s_next_state == m.TERMINAL else 0
print(f"Sampled transition: (s={s_state}, a={a_action}) -> cost={cost}, s'={s_next_state}, d={done}")

si = m.STATE_IDX[s_state]
ai = m.ACTION_IDX[a_action]
sni = m.STATE_IDX[s_next_state] if not done else m.STATE_IDX["a"]

with torch.no_grad():
    x = m.one_hot(torch.tensor([si]))
    z1_t = online.fc1(x)
    h1_t = torch.relu(z1_t)
    z2_t = online.fc2(h1_t)
    h2_t = torch.relu(z2_t)
    Q_all = online.fc3(h2_t)
    Q_sa = Q_all.squeeze(0)[ai].item()

    xp = m.one_hot(torch.tensor([sni]))
    Q_next_all = online(xp).squeeze(0).numpy()  # target net == online net here (post-training, no further sync)
    masked_next = Q_next_all + m.MASK_ADD[sni]
    min_next_real = masked_next.min() if not done else 0.0

gamma = m.GAMMA
y_real = cost + gamma * (1 - done) * (min_next_real if not done else 0.0)
delta_real = y_real - Q_sa
loss_real = delta_real ** 2

print(f"z1 (32-dim) first 5 entries: {z1_t.squeeze(0).numpy()[:5]}")
print(f"h1 (32-dim) first 5 entries: {h1_t.squeeze(0).numpy()[:5]}")
print(f"z2 (32-dim) first 5 entries: {z2_t.squeeze(0).numpy()[:5]}")
print(f"h2 (32-dim) first 5 entries: {h2_t.squeeze(0).numpy()[:5]}")
print(f"Q_theta(s,.) (7 actions) = {Q_all.squeeze(0).numpy()}")
print(f"Q_theta(s, a={a_action}) = {Q_sa:.6f}")
print(f"Q_theta-(s',.) = {Q_next_all}, masked min_a' = {min_next_real:.6f} (d={done})")
print(f"y = c + gamma(1-d)min_a' Q = {cost} + {gamma}*(1-{done})*{min_next_real:.6f} = {y_real:.6f}")
print(f"TD error delta = y - Q(s,a) = {y_real:.6f} - {Q_sa:.6f} = {delta_real:.6f}")
print(f"Single-sample squared loss = delta^2 = {loss_real:.6f}")

real_example = {
    "state": s_state, "action": a_action, "cost": cost, "next_state": s_next_state, "done": done,
    "Q_all": Q_all.squeeze(0).numpy().tolist(),
    "Q_sa": Q_sa,
    "Q_next_all": Q_next_all.tolist(),
    "min_next": float(min_next_real),
    "y": float(y_real), "delta": float(delta_real), "loss": float(loss_real),
    "z1_first5": z1_t.squeeze(0).numpy()[:5].tolist(),
    "h1_first5": h1_t.squeeze(0).numpy()[:5].tolist(),
    "z2_first5": z2_t.squeeze(0).numpy()[:5].tolist(),
    "h2_first5": h2_t.squeeze(0).numpy()[:5].tolist(),
}

# Also verify a REAL minibatch loss with the actual loss_fn used in training,
# drawn fresh from a documented replay-like sample (rebuild a small buffer
# from a short fresh rollout under the final greedy policy, eps=0, to keep
# this deterministic and reproducible without touching the training replay
# buffer which is not persisted).
print()
print("Real minibatch MSE check (batch of 8 fresh transitions, greedy policy, eps=0):")
batch_transitions = []
rng2 = np.random.default_rng(424242)
s_cur = "a"
for _ in range(8):
    if s_cur == m.TERMINAL:
        s_cur = "a"
    si_b = m.STATE_IDX[s_cur]
    with torch.no_grad():
        q = online(m.one_hot(torch.tensor([si_b]))).squeeze(0).numpy()
    a_idx_b = m.masked_greedy_action(q, si_b, rng2)
    a_b = m.ACTION_LIST[a_idx_b]
    s_next_b, cost_b = m.step(rng2, s_cur, a_b)
    done_b = 1 if s_next_b == m.TERMINAL else 0
    batch_transitions.append((si_b, a_idx_b, cost_b, m.STATE_IDX[s_next_b] if not done_b else m.STATE_IDX["a"], done_b))
    s_cur = s_next_b

s_b = torch.tensor([t[0] for t in batch_transitions], dtype=torch.long)
a_b = torch.tensor([t[1] for t in batch_transitions], dtype=torch.long)
c_b = torch.tensor([t[2] for t in batch_transitions], dtype=torch.float32)
sn_b = torch.tensor([t[3] for t in batch_transitions], dtype=torch.long)
d_b = torch.tensor([t[4] for t in batch_transitions], dtype=torch.float32)

with torch.no_grad():
    next_q = online(m.one_hot(sn_b))
    next_q_masked = next_q + m.MASK_ADD_T[sn_b]
    min_next_q, _ = next_q_masked.min(dim=1)
    y_b = c_b + gamma * (1.0 - d_b) * min_next_q
    pred_q = online(m.one_hot(s_b)).gather(1, a_b.unsqueeze(1)).squeeze(1)
    batch_loss = torch.nn.functional.mse_loss(pred_q, y_b).item()

print("y_b   =", y_b.numpy())
print("pred_q=", pred_q.numpy())
print("batch MSE loss =", batch_loss)

with open(os.path.join(RESULTS_DIR, "hand_verification_numbers.json"), "w") as f:
    json.dump({
        "illustrative_small_network": illustrative,
        "real_single_transition": real_example,
        "real_minibatch_check": {
            "y_b": y_b.numpy().tolist(),
            "pred_q": pred_q.numpy().tolist(),
            "batch_loss": batch_loss,
        },
    }, f, indent=2)

print("\nSaved to results/hand_verification_numbers.json")
