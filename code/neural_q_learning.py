"""
Episodic stochastic shortest-path MDP: neural Q-function approximation (DQN-style)
EEE598 Assignment 3

Reuses the exact environment (states, actions, edge costs, stochastic transition
probabilities, start/terminal state, gamma=1) defined for Assignment 2. The
network never sees TRANSITIONS or the Assignment-2 Q-table/policy-iteration
values; it only ever receives sampled transitions (s, a, c, s', d) returned by
step(). TRANSITIONS/ACTIONS are used solely to (a) sample environment dynamics,
exactly as Assignment 2 did, and (b) for the nominal-path report and discussion
in the written report, never as a learning target.

Run:
    python neural_q_learning.py

Produces (written to ../results/ and ../figures/):
    architecture.json              - layer sizes, parameter count, hyperparameters
    seed{0..4}_episode_costs.csv    - per-episode total cost, one row per episode
    seed{0..4}_policy.json          - greedy policy, nominal path, eval stats
    all_seeds_evaluation.csv        - 5-seed evaluation summary table
    all_seeds_policy_summary.csv    - 5-seed greedy policy / nominal path table
    seed4_training_loss.csv         - per-optimization-step loss (seed 4 only)
    figures/fig1_episode_cost_seed4.png
    figures/fig2_training_loss_seed4.png

Software: Python 3.13, PyTorch 2.15 (CPU), numpy 2.4, matplotlib 3.10.
"""

import os
import csv
import json
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CODE_DIR, "..", "results")
FIGURES_DIR = os.path.join(CODE_DIR, "..", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Environment definition -- reused verbatim from Assignment 2
#    (Assign2/code/qlearning_shortest_path.py). Branching states a, b, c each
#    have two actions; the intended transition succeeds w.p. 0.8 and the
#    alternative branch is taken w.p. 0.2. d, e, f, g have a single action
#    (deterministic successor). h is terminal, V(h) = 0. The realized cost is
#    the edge cost of the branch that actually happens, not the intended one.
# ---------------------------------------------------------------------------
ACTIONS = {
    "a": ["Ab", "Ac"],
    "b": ["Ad", "Ae"],
    "c": ["Ae", "Af"],
    "d": ["Ag"],
    "e": ["Ag"],
    "f": ["Ag"],
    "g": ["Ah"],
}

# transition[s][a] = list of (next_state, prob, cost); outcomes[0] is always
# the INTENDED (higher-probability, 0.8) branch, used later to build the
# "nominal path" (Task 3) under intended transitions.
TRANSITIONS = {
    "a": {
        "Ab": [("b", 0.8, 2), ("c", 0.2, 2)],
        "Ac": [("c", 0.8, 2), ("b", 0.2, 2)],
    },
    "b": {
        "Ad": [("d", 0.8, 3), ("e", 0.2, 2)],
        "Ae": [("e", 0.8, 2), ("d", 0.2, 3)],
    },
    "c": {
        "Ae": [("e", 0.8, 5), ("f", 0.2, 4)],
        "Af": [("f", 0.8, 4), ("e", 0.2, 5)],
    },
    "d": {"Ag": [("g", 1.0, 3)]},
    "e": {"Ag": [("g", 1.0, 3)]},
    "f": {"Ag": [("g", 1.0, 2)]},
    "g": {"Ah": [("h", 1.0, 1)]},
}

STATES = ["a", "b", "c", "d", "e", "f", "g", "h"]
STATE_IDX = {s: i for i, s in enumerate(STATES)}
TERMINAL = "h"
N_STATES = len(STATES)  # 8

ACTION_LIST = ["Ab", "Ac", "Ad", "Ae", "Af", "Ag", "Ah"]
ACTION_IDX = {a: i for i, a in enumerate(ACTION_LIST)}
N_ACTIONS = len(ACTION_LIST)  # 7

GAMMA = 1.0  # discount factor, per the assignment (finite-horizon episodic task)

# Additive action mask: MASK_ADD[state_idx, action_idx] = 0 if the action is
# feasible at that state, else a large positive number. Because this is a
# cost-minimization problem, adding a large positive constant makes an
# infeasible action's Q-value lose every min() comparison, which is the
# cost-minimization analogue of the usual "-inf before max" masking trick.
BIG = 1.0e6
MASK_ADD = np.full((N_STATES, N_ACTIONS), BIG, dtype=np.float32)
for s, acts in ACTIONS.items():
    si = STATE_IDX[s]
    for a in acts:
        MASK_ADD[si, ACTION_IDX[a]] = 0.0
MASK_ADD_T = torch.from_numpy(MASK_ADD)


def step(rng, s, a):
    """Sample a transition (s, a) -> (s_next, cost). This is the ONLY way the
    learner is allowed to interact with the environment -- it never reads
    TRANSITIONS[s][a] directly to compute a target."""
    outcomes = TRANSITIONS[s][a]
    probs = [o[1] for o in outcomes]
    idx = rng.choice(len(outcomes), p=probs)
    s_next, _, cost = outcomes[idx]
    return s_next, cost


def one_hot(state_indices, n=N_STATES):
    """state_indices: 1-D long tensor -> float one-hot tensor (B, n)."""
    return F.one_hot(state_indices, num_classes=n).float()


# ---------------------------------------------------------------------------
# 2. Neural Q-network
#    Input: one-hot encoding of the current state (8-dim).
#    Two hidden layers, 32 ReLU units each.
#    Output: one Q-value (estimated cost-to-go) per action (7-dim); invalid
#    actions at a given state are masked out before argmin / min is taken.
# ---------------------------------------------------------------------------
HIDDEN1 = 32
HIDDEN2 = 32


class QNetwork(nn.Module):
    def __init__(self, n_states=N_STATES, n_actions=N_ACTIONS,
                 hidden1=HIDDEN1, hidden2=HIDDEN2):
        super().__init__()
        self.fc1 = nn.Linear(n_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, n_actions)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        q = self.fc3(h2)
        return q


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 3. Replay buffer
#    WHY experience replay: consecutive transitions sampled online are highly
#    correlated (every episode walks the same small DAG a->{b,c}->...->h), so
#    training on them in order would feed the optimizer a biased, narrow
#    stream of gradients. Storing transitions and sampling minibatches at
#    random breaks that temporal correlation and lets each transition be
#    reused multiple times, both of which stabilize the regression in
#    Lecture 9/10's sense (DQN, Mnih et al. 2015).
# ---------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s_idx, a_idx, cost, s_next_idx, done):
        self.buffer.append((s_idx, a_idx, cost, s_next_idx, done))

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size, rng):
        idxs = rng.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in idxs]
        s, a, c, sn, d = zip(*batch)
        return (torch.tensor(s, dtype=torch.long),
                torch.tensor(a, dtype=torch.long),
                torch.tensor(c, dtype=torch.float32),
                torch.tensor(sn, dtype=torch.long),
                torch.tensor(d, dtype=torch.float32))


# ---------------------------------------------------------------------------
# 4. Hyperparameters (starting configuration from the assignment, used as-is)
# ---------------------------------------------------------------------------
HYPERPARAMS = dict(
    hidden1=HIDDEN1,
    hidden2=HIDDEN2,
    replay_capacity=10_000,
    replay_warmup=200,
    batch_size=32,
    optimizer="Adam",
    learning_rate=1e-3,
    target_update_every=100,   # hard copy every M optimization steps
    gamma=GAMMA,
    n_episodes=5000,
    eps_start=1.0,
    eps_min=0.05,
    eps_decay_episodes=4000,
    seeds=[0, 1, 2, 3, 4],
)


def epsilon_for_episode(ep_idx, total_decay_episodes=HYPERPARAMS["eps_decay_episodes"],
                         eps_start=HYPERPARAMS["eps_start"], eps_min=HYPERPARAMS["eps_min"]):
    """ep_idx is 0-indexed. Episodes 1..total_decay_episodes (1-indexed, i.e.
    ep_idx 0..total_decay_episodes-1) decay linearly from eps_start to
    eps_min; held at eps_min afterward."""
    if ep_idx >= total_decay_episodes - 1:
        return eps_min
    frac = ep_idx / (total_decay_episodes - 1)
    return eps_start + frac * (eps_min - eps_start)


def masked_greedy_action(qvals_np, state_idx, rng):
    """qvals_np: (N_ACTIONS,) numpy array of raw Q predictions for this
    state. Adds the mask, takes argmin, breaks exact ties uniformly at
    random among the feasible actions attaining the minimum."""
    masked = qvals_np + MASK_ADD[state_idx]
    min_val = masked.min()
    candidates = np.flatnonzero(np.isclose(masked, min_val, atol=1e-6))
    return int(rng.choice(candidates))


def select_action(online_net, s, eps, rng):
    """epsilon-greedy over the FEASIBLE actions at state s."""
    si = STATE_IDX[s]
    feasible = [ACTION_IDX[a] for a in ACTIONS[s]]
    if rng.random() < eps:
        return int(rng.choice(feasible))
    with torch.no_grad():
        x = one_hot(torch.tensor([si]))
        q = online_net(x).squeeze(0).numpy()
    return masked_greedy_action(q, si, rng)


# ---------------------------------------------------------------------------
# 5. Training loop for one independent seed
# ---------------------------------------------------------------------------
def train_one_seed(seed, track_loss=False, verbose=False):
    # --- full reset of all stochastic state for this seed -----------------
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)  # drives env sampling, exploration, ties

    online_net = QNetwork()
    target_net = QNetwork()
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(online_net.parameters(), lr=HYPERPARAMS["learning_rate"])
    loss_fn = nn.MSELoss()  # L(theta) = (1/B) * sum (y_i - Q_theta(s_i,a_i))^2
    replay = ReplayBuffer(HYPERPARAMS["replay_capacity"])

    opt_steps = 0
    episode_costs = np.zeros(HYPERPARAMS["n_episodes"])
    loss_history = [] if track_loss else None

    for ep in range(HYPERPARAMS["n_episodes"]):
        eps = epsilon_for_episode(ep)
        s = "a"
        total_cost = 0.0
        while s != TERMINAL:
            a_idx = select_action(online_net, s, eps, rng)
            a = ACTION_LIST[a_idx]
            s_next, cost = step(rng, s, a)
            done = 1 if s_next == TERMINAL else 0
            total_cost += cost

            replay.push(STATE_IDX[s], a_idx, cost,
                        STATE_IDX[s_next] if not done else STATE_IDX["a"],  # dummy, masked by (1-done)
                        done)

            if len(replay) >= HYPERPARAMS["replay_warmup"]:
                s_b, a_b, c_b, sn_b, d_b = replay.sample(HYPERPARAMS["batch_size"], rng)

                # --- DQN target: y = c + gamma*(1-d)*min_a' Q_theta-(s',a')
                with torch.no_grad():
                    next_q = target_net(one_hot(sn_b))                 # (B, 7)
                    next_q_masked = next_q + MASK_ADD_T[sn_b]            # mask infeasible next actions
                    min_next_q, _ = next_q_masked.min(dim=1)
                    y = c_b + GAMMA * (1.0 - d_b) * min_next_q           # terminal bootstrap forced to 0 by d_b=1
                    # y is detached (no_grad) -- gradients never flow through the target

                pred_q = online_net(one_hot(s_b)).gather(1, a_b.unsqueeze(1)).squeeze(1)
                loss = loss_fn(pred_q, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                opt_steps += 1

                if track_loss:
                    loss_history.append(loss.item())

                # --- hard target-network update ---
                # WHY a target network: bootstrapping against a Q-network that
                # changes on every gradient step means the regression target
                # y moves every time the prediction moves, which can make the
                # optimization diverge or oscillate (a moving target problem).
                # Freezing theta^- for a fixed number of steps keeps y stable
                # between syncs, consistent with Lecture 9/10's DQN target.
                if opt_steps % HYPERPARAMS["target_update_every"] == 0:
                    target_net.load_state_dict(online_net.state_dict())

            s = s_next

        episode_costs[ep] = total_cost

    if verbose:
        print(f"seed {seed}: training done, {opt_steps} optimization steps, "
              f"final 100-episode mean cost = {episode_costs[-100:].mean():.3f}")

    return online_net, episode_costs, loss_history


# ---------------------------------------------------------------------------
# 6. Policy recovery: greedy action at every nonterminal state, nominal path
# ---------------------------------------------------------------------------
def greedy_policy(net):
    """pi_theta(s) = argmin_{a in A(s)} Q_theta(s,a) at every nonterminal
    state, evaluated with epsilon=0 (pure greedy, network frozen)."""
    net.eval()
    policy = {}
    with torch.no_grad():
        for s in ["a", "b", "c", "d", "e", "f", "g"]:
            si = STATE_IDX[s]
            q = net(one_hot(torch.tensor([si]))).squeeze(0).numpy()
            masked = q + MASK_ADD[si]
            best_idx = int(np.argmin(masked))
            policy[s] = ACTION_LIST[best_idx]
    return policy


def nominal_path(policy):
    """Path from a to h assuming the INTENDED transition always succeeds
    (i.e., follow outcomes[0], the 0.8-probability branch, for every chosen
    action). This is NOT the expected stochastic path -- see report."""
    path = ["a"]
    s = "a"
    while s != TERMINAL:
        a = policy[s]
        s_next = TRANSITIONS[s][a][0][0]  # intended (highest-probability) outcome
        path.append(s_next)
        s = s_next
    return path


# ---------------------------------------------------------------------------
# 7. Stochastic evaluation: 1000 episodes, epsilon = 0, frozen network,
#    evaluation RNG independent of the training RNG.
# ---------------------------------------------------------------------------
def evaluate_policy(net, seed, n_episodes=1000):
    net.eval()
    eval_rng = np.random.default_rng(100_000 + seed)  # independent stream from training
    costs = np.zeros(n_episodes)
    reached_h = np.zeros(n_episodes, dtype=bool)

    with torch.no_grad():
        for ep in range(n_episodes):
            s = "a"
            total_cost = 0.0
            n_steps = 0
            while s != TERMINAL and n_steps < 100:  # safety cap; graph is acyclic, always terminates in 4 steps
                si = STATE_IDX[s]
                q = net(one_hot(torch.tensor([si]))).squeeze(0).numpy()
                a_idx = masked_greedy_action(q, si, eval_rng)
                a = ACTION_LIST[a_idx]
                s_next, cost = step(eval_rng, s, a)
                total_cost += cost
                s = s_next
                n_steps += 1
            costs[ep] = total_cost
            reached_h[ep] = (s == TERMINAL)

    return {
        "mean_cost": float(costs.mean()),
        "std_cost": float(costs.std(ddof=1)),
        "min_cost": float(costs.min()),
        "max_cost": float(costs.max()),
        "pct_reached_h": float(100.0 * reached_h.mean()),
        "n_episodes": n_episodes,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    n_params = count_parameters(QNetwork())
    arch = {
        "input_dim": N_STATES,
        "hidden1": HIDDEN1,
        "hidden2": HIDDEN2,
        "output_dim": N_ACTIONS,
        "activation": "ReLU",
        "trainable_parameters": n_params,
        "parameter_breakdown": {
            "fc1 (8->32) weight": 8 * HIDDEN1, "fc1 bias": HIDDEN1,
            "fc2 (32->32) weight": HIDDEN1 * HIDDEN2, "fc2 bias": HIDDEN2,
            "fc3 (32->7) weight": HIDDEN2 * N_ACTIONS, "fc3 bias": N_ACTIONS,
        },
        "hyperparameters": HYPERPARAMS,
    }
    with open(os.path.join(RESULTS_DIR, "architecture.json"), "w") as f:
        json.dump(arch, f, indent=2)
    print(f"Trainable parameters: {n_params}")

    all_eval_rows = []
    all_policy_rows = []
    seed4_costs = None
    seed4_losses = None

    for seed in HYPERPARAMS["seeds"]:
        track = (seed == 4)
        net, episode_costs, loss_history = train_one_seed(seed, track_loss=track, verbose=True)

        # per-episode cost CSV
        with open(os.path.join(RESULTS_DIR, f"seed{seed}_episode_costs.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "total_cost"])
            for i, c in enumerate(episode_costs, start=1):
                writer.writerow([i, c])

        policy = greedy_policy(net)
        path = nominal_path(policy)
        eval_stats = evaluate_policy(net, seed)

        row_policy = {"seed": seed, **{f"pi({s})": policy[s] for s in ["a", "b", "c", "d", "e", "f", "g"]},
                      "nominal_path": "->".join(path)}
        all_policy_rows.append(row_policy)

        row_eval = {"seed": seed, **eval_stats}
        all_eval_rows.append(row_eval)

        with open(os.path.join(RESULTS_DIR, f"seed{seed}_policy.json"), "w") as f:
            json.dump({"policy": policy, "nominal_path": path, "evaluation": eval_stats}, f, indent=2)

        print(f"seed {seed}: policy={policy} nominal_path={'->'.join(path)}")
        print(f"seed {seed}: eval mean={eval_stats['mean_cost']:.3f} "
              f"std={eval_stats['std_cost']:.3f} min={eval_stats['min_cost']:.1f} "
              f"max={eval_stats['max_cost']:.1f} reach_h={eval_stats['pct_reached_h']:.1f}%")

        if seed == 4:
            seed4_costs = episode_costs
            seed4_losses = loss_history
            # save network weights for the hand-verification section
            torch.save(net.state_dict(), os.path.join(RESULTS_DIR, "seed4_qnetwork_weights.pt"))

    # --- 5-seed summary tables ---
    with open(os.path.join(RESULTS_DIR, "all_seeds_evaluation.csv"), "w", newline="") as f:
        fieldnames = list(all_eval_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_eval_rows)

    with open(os.path.join(RESULTS_DIR, "all_seeds_policy_summary.csv"), "w", newline="") as f:
        fieldnames = list(all_policy_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_policy_rows)

    # --- seed4 loss CSV ---
    with open(os.path.join(RESULTS_DIR, "seed4_training_loss.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["opt_step", "loss"])
        for i, l in enumerate(seed4_losses, start=1):
            writer.writerow([i, l])

    # ----------------------------------------------------------------------
    # Figure 1: seed 4 episode cost + 100-episode rolling mean
    # ----------------------------------------------------------------------
    RAW_BLUE = "#9ec5f4"
    MEAN_BLUE = "#1c5cab"
    REF_GRAY = "#52514e"
    rolling_window = 100

    def rolling(x, window):
        return np.convolve(x, np.ones(window) / window, mode="valid")

    seed4_mean = rolling(seed4_costs, rolling_window)
    episodes = np.arange(1, len(seed4_costs) + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(episodes, seed4_costs, color=RAW_BLUE, linewidth=0.5, label="Episode total cost")
    ax.plot(np.arange(rolling_window, len(seed4_costs) + 1), seed4_mean,
             color=MEAN_BLUE, linewidth=1.8, label=f"{rolling_window}-episode rolling mean")
    ax.axhline(8.44, color=REF_GRAY, linestyle="--", linewidth=1.2,
               label="Assignment-2 optimal cost (8.44)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total episode cost")
    ax.set_title("Seed 4: neural Q-learning episode cost during training")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig1_episode_cost_seed4.png"), dpi=200)
    plt.close(fig)

    # ----------------------------------------------------------------------
    # Figure 2: seed 4 training loss + rolling average
    # ----------------------------------------------------------------------
    loss_arr = np.array(seed4_losses)
    loss_window = 200
    loss_mean = rolling(loss_arr, loss_window)
    steps = np.arange(1, len(loss_arr) + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, loss_arr, color=RAW_BLUE, linewidth=0.3, alpha=0.7, label="Per-step loss")
    ax.plot(np.arange(loss_window, len(loss_arr) + 1), loss_mean,
             color=MEAN_BLUE, linewidth=1.8, label=f"{loss_window}-step rolling mean")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Minibatch MSE loss")
    ax.set_yscale("log")
    ax.set_title("Seed 4: neural Q-learning training loss")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig2_training_loss_seed4.png"), dpi=200)
    plt.close(fig)

    print("Done. Figures written to", FIGURES_DIR)
    print("Results written to", RESULTS_DIR)


if __name__ == "__main__":
    main()
