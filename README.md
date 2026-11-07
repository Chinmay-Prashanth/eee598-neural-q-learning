# EEE598 Assignment 3 -- Neural Q-Function Approximation

Replaces the Assignment 2 Q-table with a feedforward neural Q-function
(DQN-style: experience replay, a target network, detached Bellman targets)
on the same 8-state stochastic shortest-path problem. Full writeup in
[`report/Assignment3_Report.pdf`](report/Assignment3_Report.pdf).

## Results at a glance

All 5 independent seeds (0-4, 5000 episodes each) recover the exact
Assignment-2 optimal policy -- $a\to A_b$, $b\to A_e$, $c\to A_f$ -- with a
1000-episode stochastic evaluation mean cost within 0.4% of the known
optimum (8.44). See `results/all_seeds_evaluation.csv` and
`results/all_seeds_policy_summary.csv`.

## Repo layout

```
code/
  neural_q_learning.py   main implementation: environment, QNetwork,
                          replay buffer, training loop, evaluation, figures
  hand_verification.py   regenerates the numbers in verification.md
results/                 per-seed CSV/JSON outputs, architecture.json,
                          5-seed summary tables, trained seed-4 weights
figures/                 fig1 (episode cost) and fig2 (training loss),
                          seed 4 only, as required by the assignment
report/
  Assignment3_Report.tex IEEE conference format, <=3 pages excl. code
  Assignment3_Report.pdf compiled report
verification.md          hand-worked forward pass / Bellman target / TD
                          error / loss, illustrative + real numbers
README.md                this file
```

## Environment

- Python 3.13.9
- PyTorch 2.15.0 (CPU build)
- NumPy 2.4.2
- Matplotlib 3.10.8
- Hardware: Windows 11, CPU only, no GPU used

Install:
```
pip install torch numpy matplotlib
```

## Reproducing everything

```
cd code
python neural_q_learning.py     # trains all 5 seeds (~30 s/seed on CPU),
                                 # writes results/ and figures/
python hand_verification.py     # regenerates verification.md's numbers
```

`neural_q_learning.py` is deterministic given the seed list `[0,1,2,3,4]`
in `HYPERPARAMS` -- each seed resets Python's `random`, NumPy's global
RNG, PyTorch's RNG, and a dedicated `numpy.random.Generator` used for
environment sampling, exploration, and tie-breaking, so re-running
reproduces the same numbers reported here.

## Compiling the report

```
cd report
pdflatex Assignment3_Report.tex
pdflatex Assignment3_Report.tex   # second pass for references
```

## What the learner does and does not see

The network trains only on sampled transitions `(s, a, c, s', d)` returned
by `step()`. It never reads the transition-probability table or any
Assignment-2 Q-values; those are used only afterward, to report the known
optimal cost (8.44) as a comparison point, exactly as the assignment
requires.

## AI Assistance Disclosure

The team developed the solution, performed the experiments, and analyzed
the results independently. Generative AI tools were used to assist with
code optimization and to improve the vocabulary, grammar, and readability
of the report. The final implementation and results were reviewed and
verified by the team.
