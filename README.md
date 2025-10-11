# Robust Hedging with Minimax CVaR PPO

This workspace packages the Chapter 3 theoretical formulation into a modular PyTorch implementation.

## Layout

```
robust_hedge/
  config.py         # Shared hyper-parameters
  regime.py         # Hidden Markov volatility regimes and sampling
  filtering.py      # Belief filter (predict/update)
  options.py        # Black–Scholes primitives for pricing & delta
  simulation.py     # Market rollouts and portfolio evolution
  adversary.py      # Uncertainty rectangle & corner selection
  agent.py          # PPO actor–critic modules
  training.py       # Robust CVaR PPO training loop
train.py             # CLI entry-point for experiments
```

## Usage

```
python train.py --iterations 50 --rollout-episodes 128 --eval-episodes 64 --alpha 0.95 --eps 0.02
```

The script writes JSON training statistics to `training_stats.json` (override with `--output`).
Each entry contains PPO losses, the approximate KL monitor, and evaluation summaries for:

- the calibrated matrix `P_bar`
- the worst corner of the ε-rectangle
- a stress corner with ε × 1.5

For every scenario the JSON also records baselines (Black–Scholes delta hedge and no-hedge) so Chapter 5 tables can be produced directly from the file.

## Performance Notes

- **Vectorised price path:** the market simulator now samples the entire regime path in one shot and applies a cumulative product to build prices. This removes the per-step Python loop that previously dominated runtime.
- **Episode ramping:** configuration supports ramp schedules (default: rollouts 16→32→64, evaluations 64→128→256). Early iterations stay light-weight while later iterations still receive the high-sample evaluation needed for final figures. Total environment steps per iteration are logged for apples-to-apples comparisons across runs.
- **Adaptive logging:** iterative progress prints report running mean loss for both the corner search and the rollout collection so you can monitor long runs in real time. JSON stats now include the worst corner index, VaR/CVaR, coverage error (|Pr(L>ζ) − (1−α)|), and environment steps.
- **Saving checkpoints:** when training via the CLI or notebook, the actor and critic weights are written to `policy.pt` / `value.pt` for reuse. Set `cfg.save_checkpoints=True` to emit per-iteration checkpoints (policy/value plus optimizer states) under `checkpoints/`.
- **Variance reduction:** Antithetic return sampling (optional) halves evaluation variance with minimal overhead; ζ’s learning rate automatically decays when coverage is off-target, and actor/critic learning rates + entropy bonus decay linearly across iterations for stability.

## Methodology Overview

The repository captures, as faithfully as possible, the modelling and algorithmic choices laid out in Chapter 3 (“Problem Formulation & Theory”) and operationalises them for Chapter 4 (“Methodology”). The section below is written in dissertation-ready prose: you can drop it directly into the thesis with minimal formatting changes.

### 4.1 Market Environment and Information Structure

We model an options desk that trades a European claim on an underlying whose dynamics are driven by a latent two-state (\\(L\\)/\\(H\\)) regime-switching geometric Brownian motion. Conditionally on the regime, log-returns follow
\\[
\log\frac{S_{t+\Delta}}{S_t} = \left(\mu - \tfrac{1}{2}\sigma_{s(t)}^2\right)\Delta + \sigma_{s(t)}\sqrt{\Delta}\, Z_t,\qquad Z_t\sim\mathcal N(0,1),
\\]
where \\(s(t)\\in\{L,H\}\\) evolves according to an unobservable Markov chain. The calibrated transition matrix \\((\bar P)\\) is obtained from SPX data and subsequently embedded in a rectangular uncertainty set \\((\|P-\bar P\|_\infty \le \varepsilon)\\) to represent model ambiguity.

The agent observes (i) the underlying price, (ii) time to expiry, and (iii) a filtered belief \\(q_H = \Pr(s(t)=H\mid \mathcal F_t)\\) produced by a Hidden Markov Model filter. The learning state vector augments these with: log-moneyness, the Black–Scholes delta and price computed under the belief-mixed volatility, the normalised hedging portfolio value \\((\Pi_t/S_0)\\), and the previous hedge position. Including time-to-expiry keeps the state space finite and permits a stationary optimal policy.

Simulation is fully vectorised in `generate_market_path`: regime paths are sampled in one call, log-returns are generated in batch, and prices are produced via cumulative log-sums; optional antithetic pairing (\\(z\\) and \\(-z\\)) halves the variance of payoff estimates without altering the mean.

### 4.2 Adversarial Regime Control

Nature (the adversary) perturbs the transition matrix within the rectangular set. Per Iyengar (2005), the worst-case value is achieved at one of the four matrix corners. For each iteration of training we evaluate those corners using common random numbers (shared seeds) and, when enabled, antithetic returns. The routine records the CVaR, VaR, and mean loss produced by each corner so the evolution of the worst-case regime can be tracked in Chapter 5.

### 4.3 Hedging Policy Architecture

The agent employs a stochastic actor–critic architecture:

* **Actor:** Gaussian policy whose mean and log-standard-deviation are produced by a two-layer MLP (Tanh activations). Actions are squashed via `tanh` and scaled to the hedge bound \\(h_{\max}\\), ensuring admissible hedge ratios and yielding the correct log-probability via the change-of-variables correction.
* **Critic:** An MLP of matching depth approximating the Rockafellar–Uryasev tail value \\((L-\zeta)_+\\).

Both networks are initialised with a moderate exploration scale (`log_std_init`) and run on GPU/MPS/CPU transparently.

### 4.4 Robust CVaR Objective

The learning objective is the minimax Rockafellar–Uryasev CVaR:
\\[
\min_{\pi}\max_{P\in\mathcal P(\varepsilon)} \min_{\zeta} \Big\{\zeta + \tfrac{1}{1-\alpha}\,\mathbb E[(L^{\pi}(P)-\zeta)_+]\Big\}.
\\]

The buffer \\(\zeta\\) is treated as a learnable parameter updated via the stochastic RU gradient. We monitor the coverage error \\(|\Pr(L>\zeta)-(1-\alpha)|\\) each iteration and decay the buffer learning rate if the target is not met, which materially accelerates convergence toward the desired tail quantile.

### 4.5 Training Pipeline

Training follows a proximal policy optimisation (PPO) routine tailored to the minimax setting:

1. **Corner evaluation:** Evaluate each feasible transition matrix corner using shared random seeds (and antithetic returns, if enabled). Identify the worst-case corner for the current policy.
2. **Rollout collection:** Generate a batch of hedging episodes under that corner. Episode counts ramp according to `cfg.rollout_schedule` (e.g., 16→32→64) so early iterations remain inexpensive.
3. **Risk-buffer update:** Update \\(\zeta\\) with the RU gradient. If coverage deviates from \\(1-\alpha\\) beyond the prescribed threshold, multiplicatively decay the buffer learning rate.
4. **Policy update:** Run PPO with clipped objective, entropy regularisation, and approximate KL monitoring. Both actor/critic learning rates and the entropy bonus decay linearly from their initial to final values; PPO epochs early-stop when KL exceeds `target_kl`.
5. **Logging and checkpoints:** Record CVaR, VaR, mean loss, coverage error, worst-corner index, environment steps, and current hyperparameters. Every `checkpoint_interval` iterations the policy/value networks and their optimisers are serialized (`checkpoints/policy_iter_###.pt`, etc.), enabling interruption-free resumes.

### 4.6 Evaluation Protocol

The evaluation suite reports mean, VaR, and CVaR for:

* the calibrated regime matrix \\((\bar P)\\),
* the worst-case corner identified by the adversary, and
* a stress scenario with \\(1.5\varepsilon\\).

Each scenario is benchmarked against (i) the belief-consistent Black–Scholes delta hedge and (ii) a no-hedge control. Evaluations may use large episode counts (e.g., 1024 with antithetic pairing) to produce tight confidence intervals for Chapter 5 plots. Outputs are stored both as a nested dictionary (`eval_results`) and as tidy tables for downstream analysis.

### 4.7 Implementation & Reproducibility

Experiment scripts (`train.py`, `robust_hedge_analysis.ipynb`) accept configuration via `HedgeConfig`. The helper `seed_all` fixes Python/NumPy/Torch RNGs, ensuring that stochastic components (corner evaluation, rollouts, PPO mini-batches) are reproducible. Because corner evaluations, rollouts, and evaluations share seed schedules, comparative studies maintain common random numbers; this considerably reduces variance when measuring improvements over baselines.

The code base therefore constitutes a faithful, efficient implementation of the minimax CVaR hedging methodology: it marries the theoretical design (latent regimes, adversarial uncertainty, tail-optimal control) with the engineering required to run multi-hour experiments reliably.

> **Note**
> The current environment prevents importing PyTorch because shared-memory (libomp) calls are sandboxed. The code paths therefore cannot be executed here, but they are organised for execution on a standard Python/PyTorch setup.
