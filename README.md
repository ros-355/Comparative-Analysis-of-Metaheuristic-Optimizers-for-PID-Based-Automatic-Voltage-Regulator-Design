# Metaheuristic Tuning of a PID Controller for an AVR

Reproduction package for the paper **"Comparative Analysis of Metaheuristic Optimizers for PID-Based Automatic Voltage Regulator Design"** by Roshan Kumar Sharma and Ujjwal Prasad Dahal (Department of Electrical Engineering, IOE Purwanchal Campus, Dharan, Nepal).

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Runs](https://img.shields.io/badge/independent%20runs-30-informational)
![Status](https://img.shields.io/badge/status-manuscript%20in%20preparation-yellow)

## Overview

Six established metaheuristics tune a PID controller (with a filtered derivative term) for a linear Automatic Voltage Regulator (AVR) under identical conditions:

| Abbrev. | Algorithm |
|---|---|
| GA | Genetic Algorithm |
| PSO | Particle Swarm Optimization |
| DE | Differential Evolution |
| GWO | Grey Wolf Optimizer |
| WOA | Whale Optimization Algorithm |
| SSA | Salp Swarm Algorithm |

The paper does **not** propose a new optimizer and does **not** claim general superiority. It is a controlled comparison on one specified AVR model.

## Experimental setup

| Item | Value |
|---|---|
| Plant (amplifier, exciter, generator, sensor) | Ka=10, Ta=0.1 s, Ke=1, Te=0.4 s, Kg=1, Tg=1 s, Kr=1, Tr=0.01 s |
| Controller | PID with derivative filter, N = 100 rad/s |
| Decision variables | Kp, Ki, Kd, each bounded in [0.2, 2.0] |
| Main objective | ITAE (IAE, ISE, ITSE also compared) |
| Independent runs | 30 (seeds 0–29) |
| Population / updates | 30 agents, 50 updates |
| Evaluations per run | 1,530 (identical for all algorithms) |
| Simulation horizon | 0–10 s on a 2,000-point grid |
| Unstable-candidate penalty | 1e6 |

All settings live in [`config.py`](config.py).

## Key results (full run, ITAE objective)

| Rank | Algorithm | Mean ITAE | Std. dev. |
|---|---|---|---|
| 1 | **DE** | 0.0400877 | 3.8e-06 |
| 2 | WOA | 0.0401246 | 8.8e-05 |
| 3 | GWO | 0.0404722 | 3.7e-04 |
| 4 | PSO | 0.0417043 | 2.6e-03 |
| 5 | SSA | 0.0423335 | 2.5e-03 |
| 6 | GA | 0.0426920 | 1.1e-03 |

Best DE controller: **Kp = 0.9398, Ki = 0.6540, Kd = 0.2910**

| Metric | Uncontrolled | Ziegler–Nichols | DE (ITAE) |
|---|---|---|---|
| Overshoot | 50.65 % | 53.05 % | **13.24 %** |
| Settling time | not settled | 2.991 s | **0.980 s** |
| Rise time | 0.279 s | 0.227 s | 0.197 s |

Other findings:

- DE beats every other algorithm in Holm-adjusted rank-sum tests (largest p_Holm = 0.0093, against WOA).
- Paired signed-rank checks on matched seeds agree; the DE–WOA gap is small (about 0.09 % in the mean).
- The ranking is unchanged when the saved gains are re-evaluated on 4,000- and 8,000-point integration grids.
- Choice of objective (IAE / ISE / ITAE / ITSE) changes the transient trade-off (overshoot vs. settling time).
- All tested ±20 % perturbations of the time constants (one-at-a-time and simultaneous extremes) stay stable.
- The DE controller has nominal gain margin 7.60 (17.6 dB) and phase margin 53.8° with sensor dynamics retained.

These conclusions apply only to the specified model, bounds, seeds, filtered controller and finite 10 s horizon.

<p align="center">
  <img src="results/full/figures/step_responses.png" width="48%" alt="Step responses of tuned controllers">
  <img src="results/full/figures/convergence.png" width="48%" alt="Convergence curves">
</p>

## Repository layout

```
.
├── config.py               # all plant, bounds, seeds, algorithm parameters
├── avr_model.py            # transfer functions, step response, metrics, Ziegler–Nichols
├── objectives.py           # IAE/ISE/ITAE/ITSE, penalty, evaluation counter
├── optimizers.py           # GA, PSO, DE, GWO, WOA, SSA
├── analysis.py             # summaries, statistical tests, robustness, margins
├── plots.py                # figures (300 dpi)
├── run_experiments.py      # main entry point
├── results/                # saved outputs (quick/ and full/)
├── paper/                  # manuscript (Word + LaTeX), figures, supplementary data
├── validation_suite/       # separate extended validation (partially run)
├── reference_data/         # frozen logs and partial validation snapshot
├── docs/                   # protocol notes, validation, checksums
```

## Quick start

```sh
python -m venv .venv
# Windows PowerShell: .venv/Scripts/Activate.ps1
# macOS/Linux:        source .venv/bin/activate
python -m pip install -r requirements.txt

python tools/verify_saved_results.py   # read-only check of saved results
python run_experiments.py --quick      # 5 runs x 10 updates (~20 s)
python run_experiments.py              # full: 30 runs x 50 updates (~7 min)
```

> **Warning:** `run_experiments.py` overwrites `results/quick` or `results/full`. Keep a copy of the published data first.

Pinned versions of the recorded run: Python 3.14.7, NumPy 2.3.4, SciPy 1.16.3, pandas 2.3.3, Matplotlib 3.10.7. Other versions may give tiny numerical differences, and timing columns depend on your machine.

## Integrity checks

- `tools/verify_saved_results.py` confirms 180 main runs (6 algorithms x 30 seeds), 1,530 evaluations per run, and that saved best/mean/worst/SD match the raw runs.
- It also checks SHA-256 hashes of the original experiment modules ([`docs/ORIGINAL_CORE_SHA256.json`](docs/ORIGINAL_CORE_SHA256.json)).
- Sphere-function sanity checks and a comparison of the fast step-response routine against `scipy.signal.step` are described in [`docs/ORIGINAL_VALIDATION.md`](docs/ORIGINAL_VALIDATION.md).

## Extended validation (incomplete)

`validation_suite/` contains additional checks (reference search, control effort, filter retuning, robustness, effect sizes, and others). The saved snapshot is **partial** and is **not** part of the paper's results. See [`docs/EXTENDED_VALIDATION_STATUS.md`](docs/EXTENDED_VALIDATION_STATUS.md) and [`validation_suite/README.md`](validation_suite/README.md).

## Rebuilding the paper

```sh
python -m pip install -r requirements-paper.txt
python paper/build_latex.py
python paper/build_docx.py
cd paper/latex && pdflatex main.tex && pdflatex main.tex
```

Details in [`paper/README.md`](paper/README.md).

## Citation

```bibtex
@unpublished{sharma_dahal_avr_pid,
  author = {Sharma, Roshan Kumar and Dahal, Ujjwal Prasad},
  title  = {Comparative Analysis of Metaheuristic Optimizers for PID-Based Automatic Voltage Regulator Design},
  note   = {Manuscript in preparation}
}
```



## Contact

Roshan Kumar Sharma, pur078bel042@ioepc.edu.np, ORCID [0009-0005-5055-4735](https://orcid.org/0009-0005-5055-4735)
