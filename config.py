"""All experiment and algorithm settings. Units are seconds and rad/s."""
from pathlib import Path
import numpy as np

PLANT = dict(Ka=10., Ta=0.1, Ke=1., Te=0.4, Kg=1., Tg=1., Kr=1., Tr=0.01)
DERIVATIVE_N = 100.
TIME = np.linspace(0., 10., 2000)
LOWER = np.full(3, 0.2)
UPPER = np.full(3, 2.0)
POP_SIZE = 30
MAX_ITER = 50
RUNS = 30
QUICK_RUNS = 5
QUICK_ITER = 10
SEED_START = 0
MAIN_OBJECTIVE = 'ITAE'
OBJECTIVES = ('IAE', 'ISE', 'ITAE', 'ITSE')
PENALTY = 1e6
STABILITY_TOL = 1e-9
GA_TOURNAMENT = 3
GA_CROSSOVER = 0.9
GA_MUTATION = 1 / 3  # Per-coordinate probability.
GA_SIGMA = 0.1  # Fraction of search range.
PSO_W = (0.9, 0.4)
PSO_C1 = PSO_C2 = 2.
DE_F = 0.5
DE_CR = 0.9
WOA_B = 1.
ROBUST_FACTORS = (0.8, 1.2)
FREQUENCIES = np.logspace(-4, 5, 12000)
DPI = 300
RESULTS = Path(__file__).resolve().parent / 'results'
