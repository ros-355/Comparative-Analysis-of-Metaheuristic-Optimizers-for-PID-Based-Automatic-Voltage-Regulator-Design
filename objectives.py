"""Error integral objectives with an explicit evaluation counter."""
import numpy as np
from scipy.integrate import trapezoid
import config as cfg
from avr_model import transfer_functions, is_stable, step_response


class Objective:
    def __init__(self, kind='ITAE'):
        if kind not in cfg.OBJECTIVES:
            raise ValueError(f'Unknown objective {kind}')
        self.kind = kind
        self.evaluations = 0
        self.failures = 0

    def __call__(self, gains):
        self.evaluations += 1
        try:
            system, _ = transfer_functions(gains)
            if not is_stable(system):
                self.failures += 1
                return cfg.PENALTY
            y = step_response(system, fast=True)
            if not np.all(np.isfinite(y)):
                raise ValueError('Nonfinite response')
            e = 1. - y
            integrand = np.abs(e) if self.kind in ('IAE', 'ITAE') else e**2
            if self.kind in ('ITAE', 'ITSE'):
                integrand = cfg.TIME * integrand
            value = float(trapezoid(integrand, cfg.TIME))
            if not np.isfinite(value) or value >= cfg.PENALTY:
                raise ValueError('Invalid or excessive integral')
            return value
        except (ValueError, FloatingPointError, np.linalg.LinAlgError, OverflowError):
            self.failures += 1
            return cfg.PENALTY
