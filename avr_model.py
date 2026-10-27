"""AVR transfer functions, stable step simulation, and response measurements."""
import numpy as np
from scipy import signal, optimize
import config as cfg


def plant_polynomials(params=None):
    p = cfg.PLANT.copy()
    if params:
        p.update(params)
    den = np.array([1.])
    for key in ('Ta', 'Te', 'Tg'):
        den = np.polymul(den, [p[key], 1.])
    return np.array([p['Ka'] * p['Ke'] * p['Kg']]), den, p


def controller(gains=None):
    if gains is None:  # No PID means forward controller C(s)=1.
        return np.array([1.]), np.array([1.])
    kp, ki, kd = gains
    n = cfg.DERIVATIVE_N
    return np.array([kp + kd*n, kp*n + ki, ki*n]), np.array([1., n, 0.])


def transfer_functions(gains=None, params=None):
    """Return terminal-voltage closed loop T=CG/(1+CGH), and loop L=CGH.

    Sensor H is in the feedback path; its numerator must not enter T's numerator.
    """
    gn, gd, p = plant_polynomials(params)
    cn, cd = controller(gains)
    fn, fd = np.polymul(gn, cn), np.polymul(gd, cd)
    hd = [p['Tr'], 1.]
    denominator = np.polyadd(np.polymul(fd, hd), fn * p['Kr'])
    closed = signal.TransferFunction(np.polymul(fn, hd), denominator)
    loop = signal.TransferFunction(fn * p['Kr'], np.polymul(fd, hd))
    return closed, loop


def is_stable(system):
    return bool(np.all(np.real(np.roots(system.den)) < -cfg.STABILITY_TOL))


def step_response(system, time=None, fast=False):
    """Use SciPy step, or equivalent partial fractions for fast repeated tuning.

    The fast path inverts T(s)/s analytically on the SAME time grid. Near repeated
    poles fall back to scipy.signal.step to avoid ill-conditioned residues.
    """
    t = cfg.TIME if time is None else np.asarray(time)
    if not fast:
        return signal.step(system, T=t)[1]
    step_den = np.polymul(system.den, [1., 0.])
    poles = np.roots(step_den)
    distances = np.abs(poles[:, None] - poles[None, :])
    np.fill_diagonal(distances, np.inf)
    if np.min(distances) < 1e-5:
        return signal.step(system, T=t)[1]
    # For simple poles, residue r_i = numerator(p_i) / denominator_derivative(p_i).
    r = np.polyval(system.num, poles) / np.polyval(np.polyder(step_den), poles)
    return np.real(np.sum(r[:, None] * np.exp(poles[:, None] * t), axis=0))


def metrics(system, y=None):
    """Target-referenced 10-90% rise and 2% settling; NaN if not observed.

    SSE is the analytical DC tracking error. Endpoint error is reported separately.
    Settling means all remaining samples are in band, not proof beyond 10 seconds.
    """
    t = cfg.TIME
    stable = is_stable(system)
    blank = dict(rise_time=np.nan, settling_time=np.nan, overshoot=np.nan,
                 peak_time=np.nan, steady_state_error=np.nan, endpoint_error=np.nan,
                 stable=stable)
    if not stable:
        return blank
    y = step_response(system) if y is None else y
    if not np.all(np.isfinite(y)):
        return blank
    def crossing(level):
        ids = np.flatnonzero(y >= level)
        if not len(ids):
            return np.nan
        i = ids[0]
        return t[i] if i == 0 else np.interp(level, y[i-1:i+1], t[i-1:i+1])
    out = np.flatnonzero(np.abs(y - 1.) > 0.02)
    settling = 0. if not len(out) else (t[out[-1]+1] if out[-1] < len(t)-1 else np.nan)
    peak = int(np.argmax(y))
    dc = system.num[-1] / system.den[-1]
    return dict(rise_time=crossing(.9)-crossing(.1), settling_time=settling,
                overshoot=max(0., 100.*(y[peak]-1.)), peak_time=t[peak],
                steady_state_error=abs(1.-dc), endpoint_error=abs(1.-y[-1]), stable=True)


def ziegler_nichols():
    """Find exact P-only imaginary-axis crossing; verify stable sides numerically."""
    gn, gd, p = plant_polynomials()
    den = np.polymul(gd, [p['Tr'], 1.])
    # For positive lags, phase decreases monotonically from 0 to -2*pi.
    lags = [p[k] for k in ('Ta', 'Te', 'Tg', 'Tr')]
    wu = optimize.brentq(lambda w: sum(np.arctan(w*a) for a in lags)-np.pi, 1e-8, 1e6)
    ku = abs(np.polyval(den, 1j*wu)) / (gn[0]*p['Kr'])
    def poles(k):
        return np.roots(np.polyadd(den, [k*gn[0]*p['Kr']]))
    if not (np.max(poles(ku*.999).real) < 0 < np.max(poles(ku*1.001).real)):
        raise RuntimeError('No verified P-only ultimate crossing; do not invent ZN gains.')
    pu = 2*np.pi/wu
    kp = .6*ku
    return np.array([kp, 2*kp/pu, kp*pu/8]), dict(Ku=ku, Pu=pu, omega_u=wu)
