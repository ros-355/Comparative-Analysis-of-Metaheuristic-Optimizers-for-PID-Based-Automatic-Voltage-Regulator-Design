"""Summary statistics, independent-sample rank-sum tests, robustness and margins."""
import numpy as np
import pandas as pd
from scipy import stats, optimize
import config as cfg
from avr_model import transfer_functions, metrics, is_stable


def summarize(runs):
    table = runs.groupby('algorithm')['fitness'].agg(best='min', mean='mean', worst='max', median='median', std='std')
    table['mean_time_s'] = runs.groupby('algorithm')['time_s'].mean()
    table['mean_evaluations'] = runs.groupby('algorithm')['evaluations'].mean()
    table = table.sort_values('mean')
    table.insert(0, 'rank', np.arange(1, len(table)+1))
    winner = table.index[0]
    tests = []
    for name in table.index[1:]:
        # ranksums is Wilcoxon rank-sum, not paired signed-rank.
        a = runs.loc[runs.algorithm == winner, 'fitness']
        b = runs.loc[runs.algorithm == name, 'fitness']
        result = stats.ranksums(a, b, alternative='two-sided')
        tests.append(dict(winner=winner, other=name, statistic=result.statistic, p_value=result.pvalue))
    tests = pd.DataFrame(tests)
    # Holm adjustment controls the family of winner-vs-other comparisons.
    order = np.argsort(tests.p_value.to_numpy())
    adjusted = np.maximum.accumulate((len(order)-np.arange(len(order)))*tests.p_value.to_numpy()[order])
    tests['p_holm'] = np.nan
    tests.loc[order, 'p_holm'] = np.minimum(1., adjusted)
    return table, tests, winner


def controller_table(controllers):
    rows = []
    for name, gains in controllers.items():
        system, _ = transfer_functions(gains)
        row = dict(method=name, Kp=np.nan, Ki=np.nan, Kd=np.nan)
        if gains is not None:
            row.update(zip(('Kp', 'Ki', 'Kd'), gains))
        row.update(metrics(system))
        rows.append(row)
    return pd.DataFrame(rows)


def robustness_cases():
    """One-at-a-time +/-20% changes, plus simultaneous extremes for the plot."""
    cases = {'nominal': {}}
    for key in ('Ta', 'Te', 'Tg'):
        for factor in cfg.ROBUST_FACTORS:
            cases[f'{key}_{factor:.1f}'] = {key: cfg.PLANT[key]*factor}
    for factor in cfg.ROBUST_FACTORS:
        cases[f'all_{factor:.1f}'] = {key: cfg.PLANT[key]*factor for key in ('Ta', 'Te', 'Tg')}
    return cases


def robustness(controllers):
    rows = []
    for method, gains in controllers.items():
        for case, changes in robustness_cases().items():
            system, _ = transfer_functions(gains, changes)
            rows.append(dict(method=method, case=case, **metrics(system)))
    return pd.DataFrame(rows)


def stability_margins(loop):
    """Refine every sampled crossover; select the smallest gain/phase margin.

    Covers the configured frequency interval. Infinite means no crossing found
    in that interval, rather than a claim about an arbitrary-frequency model.
    """
    w = cfg.FREQUENCIES
    def response(v):
        return np.polyval(loop.num, 1j*v)/np.polyval(loop.den, 1j*v)
    z = response(w)
    mag = np.abs(z)
    phase = np.unwrap(np.angle(z))
    gain_crossings, phase_crossings = [], []
    for i in np.flatnonzero((mag[:-1]-1)*(mag[1:]-1) < 0):
        wc = optimize.brentq(lambda v: abs(response(v))-1, w[i], w[i+1])
        # Use continuous phase branch from nearby unwrapped sampled phase.
        ph = np.angle(response(wc))
        ph += 2*np.pi*round((phase[i]-ph)/(2*np.pi))
        gain_crossings.append((180+np.degrees(ph), wc))
    for target in np.arange(-9, 10, 2)*np.pi:
        for i in np.flatnonzero((phase[:-1]-target)*(phase[1:]-target) < 0):
            def branch(v):
                ph = np.angle(response(v))
                return ph+2*np.pi*round((phase[i]-ph)/(2*np.pi))-target
            wp = optimize.brentq(branch, w[i], w[i+1])
            phase_crossings.append((1/abs(response(wp)), wp))
    gm, wp = min(phase_crossings, default=(np.inf, np.nan))
    pm, wc = min(gain_crossings, default=(np.inf, np.nan))
    closed_den = np.polyadd(loop.den, loop.num)
    return dict(gain_margin=gm, gain_margin_db=20*np.log10(gm), phase_margin_deg=pm,
                gain_crossover_rad_s=wc, phase_crossover_rad_s=wp,
                frequency_min=w[0], frequency_max=w[-1],
                nominal_closed_loop_stable=bool(np.all(np.roots(closed_den).real < 0)))
