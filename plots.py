"""Publication-readable monochrome figures; every output is a 300 dpi PNG."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config as cfg
from avr_model import transfer_functions, step_response, is_stable
from analysis import robustness_cases

STYLES = ['-', '--', '-.', ':']
MARKERS = ['o', 's', '^', 'v', 'D', 'x', '+', '*']


def save(fig, path):
    for ax in fig.axes:
        ax.grid(True, alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=cfg.DPI, bbox_inches='tight')
    plt.close(fig)


def responses(controllers, path, limit=None, title='Unit step responses'):
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (name, gains) in enumerate(controllers.items()):
        system, _ = transfer_functions(gains)
        if is_stable(system):
            ax.plot(cfg.TIME, step_response(system), color=str(.08+.065*i),
                    linestyle=STYLES[i % 4], marker=MARKERS[i % 8],
                    markevery=150+i*7, markersize=3, label=name)
    ax.axhline(1., color='.5', lw=.7)
    ax.set(xlabel='Time (s)', ylabel='Terminal voltage (p.u.)', title=title)
    if limit:
        ax.set_xlim(0, limit)
    ax.legend(fontsize=8, ncol=2)
    save(fig, path)


def convergence(curves, path, pop_size):
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (name, values) in enumerate(curves.items()):
        mean = np.mean(values, axis=0)
        ax.semilogy(pop_size*(np.arange(len(mean))+1), mean, color='black',
                    linestyle=STYLES[i % 4], marker=MARKERS[i], markevery=max(1, len(mean)//8),
                    markersize=4, label=name)
    ax.set(xlabel='Function evaluations (including initialization)', ylabel='Mean best-so-far objective')
    ax.legend()
    save(fig, path)


def boxplot(runs, path):
    names = list(runs.algorithm.unique())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([runs.loc[runs.algorithm == n, 'fitness'] for n in names], tick_labels=names)
    ax.set(ylabel='Final objective value', title='Independent optimization runs')
    save(fig, path)


def metric_bars(table, path):
    fig, axes = plt.subplots(2, 1, figsize=(9, 7))
    for ax, col, label in zip(axes, ['overshoot', 'settling_time'], ['Overshoot (%)', 'Settling time (s)']):
        ax.bar(table.method, table[col], color='white', edgecolor='black', hatch='//')
        ax.set_ylabel(label)
        for i, value in enumerate(table[col]):
            if not np.isfinite(value):
                ax.text(i, 0, 'N/A', rotation=90, ha='center', va='bottom')
        ax.tick_params(axis='x', labelrotation=30)
    save(fig, path)


def robustness_plot(gains, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    cases = robustness_cases()
    for i, name in enumerate(['nominal', 'all_0.8', 'all_1.2']):
        system, _ = transfer_functions(gains, cases[name])
        label = name if is_stable(system) else name+' (unstable)'
        ax.plot(cfg.TIME, step_response(system), color='black', linestyle=STYLES[i], label=label)
    ax.set(xlabel='Time (s)', ylabel='Terminal voltage (p.u.)', title='All three time constants varied together')
    ax.legend()
    save(fig, path)


def frequency_plots(gains, folder):
    closed, loop = transfer_functions(gains)
    w = cfg.FREQUENCIES
    z = np.polyval(loop.num, 1j*w)/np.polyval(loop.den, 1j*w)
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].semilogx(w, 20*np.log10(np.abs(z)), 'k')
    axes[0].axhline(0, color='.5', linestyle='--')
    axes[0].set_ylabel('Loop magnitude (dB)')
    axes[1].semilogx(w, np.degrees(np.unwrap(np.angle(z))), 'k')
    axes[1].axhline(-180, color='.5', linestyle='--')
    axes[1].set(xlabel='Angular frequency (rad/s)', ylabel='Loop phase (degrees)')
    save(fig, folder/'bode.png')
    fig, ax = plt.subplots(figsize=(7, 5))
    poles, zeros = np.roots(closed.den), np.roots(closed.num)
    ax.scatter(poles.real, poles.imag, marker='x', color='black', label='Poles')
    ax.scatter(zeros.real, zeros.imag, facecolors='none', edgecolors='black', label='Zeros')
    ax.axvline(0, color='.5', lw=.7)
    ax.axhline(0, color='.5', lw=.7)
    ax.set(xlabel='Real part (1/s)', ylabel='Imaginary part (rad/s)', title='Closed-loop poles and zeros')
    ax.legend()
    save(fig, folder/'pole_zero.png')
