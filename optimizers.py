"""Conventional real-valued algorithms; N initialization + N*iterations calls.

Every generation evaluates N candidates. GA evaluates N offspring then retains
one elite by replacement. No objective caching or hidden local search is used.
"""
import numpy as np
import config as cfg


def _start(obj, lb, ub, pop_size, seed):
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    if pop_size < 4 or np.any(lb >= ub):
        raise ValueError('Use population >=4 and ordered bounds')
    rng = np.random.default_rng(seed)
    x = rng.uniform(lb, ub, (pop_size, len(lb)))
    f = np.array([obj(v) for v in x])
    i = np.argmin(f)
    return rng, x, f, x[i].copy(), float(f[i])


def ga(obj_func, lb, ub, pop_size, max_iter, seed):
    """Tournament GA: child=alpha*p1+(1-alpha)*p2; Gaussian mutation and elitism."""
    r, x, f, best, value = _start(obj_func, lb, ub, pop_size, seed)
    curve = [value]
    for _ in range(max_iter):
        children = []
        def parent():
            ids = r.integers(pop_size, size=cfg.GA_TOURNAMENT)
            return x[ids[np.argmin(f[ids])]]
        while len(children) < pop_size:
            p, q = parent(), parent()
            alpha = r.random()
            if r.random() < cfg.GA_CROSSOVER:
                pair = [alpha*p+(1-alpha)*q, alpha*q+(1-alpha)*p]
            else:
                pair = [p.copy(), q.copy()]
            for child in pair:
                mask = r.random(len(lb)) < cfg.GA_MUTATION
                child = child + mask*r.normal(0, cfg.GA_SIGMA, len(lb))*(ub-lb)
                children.append(np.clip(child, lb, ub))
        x = np.array(children[:pop_size])
        f = np.array([obj_func(v) for v in x])
        worst = np.argmax(f)
        if value < f[worst]:
            x[worst], f[worst] = best.copy(), value
        i = np.argmin(f)
        if f[i] < value:
            best, value = x[i].copy(), float(f[i])
        curve.append(value)
    return best, value, np.array(curve)


def pso(obj_func, lb, ub, pop_size, max_iter, seed):
    """v=w*v+c1*r1*(pbest-x)+c2*r2*(gbest-x); x=clip(x+v)."""
    r, x, f, best, value = _start(obj_func, lb, ub, pop_size, seed)
    v = np.zeros_like(x)
    personal, scores = x.copy(), f.copy()
    curve = [value]
    for k in range(max_iter):
        w = cfg.PSO_W[0] + (cfg.PSO_W[1]-cfg.PSO_W[0])*k/max(1, max_iter-1)
        v = w*v + cfg.PSO_C1*r.random(x.shape)*(personal-x) + cfg.PSO_C2*r.random(x.shape)*(best-x)
        x = np.clip(x+v, lb, ub)
        f = np.array([obj_func(z) for z in x])
        improve = f < scores
        personal[improve], scores[improve] = x[improve], f[improve]
        i = np.argmin(scores)
        if scores[i] < value:
            best, value = personal[i].copy(), float(scores[i])
        curve.append(value)
    return best, value, np.array(curve)


def de(obj_func, lb, ub, pop_size, max_iter, seed):
    """DE/rand/1/bin: mutant=a+F*(b-c); binomial crossover, greedy selection.

    Generations are synchronous: every donor comes from the previous population.
    """
    r, x, f, best, value = _start(obj_func, lb, ub, pop_size, seed)
    curve = [value]
    for _ in range(max_iter):
        new, scores = x.copy(), f.copy()
        for i in range(pop_size):
            a, b, c = r.choice(np.delete(np.arange(pop_size), i), 3, replace=False)
            mutant = np.clip(x[a]+cfg.DE_F*(x[b]-x[c]), lb, ub)
            mask = r.random(len(lb)) < cfg.DE_CR
            mask[r.integers(len(lb))] = True
            trial = np.clip(np.where(mask, mutant, x[i]), lb, ub)
            score = obj_func(trial)
            if score <= f[i]:
                new[i], scores[i] = trial, score
        x, f = new, scores
        i = np.argmin(f)
        if f[i] < value:
            best, value = x[i].copy(), float(f[i])
        curve.append(value)
    return best, value, np.array(curve)


def gwo(obj_func, lb, ub, pop_size, max_iter, seed):
    """A=2*a*r-a, C=2*r; x'=mean(leader-A*abs(C*leader-x)).

    Persistent alpha/beta/delta and strict score comparisons follow author code.
    """
    r, x, f, best, value = _start(obj_func, lb, ub, pop_size, seed)
    leaders = np.zeros((3, len(lb)))
    scores = np.full(3, np.inf)

    def update_leaders():
        # These three comparisons intentionally follow the original reference
        # implementation, including its persistent secondary leader convention.
        for position, fitness in zip(x, f):
            if fitness < scores[0]:
                leaders[0], scores[0] = position.copy(), fitness
            if scores[0] < fitness < scores[1]:
                leaders[1], scores[1] = position.copy(), fitness
            if scores[0] < fitness and scores[1] < fitness < scores[2]:
                leaders[2], scores[2] = position.copy(), fitness

    update_leaders()
    curve = [float(scores[0])]
    for k in range(max_iter):
        a = 2*(1-k/max_iter)
        proposals = []
        for leader in leaders:
            A = 2*a*r.random(x.shape)-a
            C = 2*r.random(x.shape)
            proposals.append(leader-A*np.abs(C*leader-x))
        x = np.clip(np.mean(proposals, axis=0), lb, ub)
        f = np.array([obj_func(z) for z in x])
        update_leaders()
        curve.append(float(scores[0]))
    return leaders[0].copy(), float(scores[0]), np.array(curve)


def woa(obj_func, lb, ub, pop_size, max_iter, seed):
    """Encircle/search: x'=target-A*abs(C*target-x); spiral: D*exp(b*l)*cos(2*pi*l)+best.

    Scalar A,C,p per whale and l uniform on [a2,1], a2=-1-k/T.
    """
    r, x, f, best, value = _start(obj_func, lb, ub, pop_size, seed)
    curve = [value]
    for k in range(max_iter):
        a, a2 = 2*(1-k/max_iter), -1-k/max_iter
        for i in range(pop_size):
            A, C = 2*a*r.random()-a, 2*r.random()
            l = (a2-1)*r.random()+1
            if r.random() < .5:
                target = best if abs(A) < 1 else x[r.integers(pop_size)].copy()
                x[i] = target-A*np.abs(C*target-x[i])
            else:
                x[i] = np.abs(best-x[i])*np.exp(cfg.WOA_B*l)*np.cos(2*np.pi*l)+best
            x[i] = np.clip(x[i], lb, ub)
        f = np.array([obj_func(z) for z in x])
        i = np.argmin(f)
        if f[i] < value:
            best, value = x[i].copy(), float(f[i])
        curve.append(value)
    return best, value, np.array(curve)


def ssa(obj_func, lb, ub, pop_size, max_iter, seed):
    """Author-code SSA: first half leaders, remaining salps sequential followers.

    c1=2*exp(-(4*t/T)^2); leaders=food +/- c1*((ub-lb)*c2+lb);
    followers x_i=(x_i+x_(i-1))/2. This is the published reference-code convention.
    """
    r, x, f, best, value = _start(obj_func, lb, ub, pop_size, seed)
    curve = [value]
    for k in range(max_iter):
        c1 = 2*np.exp(-(4*(k+1)/max_iter)**2)
        for i in range(pop_size):
            if i < pop_size/2:
                step = c1*((ub-lb)*r.random(len(lb))+lb)
                x[i] = best + np.where(r.random(len(lb)) < .5, step, -step)
            else:
                x[i] = .5*(x[i]+x[i-1])
            x[i] = np.clip(x[i], lb, ub)
        f = np.array([obj_func(z) for z in x])
        i = np.argmin(f)
        if f[i] < value:
            best, value = x[i].copy(), float(f[i])
        curve.append(value)
    return best, value, np.array(curve)


ALGORITHMS = dict(GA=ga, PSO=pso, DE=de, GWO=gwo, WOA=woa, SSA=ssa)
