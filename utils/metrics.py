import math
import numpy as np      
from scipy.stats import entropy
from scipy.special import rel_entr

def relative_entropy(p, q):
    return p * np.log(p/q)

def kl_divergence(p, q, is_zero=True):
    if is_zero:
        kl = np.where(q != 0, rel_entr(p, q), 0.0)
        kl = kl[kl != np.inf]
    else:
        q = np.where(q == 0., 1e-10, q)
        kl = rel_entr(p, q)
    return np.sum(kl)

def jensen_shannon(p, q):
    m = (p + q) / 2
    return math.sqrt((kl_divergence(p, m) + kl_divergence(q, m)) / 2)

def ent_ce(p, q):
    return entropy(q) - entropy(p)

def rank_cs(p, q):
    return np.mean(q.argsort() == p.argsort())

def tvd(p, q):
    return np.sum(np.abs(q - p)) / 2