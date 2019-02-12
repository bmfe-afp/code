import numpy as np


def ttest(a, b):
    s1 = a.var()
    s2 = b.var()
    n1 = a.index.shape[0]
    n2 = b.index.shape[0]
    x1 = a.mean()
    x2 = b.mean()
    sp = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    t = (x1 - x2) / (sp * np.sqrt(1 / n1 + 1 / n2))

    return t