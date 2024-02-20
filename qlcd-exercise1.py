# Solution to the exercise 1 in "Lattice QCD for Novices" by G. Peter Lapage

import vegas
import numpy as np
N = 8
a = 1/2
m = 1

T = a * N

def f(x, start, end):
    A = np.power(m / (2 * np.pi * a), N / 2) # normalization factor
    cur = np.array([*x, end])
    prev = np.array([start, *x])
    v = (cur - prev) / a
    L = 1/2 * (m * np.power(v, 2) + np.power(cur, 2)) # lagrangian evaluated at each of the N points
    S = np.sum(L) * a
    return A * np.exp(-S)


def solution(x):
    # using known results from QM
    # probability that a particle at position x at T=0 will again be measured
    # at postion x at T=8
    return np.power(np.exp(-np.power(x, 2) / 2) / np.power(np.pi, 0.25), 2) * np.exp(-1/2 * T)

for x in np.linspace(-1, 1, 10):
    print(f"classical result for position {x}: ", solution(x))

integ = vegas.Integrator([[-20, 20] for _ in range(N - 1)])

for x in np.linspace(-1, 1, 10):
    result = integ(lambda i: f(i, x, x), nitn=10, neval=10000)
    print(f"path integral result for position {x}: ", result)
