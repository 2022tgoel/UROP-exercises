import halide as hl
import numpy as np

dim = 10

a = np.random.randint(1, 100, (dim, dim)).astype(np.float32)

a = hl.Buffer(a)

input = hl.Func("input")
i = hl.Var("i")
j = hl.Var("j")
input[i, j] = a[i, j]

def rotate(a: hl.Func, i: int, j: int) -> hl.Func:
    # eliminate the i,j element (i > j), lower traingular matrix
    g = hl.Func("g")
    _i = hl.Var("i")
    _j = hl.Var("j")
    g[_i, _i] = 1
    # vector is [a[j, j], a[i, j]]
    theta = hl.atan2(-a[j, j], a[i, j])
    g[j, j] = hl.cos(theta)
    g[j, i] = hl.sin(theta)
    g[i, j] = -hl.sin(theta)
    g[i, i] = hl.cos(theta)

    r = hl.RDom([hl.Range(0, dim)])

    q = hl.Func("q")
    q[_i, _j] += a[_i, r.x] * g[r.x, _j]
    return q

func = input
# for j in range(0, dim):
#     for i in range(j + 1, dim):
#         func = rotate(func, i, j)

func = rotate(func, 0, 1)

func.realize([dim, dim])