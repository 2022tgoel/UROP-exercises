import halide as hl
import numpy as np

dim = 3

a = np.random.randint(1, 10, (dim, dim)).astype(np.float32)
print(a)
a = hl.Buffer(a)

input = hl.Func("input")
i = hl.Var("i")
j = hl.Var("j")
input[i, j] = a[i, j]

def rotate(a: hl.Func, r: int, c: int) -> hl.Func:
    # eliminate the element in row r and column c
    assert(r > c)
    g = hl.Func("g")
    x = hl.Var("x")
    y = hl.Var("y")
    g[x, y] = 0.0
    g[x, x] = 1.0
    # vector is [a[j, j], a[i, j]]
    theta = hl.atan2(-a[c, r], a[c, c])
    g[c, c] = hl.cos(theta)
    g[r, c] = -hl.sin(theta)
    g[c, r] = hl.sin(theta)
    g[r, r] = hl.cos(theta)
    # g[1, 0] = 1.0

    r = hl.RDom([hl.Range(0, dim)])

    q = hl.Func("q")
    q[x, y] = 0.0
    q[x, y] += g[r.x, y] * a[x, r.x]
    return q

func = input
for j in range(0, dim):
    for i in range(j + 1, dim):
        func = rotate(func, i, j)

# func = rotate(func, 1, 0)

output = func.realize([dim, dim])

print(np.array(output))