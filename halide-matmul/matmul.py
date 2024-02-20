import halide as hl
import numpy as np
from timeit import default_timer as timer


def time(func):
    def wrapper():
        s = timer()
        output = func()
        e = timer()
        print("Execution time: ", e - s)
        return output
    return wrapper

dim = 100

a = np.random.randint(0, 10, (dim, dim)).astype(np.float32)
b = np.random.randint(0, 10, (dim, dim)).astype(np.float32)

print("Input A:")
print(a)
print("Input B:")
print(b)

a = hl.Buffer(a)
b = hl.Buffer(b)

c = hl.Func("c")

r = hl.RDom([hl.Range(0, dim)])

i = hl.Var("i")
j = hl.Var("j")

c[i, j] = 0.0

c[i, j] += a[i, r.x] * b[r.x, j]

@time
def matmul_halide():
    return c.realize([dim, dim])

output = matmul_halide()

output = np.array(output)

@time
def matmul_numpy():
    return np.matmul(b, a)

expected = matmul_numpy()

print("Output:")
print(output)
print("Expected:")
print(expected)



