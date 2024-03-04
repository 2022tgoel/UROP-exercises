import halide as hl
import numpy as np

dim = 2

a = np.random.randint(1, 10, (dim, dim)).astype(np.float32)
print(a)
a = hl.Buffer(a)

func = hl.Func("function")
x = hl.Var("x")
y = hl.Var("y")
t = hl.Var("t")
func[t, x, y] = 0.0
func[0, x, y] = a[x, y]

r = hl.RDom([hl.Range(0, dim), hl.Range(0, dim), hl.Range(0, dim)])

r.where(r.z > r.y)

def ind(x, y):
    return (y - x - 1) + dim * x - x * (x + 1) / 2

time = ind(r.y, r.z) # (r.z - r.y - 1) + dim * r.y - r.y * (r.y + 1) / 2

theta = hl.atan2(-func[time, r.y, r.z], func[time, r.y, r.y])
# func[time + 1, r.x, y] = func[time, r.x, y]
func[time + 1, r.x, r.y] = hl.cos(theta) * func[time, r.x, r.y] - hl.sin(theta) * func[time, r.x, r.z]
func[time + 1, r.x, r.z] = hl.sin(theta) * func[time, r.x, r.y] + hl.cos(theta) * func[time, r.x, r.z]

# func2[1, 0] = 1.0

# func = rotate(func, 1, 0)
print(ind(dim-2, dim-1) + 2)
output = func.realize([int(ind(dim-2, dim-1) + 2), dim, dim])

print(np.array(output)[:, :, -1])
# print(np.array(output)[:, :, 2])
