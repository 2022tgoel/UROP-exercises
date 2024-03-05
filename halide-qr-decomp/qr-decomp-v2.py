import halide as hl
import numpy as np

dim = 4

a = np.random.randint(1, 10, (dim, dim)).astype(np.float32)
print(a)
a = hl.Buffer(a)

func = hl.Func("function")
x = hl.Var("x")
y = hl.Var("y")
t = hl.Var("t")
func[t, x, y] = a[x, y]

def ind(x, y):
    return (y - x - 1) + dim * x - x * (x + 1) / 2

num_iters = int(ind(dim-2, dim-1) + 2)

# row iter, y, x
r = hl.RDom([hl.Range(0, dim), hl.Range(0, num_iters), hl.Range(0, dim), hl.Range(0, dim)])

r.where(r.z > r.w)

time = ind(r.w, r.z) 

r.where(r.y > time)

theta = hl.atan2(-func[time, r.w, r.z], func[time, r.w, r.w])
# func[time + 1, r.x, y] = func[time, r.x, y]
func[r.y, r.x, r.w] = hl.cos(theta) * func[time, r.x, r.w] - hl.sin(theta) * func[time, r.x, r.z]
func[r.y, r.x, r.z] = hl.sin(theta) * func[time, r.x, r.w] + hl.cos(theta) * func[time, r.x, r.z]

output = func.realize([int(num_iters), dim, dim])

print(np.array(output)[:, :, -1])
