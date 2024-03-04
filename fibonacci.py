import halide as hl
import numpy as np

dim = 40

r = hl.RDom([hl.Range(0, dim)])
func = hl.Func("function")
x = hl.Var("x")
func[x] = 0
func[0] = 1
func[1] = 1

r = hl.RDom([hl.Range(2, dim)])
func[r.x] = func[r.x - 1] + func[r.x - 2]

print(np.array(func.realize([dim])))

