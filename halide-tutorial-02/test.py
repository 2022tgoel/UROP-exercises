# By convention, we import halide as 'hl' for terseness
import halide as hl
import numpy as np
# Some constants
edge = 512
k = 20.0 / float(edge)

# Simple formula
x, y, c = hl.Var('x'), hl.Var('y'), hl.Var('c')
f = hl.Func('f')
e = hl.sin(x * ((c + 1) / 3.0) * k) * hl.cos(y * ((c + 1) / 3.0) * k)
f[x, y, c] = hl.cast(hl.UInt(8), e * 255.0)
f.vectorize(x, 8).parallel(y)

# Realize into a Buffer.
buf = f.realize([edge, edge, 3])

npbuf = np.array(buf)

# Do something with the image. We'll just save it to a PNG.
# from halide import imageio
# imageio.imwrite("/tmp/example.png", buf)