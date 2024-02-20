# https://halide-lang.org/tutorials/tutorial_lesson_02_input_image.html translated into python
import halide as hl
import numpy as np

buf = hl.Buffer(np.load("bird.npy"))

brighter = hl.Func("brighter")

x, y, z = hl.Var("x"), hl.Var("y"), hl.Var("z")

value = hl.cast(hl.UInt(8), hl.min(hl.cast(hl.Float(32), buf[x, y, z]) * 1.5, 255))

brighter[x, y, z] = value

output = brighter.realize([buf.width(), buf.height(), buf.channels()])

np.save("brighter.npy", output)