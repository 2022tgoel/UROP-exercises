import halide as hl

i, j, k = hl.Var('i'), hl.Var('j'), hl.Var('k')

f = hl.Func('f')

# f[i, j] = 