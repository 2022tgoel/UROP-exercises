dim = 10

for x in range(dim):
    for y in range(dim):
        if y > x:
            print("({}, {}): {}".format(x, y, (y - x - 1) + dim * x - x * (x + 1) / 2))