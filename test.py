from math import comb

def calc(n):
    return comb(n, 4) + comb(5, 1) * comb(4, 1) * comb(n, 5) + comb(6, 2) * comb(4, 2) * comb(n, 6) + comb(7, 3) * comb(4, 3) * comb(n, 7) + comb(8, 4) * comb(4, 4) * comb(n, 8) 

n = 8
print(calc(n))

print(comb(n, 4)**2)