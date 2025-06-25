#!/usr/bin/python3

import numpy as np

factorizations_r = None

with open('factorizations_r.npz', 'rb') as f:
    factorizations_r = dict(np.load(f, allow_pickle=True))

print(factorizations_r['4,5,5'])

names = ["P", "Q", "R"]

pqr = factorizations_r['4,5,5']

for i in range(len(pqr)):
    print(f"__constant__ int8_t {names[i]}[][{len(pqr[i])}] = {{")
    for x in range(len(pqr[i][0])):
        print("    {", end="")
        for y in range(len(pqr[i])):
            print(f"{pqr[i][y][x]}, ", end="")
        print("},")
    print("};");
