import numpy as np

factorizations_r = dict(np.load('factorizations_r.npz', allow_pickle=True))
pqr = factorizations_r['4,5,5']
names = ["P", "Q", "R"]

for i, name in enumerate(names):
    rows = len(pqr[i])
    cols = len(pqr[i][0])
    print(f"{name} tiene dimensiones: {rows}x{cols}")

for i, name in enumerate(names):
    rows = len(pqr[i])
    cols = len(pqr[i][0])
    print(f"__constant__ int8_t {name}[{rows}][{cols}] = {{")
    for y in range (rows):
        print("    {", end="")
        for x in range(cols):
            print(f"{pqr[i][y][x]}, ", end="")
        print("},")
    print("};\n")
