import re
from pprint import pprint

# Read input expression file
with open('deepmatmul.cu') as f:
    text = f.read()

# === Hs formula table ============================================================================
# Patterns
h_assignments = re.findall(r'h\[(\d+)\]\s*=\s*(.*?);', text, re.DOTALL)
a_pat = re.compile(r'([+-]?)\s*a\[(\d+)\]\[(\d+)\]')
b_pat = re.compile(r'([+-]?)\s*b\[(\d+)\]\[(\d+)\]')

# Create lookup_table[h_idx][2] → lists of entries for a[4][5] and b[5][5]
lookup_table = [[{'+': [], '-': []}, {'+': [], '-': []}] for _ in range(len(h_assignments))]

for h_idx_str, expr in h_assignments:
    h_idx = int(h_idx_str)

    for match in a_pat.finditer(expr):
        sign, y, x = match.groups()
        sign = '-' if sign == '-' else '+'
        lookup_table[h_idx][0][sign].append(int(y) * 5 + int(x))

    for match in b_pat.finditer(expr):
        sign, y, x = match.groups()
        sign = '-' if sign == '-' else '+'
        lookup_table[h_idx][1][sign].append(int(y) * 5 + int(x))

tables = [
    (0, '+', "a_pos"),
    (0, '-', "a_neg"),
    (1, '+', "b_pos"),
    (1, '-', "b_neg"),
]

for (tile_type_idx, sign, table_name) in tables:
    max_elem = max(lookup_table, key=lambda value: len(value[tile_type_idx][sign]))
    max_len = max(8, len(max_elem[tile_type_idx][sign]))
    print(f"__constant__ int8_t {table_name}[][{max_len}] = {{")
    for h_idx in range(len(lookup_table)):
        print("    {", end="")
        for a_sums in lookup_table[h_idx][tile_type_idx][sign]:
            print(f"{a_sums}, ", end="")
        for i in range(max_len - len(lookup_table[h_idx][tile_type_idx][sign])):
            print("-1, ", end="")
        print("}, ")
    print("};")
    print()

# === C formula table =============================================================================

c_assignments = re.findall(r'c\[(\d+)\]\[(\d+)\]\s*=\s*(.*?);', text, re.DOTALL)
h_pat = re.compile(r'([+-]?)\s*h\[(\d+)\]')

c_lookup_table = [{'+': [], '-': []} for _ in range(len(c_assignments))]

for c_y_idx_str, c_x_idx_str, expr in c_assignments:
    c_idx = int(c_y_idx_str) * 5 + int(c_x_idx_str)

    for match in h_pat.finditer(expr):
        sign, h_idx = match.groups()
        sign = '-' if sign == '-' else '+'
        c_lookup_table[c_idx][sign].append(h_idx)

tables = [
    ('+', "h_pos"),
    ('-', "h_neg"),
]

for (sign, table_name) in tables:
    max_elem = max(c_lookup_table, key=lambda value: len(value[sign]))
    max_len = max(8, len(max_elem[sign]))
    print(f"__constant__ int8_t {table_name}[][{max_len}] = {{")
    for c_idx in range(len(c_lookup_table)):
        print("    {", end="")
        for h_sums in c_lookup_table[c_idx][sign]:
            print(f"{h_sums}, ", end="")
        for i in range(max_len - len(c_lookup_table[c_idx][sign])):
            print("-1, ", end="")
        print("}, ")
    print("};")
    print()
