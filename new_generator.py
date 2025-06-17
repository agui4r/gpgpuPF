import re
from collections import defaultdict

# Struct definition
print("typedef struct { char is_positive; char idx_y; char idx_x; char valid; } Entry;")

# Read input expression file
with open('input.txt') as f:
    text = f.read()

# Patterns
h_assignments = re.findall(r'h\[(\d+)\]\s*=\s*(.*?);', text, re.DOTALL)
a_pat = re.compile(r'([+-]?)\s*a\[(\d+)\]\[(\d+)\]')
b_pat = re.compile(r'([+-]?)\s*b\[(\d+)\]\[(\d+)\]')

# Create lookup_table[h_idx][2] → lists of entries for a[4][5] and b[5][5]
lookup_table = defaultdict(lambda: [[], []])

for h_idx_str, expr in h_assignments:
    h_idx = int(h_idx_str)

    for match in a_pat.finditer(expr):
        sign, y, x = match.groups()
        is_positive = 0 if sign == '-' else 1
        lookup_table[h_idx][0].append((is_positive, int(y), int(x)))

    for match in b_pat.finditer(expr):
        sign, y, x = match.groups()
        is_positive = 0 if sign == '-' else 1
        lookup_table[h_idx][1].append((is_positive, int(y), int(x)))

# Compute max lengths for padding
max_len_a = max((len(v[0]) for v in lookup_table.values()), default=0)
max_len_b = max((len(v[1]) for v in lookup_table.values()), default=0)
max_len = max(max_len_a, max_len_b)

# Emit lookup_table as C array
print(f"\nEntry lookup_table[][2][{max_len}] = {{")

for h_idx in sorted(lookup_table.keys()):
    print("    {")  # h_idx entry

    for array in lookup_table[h_idx]:  # 0: a, 1: b
        print("        {")
        for entry in array:
            is_pos, y, x = entry
            print(f"            {{ {is_pos}, {y}, {x}, 1 }},")
        for _ in range(max_len - len(array)):  # pad with invalid
            print(f"            {{ 0, 0, 0, 0 }},")
        print("        },")
    print("    },")
print("};")

