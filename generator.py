import re
from collections import namedtuple

Element = namedtuple('Element', ['sign', 'idx_y', 'idx_x'])
Lookup  = namedtuple('Lookup',  ['a_start', 'b_start', 'end'])

# Read input exprs
with open('input.txt') as f:
    text = f.read()

# Regex to extract a[y][x] and b[y][x] with sign
a_pat = re.compile(r'(?P<sign>[-+]?)\s*a\[(?P<y>\d)\]\[(?P<x>\d)\]')
b_pat = re.compile(r'(?P<sign>[-+]?)\s*b\[(?P<y>\d)\]\[(?P<x>\d)\]')

elements = []
lookup = []

for m in re.finditer(r'h\[(\d+)\]\s*=\s*(.*?);', text, re.DOTALL):
    expr = m.group(2)
    a_hits = list(a_pat.finditer(expr))
    b_hits = list(b_pat.finditer(expr))

    a_start = len(elements)
    for hit in a_hits:
        sign = '-' if hit.group('sign') == '-' else '+'
        elements.append(Element(sign, hit.group('y'), hit.group('x')))

    b_start = len(elements)
    for hit in b_hits:
        sign = '-' if hit.group('sign') == '-' else '+'
        elements.append(Element(sign, hit.group('y'), hit.group('x')))

    lookup.append(Lookup(a_start, b_start, len(elements)))


# Emit C-compatible arrays
print('Element elements[] = {')
for e in elements:
    print(f"    {{'{e.sign}', {e.idx_y}, {e.idx_x}}},")
print('};\n')

print('Lookup lookup_table[] = {')
for l in lookup:
    print(f"    {{{l.a_start}, {l.b_start}, {l.end}}},")
print('};')
