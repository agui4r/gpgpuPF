import re
from collections import defaultdict

# Your input expressions
code = """
c[0][0] = -h[9] + h[11] + h[13] - h[14] - h[15] + h[52] + h[4] - h[65] - h[6];
c[0][1] = h[12] + h[14] + h[19] + h[20] - h[21] + h[22] + h[24] - h[42] + h[48] + h[49];
c[0][2] = h[14] + h[22] + h[23] + h[33] - h[36] + h[39] - h[40] + h[54] - h[55] - h[8];
c[0][3] = -h[9] + h[11] + h[13] - h[15] + h[22] + h[23] + h[24] + h[25] + h[4] - h[65] - h[6];
c[0][4] = h[14] + h[23] + h[24] + h[26] - h[27] + h[29] + h[30] - h[3] + h[60] + h[63];
c[1][0] = h[9] + h[10] - h[11] + h[12] + h[14] + h[15] - h[16] - h[43] + h[50];
c[1][1] = -h[10] + h[11] - h[12] - h[14] - h[15] + h[16] + h[17] - h[18] - h[20] + h[42] + h[43];
c[1][2] = -h[9] + h[18] + h[31] + h[34] + h[35] + h[36] - h[42] - h[59] - h[5] - h[71];
c[1][3] = h[9] + h[17] - h[18] + h[19] - h[21] - h[23] - h[25] - h[4] - h[68] + h[72];
c[1][4] = -h[9] - h[17] - h[1] - h[29] - h[37] + h[41] - h[42] + h[45] + h[66] + h[73];
c[2][0] = h[9] - h[11] + h[14] + h[15] - h[0] + h[1] + h[2] - h[3] + h[74];
c[2][1] = -h[15] - h[18] - h[20] - h[27] - h[28] - h[37] + h[41] + h[43] - h[46] + h[47];
c[2][2] = -h[15] - h[27] + h[32] + h[36] - h[38] + h[44] - h[45] + h[62] - h[70] - h[7];
c[2][3] = -h[13] + h[15] - h[22] - h[25] + h[26] + h[28] + h[30] + h[45] - h[57] + h[75];
c[2][4] = -h[9] + h[11] - h[14] + h[27] + h[28] - h[1] - h[29] - h[2] + h[45] + h[3] - h[74];
c[3][0] = -h[9] + h[11] - h[14] - h[15] + h[51] + h[53] - h[5] - h[7] + h[8];
c[3][1] = h[10] - h[11] - h[17] + h[20] - h[31] + h[32] - h[33] - h[35] + h[61] - h[69];
c[3][2] = h[9] + h[14] + h[15] - h[32] + h[33] - h[34] - h[36] - h[53] + h[5] + h[7] - h[8];
c[3][3] = h[11] + h[24] + h[25] - h[32] - h[34] - h[39] + h[40] + h[64] - h[67] - h[6];
c[3][4] = -h[11] - h[28] + h[29] - h[33] + h[34] + h[38] + h[2] - h[44] + h[56] + h[58];
"""

# Regex to find c[y][x] and associated h[i] expressions
c_line_re = re.compile(r'c\[(\d+)]\[(\d+)]\s*=\s*([^;]+);')
h_index_re = re.compile(r'h\[(\d+)]')

# Dictionary to store: (y, x) -> list of h indices
c_to_h_indices = {}

# Parse
for match in c_line_re.finditer(code):
    y, x, expr = int(match.group(1)), int(match.group(2)), match.group(3)
    h_indices = [int(i) for i in h_index_re.findall(expr)]
    c_to_h_indices[(y, x)] = h_indices

# Output
for (y, x), h_list in c_to_h_indices.items():
    print(f"c[{y}][{x}] uses h indices: {h_list}")
