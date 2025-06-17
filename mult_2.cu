#include "mult_2.cuh"

#include "util.cuh"
#include <cstdio>

constexpr char pack_elem(char is_positive, char y, char x)
{
    return (is_positive << 7) | (y << 3) | x;
}

inline __device__ constexpr void unpack_elem_into(
    const char packed,
    char &sum,
    char &y,
    char &x
) {
    sum = (packed & 0b10000000);
    y    = (packed & 0b00111000) >> 3;
    x    = (packed & 0b00000111);
}

__constant__ char elements[] = {
    pack_elem(1, 2, 1),
    pack_elem(0, 1, 0),
    pack_elem(0, 1, 4),
    pack_elem(0, 2, 0),
    pack_elem(1, 1, 1),
    pack_elem(1, 1, 4),
    pack_elem(0, 2, 4),
    pack_elem(0, 1, 4),
    pack_elem(0, 4, 0),
    pack_elem(0, 2, 0),
    pack_elem(0, 3, 0),
    pack_elem(1, 3, 1),
    pack_elem(0, 0, 0),
    pack_elem(1, 1, 4),
    pack_elem(1, 0, 1),
    pack_elem(1, 0, 3),
    pack_elem(1, 2, 3),
    pack_elem(0, 1, 4),
    pack_elem(0, 3, 0),
    pack_elem(1, 0, 4),
    pack_elem(1, 1, 1),
    pack_elem(1, 1, 4),
    pack_elem(0, 1, 3),
    pack_elem(1, 4, 0),
    pack_elem(0, 1, 1),
    pack_elem(0, 1, 4),
    pack_elem(0, 3, 4),
    pack_elem(1, 1, 2),
    pack_elem(1, 4, 0),
    pack_elem(0, 0, 0),
    pack_elem(1, 3, 0),
    pack_elem(0, 3, 1),
    pack_elem(1, 0, 0),
    pack_elem(1, 1, 3),
    pack_elem(1, 2, 1),
    pack_elem(0, 2, 2),
    pack_elem(0, 3, 2),
    pack_elem(0, 1, 2),
    pack_elem(1, 2, 0),
    pack_elem(0, 0, 1),
    pack_elem(0, 0, 3),
    pack_elem(1, 3, 3),
    pack_elem(1, 1, 2),
    pack_elem(1, 3, 0),
    pack_elem(1, 1, 1),
    pack_elem(1, 1, 4),
    pack_elem(1, 4, 0),
    pack_elem(0, 1, 0),
    pack_elem(0, 3, 0),
    pack_elem(1, 3, 1),
    pack_elem(0, 0, 0),
    pack_elem(1, 1, 1),
    pack_elem(1, 3, 0),
    pack_elem(0, 3, 1),
    pack_elem(1, 0, 0),
    pack_elem(1, 0, 1),
    pack_elem(1, 0, 3),
    pack_elem(1, 1, 3),
    pack_elem(1, 1, 1),
    pack_elem(1, 3, 0),
    pack_elem(1, 0, 2),
    pack_elem(0, 2, 1),
    pack_elem(1, 2, 2),
    pack_elem(1, 1, 3),
    pack_elem(1, 2, 0),
    pack_elem(0, 0, 1),
    pack_elem(0, 0, 3),
    pack_elem(1, 3, 0),
    pack_elem(0, 2, 1),
    pack_elem(1, 2, 2),
    pack_elem(1, 2, 0),
    pack_elem(1, 0, 1),
    pack_elem(1, 0, 3),
    pack_elem(0, 1, 0),
    pack_elem(1, 1, 1),
    pack_elem(0, 1, 2),
    pack_elem(1, 1, 3),
    pack_elem(0, 2, 1),
    pack_elem(1, 2, 2),
    pack_elem(0, 3, 0),
    pack_elem(1, 3, 1),
    pack_elem(1, 1, 1),
    pack_elem(1, 1, 0),
    pack_elem(1, 0, 0),
    pack_elem(1, 0, 1),
    pack_elem(1, 4, 1),
    pack_elem(0, 1, 2),
    pack_elem(1, 2, 0),
    pack_elem(1, 2, 1),
    pack_elem(1, 4, 1),
    pack_elem(0, 0, 4),
    pack_elem(1, 1, 0),
    pack_elem(1, 1, 2),
    pack_elem(0, 1, 4),
    pack_elem(0, 0, 0),
    pack_elem(0, 0, 1),
    pack_elem(1, 0, 3),
    pack_elem(0, 4, 1),
    pack_elem(1, 1, 0),
    pack_elem(1, 1, 2),
    pack_elem(0, 1, 4),
    pack_elem(1, 4, 1),
    pack_elem(1, 0, 2),
    pack_elem(0, 0, 3),
    pack_elem(0, 1, 3),
    pack_elem(1, 0, 0),
    pack_elem(1, 0, 1),
    pack_elem(0, 0, 3),
    pack_elem(0, 2, 0),
    pack_elem(0, 2, 1),
    pack_elem(1, 2, 3),
    pack_elem(1, 3, 3),
    pack_elem(1, 0, 2),
    pack_elem(0, 2, 0),
    pack_elem(1, 2, 3),
    pack_elem(1, 3, 3),
    pack_elem(1, 0, 4),
    pack_elem(0, 3, 3),
    pack_elem(0, 4, 0),
    pack_elem(1, 4, 3),
    pack_elem(0, 0, 0),
    pack_elem(1, 0, 0),
    pack_elem(0, 0, 3),
    pack_elem(0, 0, 2),
    pack_elem(1, 0, 3),
    pack_elem(1, 0, 4),
    pack_elem(1, 3, 3),
    pack_elem(1, 0, 2),
    pack_elem(0, 2, 0),
    pack_elem(1, 2, 2),
    pack_elem(1, 0, 0),
    pack_elem(0, 0, 3),
    pack_elem(1, 0, 4),
    pack_elem(1, 2, 4),
    pack_elem(0, 2, 3),
    pack_elem(0, 2, 4),
    pack_elem(0, 3, 0),
    pack_elem(0, 3, 4),
    pack_elem(1, 2, 0),
    pack_elem(1, 0, 0),
    pack_elem(1, 0, 4),
    pack_elem(1, 2, 4),
    pack_elem(1, 2, 0),
    pack_elem(0, 2, 2),
    pack_elem(1, 2, 3),
    pack_elem(1, 2, 4),
    pack_elem(0, 0, 3),
    pack_elem(0, 0, 4),
    pack_elem(0, 2, 3),
    pack_elem(0, 3, 3),
    pack_elem(0, 4, 0),
    pack_elem(1, 4, 3),
    pack_elem(0, 4, 4),
    pack_elem(1, 1, 0),
    pack_elem(1, 3, 0),
    pack_elem(1, 3, 3),
    pack_elem(1, 0, 2),
    pack_elem(0, 3, 0),
    pack_elem(0, 3, 1),
    pack_elem(0, 3, 2),
    pack_elem(1, 3, 2),
    pack_elem(0, 2, 0),
    pack_elem(0, 2, 2),
    pack_elem(1, 3, 3),
    pack_elem(0, 0, 2),
    pack_elem(1, 3, 0),
    pack_elem(1, 3, 2),
    pack_elem(0, 3, 4),
    pack_elem(1, 0, 2),
    pack_elem(1, 4, 0),
    pack_elem(1, 4, 2),
    pack_elem(1, 1, 2),
    pack_elem(0, 1, 4),
    pack_elem(0, 3, 4),
    pack_elem(1, 2, 0),
    pack_elem(1, 2, 1),
    pack_elem(1, 2, 2),
    pack_elem(1, 4, 1),
    pack_elem(0, 3, 0),
    pack_elem(0, 3, 3),
    pack_elem(1, 3, 4),
    pack_elem(1, 0, 2),
    pack_elem(0, 1, 2),
    pack_elem(0, 2, 0),
    pack_elem(1, 2, 2),
    pack_elem(0, 2, 3),
    pack_elem(1, 2, 4),
    pack_elem(1, 3, 0),
    pack_elem(1, 3, 1),
    pack_elem(1, 3, 4),
    pack_elem(0, 2, 0),
    pack_elem(0, 3, 0),
    pack_elem(0, 3, 3),
    pack_elem(1, 3, 4),
    pack_elem(1, 0, 2),
    pack_elem(1, 4, 0),
    pack_elem(1, 4, 2),
    pack_elem(1, 4, 4),
    pack_elem(0, 0, 2),
    pack_elem(1, 0, 3),
    pack_elem(1, 0, 4),
    pack_elem(0, 3, 3),
    pack_elem(0, 2, 0),
    pack_elem(0, 2, 2),
    pack_elem(1, 2, 3),
    pack_elem(1, 3, 3),
    pack_elem(0, 0, 0),
    pack_elem(1, 3, 0),
    pack_elem(0, 3, 4),
    pack_elem(1, 0, 2),
    pack_elem(1, 2, 0),
    pack_elem(1, 2, 2),
    pack_elem(0, 2, 3),
    pack_elem(1, 4, 0),
    pack_elem(1, 4, 2),
    pack_elem(0, 4, 3),
    pack_elem(0, 1, 0),
    pack_elem(1, 1, 4),
    pack_elem(0, 2, 4),
    pack_elem(0, 0, 0),
    pack_elem(0, 0, 1),
    pack_elem(0, 0, 4),
    pack_elem(1, 3, 0),
    pack_elem(1, 3, 1),
    pack_elem(1, 3, 4),
    pack_elem(0, 4, 1),
    pack_elem(1, 1, 3),
    pack_elem(1, 3, 0),
    pack_elem(1, 3, 1),
    pack_elem(1, 1, 2),
    pack_elem(1, 2, 1),
    pack_elem(0, 2, 2),
    pack_elem(1, 1, 1),
    pack_elem(0, 2, 0),
    pack_elem(0, 2, 2),
    pack_elem(1, 2, 3),
    pack_elem(0, 3, 2),
    pack_elem(1, 2, 4),
    pack_elem(1, 3, 0),
    pack_elem(1, 3, 2),
    pack_elem(1, 3, 4),
    pack_elem(1, 4, 0),
    pack_elem(1, 4, 2),
    pack_elem(1, 4, 4),
    pack_elem(0, 2, 4),
    pack_elem(0, 4, 0),
    pack_elem(0, 4, 4),
    pack_elem(1, 1, 0),
    pack_elem(0, 1, 4),
    pack_elem(0, 2, 0),
    pack_elem(1, 2, 4),
    pack_elem(1, 0, 0),
    pack_elem(1, 0, 1),
    pack_elem(1, 0, 4),
    pack_elem(0, 3, 0),
    pack_elem(0, 3, 1),
    pack_elem(0, 3, 4),
    pack_elem(0, 1, 2),
    pack_elem(1, 2, 2),
    pack_elem(1, 1, 1),
    pack_elem(1, 2, 1),
    pack_elem(1, 2, 4),
    pack_elem(1, 3, 0),
    pack_elem(1, 3, 1),
    pack_elem(1, 3, 4),
    pack_elem(0, 0, 0),
    pack_elem(0, 0, 2),
    pack_elem(1, 0, 3),
    pack_elem(1, 0, 4),
    pack_elem(0, 1, 0),
    pack_elem(0, 1, 2),
    pack_elem(1, 1, 3),
    pack_elem(1, 1, 4),
    pack_elem(0, 0, 0),
    pack_elem(0, 0, 1),
    pack_elem(1, 0, 3),
    pack_elem(0, 0, 3),
    pack_elem(0, 1, 3),
    pack_elem(1, 1, 1),
    pack_elem(0, 2, 0),
    pack_elem(0, 2, 1),
    pack_elem(1, 2, 3),
    pack_elem(0, 3, 1),
    pack_elem(1, 3, 3),
    pack_elem(1, 1, 1),
    pack_elem(1, 1, 0),
    pack_elem(1, 1, 1),
    pack_elem(0, 4, 0),
    pack_elem(1, 3, 1),
    pack_elem(1, 0, 0),
    pack_elem(1, 1, 0),
    pack_elem(1, 1, 2),
    pack_elem(0, 0, 1),
    pack_elem(0, 1, 0),
    pack_elem(1, 1, 3),
    pack_elem(1, 3, 0),
    pack_elem(1, 0, 1),
    pack_elem(1, 0, 3),
    pack_elem(0, 1, 1),
    pack_elem(0, 1, 4),
    pack_elem(0, 2, 1),
    pack_elem(1, 2, 2),
    pack_elem(0, 3, 1),
    pack_elem(1, 3, 2),
    pack_elem(0, 3, 3),
    pack_elem(0, 3, 4),
    pack_elem(1, 1, 2),
    pack_elem(1, 0, 3),
    pack_elem(0, 3, 3),
    pack_elem(0, 1, 2),
    pack_elem(1, 2, 0),
    pack_elem(1, 2, 2),
    pack_elem(0, 2, 3),
    pack_elem(1, 3, 2),
    pack_elem(0, 3, 3),
    pack_elem(1, 0, 0),
    pack_elem(0, 0, 4),
    pack_elem(0, 3, 0),
    pack_elem(1, 3, 4),
    pack_elem(1, 2, 0),
    pack_elem(1, 2, 2),
    pack_elem(0, 2, 3),
    pack_elem(1, 4, 0),
    pack_elem(1, 4, 2),
    pack_elem(0, 4, 3),
    pack_elem(0, 2, 0),
    pack_elem(0, 3, 0),
    pack_elem(0, 0, 2),
    pack_elem(0, 0, 4),
    pack_elem(0, 1, 4),
    pack_elem(0, 4, 0),
    pack_elem(0, 4, 2),
    pack_elem(0, 4, 4),
    pack_elem(0, 0, 3),
    pack_elem(0, 0, 4),
    pack_elem(0, 2, 3),
    pack_elem(0, 2, 4),
    pack_elem(0, 4, 0),
    pack_elem(1, 4, 3),
    pack_elem(0, 4, 4),
    pack_elem(0, 2, 2),
    pack_elem(1, 2, 3),
    pack_elem(0, 3, 2),
    pack_elem(1, 3, 3),
    pack_elem(1, 3, 0),
    pack_elem(1, 3, 2),
    pack_elem(1, 3, 4),
    pack_elem(1, 4, 0),
    pack_elem(1, 4, 2),
    pack_elem(1, 4, 4),
    pack_elem(1, 1, 4),
    pack_elem(1, 3, 4),
    pack_elem(1, 1, 2),
    pack_elem(0, 2, 0),
    pack_elem(0, 2, 1),
    pack_elem(0, 2, 2),
    pack_elem(0, 4, 1),
    pack_elem(0, 4, 2),
    pack_elem(1, 0, 3),
    pack_elem(1, 2, 3),
    pack_elem(1, 0, 0),
    pack_elem(0, 0, 3),
    pack_elem(1, 0, 4),
    pack_elem(0, 1, 4),
    pack_elem(0, 3, 3),
    pack_elem(1, 3, 4),
    pack_elem(0, 4, 0),
    pack_elem(1, 4, 3),
    pack_elem(0, 4, 4),
    pack_elem(1, 1, 0),
    pack_elem(1, 3, 0),
    pack_elem(1, 0, 1),
    pack_elem(1, 0, 2),
    pack_elem(1, 1, 1),
    pack_elem(0, 3, 0),
    pack_elem(0, 3, 1),
    pack_elem(0, 3, 2),
    pack_elem(0, 2, 2),
    pack_elem(0, 3, 2),
    pack_elem(0, 1, 2),
    pack_elem(0, 2, 2),
    pack_elem(0, 2, 4),
    pack_elem(0, 3, 0),
    pack_elem(0, 3, 2),
    pack_elem(0, 3, 4),
    pack_elem(1, 0, 0),
    pack_elem(0, 0, 2),
    pack_elem(0, 0, 3),
    pack_elem(1, 2, 0),
    pack_elem(0, 2, 2),
    pack_elem(0, 2, 3),
    pack_elem(1, 0, 0),
    pack_elem(0, 0, 3),
    pack_elem(1, 0, 4),
    pack_elem(0, 0, 0),
    pack_elem(1, 3, 0),
    pack_elem(0, 0, 2),
    pack_elem(1, 0, 3),
    pack_elem(1, 1, 3),
    pack_elem(0, 4, 0),
    pack_elem(0, 4, 2),
    pack_elem(1, 4, 3),
    pack_elem(1, 0, 0),
    pack_elem(0, 0, 1),
    pack_elem(1, 0, 2),
    pack_elem(0, 0, 4),
    pack_elem(0, 1, 1),
    pack_elem(0, 1, 4),
    pack_elem(0, 2, 1),
    pack_elem(1, 2, 2),
    pack_elem(0, 3, 0),
    pack_elem(1, 3, 1),
    pack_elem(1, 1, 3),
    pack_elem(1, 1, 4),
    pack_elem(0, 2, 4),
    pack_elem(1, 0, 0),
    pack_elem(1, 0, 1),
    pack_elem(1, 0, 4),
    pack_elem(0, 1, 4),
    pack_elem(0, 3, 0),
    pack_elem(0, 3, 1),
    pack_elem(0, 3, 4),
    pack_elem(1, 4, 1),
    pack_elem(1, 4, 4),
    pack_elem(1, 0, 0),
    pack_elem(1, 0, 2),
    pack_elem(0, 0, 3),
    pack_elem(0, 0, 4),
    pack_elem(0, 3, 0),
    pack_elem(0, 3, 2),
    pack_elem(1, 3, 3),
    pack_elem(1, 3, 4),
    pack_elem(0, 2, 0),
    pack_elem(0, 2, 2),
    pack_elem(1, 2, 3),
    pack_elem(0, 0, 2),
    pack_elem(1, 0, 3),
    pack_elem(0, 1, 2),
    pack_elem(1, 1, 3),
    pack_elem(0, 1, 3),
    pack_elem(0, 2, 0),
    pack_elem(0, 2, 1),
    pack_elem(1, 2, 3),
    pack_elem(0, 4, 1),
    pack_elem(1, 4, 3),
    pack_elem(1, 1, 2),
    pack_elem(0, 1, 4),
    pack_elem(1, 3, 2),
    pack_elem(0, 3, 4),
    pack_elem(0, 2, 0),
    pack_elem(0, 2, 1),
    pack_elem(0, 2, 2),
    pack_elem(0, 2, 0),
    pack_elem(1, 2, 2),
    pack_elem(0, 2, 3),
    pack_elem(1, 2, 4),
    pack_elem(0, 3, 0),
    pack_elem(1, 3, 2),
    pack_elem(0, 3, 3),
    pack_elem(1, 3, 4),
    pack_elem(0, 4, 0),
    pack_elem(0, 4, 2),
    pack_elem(0, 4, 4),
    pack_elem(0, 1, 0),
    pack_elem(0, 1, 3),
    pack_elem(0, 3, 0),
    pack_elem(0, 3, 3),
    pack_elem(1, 3, 0),
    pack_elem(1, 3, 1),
    pack_elem(1, 3, 2),
    pack_elem(1, 0, 2),
    pack_elem(0, 0, 3),
    pack_elem(0, 0, 4),
    pack_elem(1, 1, 2),
    pack_elem(0, 1, 3),
    pack_elem(0, 1, 4),
    pack_elem(1, 0, 0),
    pack_elem(1, 0, 1),
    pack_elem(0, 0, 3),
    pack_elem(1, 1, 3),
    pack_elem(1, 4, 1),
    pack_elem(0, 4, 3),
    pack_elem(1, 1, 0),
    pack_elem(0, 1, 2),
    pack_elem(1, 1, 3),
    pack_elem(0, 2, 0),
    pack_elem(1, 2, 2),
    pack_elem(0, 2, 3),
    pack_elem(1, 3, 0),
    pack_elem(1, 3, 1),
    pack_elem(1, 3, 4),
    pack_elem(0, 0, 1),
    pack_elem(0, 0, 3),
    pack_elem(1, 1, 1),
    pack_elem(1, 1, 4),
    pack_elem(1, 2, 0),
    pack_elem(0, 2, 1),
    pack_elem(0, 2, 3),
    pack_elem(0, 2, 4),
    pack_elem(1, 3, 0),
    pack_elem(0, 3, 1),
    pack_elem(1, 1, 4),
    pack_elem(1, 0, 2),
    pack_elem(1, 2, 2),
    pack_elem(0, 0, 0),
    pack_elem(1, 0, 3),
    pack_elem(0, 0, 4),
    pack_elem(1, 1, 3),
    pack_elem(1, 2, 3),
    pack_elem(0, 2, 4),
};

__constant__ struct {
    unsigned short a_start, b_start;
} lookup_table[] = {
    {0, 1},
    {4, 7},
    {9, 12},
    {14, 17},
    {19, 22},
    {24, 27},
    {29, 32},
    {34, 37},
    {39, 42},
    {44, 46},
    {47, 50},
    {52, 54},
    {55, 58},
    {60, 63},
    {65, 67},
    {68, 70},
    {71, 81},
    {82, 83},
    {86, 87},
    {90, 94},
    {98, 101},
    {102, 105},
    {112, 113},
    {116, 117},
    {120, 121},
    {123, 126},
    {127, 130},
    {134, 135},
    {138, 139},
    {142, 145},
    {146, 149},
    {153, 156},
    {160, 161},
    {163, 164},
    {167, 168},
    {171, 174},
    {178, 181},
    {182, 186},
    {190, 194},
    {198, 202},
    {206, 209},
    {216, 219},
    {226, 227},
    {229, 232},
    {234, 237},
    {244, 245},
    {247, 251},
    {257, 259},
    {265, 273},
    {276, 278},
    {284, 285},
    {288, 289},
    {292, 293},
    {296, 306},
    {307, 309},
    {315, 319},
    {325, 327},
    {333, 337},
    {340, 344},
    {350, 352},
    {358, 360},
    {369, 371},
    {377, 379},
    {385, 391},
    {394, 396},
    {402, 412},
    {413, 415},
    {424, 432},
    {435, 439},
    {445, 449},
    {452, 460},
    {463, 467},
    {470, 476},
    {482, 488},
    {491, 501},
    {502, 504},
    {510, 0},
};

inline __device__ float calc_h(const float a[4][5], const float b[5][5], int tid)
{
    float sum_a = 0, sum_b = 0;

    char sign, y, x;
    for (int j = lookup_table[tid].a_start; j < lookup_table[tid].b_start; ++j)
    {
        unpack_elem_into(elements[j], sign, y, x);
        sum_a += sign
            ? +a[y][x]
            : -a[y][x];
    }
    for (int j = lookup_table[tid].b_start; j < lookup_table[tid+1].a_start; ++j)
    {
        unpack_elem_into(elements[j], sign, y, x);
        sum_b += sign
            ? +b[y][x]
            : -b[y][x];
    }

    return sum_a * sum_b;
}

struct LutStruct {char sign; char idx;};
__constant__ LutStruct lut[][12] = {
    {{-1, 9}, {+1, 11}, {+1, 13}, {-1, 14}, {-1, 15}, {+1, 52}, {+1, 4}, {-1, 65}, {-1, 6}, {0, -1}},
    {{+1, 12}, {+1, 14}, {+1, 19}, {+1, 20}, {-1, 21}, {+1, 22}, {+1, 24}, {-1, 42}, {+1, 48}, {+1, 49}, {0, -1}},
    {{+1, 14}, {+1, 22}, {+1, 23}, {+1, 33}, {-1, 36}, {+1, 39}, {-1, 40}, {+1, 54}, {-1, 55}, {-1, 8}, {0, -1}},
    {{-1, 9}, {+1, 11}, {+1, 13}, {-1, 15}, {+1, 22}, {+1, 23}, {+1, 24}, {+1, 25}, {+1, 4}, {-1, 65}, {-1, 6}, {0, -1}},
    {{+1, 14}, {+1, 23}, {+1, 24}, {+1, 26}, {-1, 27}, {+1, 29}, {+1, 30}, {-1, 3}, {+1, 60}, {+1, 63}, {0, -1}},
    {{+1, 9}, {+1, 10}, {-1, 11}, {+1, 12}, {+1, 14}, {+1, 15}, {-1, 16}, {-1, 43}, {+1, 50}, {0, -1}},
    {{-1, 10}, {+1, 11}, {-1, 12}, {-1, 14}, {-1, 15}, {+1, 16}, {+1, 17}, {-1, 18}, {-1, 20}, {+1, 42}, {+1, 43}, {0, -1}},
    {{-1, 9}, {+1, 18}, {+1, 31}, {+1, 34}, {+1, 35}, {+1, 36}, {-1, 42}, {-1, 59}, {-1, 5}, {-1, 71}, {0, -1}},
    {{+1, 9}, {+1, 17}, {-1, 18}, {+1, 19}, {-1, 21}, {-1, 23}, {-1, 25}, {-1, 4}, {-1, 68}, {+1, 72}, {0, -1}},
    {{-1, 9}, {-1, 17}, {-1, 1}, {-1, 29}, {-1, 37}, {+1, 41}, {-1, 42}, {+1, 45}, {+1, 66}, {+1, 73}, {0, -1}},
    {{+1, 9}, {-1, 11}, {+1, 14}, {+1, 15}, {-1, 0}, {+1, 1}, {+1, 2}, {-1, 3}, {+1, 74}, {0, -1}},
    {{-1, 15}, {-1, 18}, {-1, 20}, {-1, 27}, {-1, 28}, {-1, 37}, {+1, 41}, {+1, 43}, {-1, 46}, {+1, 47}, {0, -1}},
    {{-1, 15}, {-1, 27}, {+1, 32}, {+1, 36}, {-1, 38}, {+1, 44}, {-1, 45}, {+1, 62}, {-1, 70}, {-1, 7}, {0, -1}},
    {{-1, 13}, {+1, 15}, {-1, 22}, {-1, 25}, {+1, 26}, {+1, 28}, {+1, 30}, {+1, 45}, {-1, 57}, {+1, 75}, {0, -1}},
    {{-1, 9}, {+1, 11}, {-1, 14}, {+1, 27}, {+1, 28}, {-1, 1}, {-1, 29}, {-1, 2}, {+1, 45}, {+1, 3}, {-1, 74}, {0, -1}},
    {{-1, 9}, {+1, 11}, {-1, 14}, {-1, 15}, {+1, 51}, {+1, 53}, {-1, 5}, {-1, 7}, {+1, 8}, {0, -1}},
    {{+1, 10}, {-1, 11}, {-1, 17}, {+1, 20}, {-1, 31}, {+1, 32}, {-1, 33}, {-1, 35}, {+1, 61}, {-1, 69}, {0, -1}},
    {{+1, 9}, {+1, 14}, {+1, 15}, {-1, 32}, {+1, 33}, {-1, 34}, {-1, 36}, {-1, 53}, {+1, 5}, {+1, 7}, {-1, 8}, {0, -1}},
    {{+1, 11}, {+1, 24}, {+1, 25}, {-1, 32}, {-1, 34}, {-1, 39}, {+1, 40}, {+1, 64}, {-1, 67}, {-1, 6}, {0, -1}},
    {{-1, 11}, {-1, 28}, {+1, 29}, {-1, 33}, {+1, 34}, {+1, 38}, {+1, 2}, {-1, 44}, {+1, 56}, {+1, 58}, {0, -1}},
};

inline __device__ float calc_c(const float h[76], int tid)
{
    float res = 0.0f;
    auto *it = lut[tid];
    while (it->sign != 0) {
        res += it->sign * h[it->idx];
        it++;
    }
    return res;
}

__device__ void deepmatmul_only_h(const float a[4][5], const float b[5][5], float h[76])
{
    h[0] = a[2][1] * ( -b[1][0] - b[1][4] - b[2][0] );
    h[1] = (a[1][1] + a[1][4] - a[2][4]) * ( -b[1][4] - b[4][0] );
    h[2] = (-a[2][0] - a[3][0] + a[3][1]) * ( -b[0][0] + b[1][4] );
    h[3] = (a[0][1] + a[0][3] + a[2][3]) * ( -b[1][4] - b[3][0] );
    h[4] = (a[0][4] + a[1][1] + a[1][4]) * ( -b[1][3] + b[4][0] );
    h[5] = (-a[1][1] - a[1][4] - a[3][4]) * ( b[1][2] + b[4][0] );
    h[6] = (-a[0][0] + a[3][0] - a[3][1]) * ( b[0][0] + b[1][3] );
    h[7] = (a[2][1] - a[2][2] - a[3][2]) * ( -b[1][2] + b[2][0] );
    h[8] = (-a[0][1] - a[0][3] + a[3][3]) * ( b[1][2] + b[3][0] );
    h[9] = (a[1][1] + a[1][4]) * b[4][0];
    h[10] = (-a[1][0] - a[3][0] + a[3][1]) * ( -b[0][0] + b[1][1] );
    h[11] = (a[3][0] - a[3][1]) * b[0][0];
    h[12] = (a[0][1] + a[0][3] + a[1][3]) * ( b[1][1] + b[3][0] );
    h[13] = (a[0][2] - a[2][1] + a[2][2]) * ( b[1][3] + b[2][0] );
    h[14] = (-a[0][1] - a[0][3]) * b[3][0];
    h[15] = (-a[2][1] + a[2][2]) * b[2][0];
    h[16] = (a[0][1] + a[0][3] - a[1][0] + a[1][1] - a[1][2] + a[1][3] - a[2][1] + a[2][2] - a[3][0] + a[3][1]) * b[1][1];
    h[17] = a[1][0] * ( b[0][0] + b[0][1] + b[4][1] );
    h[18] = -a[1][2] * ( b[2][0] + b[2][1] + b[4][1] );
    h[19] = (-a[0][4] + a[1][0] + a[1][2] - a[1][4]) * ( -b[0][0] - b[0][1] + b[0][3] - b[4][1] );
    h[20] = (a[1][0] + a[1][2] - a[1][4]) * b[4][1];
    h[21] = (a[0][2] - a[0][3] - a[1][3]) * ( b[0][0] + b[0][1] - b[0][3] - b[2][0] - b[2][1] + b[2][3] + b[3][3] );
    h[22] = a[0][2] * ( -b[2][0] + b[2][3] + b[3][3] );
    h[23] = a[0][4] * ( -b[3][3] - b[4][0] + b[4][3] );
    h[24] = -a[0][0] * ( b[0][0] - b[0][3] );
    h[25] = (-a[0][2] + a[0][3] + a[0][4]) * b[3][3];
    h[26] = (a[0][2] - a[2][0] + a[2][2]) * ( b[0][0] - b[0][3] + b[0][4] + b[2][4] );
    h[27] = -a[2][3] * ( -b[2][4] - b[3][0] - b[3][4] );
    h[28] = a[2][0] * ( b[0][0] + b[0][4] + b[2][4] );
    h[29] = (a[2][0] - a[2][2] + a[2][3]) * b[2][4];
    h[30] = (-a[0][3] - a[0][4] - a[2][3]) * ( -b[3][3] - b[4][0] + b[4][3] - b[4][4] );
    h[31] = (a[1][0] + a[3][0] + a[3][3]) * ( b[0][2] - b[3][0] - b[3][1] - b[3][2] );
    h[32] = a[3][2] * ( -b[2][0] - b[2][2] );
    h[33] = a[3][3] * ( -b[0][2] + b[3][0] + b[3][2] );
    h[34] = -a[3][4] * ( b[0][2] + b[4][0] + b[4][2] );
    h[35] = (a[1][2] - a[1][4] - a[3][4]) * ( b[2][0] + b[2][1] + b[2][2] + b[4][1] );
    h[36] = (-a[3][0] - a[3][3] + a[3][4]) * b[0][2];
    h[37] = (-a[1][2] - a[2][0] + a[2][2] - a[2][3]) * ( b[2][4] + b[3][0] + b[3][1] + b[3][4] );
    h[38] = (-a[2][0] - a[3][0] - a[3][3] + a[3][4]) * ( b[0][2] + b[4][0] + b[4][2] + b[4][4] );
    h[39] = (-a[0][2] + a[0][3] + a[0][4] - a[3][3]) * ( -b[2][0] - b[2][2] + b[2][3] + b[3][3] );
    h[40] = (-a[0][0] + a[3][0] - a[3][4]) * ( b[0][2] + b[2][0] + b[2][2] - b[2][3] + b[4][0] + b[4][2] - b[4][3] );
    h[41] = (-a[1][0] + a[1][4] - a[2][4]) * ( -b[0][0] - b[0][1] - b[0][4] + b[3][0] + b[3][1] + b[3][4] - b[4][1] );
    h[42] = a[1][3] * ( b[3][0] + b[3][1] );
    h[43] = (a[1][2] + a[2][1] - a[2][2]) * ( b[1][1] - b[2][0] );
    h[44] = (-a[2][2] + a[2][3] - a[3][2]) * ( b[2][4] + b[3][0] + b[3][2] + b[3][4] + b[4][0] + b[4][2] + b[4][4] );
    h[45] = -a[2][4] * ( -b[4][0] - b[4][4] );
    h[46] = (a[1][0] - a[1][4] - a[2][0] + a[2][4]) * ( b[0][0] + b[0][1] + b[0][4] - b[3][0] - b[3][1] - b[3][4] );
    h[47] = (-a[1][2] + a[2][2]) * ( b[1][1] + b[2][1] + b[2][4] + b[3][0] + b[3][1] + b[3][4] );
    h[48] = (-a[0][0] - a[0][2] + a[0][3] + a[0][4] - a[1][0] - a[1][2] + a[1][3] + a[1][4]) * ( -b[0][0] - b[0][1] + b[0][3] );
    h[49] = (-a[0][3] - a[1][3]) * ( b[1][1] - b[2][0] - b[2][1] + b[2][3] - b[3][1] + b[3][3] );
    h[50] = a[1][1] * ( b[1][0] + b[1][1] - b[4][0] );
    h[51] = a[3][1] * (b[0][0] + b[1][0] + b[1][2]);
    h[52] = -a[0][1] * (-b[1][0] + b[1][3] + b[3][0]);
    h[53] = (a[0][1] + a[0][3] - a[1][1] - a[1][4] - a[2][1] + a[2][2] - a[3][1] + a[3][2] - a[3][3] - a[3][4]) * b[1][2];
    h[54] = (a[0][3] - a[3][3]) * (-b[1][2] + b[2][0] + b[2][2] - b[2][3] + b[3][2] - b[3][3]);
    h[55] = (a[0][0] - a[0][4] - a[3][0] + a[3][4]) * (b[2][0] + b[2][2] - b[2][3] + b[4][0] + b[4][2] - b[4][3]);
    h[56] = (-a[2][0] - a[3][0]) * (-b[0][2] - b[0][4] - b[1][4] - b[4][0] - b[4][2] - b[4][4]);
    h[57] = (-a[0][3] - a[0][4] - a[2][3] - a[2][4]) * (-b[4][0] + b[4][3] - b[4][4]);
    h[58] = (-a[2][2] + a[2][3] - a[3][2] + a[3][3]) * (b[3][0] + b[3][2] + b[3][4] + b[4][0] + b[4][2] + b[4][4]);
    h[59] = (a[1][4] + a[3][4]) * (b[1][2] - b[2][0] - b[2][1] - b[2][2] - b[4][1] - b[4][2]);
    h[60] = (a[0][3] + a[2][3]) * (b[0][0] - b[0][3] + b[0][4] - b[1][4] - b[3][3] + b[3][4] - b[4][0] + b[4][3] - b[4][4]);
    h[61] = (a[1][0] + a[3][0]) * (b[0][1] + b[0][2] + b[1][1] - b[3][0] - b[3][1] - b[3][2]);
    h[62] = (-a[2][2] - a[3][2]) * (-b[1][2] - b[2][2] - b[2][4] - b[3][0] - b[3][2] - b[3][4]);
    h[63] = (a[0][0] - a[0][2] - a[0][3] + a[2][0] - a[2][2] - a[2][3]) * (b[0][0] - b[0][3] + b[0][4]);
    h[64] = (-a[0][0] + a[3][0]) * (-b[0][2] + b[0][3] + b[1][3] - b[4][0] - b[4][2] + b[4][3]);
    h[65] = (a[0][0] - a[0][1] + a[0][2] - a[0][4] - a[1][1] - a[1][4] - a[2][1] + a[2][2] - a[3][0] + a[3][1]) * b[1][3];
    h[66] = (a[1][4] - a[2][4]) * (b[0][0] + b[0][1] + b[0][4] - b[1][4] - b[3][0] - b[3][1] - b[3][4] + b[4][1] + b[4][4]);
    h[67] = (a[0][0] + a[0][2] - a[0][3] - a[0][4] - a[3][0] - a[3][2] + a[3][3] + a[3][4]) * (-b[2][0] - b[2][2] + b[2][3]);
    h[68] = (-a[0][2] + a[0][3] - a[1][2] + a[1][3]) * (-b[1][3] - b[2][0] - b[2][1] + b[2][3] - b[4][1] + b[4][3]);
    h[69] = (a[1][2] - a[1][4] + a[3][2] - a[3][4]) * (-b[2][0] - b[2][1] - b[2][2]);
    h[70] = (-a[2][0] + a[2][2] - a[2][3] + a[2][4] - a[3][0] + a[3][2] - a[3][3] + a[3][4]) * (-b[4][0] - b[4][2] - b[4][4]);
    h[71] = (-a[1][0] - a[1][3] - a[3][0] - a[3][3]) * (b[3][0] + b[3][1] + b[3][2]);
    h[72] = (a[0][2] - a[0][3] - a[0][4] + a[1][2] - a[1][3] - a[1][4]) * (b[0][0] + b[0][1] - b[0][3] + b[1][3] + b[4][1] - b[4][3]);
    h[73] = (a[1][0] - a[1][2] + a[1][3] - a[2][0] + a[2][2] - a[2][3]) * (b[3][0] + b[3][1] + b[3][4]);
    h[74] = - (a[0][1] + a[0][3] - a[1][1] - a[1][4] - a[2][0] + a[2][1] + a[2][3] + a[2][4] - a[3][0] + a[3][1]) * b[1][4];
    h[75] = (a[0][2] + a[2][2]) * (-b[0][0] + b[0][3] - b[0][4] + b[1][3] + b[2][3] - b[2][4]);
}

__device__ void deepmatmul_only_c(const float h[76], float c[4][5]) {
    c[0][0] += -h[9] + h[11] + h[13] - h[14] - h[15] + h[52] + h[4] - h[65] - h[6];
    c[0][1] += h[12] + h[14] + h[19] + h[20] - h[21] + h[22] + h[24] - h[42] + h[48] + h[49];
    c[0][2] += h[14] + h[22] + h[23] + h[33] - h[36] + h[39] - h[40] + h[54] - h[55] - h[8];
    c[0][3] += -h[9] + h[11] + h[13] - h[15] + h[22] + h[23] + h[24] + h[25] + h[4] - h[65] - h[6];
    c[0][4] += h[14] + h[23] + h[24] + h[26] - h[27] + h[29] + h[30] - h[3] + h[60] + h[63];
    c[1][0] += h[9] + h[10] - h[11] + h[12] + h[14] + h[15] - h[16] - h[43] + h[50];
    c[1][1] += -h[10] + h[11] - h[12] - h[14] - h[15] + h[16] + h[17] - h[18] - h[20] + h[42] + h[43];
    c[1][2] += -h[9] + h[18] + h[31] + h[34] + h[35] + h[36] - h[42] - h[59] - h[5] - h[71];
    c[1][3] += h[9] + h[17] - h[18] + h[19] - h[21] - h[23] - h[25] - h[4] - h[68] + h[72];
    c[1][4] += -h[9] - h[17] - h[1] - h[29] - h[37] + h[41] - h[42] + h[45] + h[66] + h[73];
    c[2][0] += h[9] - h[11] + h[14] + h[15] - h[0] + h[1] + h[2] - h[3] + h[74];
    c[2][1] += -h[15] - h[18] - h[20] - h[27] - h[28] - h[37] + h[41] + h[43] - h[46] + h[47];
    c[2][2] += -h[15] - h[27] + h[32] + h[36] - h[38] + h[44] - h[45] + h[62] - h[70] - h[7];
    c[2][3] += -h[13] + h[15] - h[22] - h[25] + h[26] + h[28] + h[30] + h[45] - h[57] + h[75];
    c[2][4] += -h[9] + h[11] - h[14] + h[27] + h[28] - h[1] - h[29] - h[2] + h[45] + h[3] - h[74];
    c[3][0] += -h[9] + h[11] - h[14] - h[15] + h[51] + h[53] - h[5] - h[7] + h[8];
    c[3][1] += h[10] - h[11] - h[17] + h[20] - h[31] + h[32] - h[33] - h[35] + h[61] - h[69];
    c[3][2] += h[9] + h[14] + h[15] - h[32] + h[33] - h[34] - h[36] - h[53] + h[5] + h[7] - h[8];
    c[3][3] += h[11] + h[24] + h[25] - h[32] - h[34] - h[39] + h[40] + h[64] - h[67] - h[6];
    c[3][4] += -h[11] - h[28] + h[29] - h[33] + h[34] + h[38] + h[2] - h[44] + h[56] + h[58];
}

__global__ void mult_2_kernel(
    const float* mat_a,
    const float* mat_b,
    float* mat_c,
    int N)
{
    int tile_idx_x = blockIdx.x;
    int tile_idx_y = blockIdx.y * 3;

    __shared__ float tile_a[4 * 3][5];
    __shared__ float tile_b[5][5];

    float value = 0.0f;

    __shared__ float h[3][76];

    clock_t start_time, stop_time;

    int y = threadIdx.y;
    int x = threadIdx.x;
    for (int offset = 0; offset < N / 5; offset++)
    {
        start_time = clock();
        // Copy tiles of mat_a to shared memory
        tile_a[y][x] = mat_a[(tile_idx_y * 4 + y) * N + (offset * 5 + x)];

        // Copy tile of mat_b to shared memory
        if (y < 5)
        {
            tile_b[y][x] = mat_b[(offset * 5 + y) * N + (tile_idx_x * 5 + x)];
        }

        __syncthreads();
        stop_time = clock();
        if (x == 0 && y == 0 && blockIdx.x == 0 && blockIdx.y == 0)
            printf("Time copy tile to shared = %ld\n", stop_time - start_time);

        start_time = clock();
        // if (y == 0 && x < 3)
        // {
        //     deepmatmul_only_h(tile_a, tile_b, h[x]);
        //
        // }
       
        int tile = y / 4; // 0,1,2 (id del tile dentro del bloque de threads de cuda)
        int tile_local_id = (y % 4) * blockDim.x + x;
        // printf("%d\n", tile_local_id);

        // Hay 76 hs entonces si utilizamos solo 19 threads por tile (19 de 20)
        // podemos calcular 4 hs por thread
        if (tile_local_id < 19)
        {
            #pragma unroll 4
            for (int i = 0; i < 4; i++)
            {
                int h_idx = tile_local_id * 4 + i;
                // printf("%d\n", h_idx);
                h[tile][h_idx] = calc_h(tile_a, tile_b, h_idx);
            }
        }

        // correct: -3 -2 -0 -6 0 -6 -2 -0 -2 2 -0 0 6 2 -2 0 2 3 -3 -0 1 -1 1 -1 -0 1 2 3 3 1 6 -6 -2 1 -3 -4 -1 -8 -8 0 -3 1 2 0 -7 2 0 0 -0 -0 1 3 -1 -2 0 0 12 4 0 -8 -2 0 12 -2 0 -2 0 -0 -0 -0 -0 -12 -4 0 -2 0
        // if (x == 0 && y == 0 && tile_idx_x == 0 && tile_idx_y == 0 && offset == 0)
        // {
        //     for (int i = 0; i < 76; i++)
        //         printf("%1.0f ", h[0][i]);
        //     printf("\n");
        // }

        __syncthreads();

        stop_time = clock();
        if (x == 0 && y == 0 && blockIdx.x == 0 && blockIdx.y == 0)
            printf("Time calc h[] = %ld\n", stop_time - start_time);

        start_time = clock();

        // if (y == 0 && x < 3)
        // {
        //     deepmatmul_only_c(h[x], &tile_c[x*4]);
        // }
        
        value += calc_c(h[tile], tile_local_id);
        
        __syncthreads();

        stop_time = clock();
        if (x == 0 && y == 0 && blockIdx.x == 0 && blockIdx.y == 0)
            printf("Time calc_c = %ld\n", stop_time - start_time);
    }

    mat_c[(tile_idx_y * 4 + y) * N + (tile_idx_x * 5 + x)] = value;
}

void mult_2(const float* mat_a, const float* mat_b, float* mat_c, int N)
{
    constexpr int TILE_SIZE_X = 5;
    constexpr int TILE_SIZE_Y = 4;
    dim3 gridDim(N / 5, N / (4 * 3), 1);
    dim3 blockDim(TILE_SIZE_X, TILE_SIZE_Y * 3, 1);
    mult_2_kernel<<<gridDim, blockDim>>>(mat_a, mat_b, mat_c, N);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());
}
