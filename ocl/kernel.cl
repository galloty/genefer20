/*
Copyright 2020, Yves Gallot

genefer20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

typedef uint	uint32;
typedef int 	int32;
typedef ulong	uint64;
typedef long 	int64;

#define P1		4253024257u		// 507 * 2^23 + 1
#define P2		4194304001u		// 125 * 2^25 + 1
#define P3		4076863489u		// 243 * 2^24 + 1
#define P1_INV	42356678u		// (2^64 - 1) / P1 - 2^32
#define P2_INV	103079214u		// (2^64 - 1) / P2 - 2^32
#define P3_INV	229771911u		// (2^64 - 1) / P3 - 2^32

inline uint32 _mulMod(const uint32 lhs, const uint32 rhs, const uint32 p, const uint32 p_inv)
{
	// Improved division by invariant integers, Niels Moller and Torbjorn Granlund, Algorithm 4.
	const uint64 m = lhs * (uint64)(rhs), q = (m >> 32) * p_inv + m;
	uint32 r = (uint32)(m) - (1 + (uint32)(q >> 32)) * p;
	if (r > (uint32)(q)) r += p;
	return (r >= p) ? r - p : r;
}

inline uint32 mul_P1(const uint32 lhs, const uint32 rhs) { return _mulMod(lhs, rhs, P1, P1_INV); }
inline uint32 mul_P2(const uint32 lhs, const uint32 rhs) { return _mulMod(lhs, rhs, P2, P2_INV); }
inline uint32 mul_P3(const uint32 lhs, const uint32 rhs) { return _mulMod(lhs, rhs, P3, P3_INV); }

__kernel
void square2_P1(__global uint32 * restrict const x)
{
	const size_t k = get_global_id(0);
	x[k] = mul_P1(x[k], x[k]);
}

__kernel
void square2_P2(__global uint32 * restrict const x)
{
	const size_t k = get_global_id(0);
	x[k] = mul_P2(x[k], x[k]);
}

__kernel
void square2_P3(__global uint32 * restrict const x)
{
	const size_t k = get_global_id(0);
	x[k] = mul_P3(x[k], x[k]);
}