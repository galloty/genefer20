/*
Copyright 2020, Yves Gallot

genefer20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

typedef uint	uint32;
typedef int 	int32;
typedef ulong	uint64;
typedef long 	int64;
typedef uint2	uint32_2;

#define P1			4253024257u		// 507 * 2^23 + 1
#define P2			4194304001u		// 125 * 2^25 + 1
#define P3			4076863489u		// 243 * 2^24 + 1
#define P1_INV		42356678u		// (2^64 - 1) / P1 - 2^32
#define P2_INV		103079214u		// (2^64 - 1) / P2 - 2^32
#define P3_INV		229771911u		// (2^64 - 1) / P3 - 2^32
#define InvP2_P1	1822724754u		// 1 / P2 mod P1
#define InvP3_P1	607574918u		// 1 / P3 mod P1
#define InvP3_P2	2995931465u		// 1 / P3 mod P2
#define P1P2		(P1 * (uint64)P2)
#define P2P3		(P2 * (uint64)P3)

inline uint32 _addMod(const uint32 lhs, const uint32 rhs, const uint32 p)
{
	const uint32 c = (lhs >= p - rhs) ? p : 0;
	return lhs + rhs - c;
}

inline uint32 _subMod(const uint32 lhs, const uint32 rhs, const uint32 p)
{
	const uint32 c = (lhs < rhs) ? p : 0;
	return lhs - rhs + c;
}

inline uint32 _mulMod(const uint32 lhs, const uint32 rhs, const uint32 p, const uint32 p_inv)
{
	// Improved division by invariant integers, Niels Moller and Torbjorn Granlund, Algorithm 4.
	const uint64 m = lhs * (uint64)(rhs), q = (m >> 32) * p_inv + m;
	uint32 r = (uint32)m - (1 + (uint32)(q >> 32)) * p;
	if (r > (uint32)q) r += p;
	return (r >= p) ? r - p : r;
}

inline uint32 seti_P1(const int32 i) { return (i < 0) ? (uint32)(i + P1) : (uint32)i; }
inline uint32 seti_P2(const int32 i) { return (i < 0) ? (uint32)(i + P2) : (uint32)i; }
inline uint32 seti_P3(const int32 i) { return (i < 0) ? (uint32)(i + P3) : (uint32)i; }

inline int32 geti_P1(const uint32 n) { return (n > P1 / 2) ? (int32)(n - P1) : (int32)n; }

inline uint32 add_P1(const uint32 lhs, const uint32 rhs) { return _addMod(lhs, rhs, P1); }
inline uint32 add_P2(const uint32 lhs, const uint32 rhs) { return _addMod(lhs, rhs, P2); }
inline uint32 add_P3(const uint32 lhs, const uint32 rhs) { return _addMod(lhs, rhs, P3); }

inline uint32 sub_P1(const uint32 lhs, const uint32 rhs) { return _subMod(lhs, rhs, P1); }
inline uint32 sub_P2(const uint32 lhs, const uint32 rhs) { return _subMod(lhs, rhs, P2); }
inline uint32 sub_P3(const uint32 lhs, const uint32 rhs) { return _subMod(lhs, rhs, P3); }

inline uint32 mul_P1(const uint32 lhs, const uint32 rhs) { return _mulMod(lhs, rhs, P1, P1_INV); }
inline uint32 mul_P2(const uint32 lhs, const uint32 rhs) { return _mulMod(lhs, rhs, P2, P2_INV); }
inline uint32 mul_P3(const uint32 lhs, const uint32 rhs) { return _mulMod(lhs, rhs, P3, P3_INV); }

inline int64 garner2(const uint32 r1, const uint32 r2)
{
	const uint32 u12 = mul_P1(sub_P1(r1, r2), InvP2_P1);
	const uint64 n = r2 + u12 * (uint64)P2;
	return (n > P1P2 / 2) ? (int64)(n - P1P2) : (int64)n;
}

__kernel
void set(const __global uint32 * restrict const x, __global uint32 * restrict const y)
{
	const size_t k = get_global_id(0);
	y[k] = x[k];
}

__kernel
void swap(__global uint32 * restrict const x, __global uint32 * restrict const y)
{
	const size_t k = get_global_id(0);
	const uint32 t = x[k]; x[k] = y[k]; y[k] = t;
}

__kernel
void reset(__global uint32 * restrict const x, const uint32 a)
{
	const size_t k = get_global_id(0);
	x[k] = (k < VSIZE) ? a : 0;
}

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

__kernel
void mul2_P1(const __global uint32 * restrict const y, __global uint32 * restrict const x)
{
	const size_t k = get_global_id(0);
	x[k] = mul_P1(x[k], y[k]);
}

__kernel
void mul2_P2(const __global uint32 * restrict const y, __global uint32 * restrict const x)
{
	const size_t k = get_global_id(0);
	x[k] = mul_P2(x[k], y[k]);
}

__kernel
void mul2_P3(const __global uint32 * restrict const y, __global uint32 * restrict const x)
{
	const size_t k = get_global_id(0);
	x[k] = mul_P3(x[k], y[k]);
}

__constant uint64 cMask[64] = {
	0x0000000000000001ul, 0x0000000000000002ul, 0x0000000000000004ul, 0x0000000000000008ul, 0x0000000000000010ul, 0x0000000000000020ul, 0x0000000000000040ul, 0x0000000000000080ul,
	0x0000000000000100ul, 0x0000000000000200ul, 0x0000000000000400ul, 0x0000000000000800ul, 0x0000000000001000ul, 0x0000000000002000ul, 0x0000000000004000ul, 0x0000000000008000ul,
	0x0000000000010000ul, 0x0000000000020000ul, 0x0000000000040000ul, 0x0000000000080000ul, 0x0000000000100000ul, 0x0000000000200000ul, 0x0000000000400000ul, 0x0000000000800000ul,
	0x0000000001000000ul, 0x0000000002000000ul, 0x0000000004000000ul, 0x0000000008000000ul, 0x0000000010000000ul, 0x0000000020000000ul, 0x0000000040000000ul, 0x0000000080000000ul,
	0x0000000100000000ul, 0x0000000200000000ul, 0x0000000400000000ul, 0x0000000800000000ul, 0x0000001000000000ul, 0x0000002000000000ul, 0x0000004000000000ul, 0x0000008000000000ul,
	0x0000010000000000ul, 0x0000020000000000ul, 0x0000040000000000ul, 0x0000080000000000ul, 0x0000100000000000ul, 0x0000200000000000ul, 0x0000400000000000ul, 0x0000800000000000ul,
	0x0001000000000000ul, 0x0002000000000000ul, 0x0004000000000000ul, 0x0008000000000000ul, 0x0010000000000000ul, 0x0020000000000000ul, 0x0040000000000000ul, 0x0080000000000000ul,
	0x0100000000000000ul, 0x0200000000000000ul, 0x0400000000000000ul, 0x0800000000000000ul, 0x1000000000000000ul, 0x2000000000000000ul, 0x4000000000000000ul, 0x8000000000000000ul
	};

__kernel
void mul2cond_P1(const __global uint32 * restrict const y, __global uint32 * restrict const x, const uint64 c)
{
	const size_t k = get_global_id(0);
	if ((c & cMask[k % VSIZE]) != 0) x[k] = mul_P1(x[k], y[k]);
}

__kernel
void mul2cond_P2(const __global uint32 * restrict const y, __global uint32 * restrict const x, const uint64 c)
{
	const size_t k = get_global_id(0);
	if ((c & cMask[k % VSIZE]) != 0) x[k] = mul_P2(x[k], y[k]);
}

__kernel
void mul2cond_P3(const __global uint32 * restrict const y, __global uint32 * restrict const x, const uint64 c)
{
	const size_t k = get_global_id(0);
	if ((c & cMask[k % VSIZE]) != 0) x[k] = mul_P3(x[k], y[k]);
}

__kernel
void forward2_P1(const __global uint32 * restrict const wr, __global uint32 * restrict const x, const uint32 s, const uint32 m, const int lm)
{
	const size_t id = get_global_id(0);
	const size_t vid = id / VSIZE, l = id % VSIZE;
	const size_t j = vid >> lm, i = vid & (m - 1), mj = vid - i;
	const uint32 w = wr[s + j];
	const size_t k = VSIZE * (2 * mj + i) + l;
	const uint32 u = x[k], um = mul_P1(x[k + VSIZE * m], w);
	x[k] = add_P1(u, um); x[k + VSIZE * m] = sub_P1(u, um);
}

__kernel
void forward2_P2(const __global uint32 * restrict const wr, __global uint32 * restrict const x, const uint32 s, const uint32 m, const int lm)
{
	const size_t id = get_global_id(0);
	const size_t vid = id / VSIZE, l = id % VSIZE;
	const size_t j = vid >> lm, i = vid & (m - 1), mj = vid - i;
	const uint32 w = wr[s + j];
	const size_t k = VSIZE * (2 * mj + i) + l;
	const uint32 u = x[k], um = mul_P2(x[k + VSIZE * m], w);
	x[k] = add_P2(u, um); x[k + VSIZE * m] = sub_P2(u, um);
}

__kernel
void forward2_P3(const __global uint32 * restrict const wr, __global uint32 * restrict const x, const uint32 s, const uint32 m, const int lm)
{
	const size_t id = get_global_id(0);
	const size_t vid = id / VSIZE, l = id % VSIZE;
	const size_t j = vid >> lm, i = vid & (m - 1), mj = vid - i;
	const uint32 w = wr[s + j];
	const size_t k = VSIZE * (2 * mj + i) + l;
	const uint32 u = x[k], um = mul_P3(x[k + VSIZE * m], w);
	x[k] = add_P3(u, um); x[k + VSIZE * m] = sub_P3(u, um);
}

__kernel
void backward2_P1(const __global uint32 * restrict const wri, __global uint32 * restrict const x, const uint32 s, const uint32 m, const int lm)
{
	const size_t id = get_global_id(0);
	const size_t vid = id / VSIZE, l = id % VSIZE;
	const size_t j = vid >> lm, i = vid & (m - 1), mj = vid - i;
	const uint32 wi = wri[s + j];
	const size_t k = VSIZE * (2 * mj + i) + l;
	const uint32 u = x[k], um = x[k + VSIZE * m];
	x[k] = add_P1(u, um); x[k + VSIZE * m] = mul_P1(sub_P1(u, um), wi);
}

__kernel
void backward2_P2(const __global uint32 * restrict const wri, __global uint32 * restrict const x, const uint32 s, const uint32 m, const int lm)
{
	const size_t id = get_global_id(0);
	const size_t vid = id / VSIZE, l = id % VSIZE;
	const size_t j = vid >> lm, i = vid & (m - 1), mj = vid - i;
	const uint32 wi = wri[s + j];
	const size_t k = VSIZE * (2 * mj + i) + l;
	const uint32 u = x[k], um = x[k + VSIZE * m];
	x[k] = add_P2(u, um); x[k + VSIZE * m] = mul_P2(sub_P2(u, um), wi);
}

__kernel
void backward2_P3(const __global uint32 * restrict const wri, __global uint32 * restrict const x, const uint32 s, const uint32 m, const int lm)
{
	const size_t id = get_global_id(0);
	const size_t vid = id / VSIZE, l = id % VSIZE;
	const size_t j = vid >> lm, i = vid & (m - 1), mj = vid - i;
	const uint32 wi = wri[s + j];
	const size_t k = VSIZE * (2 * mj + i) + l;
	const uint32 u = x[k], um = x[k + VSIZE * m];
	x[k] = add_P3(u, um); x[k + VSIZE * m] = mul_P3(sub_P3(u, um), wi);
}

inline uint32 barrett(const uint64 a, const uint32 b, const uint32 b_inv, const int b_s, uint32 * a_p)
{
	// n = 31, alpha = 2^{n-2} = 2^29, s = r - 2, t = n + 1 = 32 => h = 1.
	// b < 2^31, alpha = 2^29 => a < 2^29 b
	// 2^{r-1} < b <= 2^r then a < 2^{r + 29} = 2^{s + 31} and (a >> s) < 2^31
	// b_inv = [2^{s + 32} / b]
	// b_inv < 2^{s + 32} / b < 2^{s + 32} / 2^{r-1} = 2^{s + 32} / 2^{s + 1} < 2^31
	// Let h be the number of iterations in Barrett's reduction, we have h = [a / b] - [[a / 2^s] b_inv / 2^32].
	// h = ([a/b] - a/b) + a/2^{s + 32} (2^{s + 32}/b - b_inv) + b_inv/2^32 (a/2^s - [a/2^s]) + ([a/2^s] b_inv / 2^32 - [[a/2^s] b_inv / 2^32])
	// Then -1 + 0 + 0 + 0 < h < 0 + 1/2 (2^{s + 32}/b - b_inv) + b_inv/2^32 + 1,
	// 0 <= h < 1 + 1/2 + 1/2 => h = 1.

	const uint32 d = mul_hi((uint32)(a >> b_s), b_inv), r = (uint32)a - d * b;
	const bool o = (r >= b);
	*a_p = o ? d + 1 : d;
	return o ? r - b : r;
}

inline int32 reduce64(int64 * f, const uint32 b, const uint32 b_inv, const int b_s)
{
	// 1- t < 2^63 => t_h < 2^34. We must have t_h < 2^29 b => b > 32
	// 2- t < 2^22 b^2 => t_h < b^2 / 2^7. If 2 <= b < 32 then t_h < 32^2 / 2^7 = 2^8 < 2^29 b
	const uint64 t = abs(*f);
	const uint64 t_h = t >> 29;
	const uint32 t_l = (uint32)t & (((uint32)1 << 29) - 1);

	uint32 d_h, r_h = barrett(t_h, b, b_inv, b_s, &d_h);
	uint32 d_l, r_l = barrett(((uint64)r_h << 29) | t_l, b, b_inv, b_s, &d_l);
	const uint64 d = ((uint64)d_h << 29) | d_l;

	const bool s = ((*f) < 0);
	*f = s ? -(int64)d : (int64)d;
	const int32 r = s ? -(int32)r_l : (int32)r_l;
	return r;
}

__kernel
void normalize2a(const __global uint32_2 * restrict const bb_inv, __global int64 * restrict const f,
				 __global uint32 * restrict const z1, __global uint32 * restrict const z2, const int b_s)
{
	const size_t id = get_global_id(0);
	const size_t i = id % VSIZE, j = id / VSIZE, k0 = j * (VSIZE * CSIZE) + i;
	const uint32 b = bb_inv[i].s0, b_inv = bb_inv[i].s1;
	int64 a = 0;
	for (size_t c = 0; c < CSIZE; ++c)
	{
		const size_t k = k0 + c * VSIZE;
		a += garner2(mul_P1(z1[k], norm1), mul_P2(z2[k], norm2));
		const int32 r = reduce64(&a, b, b_inv, b_s);
		z1[k] = seti_P1(r); z2[k] = seti_P2(r);
	}
	f[id] = a;
}

__kernel
void normalize2b(const __global uint32_2 * restrict const bb_inv, const __global int64 * restrict const f,
				 __global uint32 * restrict const z1, __global uint32 * restrict const z2, const int b_s)
{
	const size_t id = get_global_id(0);
	const size_t i = id % VSIZE, j = id / VSIZE;
	const uint32 b = bb_inv[i].s0, b_inv = bb_inv[i].s1;
	const bool rot = (j == get_global_size(0) / VSIZE - 1);
	int64 a = rot ? -f[id] : f[id];	// a_0 = -a_n
	const size_t k0 = rot ? i : (j + 1) * (VSIZE * CSIZE) + i;	// TODO get_global_size(0) / VSIZE is a power of to => use & (N-1)
	size_t c;
	for (c = 0; c < CSIZE - 1; ++c)
	{
		const size_t k = k0 + c * VSIZE;
		a += geti_P1(z1[k]);
		const int32 r = reduce64(&a, b, b_inv, b_s);
		z1[k] = seti_P1(r); z2[k] = seti_P2(r);
		if (a == 0) break;
	}
	if (c == CSIZE - 1)
	{
		const size_t k = k0 + c * VSIZE;
		z1[k] = add_P1(z1[k], seti_P1((int32)a)); z2[k] = add_P2(z2[k], seti_P2((int32)a));
		// if (abs(a) > 1) throw;
	}
}
