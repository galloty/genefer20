/*
Copyright 2020, Yves Gallot

genefer20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

static const char * const src_ocl_kernel = \
"/*\n" \
"Copyright 2020, Yves Gallot\n" \
"\n" \
"genefer20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.\n" \
"Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.\n" \
"*/\n" \
"\n" \
"typedef uint	uint32;\n" \
"typedef int 	int32;\n" \
"typedef ulong	uint64;\n" \
"typedef long 	int64;\n" \
"typedef uint2	uint32_2;\n" \
"typedef ulong16	uint64_16;\n" \
"\n" \
"typedef struct { uint64 s0; uint32 s1; } uint96;\n" \
"typedef struct { uint64 s0; int32  s1; } int96;\n" \
"\n" \
"inline int96 int96_set_si(const int64 n) { int96 r; r.s0 = (uint64)n; r.s1 = (n < 0) ? -1 : 0; return r; }\n" \
"\n" \
"inline uint96 uint96_set(const uint64 s0, const uint32 s1) { uint96 r; r.s0 = s0; r.s1 = s1; return r; }\n" \
"\n" \
"inline int96 uint96_i(const uint96 x) { int96 r; r.s0 = x.s0; r.s1 = (int32)x.s1; return r; }\n" \
"inline uint96 int96_u(const int96 x) { uint96 r; r.s0 = x.s0; r.s1 = (uint32)x.s1; return r; }\n" \
"\n" \
"inline bool int96_is_neg(const int96 x) { return (x.s1 < 0); }\n" \
"\n" \
"inline bool uint96_is_greater(const uint96 x, const uint96 y) { return (x.s1 > y.s1) || ((x.s1 == y.s1) && (x.s0 > y.s0)); }\n" \
"\n" \
"inline int96 int96_neg(const int96 x)\n" \
"{\n" \
"	const int32 c = (x.s0 != 0) ? 1 : 0;\n" \
"	int96 r; r.s0 = -x.s0; r.s1 = -x.s1 - c;\n" \
"	return r;\n" \
"}\n" \
"\n" \
"inline int96 int96_add(const int96 x, const int96 y)\n" \
"{\n" \
"	const uint64 s0 = x.s0 + y.s0;\n" \
"	const int32 c = (s0 < y.s0) ? 1 : 0;\n" \
"	int96 r; r.s0 = s0; r.s1 = x.s1 + y.s1 + c;\n" \
"	return r;\n" \
"}\n" \
"\n" \
"inline uint96 uint96_add_64(const uint96 x, const uint64 y)\n" \
"{\n" \
"	const uint64 s0 = x.s0 + y;\n" \
"	const uint32 c = (s0 < y) ? 1 : 0;\n" \
"	uint96 r; r.s0 = s0; r.s1 = x.s1 + c;\n" \
"	return r;\n" \
"}\n" \
"\n" \
"inline int96 uint96_subi(const uint96 x, const uint96 y)\n" \
"{\n" \
"	const uint32 c = (x.s0 < y.s0) ? 1 : 0;\n" \
"	int96 r; r.s0 = x.s0 - y.s0; r.s1 = (int32)(x.s1 - y.s1 - c);\n" \
"	return r;\n" \
"}\n" \
"\n" \
"inline uint96 uint96_mul_64_32(const uint64 x, const uint32 y)\n" \
"{\n" \
"	const uint64 l = (uint32)x * (uint64)y, h = (x >> 32) * y + (l >> 32);\n" \
"	uint96 r; r.s0 = (h << 32) | (uint32)l; r.s1 = (uint32)(h >> 32);\n" \
"	return r;\n" \
"}\n" \
"\n" \
"inline uint96 int96_abs(const int96 x)\n" \
"{\n" \
"	const int96 t = (int96_is_neg(x)) ? int96_neg(x) : x;\n" \
"	return int96_u(t);\n" \
"}\n" \
"\n" \
"#define P1			4253024257u		// 507 * 2^23 + 1\n" \
"#define P2			4194304001u		// 125 * 2^25 + 1\n" \
"#define P3			4076863489u		// 243 * 2^24 + 1\n" \
"#define P1_INV		42356678u		// (2^64 - 1) / P1 - 2^32\n" \
"#define P2_INV		103079214u		// (2^64 - 1) / P2 - 2^32\n" \
"#define P3_INV		229771911u		// (2^64 - 1) / P3 - 2^32\n" \
"#define InvP2_P1	1822724754u		// 1 / P2 mod P1\n" \
"#define InvP3_P1	607574918u		// 1 / P3 mod P1\n" \
"#define InvP3_P2	2995931465u		// 1 / P3 mod P2\n" \
"#define P1P2		(P1 * (uint64)P2)\n" \
"#define P2P3		(P2 * (uint64)P3)\n" \
"#define P1P2P3l		15383592652180029441ul\n" \
"#define P1P2P3h		3942432002u\n" \
"#define P1P2P3_2l	7691796326090014720ul\n" \
"#define P1P2P3_2h	1971216001u\n" \
"\n" \
"inline uint32 _addMod(const uint32 lhs, const uint32 rhs, const uint32 p)\n" \
"{\n" \
"	const uint32 c = (lhs >= p - rhs) ? p : 0;\n" \
"	return lhs + rhs - c;\n" \
"}\n" \
"\n" \
"inline uint32 _subMod(const uint32 lhs, const uint32 rhs, const uint32 p)\n" \
"{\n" \
"	const uint32 c = (lhs < rhs) ? p : 0;\n" \
"	return lhs - rhs + c;\n" \
"}\n" \
"\n" \
"inline uint32 _mulMod(const uint32 lhs, const uint32 rhs, const uint32 p, const uint32 p_inv)\n" \
"{\n" \
"	// Improved division by invariant integers, Niels Moller and Torbjorn Granlund, Algorithm 4.\n" \
"	const uint64 m = lhs * (uint64)(rhs), q = (m >> 32) * p_inv + m;\n" \
"	uint32 r = (uint32)m - (1 + (uint32)(q >> 32)) * p;\n" \
"	if (r > (uint32)q) r += p;\n" \
"	return (r >= p) ? r - p : r;\n" \
"}\n" \
"\n" \
"inline uint32 seti_P1(const int32 i) { return (i < 0) ? (uint32)(i + P1) : (uint32)i; }\n" \
"inline uint32 seti_P2(const int32 i) { return (i < 0) ? (uint32)(i + P2) : (uint32)i; }\n" \
"inline uint32 seti_P3(const int32 i) { return (i < 0) ? (uint32)(i + P3) : (uint32)i; }\n" \
"\n" \
"inline int32 geti_P1(const uint32 n) { return (n > P1 / 2) ? (int32)(n - P1) : (int32)n; }\n" \
"\n" \
"inline uint32 add_P1(const uint32 lhs, const uint32 rhs) { return _addMod(lhs, rhs, P1); }\n" \
"inline uint32 add_P2(const uint32 lhs, const uint32 rhs) { return _addMod(lhs, rhs, P2); }\n" \
"inline uint32 add_P3(const uint32 lhs, const uint32 rhs) { return _addMod(lhs, rhs, P3); }\n" \
"\n" \
"inline uint32 sub_P1(const uint32 lhs, const uint32 rhs) { return _subMod(lhs, rhs, P1); }\n" \
"inline uint32 sub_P2(const uint32 lhs, const uint32 rhs) { return _subMod(lhs, rhs, P2); }\n" \
"inline uint32 sub_P3(const uint32 lhs, const uint32 rhs) { return _subMod(lhs, rhs, P3); }\n" \
"\n" \
"inline uint32 mul_P1(const uint32 lhs, const uint32 rhs) { return _mulMod(lhs, rhs, P1, P1_INV); }\n" \
"inline uint32 mul_P2(const uint32 lhs, const uint32 rhs) { return _mulMod(lhs, rhs, P2, P2_INV); }\n" \
"inline uint32 mul_P3(const uint32 lhs, const uint32 rhs) { return _mulMod(lhs, rhs, P3, P3_INV); }\n" \
"\n" \
"inline uint32_2 add_P12(const uint32_2 lhs, const uint32_2 rhs) { return (uint32_2)(add_P1(lhs.s0, rhs.s0), add_P2(lhs.s1, rhs.s1)); }\n" \
"inline uint32_2 sub_P12(const uint32_2 lhs, const uint32_2 rhs) { return (uint32_2)(sub_P1(lhs.s0, rhs.s0), sub_P2(lhs.s1, rhs.s1)); }\n" \
"inline uint32_2 mul_P12(const uint32_2 lhs, const uint32_2 rhs) { return (uint32_2)(mul_P1(lhs.s0, rhs.s0), mul_P2(lhs.s1, rhs.s1)); }\n" \
"\n" \
"inline int64 garner2(const uint32 r1, const uint32 r2)\n" \
"{\n" \
"	const uint32 u12 = mul_P1(sub_P1(r1, r2), InvP2_P1);\n" \
"	const uint64 n = r2 + u12 * (uint64)P2;\n" \
"	return (n > P1P2 / 2) ? (int64)(n - P1P2) : (int64)n;\n" \
"}\n" \
"\n" \
"inline static int96 garner3(const uint32 r1, const uint32 r2, const uint32 r3)\n" \
"{\n" \
"	const uint32 u13 = mul_P1(sub_P1(r1, r3), InvP3_P1);\n" \
"	const uint32 u23 = mul_P2(sub_P2(r2, r3), InvP3_P2);\n" \
"	const uint32 u123 = mul_P1(sub_P1(u13, u23), InvP2_P1);\n" \
"	const uint96 n = uint96_add_64(uint96_mul_64_32(P2P3, u123), u23 * (uint64)P3 + r3);\n" \
"	const uint96 P1P2P3 = uint96_set(P1P2P3l, P1P2P3h), P1P2P3_2 = uint96_set(P1P2P3_2l, P1P2P3_2h);\n" \
"	const int96 r = uint96_is_greater(n, P1P2P3_2) ? uint96_subi(n, P1P2P3) : uint96_i(n);\n" \
"	return r;\n" \
"}\n" \
"\n" \
"__constant uint64 cMask[64] = {\n" \
"	0x0000000000000001ul, 0x0000000000000002ul, 0x0000000000000004ul, 0x0000000000000008ul, 0x0000000000000010ul, 0x0000000000000020ul, 0x0000000000000040ul, 0x0000000000000080ul,\n" \
"	0x0000000000000100ul, 0x0000000000000200ul, 0x0000000000000400ul, 0x0000000000000800ul, 0x0000000000001000ul, 0x0000000000002000ul, 0x0000000000004000ul, 0x0000000000008000ul,\n" \
"	0x0000000000010000ul, 0x0000000000020000ul, 0x0000000000040000ul, 0x0000000000080000ul, 0x0000000000100000ul, 0x0000000000200000ul, 0x0000000000400000ul, 0x0000000000800000ul,\n" \
"	0x0000000001000000ul, 0x0000000002000000ul, 0x0000000004000000ul, 0x0000000008000000ul, 0x0000000010000000ul, 0x0000000020000000ul, 0x0000000040000000ul, 0x0000000080000000ul,\n" \
"	0x0000000100000000ul, 0x0000000200000000ul, 0x0000000400000000ul, 0x0000000800000000ul, 0x0000001000000000ul, 0x0000002000000000ul, 0x0000004000000000ul, 0x0000008000000000ul,\n" \
"	0x0000010000000000ul, 0x0000020000000000ul, 0x0000040000000000ul, 0x0000080000000000ul, 0x0000100000000000ul, 0x0000200000000000ul, 0x0000400000000000ul, 0x0000800000000000ul,\n" \
"	0x0001000000000000ul, 0x0002000000000000ul, 0x0004000000000000ul, 0x0008000000000000ul, 0x0010000000000000ul, 0x0020000000000000ul, 0x0040000000000000ul, 0x0080000000000000ul,\n" \
"	0x0100000000000000ul, 0x0200000000000000ul, 0x0400000000000000ul, 0x0800000000000000ul, 0x1000000000000000ul, 0x2000000000000000ul, 0x4000000000000000ul, 0x8000000000000000ul\n" \
"	};\n" \
"\n" \
"inline uint64 getcval(const uint64_16 c, const size_t l_64)\n" \
"{\n" \
"	const uint64 c0_mask = (l_64 == 0x00) ? 0xfffffffffffffffful : 0, c1_mask = (l_64 == 0x01) ? 0xfffffffffffffffful : 0;\n" \
"	const uint64 c2_mask = (l_64 == 0x02) ? 0xfffffffffffffffful : 0, c3_mask = (l_64 == 0x03) ? 0xfffffffffffffffful : 0;\n" \
"	const uint64 c4_mask = (l_64 == 0x04) ? 0xfffffffffffffffful : 0, c5_mask = (l_64 == 0x05) ? 0xfffffffffffffffful : 0;\n" \
"	const uint64 c6_mask = (l_64 == 0x06) ? 0xfffffffffffffffful : 0, c7_mask = (l_64 == 0x07) ? 0xfffffffffffffffful : 0;\n" \
"	const uint64 c8_mask = (l_64 == 0x08) ? 0xfffffffffffffffful : 0, c9_mask = (l_64 == 0x09) ? 0xfffffffffffffffful : 0;\n" \
"	const uint64 ca_mask = (l_64 == 0x0a) ? 0xfffffffffffffffful : 0, cb_mask = (l_64 == 0x0b) ? 0xfffffffffffffffful : 0;\n" \
"	const uint64 cc_mask = (l_64 == 0x0c) ? 0xfffffffffffffffful : 0, cd_mask = (l_64 == 0x0d) ? 0xfffffffffffffffful : 0;\n" \
"	const uint64 ce_mask = (l_64 == 0x0e) ? 0xfffffffffffffffful : 0, cf_mask = (l_64 == 0x0f) ? 0xfffffffffffffffful : 0;\n" \
"\n" \
"	return (c.s0 & c0_mask) | (c.s1 & c1_mask) | (c.s2 & c2_mask) | (c.s3 & c3_mask)\n" \
"		 | (c.s4 & c4_mask) | (c.s5 & c5_mask) | (c.s6 & c6_mask) | (c.s7 & c7_mask)\n" \
"		 | (c.s8 & c8_mask) | (c.s9 & c9_mask) | (c.sa & ca_mask) | (c.sb & cb_mask)\n" \
"		 | (c.sc & cc_mask) | (c.sd & cd_mask) | (c.se & ce_mask) | (c.sf & cf_mask);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void set_P1(const __global uint32_2 * restrict const x12, __global uint32 * restrict const y1)\n" \
"{\n" \
"	const size_t k = get_global_id(0);\n" \
"	y1[k] = x12[k].s0;\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void set_P12(const __global uint32_2 * restrict const x12, __global uint32_2 * restrict const y12)\n" \
"{\n" \
"	const size_t k = get_global_id(0);\n" \
"	y12[k] = x12[k];\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void set_P123(const __global uint32_2 * restrict const x12, const __global uint32 * restrict const x3, __global uint32_2 * restrict const y12, __global uint32 * restrict const y3)\n" \
"{\n" \
"	const size_t k = get_global_id(0);\n" \
"	const uint32_2 x12k = x12[k]; const uint32 x3k = x3[k];\n" \
"	y12[k] = x12k; y3[k] = x3k;\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void swap_P12(__global uint32_2 * restrict const x12, __global uint32_2 * restrict const y12)\n" \
"{\n" \
"	const size_t k = get_global_id(0);\n" \
"	const uint32_2 x12k = x12[k], y12k = y12[k];\n" \
"	x12[k] = y12k; y12[k] = x12k;\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void swap_P123(__global uint32_2 * restrict const x12, __global uint32 * restrict const x3, __global uint32_2 * restrict const y12, __global uint32 * restrict const y3)\n" \
"{\n" \
"	const size_t k = get_global_id(0);\n" \
"	const uint32_2 x12k = x12[k], y12k = y12[k];\n" \
"	x12[k] = y12k; y12[k] = x12k;\n" \
"	const uint32 x3k = x3[k], y3k = y3[k];\n" \
"	x3[k] = y3k; y3[k] = x3k;\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void reset_P12(__global uint32_2 * restrict const x12, const uint32 a)\n" \
"{\n" \
"	const size_t k = get_global_id(0);\n" \
"	const uint32 v = (k < VSIZE) ? a : 0;\n" \
"	x12[k] = (uint32_2)(v, v);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void reset_P123(__global uint32_2 * restrict const x12, __global uint32 * restrict const x3, const uint32 a)\n" \
"{\n" \
"	const size_t k = get_global_id(0);\n" \
"	const uint32 v = (k < VSIZE) ? a : 0;\n" \
"	x12[k] = (uint32_2)(v, v);\n" \
"	x3[k] = v;\n" \
"}\n" \
"\n" \
"inline void frwd2_P12(uint32_2 * const u_P12, const uint32_2 w12)\n" \
"{\n" \
"	const uint32_2 u1w_P12 = mul_P12(u_P12[1], w12);\n" \
"	u_P12[1] = sub_P12(u_P12[0], u1w_P12); u_P12[0] = add_P12(u_P12[0], u1w_P12);\n" \
"}\n" \
"inline void frwd2_P3(uint32 * const u_P3, const uint32 w3)\n" \
"{\n" \
"	const uint32 u1w_P3 = mul_P3(u_P3[1], w3);\n" \
"	u_P3[1] = sub_P3(u_P3[0], u1w_P3); u_P3[0] = add_P3(u_P3[0], u1w_P3);\n" \
"}\n" \
"\n" \
"inline void bkwd2_P12(uint32_2 * const u_P12, const uint32_2 wi12)\n" \
"{\n" \
"	const uint32_2 v1_P12 = sub_P12(u_P12[0], u_P12[1]);\n" \
"	u_P12[0] = add_P12(u_P12[0], u_P12[1]); u_P12[1] = mul_P12(v1_P12, wi12);\n" \
"}\n" \
"inline void bkwd2_P3(uint32 * const u_P3, const uint32 wi3)\n" \
"{\n" \
"	const uint32 v1_P3 = sub_P3(u_P3[0], u_P3[1]);\n" \
"	u_P3[0] = add_P3(u_P3[0], u_P3[1]); u_P3[1] = mul_P3(v1_P3, wi3);\n" \
"}\n" \
"\n" \
"inline void sqr2_P12(uint32_2 * const u_P12, const uint32_2 w12)\n" \
"{\n" \
"	const uint32_2 u1w_P12 = mul_P12(u_P12[1], w12);\n" \
"	const uint32_2 v0_P12 = add_P12(mul_P12(u_P12[0], u_P12[0]), mul_P12(u1w_P12, u1w_P12));\n" \
"	const uint32_2 v1_P12 = mul_P12(add_P12(u_P12[0], u_P12[0]), u_P12[1]);\n" \
"	u_P12[0] = add_P12(v0_P12, v0_P12); u_P12[1] = add_P12(v1_P12, v1_P12);\n" \
"}\n" \
"inline void sqr2_P3(uint32 * const u_P3, const uint32 w3)\n" \
"{\n" \
"	const uint32 u1w_P3 = mul_P3(u_P3[1], w3);\n" \
"	const uint32 v0_P3 = add_P3(mul_P3(u_P3[0], u_P3[0]), mul_P3(u1w_P3, u1w_P3));\n" \
"	const uint32 v1_P3 = mul_P3(add_P3(u_P3[0], u_P3[0]), u_P3[1]);\n" \
"	u_P3[0] = add_P3(v0_P3, v0_P3); u_P3[1] = add_P3(v1_P3, v1_P3);\n" \
"}\n" \
"\n" \
"inline void read2_P12(uint32_2 * const u_P12, const __global uint32_2 * const x12, const size_t k, const uint32 m)\n" \
"{\n" \
"	for (size_t h = 0; h < 2; ++h) u_P12[h] = x12[k + h * VSIZE * m];\n" \
"}\n" \
"inline void read2_P3(uint32 * const u_P3, const __global uint32 * const x3, const size_t k, const uint32 m)\n" \
"{\n" \
"	for (size_t h = 0; h < 2; ++h) u_P3[h] = x3[k + h * VSIZE * m];\n" \
"}\n" \
"\n" \
"inline void write2_P12(__global uint32_2 * const x12, const uint32_2 * const u_P12, const size_t k, const uint32 m)\n" \
"{\n" \
"	for (size_t h = 0; h < 2; ++h) x12[k + h * VSIZE * m] = u_P12[h];\n" \
"}\n" \
"inline void write2_P3(__global uint32 * const x3, const uint32 * const u_P3, const size_t k, const uint32 m)\n" \
"{\n" \
"	for (size_t h = 0; h < 2; ++h) x3[k + h * VSIZE * m] = u_P3[h];\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void forward2_P12(const __global uint32_2 * restrict const wr12, __global uint32_2 * restrict const x12, const uint32 s, const uint32 m, const int lm)\n" \
"{\n" \
"	const size_t id = get_global_id(0), vid = id / VSIZE, l = id % VSIZE;\n" \
"\n" \
"	const size_t sj = s + (vid >> lm), i = vid & (m - 1), mj = vid - i, k = VSIZE * (2 * mj + i) + l;\n" \
"\n" \
"	uint32_2 u_P12[2]; read2_P12(u_P12, x12, k, m);\n" \
"\n" \
"	frwd2_P12(u_P12, wr12[sj]);\n" \
"\n" \
"	write2_P12(x12, u_P12, k, m);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void forward2_P123(const __global uint32_2 * restrict const wr12, const __global uint32 * restrict const wr3,\n" \
"	__global uint32_2 * restrict const x12, __global uint32 * restrict const x3, const uint32 s, const uint32 m, const int lm)\n" \
"{\n" \
"	const size_t id = get_global_id(0), vid = id / VSIZE, l = id % VSIZE;\n" \
"\n" \
"	const size_t sj = s + (vid >> lm), i = vid & (m - 1), mj = vid - i, k = VSIZE * (2 * mj + i) + l;\n" \
"\n" \
"	uint32_2 u_P12[2]; read2_P12(u_P12, x12, k, m); uint32 u_P3[2]; read2_P3(u_P3, x3, k, m);\n" \
"\n" \
"	frwd2_P12(u_P12, wr12[sj]); frwd2_P3(u_P3, wr3[sj]);\n" \
"\n" \
"	write2_P12(x12, u_P12, k, m); write2_P3(x3, u_P3, k, m);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void backward2_P12(const __global uint32_2 * restrict const wri12, __global uint32_2 * restrict const x12, const uint32 s, const uint32 m, const int lm)\n" \
"{\n" \
"	const size_t id = get_global_id(0), vid = id / VSIZE, l = id % VSIZE;\n" \
"\n" \
"	const size_t sj = s + (vid >> lm), i = vid & (m - 1), mj = vid - i, k = VSIZE * (2 * mj + i) + l;\n" \
"\n" \
"	uint32_2 u_P12[2]; read2_P12(u_P12, x12, k, m);\n" \
"\n" \
"	bkwd2_P12(u_P12, wri12[sj]);\n" \
"\n" \
"	write2_P12(x12, u_P12, k, m);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void backward2_P123(const __global uint32_2 * restrict const wri12, const __global uint32 * restrict const wri3,\n" \
"	__global uint32_2 * restrict const x12, __global uint32 * restrict const x3, const uint32 s, const uint32 m, const int lm)\n" \
"{\n" \
"	const size_t id = get_global_id(0), vid = id / VSIZE, l = id % VSIZE;\n" \
"\n" \
"	const size_t sj = s + (vid >> lm), i = vid & (m - 1), mj = vid - i, k = VSIZE * (2 * mj + i) + l;\n" \
"\n" \
"	uint32_2 u_P12[2]; read2_P12(u_P12, x12, k, m); uint32 u_P3[2]; read2_P3(u_P3, x3, k, m);\n" \
"\n" \
"	bkwd2_P12(u_P12, wri12[sj]); bkwd2_P3(u_P3, wri3[sj]);\n" \
"\n" \
"	write2_P12(x12, u_P12, k, m); write2_P3(x3, u_P3, k, m);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void square2_P12(const __global uint32_2 * restrict const wr12, const __global uint32_2 * restrict const wri12,\n" \
"	__global uint32_2 * restrict const x12)\n" \
"{\n" \
"	const size_t id = get_global_id(0), vid = id / VSIZE, l = id % VSIZE;\n" \
"\n" \
"	const size_t sj = get_global_size(0) / VSIZE + vid, k = VSIZE * 2 * vid + l;\n" \
"\n" \
"	uint32_2 u_P12[2]; read2_P12(u_P12, x12, k, 1);\n" \
"\n" \
"	sqr2_P12(u_P12, wr12[sj]);\n" \
"\n" \
"	write2_P12(x12, u_P12, k, 1);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void square2_P123(const __global uint32_2 * restrict const wr12, const __global uint32 * restrict const wr3,\n" \
"	const __global uint32_2 * restrict const wri12, const __global uint32 * restrict const wri3,\n" \
"	__global uint32_2 * restrict const x12, __global uint32 * restrict const x3)\n" \
"{\n" \
"	const size_t id = get_global_id(0), vid = id / VSIZE, l = id % VSIZE;\n" \
"\n" \
"	const size_t sj = get_global_size(0) / VSIZE + vid, k = VSIZE * 2 * vid + l;\n" \
"\n" \
"	uint32_2 u_P12[2]; read2_P12(u_P12, x12, k, 1); uint32 u_P3[2]; read2_P3(u_P3, x3, k, 1);\n" \
"\n" \
"	sqr2_P12(u_P12, wr12[sj]); sqr2_P3(u_P3, wr3[sj]);\n" \
"\n" \
"	write2_P12(x12, u_P12, k, 1); write2_P3(x3, u_P3, k, 1);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mul2cond64_P12(const __global uint32_2 * restrict const wr12, const __global uint32_2 * restrict const wri12,\n" \
"	const __global uint32_2 * restrict const y12, __global uint32_2 * restrict const x12, const uint64 c)\n" \
"{\n" \
"	const size_t id = get_global_id(0), vid = id / VSIZE, l = id % VSIZE;\n" \
"\n" \
"	const size_t sj = get_global_size(0) / VSIZE + vid, k = VSIZE * 2 * vid + l;\n" \
"\n" \
"	uint32_2 u_P12[2]; read2_P12(u_P12, x12, k, 1);\n" \
"\n" \
"	frwd2_P12(u_P12, wr12[sj]);\n" \
"\n" \
"	if ((c & cMask[l]) != 0)\n" \
"	{\n" \
"		for (size_t h = 0; h < 2; ++h) u_P12[h] = mul_P12(u_P12[h], y12[k + h * VSIZE]);\n" \
"	}\n" \
"\n" \
"	bkwd2_P12(u_P12, wri12[sj]);\n" \
"\n" \
"	write2_P12(x12, u_P12, k, 1);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mul2cond64_P123(const __global uint32_2 * restrict const wr12, const __global uint32 * restrict const wr3,\n" \
"	const __global uint32_2 * restrict const wri12, const __global uint32 * restrict const wri3,\n" \
"	const __global uint32_2 * restrict const y12, __global uint32 * restrict const y3,\n" \
"	__global uint32_2 * restrict const x12,  __global uint32 * restrict const x3, const uint64 c)\n" \
"{\n" \
"	const size_t id = get_global_id(0), vid = id / VSIZE, l = id % VSIZE;\n" \
"\n" \
"	const size_t sj = get_global_size(0) / VSIZE + vid, k = VSIZE * 2 * vid + l;\n" \
"\n" \
"	uint32_2 u_P12[2]; read2_P12(u_P12, x12, k, 1); uint32 u_P3[2]; read2_P3(u_P3, x3, k, 1);\n" \
"\n" \
"	frwd2_P12(u_P12, wr12[sj]); frwd2_P3(u_P3, wr3[sj]);\n" \
"\n" \
"	if ((c & cMask[l]) != 0)\n" \
"	{\n" \
"		for (size_t h = 0; h < 2; ++h) u_P12[h] = mul_P12(u_P12[h], y12[k + h * VSIZE]);\n" \
"		for (size_t h = 0; h < 2; ++h) u_P3[h] = mul_P3(u_P3[h], y3[k + h * VSIZE]);\n" \
"	}\n" \
"\n" \
"	bkwd2_P12(u_P12, wri12[sj]); bkwd2_P3(u_P3, wri3[sj]);\n" \
"\n" \
"	write2_P12(x12, u_P12, k, 1); write2_P3(x3, u_P3, k, 1);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mul2cond1024_P12(const __global uint32_2 * restrict const wr12, const __global uint32_2 * restrict const wri12,\n" \
"	const __global uint32_2 * restrict const y12, __global uint32_2 * restrict const x12, const uint64_16 c)\n" \
"{\n" \
"	const size_t id = get_global_id(0), vid = id / VSIZE, l = id % VSIZE;\n" \
"\n" \
"	const size_t sj = get_global_size(0) / VSIZE + vid, k = VSIZE * 2 * vid + l;\n" \
"\n" \
"	uint32_2 u_P12[2]; read2_P12(u_P12, x12, k, 1);\n" \
"\n" \
"	frwd2_P12(u_P12, wr12[sj]);\n" \
"\n" \
"	if ((getcval(c, l / 64) & cMask[l % 64]) != 0)\n" \
"	{\n" \
"		for (size_t h = 0; h < 2; ++h) u_P12[h] = mul_P12(u_P12[h], y12[k + h * VSIZE]);\n" \
"	}\n" \
"\n" \
"	bkwd2_P12(u_P12, wri12[sj]);\n" \
"\n" \
"	write2_P12(x12, u_P12, k, 1);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mul2cond1024_P123(const __global uint32_2 * restrict const wr12, const __global uint32 * restrict const wr3,\n" \
"	const __global uint32_2 * restrict const wri12, const __global uint32 * restrict const wri3,\n" \
"	const __global uint32_2 * restrict const y12, __global uint32 * restrict const y3,\n" \
"	__global uint32_2 * restrict const x12,  __global uint32 * restrict const x3, const uint64_16 c)\n" \
"{\n" \
"	const size_t id = get_global_id(0), vid = id / VSIZE, l = id % VSIZE;\n" \
"\n" \
"	const size_t sj = get_global_size(0) / VSIZE + vid, k = VSIZE * 2 * vid + l;\n" \
"\n" \
"	uint32_2 u_P12[2]; read2_P12(u_P12, x12, k, 1); uint32 u_P3[2]; read2_P3(u_P3, x3, k, 1);\n" \
"\n" \
"	frwd2_P12(u_P12, wr12[sj]); frwd2_P3(u_P3, wr3[sj]);\n" \
"\n" \
"	if ((getcval(c, l / 64) & cMask[l % 64]) != 0)\n" \
"	{\n" \
"		for (size_t h = 0; h < 2; ++h) u_P12[h] = mul_P12(u_P12[h], y12[k + h * VSIZE]);\n" \
"		for (size_t h = 0; h < 2; ++h) u_P3[h] = mul_P3(u_P3[h], y3[k + h * VSIZE]);\n" \
"	}\n" \
"\n" \
"	bkwd2_P12(u_P12, wri12[sj]); bkwd2_P3(u_P3, wri3[sj]);\n" \
"\n" \
"	write2_P12(x12, u_P12, k, 1); write2_P3(x3, u_P3, k, 1);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mul2_P12(const __global uint32_2 * restrict const wr12, const __global uint32_2 * restrict const wri12,\n" \
"	const __global uint32_2 * restrict const y12, __global uint32_2 * restrict const x12)\n" \
"{\n" \
"	const size_t id = get_global_id(0), vid = id / VSIZE, l = id % VSIZE;\n" \
"\n" \
"	const size_t sj = get_global_size(0) / VSIZE + vid, k = VSIZE * 2 * vid + l;\n" \
"\n" \
"	const uint32_2 wr12_1 = wr12[sj];\n" \
"\n" \
"	uint32_2 u_P12[2]; read2_P12(u_P12, x12, k, 1);\n" \
"\n" \
"	frwd2_P12(u_P12, wr12_1);\n" \
"\n" \
"	uint32_2 v_P12[2]; read2_P12(v_P12, y12, k, 1);\n" \
"\n" \
"	frwd2_P12(v_P12, wr12_1);\n" \
"\n" \
"	for (size_t h = 0; h < 2; ++h) u_P12[h] = mul_P12(u_P12[h], v_P12[h]);\n" \
"\n" \
"	bkwd2_P12(u_P12, wri12[sj]);\n" \
"\n" \
"	write2_P12(x12, u_P12, k, 1);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mul2_P123(const __global uint32_2 * restrict const wr12, const __global uint32 * restrict const wr3,\n" \
"	const __global uint32_2 * restrict const wri12, const __global uint32 * restrict const wri3,\n" \
"	const __global uint32_2 * restrict const y12, __global uint32 * restrict const y3,\n" \
"	__global uint32_2 * restrict const x12,  __global uint32 * restrict const x3)\n" \
"{\n" \
"	const size_t id = get_global_id(0), vid = id / VSIZE, l = id % VSIZE;\n" \
"\n" \
"	const size_t sj = get_global_size(0) / VSIZE + vid, k = VSIZE * 2 * vid + l;\n" \
"\n" \
"	const uint32_2 wr12_1 = wr12[sj];\n" \
"	const uint32 wr3_1 = wr3[sj];\n" \
"\n" \
"	uint32_2 u_P12[2]; read2_P12(u_P12, x12, k, 1); uint32 u_P3[2]; read2_P3(u_P3, x3, k, 1);\n" \
"\n" \
"	frwd2_P12(u_P12, wr12_1); frwd2_P3(u_P3, wr3_1);\n" \
"\n" \
"	uint32_2 v_P12[2]; read2_P12(v_P12, y12, k, 1); uint32 v_P3[2]; read2_P3(v_P3, y3, k, 1);\n" \
"\n" \
"	frwd2_P12(v_P12, wr12_1); frwd2_P3(v_P3, wr3_1);\n" \
"\n" \
"	for (size_t h = 0; h < 2; ++h) u_P12[h] = mul_P12(u_P12[h], v_P12[h]);\n" \
"	for (size_t h = 0; h < 2; ++h) u_P3[h] = mul_P3(u_P3[h], v_P3[h]);\n" \
"\n" \
"	bkwd2_P12(u_P12, wri12[sj]); bkwd2_P3(u_P3, wri3[sj]);\n" \
"\n" \
"	write2_P12(x12, u_P12, k, 1); write2_P3(x3, u_P3, k, 1);\n" \
"}\n" \
"\n" \
"inline void frwd41_P12(uint32_2 * const u_P12, const uint32_2 w12_1)\n" \
"{\n" \
"	const uint32_2 u2w1_P12 = mul_P12(u_P12[2], w12_1), u3w1_P12 = mul_P12(u_P12[3], w12_1);\n" \
"	u_P12[2] = sub_P12(u_P12[0], u2w1_P12); u_P12[0] = add_P12(u_P12[0], u2w1_P12);\n" \
"	u_P12[3] = sub_P12(u_P12[1], u3w1_P12); u_P12[1] = add_P12(u_P12[1], u3w1_P12);\n" \
"}\n" \
"inline void frwd41_P3(uint32 * const u_P3, const uint32 w3_1)\n" \
"{\n" \
"	const uint32 u2w1_P3 = mul_P3(u_P3[2], w3_1), u3w1_P3 = mul_P3(u_P3[3], w3_1);\n" \
"	u_P3[2] = sub_P3(u_P3[0], u2w1_P3); u_P3[0] = add_P3(u_P3[0], u2w1_P3);\n" \
"	u_P3[3] = sub_P3(u_P3[1], u3w1_P3); u_P3[1] = add_P3(u_P3[1], u3w1_P3);\n" \
"}\n" \
"\n" \
"inline void frwd42_P12(uint32_2 * const u_P12, const uint32_2 w12_2, const uint32_2 w12_3)\n" \
"{\n" \
"	const uint32_2 u1w2_P12 = mul_P12(u_P12[1], w12_2), u3w3_P12 = mul_P12(u_P12[3], w12_3);\n" \
"	u_P12[1] = sub_P12(u_P12[0], u1w2_P12); u_P12[0] = add_P12(u_P12[0], u1w2_P12);\n" \
"	u_P12[3] = sub_P12(u_P12[2], u3w3_P12); u_P12[2] = add_P12(u_P12[2], u3w3_P12);\n" \
"}\n" \
"inline void frwd42_P3(uint32 * const u_P3, const uint32 w3_2, const uint32 w3_3)\n" \
"{\n" \
"	const uint32 u1w2_P3 = mul_P3(u_P3[1], w3_2), u3w3_P3 = mul_P3(u_P3[3], w3_3);\n" \
"	u_P3[1] = sub_P3(u_P3[0], u1w2_P3); u_P3[0] = add_P3(u_P3[0], u1w2_P3);\n" \
"	u_P3[3] = sub_P3(u_P3[2], u3w3_P3); u_P3[2] = add_P3(u_P3[2], u3w3_P3);\n" \
"}\n" \
"\n" \
"inline void bkwd42_P12(uint32_2 * const u_P12, const uint32_2 wi12_2, const uint32_2 wi12_3)\n" \
"{\n" \
"	const uint32_2 v1_P12 = sub_P12(u_P12[0], u_P12[1]), v3_P12 = sub_P12(u_P12[2], u_P12[3]);\n" \
"	u_P12[0] = add_P12(u_P12[0], u_P12[1]); u_P12[1] = mul_P12(v1_P12, wi12_2);\n" \
"	u_P12[2] = add_P12(u_P12[2], u_P12[3]); u_P12[3] = mul_P12(v3_P12, wi12_3);\n" \
"}\n" \
"inline void bkwd42_P3(uint32 * const u_P3, const uint32 wi3_2, const uint32 wi3_3)\n" \
"{\n" \
"	const uint32 v1_P3 = sub_P3(u_P3[0], u_P3[1]), v3_P3 = sub_P3(u_P3[2], u_P3[3]);\n" \
"	u_P3[0] = add_P3(u_P3[0], u_P3[1]); u_P3[1] = mul_P3(v1_P3, wi3_2);\n" \
"	u_P3[2] = add_P3(u_P3[2], u_P3[3]); u_P3[3] = mul_P3(v3_P3, wi3_3);\n" \
"}\n" \
"\n" \
"inline void bkwd41_P12(uint32_2 * const u_P12, const uint32_2 wi12_1)\n" \
"{\n" \
"	const uint32_2 v2_P12 = sub_P12(u_P12[0], u_P12[2]), v3_P12 = sub_P12(u_P12[1], u_P12[3]);\n" \
"	u_P12[0] = add_P12(u_P12[0], u_P12[2]); u_P12[2] = mul_P12(v2_P12, wi12_1);\n" \
"	u_P12[1] = add_P12(u_P12[1], u_P12[3]); u_P12[3] = mul_P12(v3_P12, wi12_1);\n" \
"}\n" \
"inline void bkwd41_P3(uint32 * const u_P3, const uint32 wi3_1)\n" \
"{\n" \
"	const uint32 v2_P3 = sub_P3(u_P3[0], u_P3[2]), v3_P3 = sub_P3(u_P3[1], u_P3[3]);\n" \
"	u_P3[0] = add_P3(u_P3[0], u_P3[2]); u_P3[2] = mul_P3(v2_P3, wi3_1);\n" \
"	u_P3[1] = add_P3(u_P3[1], u_P3[3]); u_P3[3] = mul_P3(v3_P3, wi3_1);\n" \
"}\n" \
"\n" \
"inline void sqr42_P12(uint32_2 * const u_P12, const uint32_2 w12_2, const uint32_2 w12_3)\n" \
"{\n" \
"	const uint32_2 u1w2_P12 = mul_P12(u_P12[1], w12_2), u3w3_P12 = mul_P12(u_P12[3], w12_3);\n" \
"	const uint32_2 v0_P12 = add_P12(mul_P12(u_P12[0], u_P12[0]), mul_P12(u1w2_P12, u1w2_P12));\n" \
"	const uint32_2 v1_P12 = mul_P12(add_P12(u_P12[0], u_P12[0]), u_P12[1]);\n" \
"	const uint32_2 v2_P12 = add_P12(mul_P12(u_P12[2], u_P12[2]), mul_P12(u3w3_P12, u3w3_P12));\n" \
"	const uint32_2 v3_P12 = mul_P12(add_P12(u_P12[2], u_P12[2]), u_P12[3]);\n" \
"	u_P12[0] = add_P12(v0_P12, v0_P12); u_P12[1] = add_P12(v1_P12, v1_P12);\n" \
"	u_P12[2] = add_P12(v2_P12, v2_P12); u_P12[3] = add_P12(v3_P12, v3_P12);\n" \
"}\n" \
"inline void sqr42_P3(uint32 * const u_P3, const uint32 w3_2, const uint32 w3_3)\n" \
"{\n" \
"	const uint32 u1w2_P3 = mul_P3(u_P3[1], w3_2), u3w3_P3 = mul_P3(u_P3[3], w3_3);\n" \
"	const uint32 v0_P3 = add_P3(mul_P3(u_P3[0], u_P3[0]), mul_P3(u1w2_P3, u1w2_P3));\n" \
"	const uint32 v1_P3 = mul_P3(add_P3(u_P3[0], u_P3[0]), u_P3[1]);\n" \
"	const uint32 v2_P3 = add_P3(mul_P3(u_P3[2], u_P3[2]), mul_P3(u3w3_P3, u3w3_P3));\n" \
"	const uint32 v3_P3 = mul_P3(add_P3(u_P3[2], u_P3[2]), u_P3[3]);\n" \
"	u_P3[0] = add_P3(v0_P3, v0_P3); u_P3[1] = add_P3(v1_P3, v1_P3);\n" \
"	u_P3[2] = add_P3(v2_P3, v2_P3); u_P3[3] = add_P3(v3_P3, v3_P3);\n" \
"}\n" \
"\n" \
"inline void read4_P12(uint32_2 * const u_P12, const __global uint32_2 * const x12, const size_t k, const uint32 m)\n" \
"{\n" \
"	for (size_t h = 0; h < 4; ++h) u_P12[h] = x12[k + h * VSIZE * m];\n" \
"}\n" \
"inline void read4_P3(uint32 * const u_P3, const __global uint32 * const x3, const size_t k, const uint32 m)\n" \
"{\n" \
"	for (size_t h = 0; h < 4; ++h) u_P3[h] = x3[k + h * VSIZE * m];\n" \
"}\n" \
"\n" \
"inline void write4_P12(__global uint32_2 * const x12, const uint32_2 * const u_P12, const size_t k, const uint32 m)\n" \
"{\n" \
"	for (size_t h = 0; h < 4; ++h) x12[k + h * VSIZE * m] = u_P12[h];\n" \
"}\n" \
"inline void write4_P3(__global uint32 * const x3, const uint32 * const u_P3, const size_t k, const uint32 m)\n" \
"{\n" \
"	for (size_t h = 0; h < 4; ++h) x3[k + h * VSIZE * m] = u_P3[h];\n" \
"}\n" \
"\n" \
"inline void read22l_P12(uint32_2 * const u_P12, const __local uint32_2 * const X12, const size_t k, const uint32 m)\n" \
"{\n" \
"	for (size_t h = 0; h < 2; ++h) u_P12[h] = X12[k + h * VSIZE * m];\n" \
"	for (size_t h = 0; h < 2; ++h) u_P12[h + 2] = X12[k + 1 + h * VSIZE * m];\n" \
"}\n" \
"inline void read22l_P3(uint32 * const u_P3, const __local uint32 * const X3, const size_t k, const uint32 m)\n" \
"{\n" \
"	for (size_t h = 0; h < 2; ++h) u_P3[h] = X3[k + h * VSIZE * m];\n" \
"	for (size_t h = 0; h < 2; ++h) u_P3[h + 2] = X3[k + 1 + h * VSIZE * m];\n" \
"}\n" \
"\n" \
"inline void write22l_P12(__local uint32_2 * const X12, const uint32_2 * const u_P12, const size_t k, const uint32 m)\n" \
"{\n" \
"	for (size_t h = 0; h < 2; ++h) X12[k + h * VSIZE * m] = u_P12[h];\n" \
"	for (size_t h = 0; h < 2; ++h) X12[k + 1 + h * VSIZE * m] = u_P12[h + 2];\n" \
"}\n" \
"inline void write22l_P3(__local uint32 * const X3, const uint32 * const u_P3, const size_t k, const uint32 m)\n" \
"{\n" \
"	for (size_t h = 0; h < 2; ++h) X3[k + h * VSIZE * m] = u_P3[h];\n" \
"	for (size_t h = 0; h < 2; ++h) X3[k + 1 + h * VSIZE * m] = u_P3[h + 2];\n" \
"}\n" \
"\n" \
"inline void read4l_P12(uint32_2 * const u_P12, const __local uint32_2 * const X12, const size_t k, const uint32 m)\n" \
"{\n" \
"	for (size_t h = 0; h < 4; ++h) u_P12[h] = X12[k + h * VSIZE * m];\n" \
"}\n" \
"inline void read4l_P3(uint32 * const u_P3, const __local uint32 * const X3, const size_t k, const uint32 m)\n" \
"{\n" \
"	for (size_t h = 0; h < 4; ++h) u_P3[h] = X3[k + h * VSIZE * m];\n" \
"}\n" \
"\n" \
"inline void write4l_P12(__local uint32_2 * const X12, const uint32_2 * const u_P12, const size_t k, const uint32 m)\n" \
"{\n" \
"	for (size_t h = 0; h < 4; ++h) X12[k + h * VSIZE * m] = u_P12[h];\n" \
"}\n" \
"inline void write4l_P3(__local uint32 * const X3, const uint32 * const u_P3, const size_t k, const uint32 m)\n" \
"{\n" \
"	for (size_t h = 0; h < 4; ++h) X3[k + h * VSIZE * m] = u_P3[h];\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void forward4_P12(const __global uint32_2 * restrict const wr12, __global uint32_2 * restrict const x12, const uint32 s, const uint32 m, const int lm)\n" \
"{\n" \
"	const size_t id = get_global_id(0), vid = id / VSIZE, l = id % VSIZE;\n" \
"\n" \
"	const size_t sj = s + (vid >> lm), i = vid & (m - 1), mj = vid - i, k = VSIZE * (4 * mj + i) + l;\n" \
"\n" \
"	uint32_2 u_P12[4]; read4_P12(u_P12, x12, k, m);\n" \
"\n" \
"	frwd41_P12(u_P12, wr12[sj]);\n" \
"	frwd42_P12(u_P12, wr12[2 * sj], wr12[2 * sj + 1]);\n" \
"\n" \
"	write4_P12(x12, u_P12, k, m);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void forward4_P123(const __global uint32_2 * restrict const wr12, const __global uint32 * restrict const wr3,\n" \
"	__global uint32_2 * restrict const x12, __global uint32 * restrict const x3, const uint32 s, const uint32 m, const int lm)\n" \
"{\n" \
"	const size_t id = get_global_id(0), vid = id / VSIZE, l = id % VSIZE;\n" \
"\n" \
"	const size_t sj = s + (vid >> lm), i = vid & (m - 1), mj = vid - i, k = VSIZE * (4 * mj + i) + l;\n" \
"\n" \
"	uint32_2 u_P12[4]; read4_P12(u_P12, x12, k, m); uint32 u_P3[4]; read4_P3(u_P3, x3, k, m);\n" \
"\n" \
"	frwd41_P12(u_P12, wr12[sj]); frwd41_P3(u_P3, wr3[sj]);\n" \
"	frwd42_P12(u_P12, wr12[2 * sj], wr12[2 * sj + 1]); frwd42_P3(u_P3, wr3[2 * sj], wr3[2 * sj + 1]);\n" \
"\n" \
"	write4_P12(x12, u_P12, k, m); write4_P3(x3, u_P3, k, m);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((work_group_size_hint(16 / 4 * VSIZE, 1, 1)))\n" \
"void forward16_P12(const __global uint32_2 * restrict const wr12, __global uint32_2 * restrict const x12, const uint32 s, const uint32 m, const int lm)\n" \
"{\n" \
"	__local uint32_2 X12[16 * VSIZE];	// 32 KB => VSIZE = 256\n" \
"\n" \
"	const size_t gid = get_global_id(0), vid = gid / VSIZE, l = gid % VSIZE;\n" \
"	const size_t lid = get_local_id(0), i = lid / VSIZE, iv = lid & (size_t)~(VSIZE - 1);\n" \
"\n" \
"	const size_t vid_blk = (vid & (size_t)~(4 * m - 1)) * 4, idl = get_group_id(0) & (m - 1);\n" \
"	const size_t k0 = VSIZE * (vid_blk + idl) + l, miv = iv << lm;\n" \
"	const size_t sj4 = s * 4 + (vid_blk >> (lm + 2)) + i, sj = sj4 / 4;\n" \
"\n" \
"	uint32_2 u_P12[4]; read4_P12(u_P12, x12, k0 + miv, 4 * m);\n" \
"	frwd41_P12(u_P12, wr12[sj]);\n" \
"	frwd42_P12(u_P12, wr12[2 * sj], wr12[2 * sj + 1]);\n" \
"	write4l_P12(X12, u_P12, iv + l, 4);\n" \
"\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	uint32_2 v_P12[4]; read4l_P12(v_P12, X12, 4 * iv + l, 1);\n" \
"	frwd41_P12(v_P12, wr12[sj4]);\n" \
"	frwd42_P12(v_P12, wr12[2 * sj4], wr12[2 * sj4 + 1]);\n" \
"	write4_P12(x12, v_P12, k0 + 4 * miv, m);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((work_group_size_hint(16 / 4 * VSIZE, 1, 1)))\n" \
"void forward16_P3(const __global uint32 * restrict const wr3, __global uint32 * restrict const x3, const uint32 s, const uint32 m, const int lm)\n" \
"{\n" \
"	__local uint32 X3[16 * VSIZE];\n" \
"\n" \
"	const size_t gid = get_global_id(0), vid = gid / VSIZE, l = gid % VSIZE;\n" \
"	const size_t lid = get_local_id(0), i = lid / VSIZE, iv = lid & (size_t)~(VSIZE - 1);\n" \
"\n" \
"	const size_t vid_blk = (vid & (size_t)~(4 * m - 1)) * 4, idl = get_group_id(0) & (m - 1);\n" \
"	const size_t k0 = VSIZE * (vid_blk + idl) + l, miv = iv << lm;\n" \
"	const size_t sj4 = s * 4 + (vid_blk >> (lm + 2)) + i, sj = sj4 / 4;\n" \
"\n" \
"	uint32 u_P3[4]; read4_P3(u_P3, x3, k0 + miv, 4 * m);\n" \
"	frwd41_P3(u_P3, wr3[sj]);\n" \
"	frwd42_P3(u_P3, wr3[2 * sj], wr3[2 * sj + 1]);\n" \
"	write4l_P3(X3, u_P3, iv + l, 4);\n" \
"\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	uint32 v_P3[4]; read4l_P3(v_P3, X3, 4 * iv + l, 1);\n" \
"	frwd41_P3(v_P3, wr3[sj4]);\n" \
"	frwd42_P3(v_P3, wr3[2 * sj4], wr3[2 * sj4 + 1]);\n" \
"	write4_P3(x3, v_P3, k0 + 4 * miv, m);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void backward4_P12(const __global uint32_2 * restrict const wri12, __global uint32_2 * restrict const x12, const uint32 s, const uint32 m, const int lm)\n" \
"{\n" \
"	const size_t id = get_global_id(0), vid = id / VSIZE, l = id % VSIZE;\n" \
"\n" \
"	const size_t sj = s + (vid >> lm), i = vid & (m - 1), mj = vid - i, k = VSIZE * (4 * mj + i) + l;\n" \
"\n" \
"	uint32_2 u_P12[4]; read4_P12(u_P12, x12, k, m);\n" \
"\n" \
"	bkwd42_P12(u_P12, wri12[2 * sj], wri12[2 * sj + 1]);\n" \
"	bkwd41_P12(u_P12, wri12[sj]);\n" \
"\n" \
"	write4_P12(x12, u_P12, k, m);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void backward4_P123(const __global uint32_2 * restrict const wri12, const __global uint32 * restrict const wri3,\n" \
"	__global uint32_2 * restrict const x12, __global uint32 * restrict const x3, const uint32 s, const uint32 m, const int lm)\n" \
"{\n" \
"	const size_t id = get_global_id(0), vid = id / VSIZE, l = id % VSIZE;\n" \
"\n" \
"	const size_t sj = s + (vid >> lm), i = vid & (m - 1), mj = vid - i, k = VSIZE * (4 * mj + i) + l;\n" \
"\n" \
"	uint32_2 u_P12[4]; read4_P12(u_P12, x12, k, m); uint32 u_P3[4]; read4_P3(u_P3, x3, k, m);\n" \
"\n" \
"	bkwd42_P12(u_P12, wri12[2 * sj], wri12[2 * sj + 1]); bkwd42_P3(u_P3, wri3[2 * sj], wri3[2 * sj + 1]);\n" \
"	bkwd41_P12(u_P12, wri12[sj]); bkwd41_P3(u_P3, wri3[sj]);\n" \
"\n" \
"	write4_P12(x12, u_P12, k, m); write4_P3(x3, u_P3, k, m);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((work_group_size_hint(16 / 4 * VSIZE, 1, 1)))\n" \
"void backward16_P12(const __global uint32_2 * restrict const wri12, __global uint32_2 * restrict const x12, const uint32 s, const uint32 m, const int lm)\n" \
"{\n" \
"	__local uint32_2 X12[16 * VSIZE];	// 32 KB => VSIZE = 256\n" \
"\n" \
"	const size_t gid = get_global_id(0), vid = gid / VSIZE, l = gid % VSIZE;\n" \
"	const size_t lid = get_local_id(0), i = lid / VSIZE, iv = lid & (size_t)~(VSIZE - 1);\n" \
"\n" \
"	const size_t vid_blk = (vid & (size_t)~(4 * m - 1)) * 4, idl = get_group_id(0) & (m - 1);\n" \
"	const size_t k0 = VSIZE * (vid_blk + idl) + l, miv = iv << lm;\n" \
"	const size_t sj4 = s * 4 + (vid_blk >> (lm + 2)) + i, sj = sj4 / 4;\n" \
"\n" \
"	uint32_2 v_P12[4]; read4_P12(v_P12, x12, k0 + 4 * miv, m);\n" \
"	bkwd42_P12(v_P12, wri12[2 * sj4], wri12[2 * sj4 + 1]);\n" \
"	bkwd41_P12(v_P12, wri12[sj4]);\n" \
"	write4l_P12(X12, v_P12, 4 * iv + l, 1);\n" \
"\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	uint32_2 u_P12[4]; read4l_P12(u_P12, X12, iv + l, 4);\n" \
"	bkwd42_P12(u_P12, wri12[2 * sj], wri12[2 * sj + 1]);\n" \
"	bkwd41_P12(u_P12, wri12[sj]);\n" \
"	write4_P12(x12, u_P12, k0 + miv, 4 * m);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((work_group_size_hint(16 / 4 * VSIZE, 1, 1)))\n" \
"void backward16_P3(const __global uint32 * restrict const wri3, __global uint32 * restrict const x3, const uint32 s, const uint32 m, const int lm)\n" \
"{\n" \
"	__local uint32 X3[16 * VSIZE];\n" \
"\n" \
"	const size_t gid = get_global_id(0), vid = gid / VSIZE, l = gid % VSIZE;\n" \
"	const size_t lid = get_local_id(0), i = lid / VSIZE, iv = lid & (size_t)~(VSIZE - 1);\n" \
"\n" \
"	const size_t vid_blk = (vid & (size_t)~(4 * m - 1)) * 4, idl = get_group_id(0) & (m - 1);\n" \
"	const size_t k0 = VSIZE * (vid_blk + idl) + l, miv = iv << lm;\n" \
"	const size_t sj4 = s * 4 + (vid_blk >> (lm + 2)) + i, sj = sj4 / 4;\n" \
"\n" \
"	uint32 v_P3[4]; read4_P3(v_P3, x3, k0 + 4 * miv, m);\n" \
"	bkwd42_P3(v_P3, wri3[2 * sj4], wri3[2 * sj4 + 1]);\n" \
"	bkwd41_P3(v_P3, wri3[sj4]);\n" \
"	write4l_P3(X3, v_P3, 4 * iv + l, 1);\n" \
"\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	uint32 u_P3[4]; read4l_P3(u_P3, X3, iv + l, 4);\n" \
"	bkwd42_P3(u_P3, wri3[2 * sj], wri3[2 * sj + 1]);\n" \
"	bkwd41_P3(u_P3, wri3[sj]);\n" \
"	write4_P3(x3, u_P3, k0 + miv, 4 * m);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void square4_P12(const __global uint32_2 * restrict const wr12, const __global uint32_2 * restrict const wri12,\n" \
"	__global uint32_2 * restrict const x12)\n" \
"{\n" \
"	const size_t id = get_global_id(0), vid = id / VSIZE, l = id % VSIZE;\n" \
"\n" \
"	const size_t sj = get_global_size(0) / VSIZE + vid, k = VSIZE * 4 * vid + l;\n" \
"\n" \
"	uint32_2 u_P12[4]; read4_P12(u_P12, x12, k, 1);\n" \
"\n" \
"	frwd41_P12(u_P12, wr12[sj]);\n" \
"	sqr42_P12(u_P12, wr12[2 * sj], wr12[2 * sj + 1]);\n" \
"	bkwd41_P12(u_P12, wri12[sj]);\n" \
"\n" \
"	write4_P12(x12, u_P12, k, 1);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void square4_P123(const __global uint32_2 * restrict const wr12, const __global uint32 * restrict const wr3,\n" \
"	const __global uint32_2 * restrict const wri12, const __global uint32 * restrict const wri3,\n" \
"	__global uint32_2 * restrict const x12, __global uint32 * restrict const x3)\n" \
"{\n" \
"	const size_t id = get_global_id(0), vid = id / VSIZE, l = id % VSIZE;\n" \
"\n" \
"	const size_t sj = get_global_size(0) / VSIZE + vid, k = VSIZE * 4 * vid + l;\n" \
"\n" \
"	uint32_2 u_P12[4]; read4_P12(u_P12, x12, k, 1); uint32 u_P3[4]; read4_P3(u_P3, x3, k, 1);\n" \
"\n" \
"	frwd41_P12(u_P12, wr12[sj]); frwd41_P3(u_P3, wr3[sj]);\n" \
"	sqr42_P12(u_P12, wr12[2 * sj], wr12[2 * sj + 1]); sqr42_P3(u_P3, wr3[2 * sj], wr3[2 * sj + 1]);\n" \
"	bkwd41_P12(u_P12, wri12[sj]); bkwd41_P3(u_P3, wri3[sj]);\n" \
"\n" \
"	write4_P12(x12, u_P12, k, 1); write4_P3(x3, u_P3, k, 1);\n" \
"}\n" \
"\n" \
"inline void frwd4_P12(const __global uint32_2 * restrict const wr12, const __global uint32_2 * restrict const x12,\n" \
"	__local uint32_2 * const X12, const size_t kg, const size_t k, const size_t m, const size_t sj)\n" \
"{\n" \
"	uint32_2 u_P12[4]; read4_P12(u_P12, x12, kg + k, m);\n" \
"	frwd41_P12(u_P12, wr12[sj]);\n" \
"	frwd42_P12(u_P12, wr12[2 * sj], wr12[2 * sj + 1]);\n" \
"	write4l_P12(X12, u_P12, k, m);\n" \
"}\n" \
"inline void frwd4_P3(const __global uint32 * restrict const wr3, const __global uint32 * restrict const x3,\n" \
"	__local uint32 * const X3, const size_t kg, const size_t k, const size_t m, const size_t sj)\n" \
"{\n" \
"	uint32 u_P3[4]; read4_P3(u_P3, x3, kg + k, m);\n" \
"	frwd41_P3(u_P3, wr3[sj]);\n" \
"	frwd42_P3(u_P3, wr3[2 * sj], wr3[2 * sj + 1]);\n" \
"	write4l_P3(X3, u_P3, k, m);\n" \
"}\n" \
"\n" \
"inline void bkwd4_P12(const __global uint32_2 * restrict const wri12, const __local uint32_2 * const X12,\n" \
"	__global uint32_2 * restrict const x12, const size_t kg, const size_t k, const size_t m, const size_t sj)\n" \
"{\n" \
"	uint32_2 u_P12[4]; read4l_P12(u_P12, X12, k, m);\n" \
"	bkwd42_P12(u_P12, wri12[2 * sj], wri12[2 * sj + 1]);\n" \
"	bkwd41_P12(u_P12, wri12[sj]);\n" \
"	write4_P12(x12, u_P12, kg + k, m);\n" \
"}\n" \
"inline void bkwd4_P3(const __global uint32 * restrict const wri3, const __local uint32 * const X3,\n" \
"	__global uint32 * restrict const x3, const size_t kg, const size_t k, const size_t m, const size_t sj)\n" \
"{\n" \
"	uint32 u_P3[4]; read4l_P3(u_P3, X3, k, m);\n" \
"	bkwd42_P3(u_P3, wri3[2 * sj], wri3[2 * sj + 1]);\n" \
"	bkwd41_P3(u_P3, wri3[sj]);\n" \
"	write4_P3(x3, u_P3, kg + k, m);\n" \
"}\n" \
"\n" \
"inline void sqr4l_P12(const __global uint32_2 * restrict const wr12, const __global uint32_2 * restrict const wri12,\n" \
"	__local uint32_2 * const X12, const size_t k, const size_t sj)\n" \
"{\n" \
"	uint32_2 u_P12[4]; read4l_P12(u_P12, X12, k, 1);\n" \
"	frwd41_P12(u_P12, wr12[sj]);\n" \
"	sqr42_P12(u_P12, wr12[2 * sj], wr12[2 * sj + 1]);\n" \
"	bkwd41_P12(u_P12, wri12[sj]);\n" \
"	write4l_P12(X12, u_P12, k, 1);\n" \
"}\n" \
"inline void sqr4l_P3(const __global uint32 * restrict const wr3, const __global uint32 * restrict const wri3,\n" \
"	__local uint32 * const X3, const size_t k, const size_t sj)\n" \
"{\n" \
"	uint32 u_P3[4]; read4l_P3(u_P3, X3, k, 1);\n" \
"	frwd41_P3(u_P3, wr3[sj]);\n" \
"	sqr42_P3(u_P3, wr3[2 * sj], wr3[2 * sj + 1]);\n" \
"	bkwd41_P3(u_P3, wri3[sj]);\n" \
"	write4l_P3(X3, u_P3, k, 1);\n" \
"}\n" \
"\n" \
"inline void sqr22l_P12(const __global uint32_2 * restrict const wr12,\n" \
"	__local uint32_2 * const X12, const size_t k, const size_t sj)\n" \
"{\n" \
"	uint32_2 u_P12[4]; read22l_P12(u_P12, X12, k, 1);\n" \
"	const uint32_2 w12 = wr12[sj];\n" \
"	sqr2_P12(&u_P12[0], w12); sqr2_P12(&u_P12[2], w12);\n" \
"	write22l_P12(X12, u_P12, k, 1);\n" \
"}\n" \
"inline void sqr22l_P3(const __global uint32 * restrict const wr3,\n" \
"	__local uint32 * const X3, const size_t k, const size_t sj)\n" \
"{\n" \
"	uint32 u_P3[4]; read22l_P3(u_P3, X3, k, 1);\n" \
"	const uint32 w3 = wr3[sj];\n" \
"	sqr2_P3(&u_P3[0], w3); sqr2_P3(&u_P3[2], w3);\n" \
"	write22l_P3(X3, u_P3, k, 1);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((work_group_size_hint(8 / 4 * VSIZE, 1, 1)))\n" \
"void square8_P12(const __global uint32_2 * restrict const wr12, const __global uint32_2 * restrict const wri12,\n" \
"	__global uint32_2 * restrict const x12)\n" \
"{\n" \
"	__local uint32_2 X12[8 * VSIZE];\n" \
"\n" \
"	const size_t n_2 = get_global_size(0) * 2 / VSIZE, sj2 = n_2 + get_global_id(0) * 2 / VSIZE;\n" \
"	const size_t k_group = get_group_id(0) * 8 * VSIZE, i = get_local_id(0);\n" \
"\n" \
"	const size_t sj8 = sj2 / 4, k8 = i;\n" \
"	const size_t k2 = 2 * ((2 * i) & (size_t)~(VSIZE - 1)) + ((2 * i) % VSIZE);\n" \
"\n" \
"	frwd4_P12(wr12, x12, X12, k_group, k8, 2, sj8);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	sqr22l_P12(wr12, X12, k2, sj2);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	bkwd4_P12(wri12, X12, x12, k_group, k8, 2, sj8);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((work_group_size_hint(8 / 4 * VSIZE, 1, 1)))\n" \
"void square8_P3(const __global uint32 * restrict const wr3, const __global uint32 * restrict const wri3,\n" \
"	__global uint32 * restrict const x3)\n" \
"{\n" \
"	__local uint32 X3[8 * VSIZE];\n" \
"\n" \
"	const size_t n_2 = get_global_size(0) * 2 / VSIZE, sj2 = n_2 + get_global_id(0) * 2 / VSIZE;\n" \
"	const size_t k_group = get_group_id(0) * 8 * VSIZE, i = get_local_id(0);\n" \
"\n" \
"	const size_t sj8 = sj2 / 4, k8 = i;\n" \
"	const size_t k2 = 2 * ((2 * i) & (size_t)~(VSIZE - 1)) + ((2 * i) % VSIZE);\n" \
"\n" \
"	frwd4_P3(wr3, x3, X3, k_group, k8, 2, sj8);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	sqr22l_P3(wr3, X3, k2, sj2);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	bkwd4_P3(wri3, X3, x3, k_group, k8, 2, sj8);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((work_group_size_hint(16 / 4 * VSIZE, 1, 1)))\n" \
"void square16_P12(const __global uint32_2 * restrict const wr12, const __global uint32_2 * restrict const wri12,\n" \
"	__global uint32_2 * restrict const x12)\n" \
"{\n" \
"	__local uint32_2 X12[16 * VSIZE];	// 32 KB => VSIZE = 256\n" \
"\n" \
"	const size_t n_4 = get_global_size(0) / VSIZE, sj4 = n_4 + get_global_id(0) / VSIZE;\n" \
"	const size_t k_group = get_group_id(0) * 16 * VSIZE, i = get_local_id(0);\n" \
"\n" \
"	const size_t sj16 = sj4 / 4, k16 = i;\n" \
"	const size_t k4 = 4 * (i & (size_t)~(VSIZE - 1)) + (i % VSIZE);\n" \
"\n" \
"	frwd4_P12(wr12, x12, X12, k_group, k16, 4, sj16);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	sqr4l_P12(wr12, wri12, X12, k4, sj4);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	bkwd4_P12(wri12, X12, x12, k_group, k16, 4, sj16);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((work_group_size_hint(16 / 4 * VSIZE, 1, 1)))\n" \
"void square16_P3(const __global uint32 * restrict const wr3, const __global uint32 * restrict const wri3,\n" \
"	__global uint32 * restrict const x3)\n" \
"{\n" \
"	__local uint32 X3[16 * VSIZE];\n" \
"\n" \
"	const size_t n_4 = get_global_size(0) / VSIZE, sj4 = n_4 + get_global_id(0) / VSIZE;\n" \
"	const size_t k_group = get_group_id(0) * 16 * VSIZE, i = get_local_id(0);\n" \
"\n" \
"	const size_t sj16 = sj4 / 4, k16 = i;\n" \
"	const size_t k4 = 4 * (i & (size_t)~(VSIZE - 1)) + (i % VSIZE);\n" \
"\n" \
"	frwd4_P3(wr3, x3, X3, k_group, k16, 4, sj16);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	sqr4l_P3(wr3, wri3, X3, k4, sj4);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	bkwd4_P3(wri3, X3, x3, k_group, k16, 4, sj16);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mul4cond64_P12(const __global uint32_2 * restrict const wr12, const __global uint32_2 * restrict const wri12,\n" \
"	const __global uint32_2 * restrict const y12, __global uint32_2 * restrict const x12, const uint64 c)\n" \
"{\n" \
"	const size_t id = get_global_id(0), vid = id / VSIZE, l = id % VSIZE;\n" \
"\n" \
"	const size_t sj = get_global_size(0) / VSIZE + vid, k = VSIZE * 4 * vid + l;\n" \
"\n" \
"	uint32_2 u_P12[4]; read4_P12(u_P12, x12, k, 1);\n" \
"\n" \
"	frwd41_P12(u_P12, wr12[sj]);\n" \
"	frwd42_P12(u_P12, wr12[2 * sj], wr12[2 * sj + 1]);\n" \
"\n" \
"	if ((c & cMask[l]) != 0)\n" \
"	{\n" \
"		for (size_t h = 0; h < 4; ++h) u_P12[h] = mul_P12(u_P12[h], y12[k + h * VSIZE]);\n" \
"	}\n" \
"\n" \
"	bkwd42_P12(u_P12, wri12[2 * sj], wri12[2 * sj + 1]);\n" \
"	bkwd41_P12(u_P12, wri12[sj]);\n" \
"\n" \
"	write4_P12(x12, u_P12, k, 1);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mul4cond64_P123(const __global uint32_2 * restrict const wr12, const __global uint32 * restrict const wr3,\n" \
"	const __global uint32_2 * restrict const wri12, const __global uint32 * restrict const wri3,\n" \
"	const __global uint32_2 * restrict const y12, const __global uint32 * restrict const y3,\n" \
"	__global uint32_2 * restrict const x12, __global uint32 * restrict const x3, const uint64 c)\n" \
"{\n" \
"	const size_t id = get_global_id(0), vid = id / VSIZE, l = id % VSIZE;\n" \
"\n" \
"	const size_t sj = get_global_size(0) / VSIZE + vid, k = VSIZE * 4 * vid + l;\n" \
"\n" \
"	uint32_2 u_P12[4]; read4_P12(u_P12, x12, k, 1); uint32 u_P3[4]; read4_P3(u_P3, x3, k, 1);\n" \
"\n" \
"	frwd41_P12(u_P12, wr12[sj]); frwd41_P3(u_P3, wr3[sj]);\n" \
"	frwd42_P12(u_P12, wr12[2 * sj], wr12[2 * sj + 1]); frwd42_P3(u_P3, wr3[2 * sj], wr3[2 * sj + 1]);\n" \
"\n" \
"	if ((c & cMask[l]) != 0)\n" \
"	{\n" \
"		for (size_t h = 0; h < 4; ++h) u_P12[h] = mul_P12(u_P12[h], y12[k + h * VSIZE]);\n" \
"		for (size_t h = 0; h < 4; ++h) u_P3[h] = mul_P3(u_P3[h], y3[k + h * VSIZE]);\n" \
"	}\n" \
"\n" \
"	bkwd42_P12(u_P12, wri12[2 * sj], wri12[2 * sj + 1]); bkwd42_P3(u_P3, wri3[2 * sj], wri3[2 * sj + 1]);\n" \
"	bkwd41_P12(u_P12, wri12[sj]); bkwd41_P3(u_P3, wri3[sj]);\n" \
"\n" \
"	write4_P12(x12, u_P12, k, 1); write4_P3(x3, u_P3, k, 1);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mul4cond1024_P12(const __global uint32_2 * restrict const wr12, const __global uint32_2 * restrict const wri12,\n" \
"	const __global uint32_2 * restrict const y12, __global uint32_2 * restrict const x12, const uint64_16 c)\n" \
"{\n" \
"	const size_t id = get_global_id(0), vid = id / VSIZE, l = id % VSIZE;\n" \
"\n" \
"	const size_t sj = get_global_size(0) / VSIZE + vid, k = VSIZE * 4 * vid + l;\n" \
"\n" \
"	uint32_2 u_P12[4]; read4_P12(u_P12, x12, k, 1);\n" \
"\n" \
"	frwd41_P12(u_P12, wr12[sj]);\n" \
"	frwd42_P12(u_P12, wr12[2 * sj], wr12[2 * sj + 1]);\n" \
"\n" \
"	if ((getcval(c, l / 64) & cMask[l % 64]) != 0)\n" \
"	{\n" \
"		for (size_t h = 0; h < 4; ++h) u_P12[h] = mul_P12(u_P12[h], y12[k + h * VSIZE]);\n" \
"	}\n" \
"\n" \
"	bkwd42_P12(u_P12, wri12[2 * sj], wri12[2 * sj + 1]);\n" \
"	bkwd41_P12(u_P12, wri12[sj]);\n" \
"\n" \
"	write4_P12(x12, u_P12, k, 1);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mul4cond1024_P123(const __global uint32_2 * restrict const wr12, const __global uint32 * restrict const wr3,\n" \
"	const __global uint32_2 * restrict const wri12, const __global uint32 * restrict const wri3,\n" \
"	const __global uint32_2 * restrict const y12, const __global uint32 * restrict const y3,\n" \
"	__global uint32_2 * restrict const x12, __global uint32 * restrict const x3, const uint64_16 c)\n" \
"{\n" \
"	const size_t id = get_global_id(0), vid = id / VSIZE, l = id % VSIZE;\n" \
"\n" \
"	const size_t sj = get_global_size(0) / VSIZE + vid, k = VSIZE * 4 * vid + l;\n" \
"\n" \
"	uint32_2 u_P12[4]; read4_P12(u_P12, x12, k, 1); uint32 u_P3[4]; read4_P3(u_P3, x3, k, 1);\n" \
"\n" \
"	frwd41_P12(u_P12, wr12[sj]); frwd41_P3(u_P3, wr3[sj]);\n" \
"	frwd42_P12(u_P12, wr12[2 * sj], wr12[2 * sj + 1]); frwd42_P3(u_P3, wr3[2 * sj], wr3[2 * sj + 1]);\n" \
"\n" \
"	if ((getcval(c, l / 64) & cMask[l % 64]) != 0)\n" \
"	{\n" \
"		for (size_t h = 0; h < 4; ++h) u_P12[h] = mul_P12(u_P12[h], y12[k + h * VSIZE]);\n" \
"		for (size_t h = 0; h < 4; ++h) u_P3[h] = mul_P3(u_P3[h], y3[k + h * VSIZE]);\n" \
"	}\n" \
"\n" \
"	bkwd42_P12(u_P12, wri12[2 * sj], wri12[2 * sj + 1]); bkwd42_P3(u_P3, wri3[2 * sj], wri3[2 * sj + 1]);\n" \
"	bkwd41_P12(u_P12, wri12[sj]); bkwd41_P3(u_P3, wri3[sj]);\n" \
"\n" \
"	write4_P12(x12, u_P12, k, 1); write4_P3(x3, u_P3, k, 1);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mul4_P12(const __global uint32_2 * restrict const wr12, const __global uint32_2 * restrict const wri12,\n" \
"	const __global uint32_2 * restrict const y12, __global uint32_2 * restrict const x12)\n" \
"{\n" \
"	const size_t id = get_global_id(0), vid = id / VSIZE, l = id % VSIZE;\n" \
"\n" \
"	const size_t sj = get_global_size(0) / VSIZE + vid, k = VSIZE * 4 * vid + l;\n" \
"\n" \
"	const uint32_2 wr12_1 = wr12[sj], wr12_2 = wr12[2 * sj], wr12_3 = wr12[2 * sj + 1];\n" \
"\n" \
"	uint32_2 u_P12[4]; read4_P12(u_P12, x12, k, 1);\n" \
"\n" \
"	frwd41_P12(u_P12, wr12_1);\n" \
"	frwd42_P12(u_P12, wr12_2, wr12_3);\n" \
"\n" \
"	uint32_2 v_P12[4]; read4_P12(v_P12, y12, k, 1);\n" \
"\n" \
"	frwd41_P12(v_P12, wr12_1);\n" \
"	frwd42_P12(v_P12, wr12_2, wr12_3);\n" \
"\n" \
"	for (size_t h = 0; h < 4; ++h) u_P12[h] = mul_P12(u_P12[h], v_P12[h]);\n" \
"\n" \
"	bkwd42_P12(u_P12, wri12[2 * sj], wri12[2 * sj + 1]);\n" \
"	bkwd41_P12(u_P12, wri12[sj]);\n" \
"\n" \
"	write4_P12(x12, u_P12, k, 1);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mul4_P123(const __global uint32_2 * restrict const wr12, const __global uint32 * restrict const wr3,\n" \
"	const __global uint32_2 * restrict const wri12, const __global uint32 * restrict const wri3,\n" \
"	const __global uint32_2 * restrict const y12, const __global uint32 * restrict const y3,\n" \
"	__global uint32_2 * restrict const x12, __global uint32 * restrict const x3)\n" \
"{\n" \
"	const size_t id = get_global_id(0), vid = id / VSIZE, l = id % VSIZE;\n" \
"\n" \
"	const size_t sj = get_global_size(0) / VSIZE + vid, k = VSIZE * 4 * vid + l;\n" \
"\n" \
"	const uint32_2 wr12_1 = wr12[sj], wr12_2 = wr12[2 * sj], wr12_3 = wr12[2 * sj + 1];\n" \
"	const uint32 wr3_1 = wr3[sj], wr3_2 = wr3[2 * sj], wr3_3 = wr3[2 * sj + 1];\n" \
"\n" \
"	uint32_2 u_P12[4]; read4_P12(u_P12, x12, k, 1); uint32 u_P3[4]; read4_P3(u_P3, x3, k, 1);\n" \
"\n" \
"	frwd41_P12(u_P12, wr12_1); frwd41_P3(u_P3, wr3_1);\n" \
"	frwd42_P12(u_P12, wr12_2, wr12_3); frwd42_P3(u_P3, wr3_2, wr3_3);\n" \
"\n" \
"	uint32_2 v_P12[4]; read4_P12(v_P12, y12, k, 1); uint32 v_P3[4]; read4_P3(v_P3, y3, k, 1);\n" \
"\n" \
"	frwd41_P12(v_P12, wr12_1); frwd41_P3(v_P3, wr3_1);\n" \
"	frwd42_P12(v_P12, wr12_2, wr12_3); frwd42_P3(v_P3, wr3_2, wr3_3);\n" \
"\n" \
"	for (size_t h = 0; h < 4; ++h) u_P12[h] = mul_P12(u_P12[h], v_P12[h]);\n" \
"	for (size_t h = 0; h < 4; ++h) u_P3[h] = mul_P3(u_P3[h], v_P3[h]);\n" \
"\n" \
"	bkwd42_P12(u_P12, wri12[2 * sj], wri12[2 * sj + 1]); bkwd42_P3(u_P3, wri3[2 * sj], wri3[2 * sj + 1]);\n" \
"	bkwd41_P12(u_P12, wri12[sj]); bkwd41_P3(u_P3, wri3[sj]);\n" \
"\n" \
"	write4_P12(x12, u_P12, k, 1); write4_P3(x3, u_P3, k, 1);\n" \
"}\n" \
"\n" \
"inline uint32 barrett(const uint64 a, const uint32 b, const uint32 b_inv, const int b_s, uint32 * a_p)\n" \
"{\n" \
"	// n = 31, alpha = 2^{n-2} = 2^29, s = r - 2, t = n + 1 = 32 => h = 1.\n" \
"	// b < 2^31, alpha = 2^29 => a < 2^29 b\n" \
"	// 2^{r-1} < b <= 2^r then a < 2^{r + 29} = 2^{s + 31} and (a >> s) < 2^31\n" \
"	// b_inv = [2^{s + 32} / b]\n" \
"	// b_inv < 2^{s + 32} / b < 2^{s + 32} / 2^{r-1} = 2^{s + 32} / 2^{s + 1} < 2^31\n" \
"	// Let h be the number of iterations in Barrett's reduction, we have h = [a / b] - [[a / 2^s] b_inv / 2^32].\n" \
"	// h = ([a/b] - a/b) + a/2^{s + 32} (2^{s + 32}/b - b_inv) + b_inv/2^32 (a/2^s - [a/2^s]) + ([a/2^s] b_inv / 2^32 - [[a/2^s] b_inv / 2^32])\n" \
"	// Then -1 + 0 + 0 + 0 < h < 0 + 1/2 (2^{s + 32}/b - b_inv) + b_inv/2^32 + 1,\n" \
"	// 0 <= h < 1 + 1/2 + 1/2 => h = 1.\n" \
"\n" \
"	const uint32 d = mul_hi((uint32)(a >> b_s), b_inv), r = (uint32)a - d * b;\n" \
"	const bool o = (r >= b);\n" \
"	*a_p = o ? d + 1 : d;\n" \
"	return o ? r - b : r;\n" \
"}\n" \
"\n" \
"inline int32 reduce64(int64 * f, const uint32 b, const uint32 b_inv, const int b_s)\n" \
"{\n" \
"	// 1- t < 2^63 => t_h < 2^34. We must have t_h < 2^29 b => b > 32\n" \
"	// 2- t < 2^22 b^2 => t_h < b^2 / 2^7. If 2 <= b < 32 then t_h < 32^2 / 2^7 = 2^8 < 2^29 b\n" \
"	const uint64 t = abs(*f);\n" \
"	const uint64 t_h = t >> 29;\n" \
"	const uint32 t_l = (uint32)t & ((1u << 29) - 1);\n" \
"\n" \
"	uint32 d_h, r_h = barrett(t_h, b, b_inv, b_s, &d_h);\n" \
"	uint32 d_l, r_l = barrett(((uint64)r_h << 29) | t_l, b, b_inv, b_s, &d_l);\n" \
"	const uint64 d = ((uint64)d_h << 29) | d_l;\n" \
"\n" \
"	const bool s = (*f < 0);\n" \
"	*f = s ? -(int64)d : (int64)d;\n" \
"	return s ? -(int32)r_l : (int32)r_l;\n" \
"}\n" \
"\n" \
"inline int32 reduce96(int96 * f, const uint32 b, const uint32 b_inv, const int b_s)\n" \
"{\n" \
"	const uint96 t = int96_abs(*f);\n" \
"	const uint64 t_h = ((uint64)t.s1 << (64 - 29)) | (t.s0 >> 29);\n" \
"	const uint32 t_l = (uint32)t.s0 & ((1u << 29) - 1);\n" \
"\n" \
"	uint32 d_h, r_h = barrett(t_h, b, b_inv, b_s, &d_h);\n" \
"	uint32 d_l, r_l = barrett(((uint64)r_h << 29) | t_l, b, b_inv, b_s, &d_l);\n" \
"	const uint64 d = ((uint64)d_h << 29) | d_l;\n" \
"\n" \
"	const bool s = int96_is_neg(*f);\n" \
"	*f = int96_set_si(s ? -(int64)d : (int64)d);\n" \
"	return s ? -(int32)r_l : (int32)r_l;\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void normalize2a(const __global uint32_2 * restrict const bb_inv, __global int64 * restrict const f,\n" \
"				 __global uint32_2 * restrict const z12, const int b_s)\n" \
"{\n" \
"	const size_t id = get_global_id(0);\n" \
"	const size_t i = id % VSIZE, j = id / VSIZE, k0 = j * (VSIZE * CSIZE) + i;\n" \
"	const uint32 b = bb_inv[i].s0, b_inv = bb_inv[i].s1;\n" \
"	int64 a = 0;\n" \
"	for (size_t c = 0; c < CSIZE; ++c)\n" \
"	{\n" \
"		const size_t k = k0 + c * VSIZE;\n" \
"		const uint32_2 z12k = z12[k];\n" \
"		a += garner2(mul_P1(z12k.s0, norm1), mul_P2(z12k.s1, norm2));\n" \
"		const int32 r = reduce64(&a, b, b_inv, b_s);\n" \
"		z12[k] = (uint32_2)(seti_P1(r), seti_P2(r));\n" \
"	}\n" \
"	f[id] = a;\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void normalize2b(const __global uint32_2 * restrict const bb_inv, const __global int64 * restrict const f,\n" \
"				 __global uint32_2 * restrict const z12, const int b_s)\n" \
"{\n" \
"	const size_t id = get_global_id(0);\n" \
"	const size_t i = id % VSIZE, j = (id / VSIZE + 1) & (get_global_size(0) / VSIZE - 1);\n" \
"	int64 a = f[id];\n" \
"	const uint32 b = bb_inv[i].s0, b_inv = bb_inv[i].s1;\n" \
"	const size_t k0 = j * (VSIZE * CSIZE) + i;\n" \
"	if (j == 0) a = -a;	// a_0 = -a_n\n" \
"	size_t c;\n" \
"	for (c = 0; c < CSIZE - 1; ++c)\n" \
"	{\n" \
"		const size_t k = k0 + c * VSIZE;\n" \
"		a += geti_P1(z12[k].s0);\n" \
"		const int32 r = reduce64(&a, b, b_inv, b_s);\n" \
"		z12[k] = (uint32_2)(seti_P1(r), seti_P2(r));\n" \
"		if (a == 0) return;\n" \
"	}\n" \
"	if (c == CSIZE - 1)\n" \
"	{\n" \
"		const size_t k = k0 + c * VSIZE;\n" \
"		z12[k] = add_P12(z12[k], (uint32_2)(seti_P1((int32)a), seti_P2((int32)a)));\n" \
"		// if (abs(a) > 1) throw;\n" \
"	}\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void normalize3a(const __global uint32_2 * restrict const bb_inv, __global int64 * restrict const f,\n" \
"				 __global uint32_2 * restrict const z12, __global uint32 * restrict const z3, const int b_s)\n" \
"{\n" \
"	const size_t id = get_global_id(0);\n" \
"	const size_t i = id % VSIZE, j = id / VSIZE, k0 = j * (VSIZE * CSIZE) + i;\n" \
"	const uint32 b = bb_inv[i].s0, b_inv = bb_inv[i].s1;\n" \
"	int96 a = int96_set_si(0);\n" \
"	for (size_t c = 0; c < CSIZE; ++c)\n" \
"	{\n" \
"		const size_t k = k0 + c * VSIZE;\n" \
"		const uint32_2 z12k = z12[k];\n" \
"		a = int96_add(a, garner3(mul_P1(z12k.s0, norm1), mul_P2(z12k.s1, norm2), mul_P3(z3[k], norm3)));\n" \
"		const int32 r = reduce96(&a, b, b_inv, b_s);\n" \
"		z12[k] = (uint32_2)(seti_P1(r), seti_P2(r)); z3[k] = seti_P3(r);\n" \
"	}\n" \
"	f[id] = (int64)a.s0;\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void normalize3b(const __global uint32_2 * restrict const bb_inv, const __global int64 * restrict const f,\n" \
"				 __global uint32_2 * restrict const z12, __global uint32 * restrict const z3, const int b_s)\n" \
"{\n" \
"	const size_t id = get_global_id(0);\n" \
"	const size_t i = id % VSIZE, j = (id / VSIZE + 1) & (get_global_size(0) / VSIZE - 1);\n" \
"	int64 a = f[id];\n" \
"	const uint32 b = bb_inv[i].s0, b_inv = bb_inv[i].s1;\n" \
"	const size_t k0 = j * (VSIZE * CSIZE) + i;\n" \
"	if (j == 0) a = -a;	// a_0 = -a_n\n" \
"	size_t c;\n" \
"	for (c = 0; c < CSIZE - 1; ++c)\n" \
"	{\n" \
"		const size_t k = k0 + c * VSIZE;\n" \
"		a += geti_P1(z12[k].s0);\n" \
"		const int32 r = reduce64(&a, b, b_inv, b_s);\n" \
"		z12[k] = (uint32_2)(seti_P1(r), seti_P2(r)); z3[k] = seti_P3(r);\n" \
"		if (a == 0) return;\n" \
"	}\n" \
"	if (c == CSIZE - 1)\n" \
"	{\n" \
"		const size_t k = k0 + c * VSIZE;\n" \
"		z12[k] = add_P12(z12[k], (uint32_2)(seti_P1((int32)a), seti_P2((int32)a))); z3[k] = add_P3(z3[k], seti_P3((int32)a));\n" \
"		// if (abs(a) > 1) throw;\n" \
"	}\n" \
"}\n" \
"";
