/*
Copyright 2020, Yves Gallot

genefer20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include "engine.h"
#include "pio.h"

#include <cstdint>
#include <cmath>
#include <sstream>
#include <array>

#include "ocl/kernel.h"

typedef std::array<uint32_t, VSIZE> vint32;

inline int ilog2(const uint32_t n) { return 31 - __builtin_clz(n); }

class transform
{
private:
	static const uint32_t P1 = 4253024257u;			// 507 * 2^23 + 1
	static const uint32_t P2 = 4194304001u;			// 125 * 2^25 + 1
	static const uint32_t P3 = 4076863489u;			// 243 * 2^24 + 1
	static const uint32_t P1_INV = 42356678u;		// (2^64 - 1) / P1 - 2^32
	static const uint32_t P2_INV = 103079214u;		// (2^64 - 1) / P2 - 2^32
	static const uint32_t P3_INV = 229771911u;		// (2^64 - 1) / P3 - 2^32
	static const uint32_t InvP2_P1 = 1822724754u;	// 1 / P2 mod P1
	static const uint32_t InvP3_P1 = 607574918u;	// 1 / P3 mod P1
	static const uint32_t InvP3_P2 = 2995931465u;	// 1 / P3 mod P2
	static const uint64_t P1P2 = (P1 * uint64_t(P2));
	static const uint64_t P2P3 = (P2 * uint64_t(P3));
	static const uint64_t P1P2P3l = 15383592652180029441ull;
	static const uint32_t P1P2P3h = 3942432002u;
	static const uint64_t P1P2P3_2l = 7691796326090014720ull;
	static const uint32_t P1P2P3_2h = 1971216001u;

private:
	template <uint32 p, uint32 p_inv, uint32 prmRoot>
	class Zp
	{
	private:
		uint32 _n;

	private:
		uint32 _mulMod(const uint32 n) const
		{
			const uint64 m = _n * uint64(n), q = (m >> 32) * p_inv + m;
			uint32 r = uint32(m) - (1 + uint32(q >> 32)) * p;
			if (r > uint32(q)) r += p;
			return (r >= p) ? r - p : r;
		}
	
	public:
		Zp() {}
		explicit Zp(const uint32 n) : _n(n) {}
		explicit Zp(const int32 i) : _n((i < 0) ? i + p : i) {}

		uint32 get() const { return _n; }
		int32 geti() const { return (_n > p / 2) ? int32(_n - p) : int32(_n); }

		Zp operator-() const { return Zp((_n != 0) ? p - _n : 0); }

		Zp operator+(const Zp & rhs) const { const uint32 c = (_n >= p - rhs._n) ? p : 0; return Zp(_n + rhs._n - c); }
		Zp operator-(const Zp & rhs) const { const uint32 c = (_n < rhs._n) ? p : 0; return Zp(_n - rhs._n + c); }
		Zp operator*(const Zp & rhs) const { return Zp(_mulMod(rhs._n)); }

		Zp & operator+=(const Zp & rhs) { *this = *this + rhs; return *this; }
		Zp & operator-=(const Zp & rhs) { *this = *this - rhs; return *this; }
		Zp & operator*=(const Zp & rhs) { *this = *this * rhs; return *this; }

		Zp pow(const uint32 e) const
		{
			if (e == 0) return Zp(1);

			Zp r = Zp(1), y = *this;
			for (uint32 i = e; i > 1; i /= 2)
			{
				if (i % 2 != 0) r *= y;
				y *= y;
			}
			r *= y;

			return r;
		}

		static Zp norm(const size_t n) { return -Zp(uint32((p - 1) / n)); }
		static const Zp prmRoot_n(const size_t n) { return Zp(prmRoot).pow(uint32((p - 1) / n)); }
	};

	typedef Zp<P1, P1_INV, 5> Zp1;
	typedef Zp<P2, P2_INV, 3> Zp2;
	typedef Zp<P3, P3_INV, 7> Zp3;

	template <typename Zp>
	struct Szp
	{
		Zp * const x;
		Zp * const y;
		// Zp * const wr;
		// Zp * const wri;
		Zp * const d;
		const Zp norm;

		Szp(const size_t n) : x(new Zp[VSIZE * n]), y(new Zp[VSIZE * n]), /*wr(new Zp[n]), wri(new Zp[n]),*/ d(new Zp[VSIZE * n]), norm(Zp::norm(n)) {}
		virtual ~Szp() 
		{
			delete[] x;
			delete[] y;
			// delete[] wr;
			// delete[] wri;
			delete[] d;
		}
	};

private:
	const size_t _n;
	const bool _isBoinc;
	engine & _engine;
	const Szp<Zp1> _z1;
	const Szp<Zp2> _z2;
	const Szp<Zp3> _z3;
	uint32 * const _x;
	bool _3primes = true;
	int _lgb = 0;
	int _b_s = 0;
	uint32_t _b[VSIZE];
	uint32_t _b_inv[VSIZE];

private:
	inline static int64 garner2(const Zp1 & r1, const Zp2 & r2)
	{
		const Zp1 u12 = (r1 - Zp1(r2.get())) * Zp1(InvP2_P1);
		const uint64 n = r2.get() + uint64(u12.get()) * P2;
		return (n > P1P2 / 2) ? int64(n - P1P2) : int64(n);
	}

private:
	inline static int96 garner3(const Zp1 & r1, const Zp2 & r2, const Zp3 & r3)
	{
		const Zp1 u13 = (r1 - Zp1(r3.get())) * Zp1(InvP3_P1);
		const Zp2 u23 = (r2 - Zp2(r3.get())) * Zp2(InvP3_P2);
		const Zp1 u123 = (u13 - Zp1(u23.get())) * Zp1(InvP2_P1);
		const uint96 n = uint96_add_64(uint96_mul_64_32(P2P3, u123.get()), u23.get() * uint64(P3) + r3.get());
		const uint96 P1P2P3 = uint96_set(P1P2P3l, P1P2P3h), P1P2P3_2 = uint96_set(P1P2P3_2l, P1P2P3_2h);
		const int96 r = uint96_is_greater(n, P1P2P3_2) ? uint96_subi(n, P1P2P3) : uint96_i(n);
		return r;
	}

private:
	static size_t bitRev(const size_t i, const size_t n)
	{
		size_t r = 0;
		for (size_t k = n, j = i; k > 1; k /= 2, j /= 2) r = (2 * r) | (j % 2);
		return r;
	}

protected:
	template<typename Zp>
	static void create(const size_t n, std::vector<uint32> & wr, std::vector<uint32> & wri)
	{
		for (size_t s = 1; s < n; s *= 2)
		{
			const size_t m = 4 * s;
			const Zp prRoot_m = Zp::prmRoot_n(m);

			for (size_t i = 0; i < s; ++i)
			{
				const size_t e = bitRev(i, 2 * s) + 1;
				const Zp wrsi = prRoot_m.pow(e);
				wr[s + i] = wrsi.get(); wri[s + s - i - 1] = Zp(-wrsi).get();
			}
		}
	}

// protected:
// 	template<typename Zp>
// 	static void reset(const size_t n, const Szp<Zp> & z, const uint32 a)
// 	{
// 		Zp * const x = z.x;
// 		for (size_t k = 0; k < VSIZE; ++k) x[k] = Zp(a);
// 		for (size_t k = VSIZE; k < VSIZE * n; ++k) x[k] = Zp(0);
// 		Zp * const d = z.d;
// 		for (size_t k = 0; k < VSIZE; ++k) d[k] = Zp(a);
// 		for (size_t k = VSIZE; k < VSIZE * n; ++k) d[k] = Zp(0);
// 	}

// protected:
// 	template<typename Zp>
// 	static void set(const size_t n, Zp * const y, const Zp * const x)
// 	{
// 		for (size_t k = 0; k < VSIZE * n; ++k) y[k] = x[k];
// 	}

// private:
// 	template<typename Zp>
// 	static void swap(const size_t n, Zp * const y, Zp * const x)
// 	{
// 		for (size_t k = 0; k < VSIZE * n; ++k)
// 		{
// 			const Zp t = y[k]; y[k] = x[k]; x[k] = t;
// 		}
// 	}

// protected:
// 	template<typename Zp>
// 	static void forward(const size_t n, Zp * const x, const Zp * const wr)
// 	{
// 		for (size_t m = n / 2, lm = ilog2(m), s = 1; m >= 1; m /= 2, --lm, s *= 2)
// 		{
// 			for (size_t id = 0; id < VSIZE * n / 2; ++id)
// 			{
// 				const size_t vid = id / VSIZE, l = id % VSIZE;
// 				const size_t j = vid >> lm, i = vid & (m - 1), mj = vid - i;
// 				const Zp w = wr[s + j];
// 				const size_t k = VSIZE * (2 * mj + i) + l;
// 				const Zp u = x[k], um = x[k + VSIZE * m] * w;
// 				x[k] = u + um; x[k + VSIZE * m] = u - um;
// 			}
// 		}
// 	}

// protected:
// 	template<typename Zp>
// 	static void backward(const size_t n, Zp * const x, const Zp * const wri)
// 	{
// 		for (size_t m = 1, lm = ilog2(m), s = n / 2; m <= n / 2; m *= 2, ++lm, s /= 2)
// 		{
// 			for (size_t id = 0; id < VSIZE * n / 2; ++id)
// 			{
// 				const size_t vid = id / VSIZE, l = id % VSIZE;
// 				const size_t j = vid >> lm, i = vid & (m - 1), mj = vid - i;
// 				const Zp wi = wri[s + j];
// 				const size_t k = VSIZE * (2 * mj + i) + l;
// 				const Zp u = x[k], um = x[k + VSIZE * m];
// 				x[k] = u + um; x[k + VSIZE * m] = (u - um) * wi;
// 			}
// 		}
// 	}

// private:
// 	template<typename Zp>
// 	static void square2(const size_t n, Zp * const x)
// 	{
// 		for (size_t k = 0; k < VSIZE * n; ++k) x[k] *= x[k];
// 	}

// private:
// 	template<typename Zp>
// 	static void mul2(const size_t n, Zp * const x, const Zp * const y)
// 	{
// 		for (size_t k = 0; k < VSIZE * n; ++k) x[k] *= y[k];
// 	}

// private:
// 	template<typename Zp>
// 	static void mul2cond(const size_t n, Zp * const x, const Zp * const y, const uint64 c)
// 	{
// 		for (size_t k = 0; k < VSIZE * n; ++k)
// 		{
// 			const size_t i = k % VSIZE;
// 			if ((c & (uint64(1) << i)) != 0) x[k] *= y[k];
// 		}
// 	}

// protected:
// 	template<typename Zp>
// 	static void square(const size_t n, const Szp<Zp> & z)
// 	{
// 		forward<Zp>(n, z.x, z.wr);
// 		square2<Zp>(n, z.x);
// 		backward<Zp>(n, z.x, z.wri);
// 	}

// protected:
// 	template<typename Zp>
// 	static void mul(const size_t n, const Szp<Zp> & z, const uint64 c)
// 	{
// 		forward<Zp>(n, z.x, z.wr);
// 		mul2cond<Zp>(n, z.x, z.y, c);
// 		backward<Zp>(n, z.x, z.wri);
// 	}

// protected:
// 	template<typename Zp>
// 	static void mul_dx(const size_t n, const Szp<Zp> & z)
// 	{
// 		forward<Zp>(n, z.d, z.wr);
// 		set<Zp>(n, z.y, z.x);
// 		forward<Zp>(n, z.y, z.wr);
// 		mul2<Zp>(n, z.d, z.y);
// 		backward<Zp>(n, z.d, z.wri);
// 	}

// protected:
// 	template<typename Zp>
// 	static void mul_xd_swap(const size_t n, const Szp<Zp> & z)
// 	{
// 		forward<Zp>(n, z.x, z.wr);
// 		set<Zp>(n, z.y, z.d);
// 		forward<Zp>(n, z.y, z.wr);
// 		mul2<Zp>(n, z.x, z.y);
// 		backward<Zp>(n, z.x, z.wri);
// 		swap<Zp>(n, z.x, z.d);
// 	}

private:
	static uint32 barrett(const uint64 a, const uint32 b, const uint32 b_inv, const int b_s, uint32 & a_p)
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

		const uint32 d = mul_hi(uint32(a >> b_s), b_inv), r = uint32(a) - d * b;
		const bool o = (r >= b);
		a_p = o ? d + 1 : d;
		return o ? r - b : r;
	}

protected:
	static int32 reduce64(int64 & f, const uint32 b, const uint32 b_inv, const int b_s)
	{
		// 1- t < 2^63 => t_h < 2^34. We must have t_h < 2^29 b => b > 32
		// 2- t < 2^22 b^2 => t_h < b^2 / 2^7. If 2 <= b < 32 then t_h < 32^2 / 2^7 = 2^8 < 2^29 b
		const uint64 t = std::abs(f);
		const uint64 t_h = t >> 29;
		const uint32 t_l = uint32(t) & ((uint32(1) << 29) - 1);

		uint32 d_h, r_h = barrett(t_h, b, b_inv, b_s, d_h);
		uint32 d_l, r_l = barrett((uint64(r_h) << 29) | t_l, b, b_inv, b_s, d_l);
		const uint64 d = (uint64(d_h) << 29) | d_l;

		const bool s = (f < 0);
		f = s ? -int64(d) : int64(d);
		const int32 r = s ? -int32(r_l) : int32(r_l);
		return r;
	}

protected:
	static int32 reduce96(int96 & f, const uint32 b, const uint32 b_inv, const int b_s)
	{
		const uint96 t = int96_abs(f);
		const uint64 t_h = (uint64(t.s1) << (64 - 29)) | (t.s0 >> 29);
		const uint32 t_l = uint32(t.s0) & ((uint32(1) << 29) - 1);

		uint32 d_h, r_h = barrett(t_h, b, b_inv, b_s, d_h);
		uint32 d_l, r_l = barrett((uint64(r_h) << 29) | t_l, b, b_inv, b_s, d_l);
		const uint64 d = (uint64(d_h) << 29) | d_l;

		const bool s = int96_is_neg(f);
		f = int96_set_si(s ? -int64(d) : int64(d));
		const int32 r = s ? -int32(r_l) : int32(r_l);
		return r;
	}

private:
	void normalize2(Zp1 * const x1, Zp2 * const x2) const
	{
		const size_t n = this->_n;
		const Zp1 norm1 = this->_z1.norm;
		const Zp2 norm2 = this->_z2.norm;

		const uint32 * const b = this->_b;
		const uint32 * const b_inv = this->_b_inv;
		const int b_s = this->_b_s;

		int64 f[VSIZE];
		for (size_t i = 0; i < VSIZE; ++i) f[i] = 0;

		for (size_t k = 0; k < VSIZE * n; ++k)
		{
			const size_t i = k % VSIZE;
			f[i] += garner2(x1[k] * norm1, x2[k] * norm2);
			const int32 r = reduce64(f[i], b[i], b_inv[i], b_s);
			x1[k] = Zp1(r); x2[k] = Zp2(r);
		}

		for (size_t i = 0; i < VSIZE; ++i)
		{
			while (f[i] != 0)
			{
				f[i] = -f[i];		// a_0 = -a_n
				for (size_t j = 0; j < n; ++j)
				{
					const size_t k = VSIZE * j + i;
					f[i] += x1[k].geti();
					const int32 r = reduce64(f[i], b[i], b_inv[i], b_s);
					x1[k] = Zp1(r); x2[k] = Zp2(r);
					if (f[i] == 0) break;
				}
			}
		}
	}

private:
	void normalize3(Zp1 * const x1, Zp2 * const x2, Zp3 * const x3) const
	{
		const size_t n = this->_n;
		const Zp1 norm1 = this->_z1.norm;
		const Zp2 norm2 = this->_z2.norm;
		const Zp3 norm3 = this->_z3.norm;

		const uint32 * const b = this->_b;
		const uint32 * const b_inv = this->_b_inv;
		const int b_s = this->_b_s;

		int96 f96[VSIZE];
		for (size_t i = 0; i < VSIZE; ++i) f96[i] = int96_set_si(0);

		for (size_t k = 0; k < VSIZE * n; ++k)
		{
			const size_t i = k % VSIZE;
			f96[i] = int96_add(f96[i], garner3(x1[k] * norm1, x2[k] * norm2, x3[k] * norm3));
			const int32 r = reduce96(f96[i], b[i], b_inv[i], b_s);
			x1[k] = Zp1(r); x2[k] = Zp2(r); x3[k] = Zp3(r);
		}

		int64 f[VSIZE];
		for (size_t i = 0; i < VSIZE; ++i) f[i] = int64(f96[i].s0);

		for (size_t i = 0; i < VSIZE; ++i)
		{
			while (f[i] != 0)
			{
				f[i] = -f[i];		// a_0 = -a_n
				for (size_t j = 0; j < n; ++j)
				{
					const size_t k = VSIZE * j + i;
					f[i] += x1[k].geti();
					const int32 r = reduce64(f[i], b[i], b_inv[i], b_s);
					x1[k] = Zp1(r); x2[k] = Zp2(r); x3[k] = Zp3(r);
					if (f[i] == 0) break;
				}
			}
		}
	}

private:
	void initMultiplicand() const
	{
		const size_t n = this->_n;
		const Szp<Zp1> & z1 = this->_z1;
		_engine.writeMemory_x1((uint32 *)z1.x);
		_engine.setxy_P1();
		for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2y_P1(s, lm);
		_engine.readMemory_y1((uint32 *)z1.y);
		const Szp<Zp2> & z2 = this->_z2;
		_engine.writeMemory_x2((uint32 *)z2.x);
		_engine.setxy_P2();
		for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2y_P2(s, lm);
		_engine.readMemory_y2((uint32 *)z2.y);
		if (this->_3primes)
		{
			const Szp<Zp3> & z3 = this->_z3;
			_engine.writeMemory_x3((uint32 *)z3.x);
			_engine.setxy_P3();
			for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2y_P3(s, lm);
			_engine.readMemory_y3((uint32 *)z3.y);
		}
	}

private:
	void squareMod() const
	{
		const size_t n = this->_n;

		const Szp<Zp1> & z1 = this->_z1;
		_engine.writeMemory_x1((uint32 *)z1.x);
		for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2x_P1(s, lm);
		_engine.square2_P1();
		for (size_t lm = 0, s = n / 2; s > 0; ++lm, s /= 2) _engine.backward2x_P1(s, lm);
		_engine.readMemory_x1((uint32 *)z1.x);

		const Szp<Zp2> & z2 = this->_z2;
		_engine.writeMemory_x2((uint32 *)z2.x);
		for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2x_P2(s, lm);
		_engine.square2_P2();
		for (size_t lm = 0, s = n / 2; s > 0; ++lm, s /= 2) _engine.backward2x_P2(s, lm);
		_engine.readMemory_x2((uint32 *)z2.x);

		if (this->_3primes)
		{
			const Szp<Zp3> & z3 = this->_z3;
			_engine.writeMemory_x3((uint32 *)z3.x);
			for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2x_P3(s, lm);
			_engine.square2_P3();
			for (size_t lm = 0, s = n / 2; s > 0; ++lm, s /= 2) _engine.backward2x_P3(s, lm);
			_engine.readMemory_x3((uint32 *)z3.x);

			normalize3(z1.x, z2.x, z3.x);
		}
		else normalize2(z1.x, z2.x);
	}

private:
	void mulMod(const uint64 c) const
	{
		const size_t n = this->_n;

		const Szp<Zp1> & z1 = this->_z1;
		_engine.writeMemory_x1((uint32 *)z1.x);
		_engine.writeMemory_y1((uint32 *)z1.y);
		for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2x_P1(s, lm);
		_engine.mul2condxy_P1(c);
		for (size_t lm = 0, s = n / 2; s > 0; ++lm, s /= 2) _engine.backward2x_P1(s, lm);
		_engine.readMemory_x1((uint32 *)z1.x);

		const Szp<Zp2> & z2 = this->_z2;
		_engine.writeMemory_x2((uint32 *)z2.x);
		_engine.writeMemory_y2((uint32 *)z2.y);
		for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2x_P2(s, lm);
		_engine.mul2condxy_P2(c);
		for (size_t lm = 0, s = n / 2; s > 0; ++lm, s /= 2) _engine.backward2x_P2(s, lm);
		_engine.readMemory_x2((uint32 *)z2.x);
		if (this->_3primes)
		{
			const Szp<Zp3> & z3 = this->_z3;
			_engine.writeMemory_x3((uint32 *)z3.x);
			_engine.writeMemory_y3((uint32 *)z3.y);
			for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2x_P3(s, lm);
			_engine.mul2condxy_P3(c);
			for (size_t lm = 0, s = n / 2; s > 0; ++lm, s /= 2) _engine.backward2x_P3(s, lm);
			_engine.readMemory_x3((uint32 *)z3.x);

			normalize3(z1.x, z2.x, z3.x);
		}
		else normalize2(z1.x, z2.x);
	}

private:
	bool readOpenCL(const char * const clFileName, const char * const headerFileName, const char * const varName, std::stringstream & src) const
	{
		if (_isBoinc) return false;

		std::ifstream clFile(clFileName);
		if (!clFile.is_open()) return false;
		
		// if .cl file exists then generate header file
		std::ofstream hFile(headerFileName, std::ios::binary);	// binary: don't convert line endings to `CRLF` 
		if (!hFile.is_open()) throw std::runtime_error("cannot write openCL header file");

		hFile << "/*" << std::endl;
		hFile << "Copyright 2020, Yves Gallot" << std::endl << std::endl;
		hFile << "genefer20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it." << std::endl;
		hFile << "Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful." << std::endl;
		hFile << "*/" << std::endl << std::endl;

		hFile << "#pragma once" << std::endl << std::endl;
		hFile << "#include <cstdint>" << std::endl << std::endl;

		hFile << "static const char * const " << varName << " = \\" << std::endl;

		std::string line;
		while (std::getline(clFile, line))
		{
			hFile << "\"";
			for (char c : line)
			{
				if ((c == '\\') || (c == '\"')) hFile << '\\';
				hFile << c;
			}
			hFile << "\\n\" \\" << std::endl;

			src << line << std::endl;
		}
		hFile << "\"\";" << std::endl;

		hFile.close();
		clFile.close();
		return true;
	}

private:
	void initEngine()
	{
		std::stringstream src;
		src << "#define\tVSIZE\t" << VSIZE << std::endl << std::endl;
		src << std::endl;

		// if xxx.cl file is not found then source is src_ocl_xxx string in src/ocl/xxx.h
		if (!readOpenCL("ocl/kernel.cl", "src/ocl/kernel.h", "src_ocl_kernel", src)) src << src_ocl_kernel;

		_engine.loadProgram(src.str());
		_engine.allocMemory(this->_n);
		_engine.createKernels();
	}

private:
	void clearEngine()
	{
		_engine.releaseKernels();
		_engine.releaseMemory();
		_engine.clearProgram();
	}

public:
	transform(const uint32_t size, engine & engine, const bool isBoinc) :
		_n(size), _isBoinc(isBoinc), _engine(engine), _z1(size), _z2(size), _z3(size), _x(new uint32[VSIZE * size])
	{
		initEngine();

		const size_t n = this->_n;
		std::vector<uint32> wr(n), wri(n);
		create<Zp1>(n, wr, wri);
		_engine.writeMemory_w1(wr.data(), wri.data());
		create<Zp2>(n, wr, wri);
		_engine.writeMemory_w2(wr.data(), wri.data());
		create<Zp3>(n, wr, wri);
		_engine.writeMemory_w3(wr.data(), wri.data());
	}

public:
	virtual ~transform()
	{
		delete[] _x;

		clearEngine();
	}

public:
	int getError() const
	{
		return 0;
	}

public:
	void init(const vint32 & b, const uint32_t a)
	{
		const size_t n = this->_n;
		const uint32_t bmax = b[VSIZE - 1];
		this->_3primes = (bmax * uint64_t(bmax) >= (P1P2 / (2 * n)));

 		this->_lgb = ilog2(bmax);
		const int s = _lgb - 1;
		this->_b_s = s;	// TODO merge _lgb & _b_s

		for (size_t i = 0; i < VSIZE; ++i)
		{
			this->_b[i] = b[i];
			this->_b_inv[i] = uint32_t((uint64_t(1) << (s + 32)) / b[i]);
		}

		_engine.reset_P1(a);
		_engine.reset_P2(a);
		_engine.readMemory_x1((uint32 *)this->_z1.x);
		_engine.readMemory_d1((uint32 *)this->_z1.d);
		_engine.readMemory_x2((uint32 *)this->_z2.x);
		_engine.readMemory_d2((uint32 *)this->_z2.d);
		if (this->_3primes)
		{
			_engine.reset_P3(a);
			_engine.readMemory_x3((uint32 *)this->_z3.x);
			_engine.readMemory_d3((uint32 *)this->_z3.d);
		}
	}

public:
	void powMod() const
	{
		initMultiplicand();
		const uint32 * const b = this->_b;
		for (int j = this->_lgb - 1; j >= 0; --j)
		{
			squareMod();
			uint64 c = 0;
			for (size_t i = 0; i < VSIZE; ++i)
			{
				const uint64 ci = ((b[i] & (uint32(1) << j)) != 0) ? 1 : 0;
				c |= ci << i;
			}
			mulMod(c);
		}
	}

public:
	void gerbiczStep() const
	{
		const size_t n = this->_n;

		const Szp<Zp1> & z1 = this->_z1;
		_engine.writeMemory_d1((uint32 *)z1.d);
		_engine.writeMemory_x1((uint32 *)z1.x);
		for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2d_P1(s, lm);
		_engine.setxy_P1();
		for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2y_P1(s, lm);
		_engine.mul2dy_P1();
		for (size_t lm = 0, s = n / 2; s > 0; ++lm, s /= 2) _engine.backward2d_P1(s, lm);
		_engine.readMemory_d1((uint32 *)z1.d);

		const Szp<Zp2> & z2 = this->_z2;
		_engine.writeMemory_d2((uint32 *)z2.d);
		_engine.writeMemory_x2((uint32 *)z2.x);
		for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2d_P2(s, lm);
		_engine.setxy_P2();
		for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2y_P2(s, lm);
		_engine.mul2dy_P2();
		for (size_t lm = 0, s = n / 2; s > 0; ++lm, s /= 2) _engine.backward2d_P2(s, lm);
		_engine.readMemory_d2((uint32 *)z2.d);

		if (this->_3primes)
		{
			const Szp<Zp3> & z3 = this->_z3;
			_engine.writeMemory_d3((uint32 *)z3.d);
			_engine.writeMemory_x3((uint32 *)z3.x);
			for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2d_P3(s, lm);
			_engine.setxy_P3();
			for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2y_P3(s, lm);
			_engine.mul2dy_P3();
			for (size_t lm = 0, s = n / 2; s > 0; ++lm, s /= 2) _engine.backward2d_P3(s, lm);
			_engine.readMemory_d3((uint32 *)z3.d);

			normalize3(z1.d, z2.d, z3.d);
		}
		else normalize2(z1.d, z2.d);
	}

public:
	void gerbiczLastStep() const
	{
		const size_t n = this->_n;

		const Szp<Zp1> & z1 = this->_z1;
		_engine.writeMemory_x1((uint32 *)z1.x);
		_engine.writeMemory_d1((uint32 *)z1.d);
		for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2x_P1(s, lm);
		_engine.setdy_P1();
		for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2y_P1(s, lm);
		_engine.mul2xy_P1();
		for (size_t lm = 0, s = n / 2; s > 0; ++lm, s /= 2) _engine.backward2x_P1(s, lm);
		_engine.swap_xd_P1();
		_engine.readMemory_x1((uint32 *)z1.x);
		_engine.readMemory_d1((uint32 *)z1.d);

		const Szp<Zp2> & z2 = this->_z2;
		_engine.writeMemory_x2((uint32 *)z2.x);
		_engine.writeMemory_d2((uint32 *)z2.d);
		for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2x_P2(s, lm);
		_engine.setdy_P2();
		for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2y_P2(s, lm);
		_engine.mul2xy_P2();
		for (size_t lm = 0, s = n / 2; s > 0; ++lm, s /= 2) _engine.backward2x_P2(s, lm);
		_engine.swap_xd_P2();
		_engine.readMemory_x2((uint32 *)z2.x);
		_engine.readMemory_d2((uint32 *)z2.d);

		if (this->_3primes)
		{
			const Szp<Zp3> & z3 = this->_z3;
			_engine.writeMemory_x3((uint32 *)z3.x);
			_engine.writeMemory_d3((uint32 *)z3.d);
			for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2x_P3(s, lm);
			_engine.setdy_P3();
			for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2y_P3(s, lm);
			_engine.mul2xy_P3();
			for (size_t lm = 0, s = n / 2; s > 0; ++lm, s /= 2) _engine.backward2x_P3(s, lm);
			_engine.swap_xd_P3();
			_engine.readMemory_x3((uint32 *)z3.x);
			_engine.readMemory_d3((uint32 *)z3.d);

			normalize3(z1.d, z2.d, z3.d);
		}
		else normalize2(z1.d, z2.d);
	}

public:
	void isPrime(bool prm[VSIZE], uint64_t res[VSIZE], uint64_t res64[VSIZE]) const
	{
		const size_t n = this->_n;
		const Zp1 * const x1 = this->_z1.x;
		uint32 * const x = this->_x;

		const uint32 * const b = this->_b;
		const uint32 * const b_inv = this->_b_inv;
		const int b_s = this->_b_s;

		int64 f[VSIZE];
		for (size_t i = 0; i < VSIZE; ++i) f[i] = 0;

		for (size_t k = 0; k < VSIZE * n; ++k)
		{
			const size_t i = k % VSIZE;
			f[i] += x1[k].geti();
			int32 r = reduce64(f[i], b[i], b_inv[i], b_s);
			if (r < 0) { r += b[i]; f[i] -= 1; }
			x[k] = uint32(r);
		}

		for (size_t i = 0; i < VSIZE; ++i)
		{
			while (f[i] != 0)
			{
				f[i] = -f[i];		// a_0 = -a_n
				for (size_t j = 0; j < n; ++j)
				{
					const size_t k = VSIZE * j + i;
					f[i] += x[k];
					int32 r = reduce64(f[i], b[i], b_inv[i], b_s);
					if (r < 0) { r += b[i]; f[i] -= 1; }
					x[k] = uint32(r);
					if (f[i] == 0) break;
				}
			}

			uint64_t r = 0;
			for (size_t j = 8; j > 0; --j)
			{
				r = (r << 8) | uint8_t(x[VSIZE * (n - j) + i]);
			}
			res[i] = r;

			const uint32 x0 = x[i];
			bool isPrime = (x0 == 1);
			uint64_t r64 = x0;
			uint64 bi = b[i];
			for (size_t j = 1; j < n; ++j)
			{
				const size_t k = VSIZE * j + i;
				const uint32 xk = x[k];
				isPrime &= (xk == 0);
				r64 += xk * bi;
				bi *= b[i];
			}
			res64[i] = r64;

			prm[i] = isPrime;
		}
	}

public:
	bool gerbiczCheck(const uint32_t a) const
	{
		const size_t n = this->_n;
		const Zp1 * const x1 = this->_z1.x;
		const Zp1 * const d1 = this->_z1.d;
		uint32 * const x = this->_x;

		const uint32 * const b = this->_b;
		const uint32 * const b_inv = this->_b_inv;
		const int b_s = this->_b_s;

		int64 f[VSIZE];
		for (size_t i = 0; i < VSIZE; ++i) f[i] = 0;

		for (size_t k = 0; k < VSIZE * n; ++k)
		{
			const size_t i = k % VSIZE;
			const int64 e = x1[k].geti() * int64(a) - d1[k].geti();
			f[i] += e;
			int32 r = reduce64(f[i], b[i], b_inv[i], b_s);
			if (r < 0) { r += b[i]; f[i] -= 1; }
			x[k] = uint32(r);
		}

		for (size_t i = 0; i < VSIZE; ++i)
		{
			while (f[i] != 0)
			{
				f[i] = -f[i];		// a_0 = -a_n
				for (size_t j = 0; j < n; ++j)
				{
					const size_t k = VSIZE * j + i;
					f[i] += x[k];
					int32 r = reduce64(f[i], b[i], b_inv[i], b_s);
					if (r < 0) { r += b[i]; f[i] -= 1; }
					x[k] = uint32(r);
					if (f[i] == 0) break;
				}
			}
		}

		for (size_t k = 0; k < VSIZE * n; ++k) if (x[k] != 0) return false;
		return true;
	}

// private:
// 	static bool _writeContext(FILE * const cFile, const char * const ptr, const size_t size)
// 	{
// 		const size_t ret = std::fwrite(ptr , sizeof(char), size, cFile);
// 		if (ret == size * sizeof(char)) return true;
// 		std::fclose(cFile);
// 		return false;
// 	}

// private:
// 	static bool _readContext(FILE * const cFile, char * const ptr, const size_t size)
// 	{
// 		const size_t ret = std::fread(ptr , sizeof(char), size, cFile);
// 		if (ret == size * sizeof(char)) return true;
// 		std::fclose(cFile);
// 		return false;
// 	}

// private:
// 	static std::string _filename(const char * const ext)
// 	{
// 		return std::string("genefer_") + std::string(ext) + std::string(".ctx");
// 	}

// public:
// 	bool saveContext(const uint32_t i, const double elapsedTime, const char * const ext)
// 	{
// 		FILE * const cFile = pio::open(_filename(ext).c_str(), "wb");
// 		if (cFile == nullptr)
// 		{
// 			std::ostringstream ss; ss << "cannot write 'genefer.ctx' file " << std::endl;
// 			pio::error(ss.str());
// 			return false;
// 		}

// 		const size_t size = _size;

// 		const uint32_t version = 0;
// 		if (!_writeContext(cFile, reinterpret_cast<const char *>(&version), sizeof(version))) return false;

// 		std::fclose(cFile);
// 		return true;
// 	}

// public:
// 	bool restoreContext(uint32_t & i, double & elapsedTime, const char * const ext, const bool restore_uv = true)
// 	{
// 		FILE * const cFile = pio::open(_filename(ext).c_str(), "rb");
// 		if (cFile == nullptr) return false;

// 		const size_t size = _size;

// 		uint32_t version = 0;
// 		if (!_readContext(cFile, reinterpret_cast<char *>(&version), sizeof(version))) return false;
// 		if (version != 0) return false;

// 		std::fclose(cFile);
// 		return true;
// 	}
};
