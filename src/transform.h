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
	static const uint64_t P1P2 = (P1 * uint64_t(P2));

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
		Zp * const d;

		Szp(const size_t n) : x(new Zp[VSIZE * n]), y(new Zp[VSIZE * n]), d(new Zp[VSIZE * n]) {}
		virtual ~Szp() 
		{
			delete[] x;
			delete[] y;
			delete[] d;
		}
	};

private:
	const size_t _n;
	const bool _isBoinc;
	engine & _engine;
	const Szp<Zp1> _z1;
	uint32 * const _x;
	bool _3primes = true;
	int _lgb = 0;
	int _b_s = 0;
	uint32_2 _bb_inv[VSIZE];

private:
	static size_t bitRev(const size_t i, const size_t n)
	{
		size_t r = 0;
		for (size_t k = n, j = i; k > 1; k /= 2, j /= 2) r = (2 * r) | (j % 2);
		return r;
	}

private:
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

private:
	static int32 reduce64f(int64 & f, const uint32 b)
	{
		uint64 t = std::abs(f);
		int32 d = 0;
		while (t >= b) { t -= b; ++d; }
		const bool s = (f < 0);
		f = s ? -d : d;
		return s ? -int32(t) : int32(t);
	}

private:
	void initMultiplicand() const
	{
		const size_t n = this->_n;
		_engine.setxy_P1();
		for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2y_P1(s, lm);
		_engine.setxy_P2();
		for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2y_P2(s, lm);
		if (this->_3primes)
		{
			_engine.setxy_P3();
			for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2y_P3(s, lm);
		}
	}

private:
	void squareMod() const
	{
		const size_t n = this->_n;

		for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2x_P1(s, lm);
		_engine.square2_P1();
		for (size_t lm = 0, s = n / 2; s > 0; ++lm, s /= 2) _engine.backward2x_P1(s, lm);

		for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2x_P2(s, lm);
		_engine.square2_P2();
		for (size_t lm = 0, s = n / 2; s > 0; ++lm, s /= 2) _engine.backward2x_P2(s, lm);

		if (this->_3primes)
		{
			for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2x_P3(s, lm);
			_engine.square2_P3();
			for (size_t lm = 0, s = n / 2; s > 0; ++lm, s /= 2) _engine.backward2x_P3(s, lm);

			_engine.normalize3ax();
			_engine.normalize3bx();
		}
		else
		{
			_engine.normalize2ax();
			_engine.normalize2bx();
		}
	}

private:
	void mulMod(const uint64 c) const
	{
		const size_t n = this->_n;

		for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2x_P1(s, lm);
		_engine.mul2condxy_P1(c);
		for (size_t lm = 0, s = n / 2; s > 0; ++lm, s /= 2) _engine.backward2x_P1(s, lm);

		for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2x_P2(s, lm);
		_engine.mul2condxy_P2(c);
		for (size_t lm = 0, s = n / 2; s > 0; ++lm, s /= 2) _engine.backward2x_P2(s, lm);

		if (this->_3primes)
		{
			for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2x_P3(s, lm);
			_engine.mul2condxy_P3(c);
			for (size_t lm = 0, s = n / 2; s > 0; ++lm, s /= 2) _engine.backward2x_P3(s, lm);

			_engine.normalize3ax();
			_engine.normalize3bx();
		}
		else
		{
			_engine.normalize2ax();
			_engine.normalize2bx();
		}
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


public:
	transform(const uint32_t size, engine & engine, const bool isBoinc) :
		_n(size), _isBoinc(isBoinc), _engine(engine), _z1(size), _x(new uint32[VSIZE * size])
	{
		const size_t n = this->_n;

		std::stringstream src;
		src << "#define\tVSIZE\t" << VSIZE << std::endl;
		src << "#define\tCSIZE\t" << CSIZE << std::endl << std::endl;
		src << "#define\tnorm1\t" << Zp1::norm(n).get() << "u" << std::endl;
		src << "#define\tnorm2\t" << Zp2::norm(n).get() << "u" << std::endl;
		src << "#define\tnorm3\t" << Zp3::norm(n).get() << "u" << std::endl;
		src << std::endl;

		// if xxx.cl file is not found then source is src_ocl_xxx string in src/ocl/xxx.h
		if (!readOpenCL("ocl/kernel.cl", "src/ocl/kernel.h", "src_ocl_kernel", src)) src << src_ocl_kernel;

		_engine.loadProgram(src.str());
		_engine.allocMemory(this->_n);
		_engine.createKernels();

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

		_engine.releaseKernels();
		_engine.releaseMemory();
		_engine.clearProgram();
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

		std::array<uint32_2, VSIZE> bb_inv;
		for (size_t i = 0; i < VSIZE; ++i)
		{
			bb_inv[i].s[0] = b[i];
			bb_inv[i].s[1] = uint32((uint64(1) << (s + 32)) / b[i]);
			this->_bb_inv[i] = bb_inv[i];
		}

		_engine.writeMemory_b(bb_inv.data());
		_engine.setParam_bs(s);

		_engine.reset_P1(a);
		_engine.reset_P2(a);
		if (this->_3primes) _engine.reset_P3(a);
	}

public:
	void powMod() const
	{
		initMultiplicand();
		const uint32_2 * const bb_inv = this->_bb_inv;
		for (int j = this->_lgb - 1; j >= 0; --j)
		{
			squareMod();
			uint64 c = 0;
			for (size_t i = 0; i < VSIZE; ++i)
			{
				const uint64 ci = ((bb_inv[i].s[0] & (uint32(1) << j)) != 0) ? 1 : 0;
				c |= ci << i;
			}
			if (c != 0) mulMod(c);
		}
	}

public:
	void gerbiczStep() const
	{
		const size_t n = this->_n;

		for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2d_P1(s, lm);
		_engine.setxy_P1();
		for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2y_P1(s, lm);
		_engine.mul2dy_P1();
		for (size_t lm = 0, s = n / 2; s > 0; ++lm, s /= 2) _engine.backward2d_P1(s, lm);

		for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2d_P2(s, lm);
		_engine.setxy_P2();
		for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2y_P2(s, lm);
		_engine.mul2dy_P2();
		for (size_t lm = 0, s = n / 2; s > 0; ++lm, s /= 2) _engine.backward2d_P2(s, lm);

		if (this->_3primes)
		{
			for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2d_P3(s, lm);
			_engine.setxy_P3();
			for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2y_P3(s, lm);
			_engine.mul2dy_P3();
			for (size_t lm = 0, s = n / 2; s > 0; ++lm, s /= 2) _engine.backward2d_P3(s, lm);

			_engine.normalize3ad();
			_engine.normalize3bd();
		}
		else
		{
			_engine.normalize2ad();
			_engine.normalize2bd();
		}
	}

public:
	void gerbiczLastStep() const
	{
		const size_t n = this->_n;

		for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2x_P1(s, lm);
		_engine.setdy_P1();
		for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2y_P1(s, lm);
		_engine.mul2xy_P1();
		for (size_t lm = 0, s = n / 2; s > 0; ++lm, s /= 2) _engine.backward2x_P1(s, lm);
		_engine.swap_xd_P1();

		for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2x_P2(s, lm);
		_engine.setdy_P2();
		for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2y_P2(s, lm);
		_engine.mul2xy_P2();
		for (size_t lm = 0, s = n / 2; s > 0; ++lm, s /= 2) _engine.backward2x_P2(s, lm);
		_engine.swap_xd_P2();

		if (this->_3primes)
		{
			for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2x_P3(s, lm);
			_engine.setdy_P3();
			for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2y_P3(s, lm);
			_engine.mul2xy_P3();
			for (size_t lm = 0, s = n / 2; s > 0; ++lm, s /= 2) _engine.backward2x_P3(s, lm);
			_engine.swap_xd_P3();

			_engine.normalize3ad();
			_engine.normalize3bd();
		}
		else
		{
			_engine.normalize2ad();
			_engine.normalize2bd();
		}
	}

public:
	bool gerbiczCheck(const uint32_t a) const
	{
		const size_t n = this->_n;
		const Zp1 * const x = this->_z1.x;
		Zp1 * const d = this->_z1.d;

		const uint32_2 * const bb_inv = this->_bb_inv;

		_engine.readMemory_x1((uint32 *)this->_z1.x);
		_engine.readMemory_d1((uint32 *)this->_z1.d);

		std::vector<int32> vf1(VSIZE / CSIZE * n);
		// std::vector<int32> vf2(VSIZE / CSIZE * n);
		int32 * f1 = vf1.data();
		// int32 * f2 = vf2.data();

		// int32 e_min = 1000, e_max = -1000;
		for (size_t id = 0; id < VSIZE / CSIZE * n; ++id)
		{
			const size_t i = id % VSIZE, j = id / VSIZE, k0 = j * (VSIZE * CSIZE) + i;
			const uint32 b = bb_inv[i].s[0];
			int64 e = 0;
			for (size_t c = 0; c < CSIZE; ++c)
			{
				const size_t k = k0 + c * VSIZE;
				e += x[k].geti() * int64(a) - d[k].geti();
				int32 r = reduce64f(e, b);
				if (r < 0) { r += b; --e; }	// TODO add to reduce64f
				d[k] = Zp1(r);
			}
			const size_t idn = (id + VSIZE) & ((VSIZE / CSIZE * n) - 1);
			if (idn < VSIZE) e = -e;	// a_0 = -a_n
			f1[idn] = int32(e);			// |e| <= 3
			// if (int32(e) < e_min) e_min = e;
			// if (int32(e) > e_max) e_max = e;
		}
		// std::cout << "e_min = " << e_min << ", e_max = " << e_max << std::endl;

#define	ESIZE	1

		int maxnbloops = 0;
		for (size_t i = 0; i < VSIZE; ++i)
		{
			const uint32 b = bb_inv[i].s[0];
			int32 e[ESIZE];
			for (size_t u = 0; u < ESIZE; ++u)
			{
				int32 eu = 0;
				for (size_t v = 0; v < n / CSIZE / ESIZE; ++v)
				{
					const size_t l = u * (n / CSIZE / ESIZE) + v;
					eu += f1[l * VSIZE + i];
					if (eu != 0)
					{
						for (size_t j = 0; j < CSIZE; ++j)
						{
							const size_t k = (l * CSIZE + j) * VSIZE + i;
							int32 r = eu + d[k].geti();
							eu = 0;
							if (r < 0) { r += b; --eu; }
							else if (uint32(r) >= b) { r -= b; ++eu; }
							d[k] = Zp1(r);
							if (eu == 0) break;
						}
					}
				}
				e[u] = eu;
			}
			int nbloops = 0;
			bool again;
			do
			{
				again = false;
				int32 e2[ESIZE];
				for (size_t u = 0; u < ESIZE; ++u)
				{
					int32 eu = e[(u - 1) & (ESIZE - 1)];
					if (eu == 0) { e2[u] = 0; continue; }
					if (u == 0) eu = -eu;		// a_0 = -a_n
					for (size_t v = 0; v < n / ESIZE; ++v)
					{
						const size_t j = u * (n / ESIZE) + v;
						const size_t k = VSIZE * j + i;
						int32 r = eu + d[k].geti();
						eu = 0;
						if (r < 0) { r += b; --eu; }
						else if (uint32(r) >= b) { r -= b; ++eu; }
						d[k] = Zp1(r);
						if (eu == 0) break;
					}
					e2[u] = eu;
					if (eu != 0) again = true;
				}
				for (size_t u = 0; u < ESIZE; ++u) e[u] = e2[u];
				++nbloops;
			} while (again);
			if (nbloops > maxnbloops) maxnbloops = nbloops;
		}
		std::cout << maxnbloops << " loop(s)" << std::endl;

		// int nbloops = 0;
		// bool again;
		// do
		// {
		// 	again = false;
		// 	for (size_t id = 0; id < VSIZE / CSIZE * n; ++id)
		// 	{
		// 		const size_t i = id % VSIZE, j = id / VSIZE, k0 = j * (VSIZE * CSIZE) + i;
		// 		const uint32 b = bb_inv[i].s[0];
		// 		int64 e = f1[id];
		// 		const size_t idn = (id + VSIZE) & ((VSIZE / CSIZE * n) - 1);
		// 		if (e == 0)
		// 		{
		// 			f2[idn] = 0;
		// 			continue;
		// 		}
		// 		size_t c;
		// 		// for (c = 0; c < CSIZE; ++c)
		// 		for (c = 0; c < CSIZE - 1; ++c)
		// 		{
		// 			const size_t k = k0 + c * VSIZE;
		// 			e += d[k].geti();
		// 			int32 r = reduce64f(e, b);
		// 			if (r < 0) { r += b; --e; }
		// 			d[k] = Zp1(r);
		// 			if (e == 0) break;
		// 		}
		// 		// if (idn < VSIZE) e = -e;	// a_0 = -a_n
		// 		// f2[idn] = int32(e);
		// 		// if (e != 0)
		// 		// {
		// 		// 	if (again == false) ++nbloops;
		// 		// 	again = true;
		// 		// }
		// 		if (e != 0)	//c == CSIZE - 1)
		// 		{
		// 			const size_t k = k0 + c * VSIZE;
		// 			e += d[k].geti(); 
		// 			d[k] = Zp1(int32(e));
		// 			if (e < -1) std::cout << "1: " << k << ": " << e << std::endl;
		// 			if (e > (int)b) std::cout << "1: " << k << ": " << e << std::endl;
		// 		}
		// 	}
		// 	int32 * f = f1; f1 = f2; f2 = f;
		// } while (again);
		// std::cout << nbloops << " loops" << std::endl;

		// nbloops = 0;
		// // bool again;
		// do
		// {
		// 	again = false;
		// 	for (size_t id = 0; id < VSIZE / CSIZE * n; ++id)
		// 	{
		// 		const size_t i = id % VSIZE, j = id / VSIZE, k0 = j * (VSIZE * CSIZE) + i;
		// 		const uint32 b = bb_inv[i].s[0];

		// 		const size_t km1 = (k0 - VSIZE) & (n * VSIZE - 1);	//(j == 0) ? n - 1 : j - 1;
		// 		int64 e = d[km1].geti();
		// 		int32 r = reduce64f(e, b);
		// 		if (r < 0) { r += b; --e; }
		// 		d[km1] = Zp1(r);
		// 		// int32 t = d[km1].geti();
		// 		// if (t < 0) { d[km1] = Zp1(t + b); --e; }
		// 		// else if (uint32(t) >= b) { d[km1] = Zp1(uint32(t) - b); ++e; }
		// 		if (j == 0) e = -e;		// a_0 = -a_n
		// 		if (e == 0) continue;
		// 		// const size_t k0 = VSIZE * j + i;
		// 		size_t c;
		// 		for (c = 0; c < CSIZE - 1; ++c)
		// 		{
		// 			const size_t k = k0 + c * VSIZE;
		// 			e += d[k].geti();
		// 			int32 r = reduce64f(e, b);
		// 			if (r < 0) { r += b; --e; }
		// 			d[k] = Zp1(r);
		// 			if (e == 0) break;
		// 		}
		// 		if (e != 0)	//= CSIZE - 1)
		// 		{
		// 			const size_t k = k0 + c * VSIZE;
		// 			e += d[k].geti(); 
		// 			d[k] = Zp1(int32(e));
		// 			if ((e < 0) || (e >= b))
		// 			{
		// 				if (again == false) ++nbloops;
		// 				again = true;
		// 			}
		// 			// if (t < 0) std::cout << "2: " << k << ": " << t << std::endl;
		// 			// if (t >= b) std::cout << "2: " << k << ": " << t << std::endl;
		// 		}
		// 	}
		// } while (again);
		// std::cout << nbloops << " loops" << std::endl;

		// for (size_t i = 0; i < VSIZE; ++i)
		// {
		// 	const uint32 b = bb_inv[i].s[0];
		// 	int64 e = 0;	//f[i];
		// 	do
		// 	{
		// 		e = -e;		// a_0 = -a_n
		// 		for (size_t j = 0; j < n; ++j)
		// 		{
		// 			const size_t k = VSIZE * j + i;
		// 			e += d[k].geti();
		// 			int32 r = reduce64f(e, b);
		// 			if (r < 0) { r += b; --e; }
		// 			d[k] = Zp1(r);
		// 			// if (d == 0) break;
		// 		}
		// 	} while (e != 0);
		// }

		bool ret = true;
		for (size_t k = 0; k < VSIZE * n; ++k)
			if (d[k].get() != 0)
			{
				ret = false;
			}

		_engine.writeMemory_d1((uint32 *)this->_z1.d);
		return ret;
	}

public:
	void isPrime(bool prm[VSIZE], uint64_t res[VSIZE], uint64_t res64[VSIZE]) const
	{
		const size_t n = this->_n;
		const Zp1 * const x1 = this->_z1.x;
		uint32 * const x = this->_x;

		const uint32_2 * const bb_inv = this->_bb_inv;

		_engine.readMemory_x1((uint32 *)this->_z1.x);

		int64 f[VSIZE];
		for (size_t i = 0; i < VSIZE; ++i) f[i] = 0;

		for (size_t k = 0; k < VSIZE * n; ++k)
		{
			const size_t i = k % VSIZE;
			f[i] += x1[k].geti();
			const uint32 b = bb_inv[i].s[0];
			int32 r = reduce64f(f[i], b);
			if (r < 0) { r += b; f[i] -= 1; }
			x[k] = uint32(r);
		}

		for (size_t i = 0; i < VSIZE; ++i)
		{
			const uint32 b = bb_inv[i].s[0];
			while (f[i] != 0)
			{
				f[i] = -f[i];		// a_0 = -a_n
				for (size_t j = 0; j < n; ++j)
				{
					const size_t k = VSIZE * j + i;
					f[i] += x[k];
					int32 r = reduce64f(f[i], b);
					if (r < 0) { r += b; f[i] -= 1; }
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
			uint64 bi = b;
			for (size_t j = 1; j < n; ++j)
			{
				const size_t k = VSIZE * j + i;
				const uint32 xk = x[k];
				isPrime &= (xk == 0);
				r64 += xk * bi;
				bi *= b;
			}
			res64[i] = r64;

			prm[i] = isPrime;
		}
	}
};
