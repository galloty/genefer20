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

private:
	const size_t _n;
	const bool _isBoinc;
	engine & _engine;
	uint32 * const _x;
	uint32_2 * const _gx;
	uint32_2 * const _gd;
	bool _3primes = true;
	int _b_s = 0;
	uint32 _b[VSIZE];
	uint64 _c[32];

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
	static uint32 reduce32(int32 & f, const uint32 b)
	{
		int32 d = 0;
		if (f < 0) { f += b; --d; }
		else if (uint32(f) >= b) { f -= b; ++d; }
		const uint32 r = uint32(f);
		f = d;
		return r;
	}

private:
	static uint32 reduce64(int64 & f, const uint32 b)
	{
		int32 d = 0;
		if (f < 0)
		{
			do { f += b; --d; } while (f < 0);
		}
		else
		{
			while (f >= b) { f -= b; ++d; }
		}
		const uint32 r = uint32(f);
		f = d;
		return r;
	}

private:
	void initMultiplicand() const
	{
		const size_t n = this->_n;

		if (this->_3primes)
		{
			_engine.setxy_P123();
			for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2y_P12(s, lm);
			for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2y_P3(s, lm);
		}
		else
		{
			_engine.setxy_P12();
			for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2y_P12(s, lm);
		}
	}

private:
	void squareMod() const
	{
		const size_t n = this->_n;
		const int ln = ilog2(uint32_t(n));
		const bool odd = (ln % 2 != 0);

		if (this->_3primes)
		{
			for (size_t lm = ln - 1, s = 1; s <= n / 2; --lm, s *= 2) _engine.forward2x_P12(s, lm);
			for (size_t lm = ln - 2, s = 1; s <= n / 4; lm -= 2, s *= 4) _engine.forward4x_P3(s, lm);
			if (odd) _engine.forward2x_P3(n / 2, 0);
			_engine.square2_P12();
			_engine.square2_P3();
			for (size_t lm = 0, s = n / 2; s > 0; ++lm, s /= 2) _engine.backward2x_P12(s, lm);
			if (odd) _engine.backward2x_P3(n / 2, 0);
			for (size_t lm = odd ? 1 : 0, s = odd ? n / 8 : n / 4; s > 0; lm += 2, s /= 4) _engine.backward4x_P3(s, lm);
			_engine.normalize3ax();
			_engine.normalize3bx();
		}
		else
		{
			for (size_t lm = ln - 1, s = 1; s < n; --lm, s *= 2) _engine.forward2x_P12(s, lm);
			_engine.square2_P12();
			for (size_t lm = 0, s = n / 2; s > 0; ++lm, s /= 2) _engine.backward2x_P12(s, lm);
			_engine.normalize2ax();
			_engine.normalize2bx();
		}
	}

private:
	void mulMod(const uint64 c) const
	{
		const size_t n = this->_n;

		for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2x_P12(s, lm);
		_engine.mul2condxy_P12(c);
		for (size_t lm = 0, s = n / 2; s > 0; ++lm, s /= 2) _engine.backward2x_P12(s, lm);

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
	transform(const size_t size, engine & engine, const bool isBoinc) :
		_n(size), _isBoinc(isBoinc), _engine(engine),
		_x(new uint32[VSIZE * size]), _gx(new uint32_2[VSIZE * size]), _gd(new uint32_2[VSIZE * size])
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

		std::vector<uint32> wr1(n), wr2(n), wr3(n), wri1(n), wri2(n), wri3(n);
		create<Zp1>(n, wr1, wri1); create<Zp2>(n, wr2, wri2); create<Zp3>(n, wr3, wri3);

		std::vector<uint32_2> wr12(n), wri12(n);
		for (size_t i = 0; i < n; ++i) { wr12[i].s[0] = wr1[i]; wr12[i].s[1] = wr2[i]; }
		for (size_t i = 0; i < n; ++i) { wri12[i].s[0] = wri1[i]; wri12[i].s[1] = wri2[i]; }
		_engine.writeMemory_w(wr12.data(), wr3.data(), wri12.data(), wri3.data());
	}

public:
	virtual ~transform()
	{
		delete[] _x;
		delete[] _gx;
		delete[] _gd;

		_engine.releaseKernels();
		_engine.releaseMemory();
		_engine.clearProgram();
	}

public:
	void copyRes() { _engine.setxres_P1(); }

public:
	void saveRes()
	{
		_engine.readMemory_res(this->_x);
		_engine.readMemory_x12(this->_gx);
		_engine.readMemory_d12(this->_gd);
	}

public:
	void init(const vint32 & b, const uint32_t a)
	{
		const size_t n = this->_n;
		const uint32_t bmax = b[VSIZE - 1];
		this->_3primes = (bmax * uint64_t(bmax) >= (P1P2 / (2 * n)));

		const int32 s = ilog2(bmax) - 1;
		this->_b_s = s;

		uint32_2 bb_inv[VSIZE];
		for (size_t i = 0; i < VSIZE; ++i)
		{
			this->_b[i] = b[i];
			bb_inv[i].s[0] = b[i];
			bb_inv[i].s[1] = uint32((uint64(1) << (s + 32)) / b[i]);
		}

		uint64 * const c = this->_c;
		for (int j = s; j >= 0; --j)
		{
			uint64 cj = 0;
			for (size_t i = 0; i < VSIZE; ++i)
			{
				const uint64 ci = ((b[i] & (uint32(1) << j)) != 0) ? 1 : 0;
				cj |= ci << i;
			}
			c[j] = cj;
		}

		_engine.writeMemory_b(bb_inv);
		_engine.setParam_bs(s);

		_engine.reset_P12(a);
		if (this->_3primes) _engine.reset_P3(a);
	}

public:
	void powMod() const
	{
		initMultiplicand();

		const uint64 * const c = this->_c;

		for (int j = this->_b_s; j >= 0; --j)
		{
			squareMod();
			const uint64 cj = c[j];
			if (cj != 0) mulMod(cj);
		}
	}

public:
	void gerbiczStep() const
	{
		const size_t n = this->_n;

		if (this->_3primes)
		{
			for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2d_P12(s, lm);
			for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2d_P3(s, lm);
			_engine.setxy_P123();
			for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2y_P12(s, lm);
			for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2y_P3(s, lm);
			_engine.mul2dy_P12();
			_engine.mul2dy_P3();
			for (size_t lm = 0, s = n / 2; s > 0; ++lm, s /= 2) _engine.backward2d_P12(s, lm);
			for (size_t lm = 0, s = n / 2; s > 0; ++lm, s /= 2) _engine.backward2d_P3(s, lm);
			_engine.normalize3ad();
			_engine.normalize3bd();
		}
		else
		{
			for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2d_P12(s, lm);
			_engine.setxy_P12();
			for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2y_P12(s, lm);
			_engine.mul2dy_P12();
			for (size_t lm = 0, s = n / 2; s > 0; ++lm, s /= 2) _engine.backward2d_P12(s, lm);
			_engine.normalize2ad();
			_engine.normalize2bd();
		}
	}

public:
	void gerbiczLastStep() const
	{
		const size_t n = this->_n;

		if (this->_3primes)
		{
			for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2x_P12(s, lm);
			for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2x_P3(s, lm);
			_engine.setdy_P123();
			for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2y_P12(s, lm);
			for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2y_P3(s, lm);
			_engine.mul2xy_P12();
			_engine.mul2xy_P3();
			for (size_t lm = 0, s = n / 2; s > 0; ++lm, s /= 2) _engine.backward2x_P12(s, lm);
			for (size_t lm = 0, s = n / 2; s > 0; ++lm, s /= 2) _engine.backward2x_P3(s, lm);
			_engine.swap_xd_P123();
			_engine.normalize3ad();
			_engine.normalize3bd();
		}
		else
		{
			for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2x_P12(s, lm);
			_engine.setdy_P12();
			for (size_t lm = ilog2(n / 2), s = 1; s < n; --lm, s *= 2) _engine.forward2y_P12(s, lm);
			_engine.mul2xy_P12();
			for (size_t lm = 0, s = n / 2; s > 0; ++lm, s /= 2) _engine.backward2x_P12(s, lm);
			_engine.swap_xd_P12();
			_engine.normalize2ad();
			_engine.normalize2bd();
		}
	}

public:
	bool gerbiczCheck(const uint32_t a) const
	{
		const size_t n = this->_n;
		const uint32_2 * const x = this->_gx;
		uint32_2 * const d = this->_gd;
		const uint32 * const b = this->_b;

		int32 f[VSIZE];
		for (size_t i = 0; i < VSIZE; ++i) f[i] = 0;

		for (size_t j = 0; j < n; ++j)
		{
			for (size_t i = 0; i < VSIZE; ++i)
			{
				const size_t k = j * VSIZE + i;
				const uint32 xk = x[k].s[0], dk = d[k].s[0];
				const uint32 cxk = (xk > P1 / 2) ? P1 : 0, cdk = (dk > P1 / 2) ? P1 : 0;
				const int32 ixk = int32(xk - cxk), idk = int32(dk - cdk);
				int64 e = f[i] + ixk * int64(a) - idk;
				d[k].s[0] = reduce64(e, b[i]);
				f[i] = int32(e);	// |e| <= 3
			}
		}

		for (size_t i = 0; i < VSIZE; ++i)
		{
			int32 e = f[i];
			while (e != 0)
			{
				e = -e;		// a_0 = -a_n
				for (size_t j = 0; j < n; ++j)
				{
					const size_t k = j * VSIZE + i;
					e += int32(d[k].s[0]);
					d[k].s[0] = reduce32(e, b[i]);
					if (e == 0) break;
				}
			}
		}

		for (size_t k = 0; k < VSIZE * n; ++k) if (d[k].s[0] != 0) return false;

		return true;
	}

public:
	void isPrime(bool prm[VSIZE], uint64_t res[VSIZE], uint64_t res64[VSIZE]) const
	{
		const size_t n = this->_n;
		uint32 * const x = this->_x;
		const uint32 * const b = this->_b;

		int32 f[VSIZE];
		for (size_t i = 0; i < VSIZE; ++i) f[i] = 0;

		for (size_t j = 0; j < n; ++j)
		{
			for (size_t i = 0; i < VSIZE; ++i)
			{
				const size_t k = j * VSIZE + i;
				const uint32 xk = x[k];
				const int32 ixk = (xk > P1 / 2) ? int32(xk - P1) : int32(xk);
				int64 e = f[i] + int64(ixk);
				x[k] = reduce64(e, b[i]);
				f[i] = int32(e);
			}
		}

		for (size_t i = 0; i < VSIZE; ++i)
		{
			int32 e = f[i];
			while (e != 0)
			{
				e = -e;		// a_0 = -a_n
				for (size_t j = 0; j < n; ++j)
				{
					const size_t k = j * VSIZE + i;
					e += int32(x[k]);
					x[k] = reduce32(e, b[i]);
					if (e == 0) break;
				}
			}

			uint64_t r = 0;
			for (size_t j = 8; j > 0; --j)
			{
				r = (r << 8) | uint8_t(x[(n - j) * VSIZE + i]);
			}
			res[i] = r;

			prm[i] = (x[i] == 1);
		}

		uint64_t bi[VSIZE];
		for (size_t i = 0; i < VSIZE; ++i) bi[i] = b[i];

		for (size_t i = 0; i < VSIZE; ++i) res64[i] = x[i];

		for (size_t j = 1; j < n; ++j)
		{
			for (size_t i = 0; i < VSIZE; ++i)
			{
				const size_t k = j * VSIZE + i;
				const uint32 xk = x[k];
				prm[i] &= (xk == 0);
				res64[i] += xk * bi[i];
				bi[i] *= b[i];
			}
		}
	}
};
