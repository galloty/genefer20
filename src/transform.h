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

typedef std::array<uint32_t, VSIZE_MAX> vint32;

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
	size_t _vsize = 0;
	size_t _csize = 0;
	bool _radix16 = true;
	bool _3primes = true;
	int _b_s = 0;
	uint32 _b[VSIZE_MAX];
	uint64_4 _c[32];

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
		const int ln = ilog2(uint32_t(n));
		const bool odd = (ln % 2 != 0);

		if (this->_3primes)
		{
			_engine.setxy_P123();
			for (size_t lm = ln - 2, s = 1; s <= n / 4; lm -= 2, s *= 4) _engine.forward4y_P123(s, lm);
			if (odd) _engine.forward2y_P123(n / 2, 0);
		}
		else
		{
			_engine.setxy_P12();
			for (size_t lm = ln - 2, s = 1; s <= n / 4; lm -= 2, s *= 4) _engine.forward4y_P12(s, lm);
			if (odd) _engine.forward2y_P12(n / 2, 0);
		}
	}

private:
	void squareMod() const
	{
		const size_t n = this->_n;
		const int ln = ilog2(uint32_t(n));

		if (this->_3primes)
		{
			size_t s = 1, lm = ln;
			for (; lm > 4; s *= 16, lm -= 4) { _engine.forward16x_P12(s, lm - 4); _engine.forward16x_P3(s, lm - 4); }
			if (lm == 1) _engine.square2x_P123();
			else if (lm == 2) _engine.square4x_P123();
			else if (lm == 3) { _engine.square8x_P12(); _engine.square8x_P3(); }
			else if (lm == 4)  { _engine.square16x_P12(); _engine.square16x_P3(); }
			for (s /= 16, lm += 4; s > 0; s /= 16, lm += 4) { _engine.backward16x_P12(s, lm - 4); _engine.backward16x_P3(s, lm - 4); }
			_engine.normalize3ax();
			_engine.normalize3bx();
		}
		else
		{
			size_t s = 1, lm = ln;
			for (; lm > 4; s *= 16, lm -= 4) _engine.forward16x_P12(s, lm - 4);
			if (lm == 1) _engine.square2x_P12();
			else if (lm == 2) _engine.square4x_P12();
			else if (lm == 3) _engine.square8x_P12();
			else if (lm == 4) _engine.square16x_P12();
			for (s /= 16, lm += 4; s > 0; s /= 16, lm += 4) _engine.backward16x_P12(s, lm - 4);
			_engine.normalize2ax();
			_engine.normalize2bx();
		}
	}

private:
	void mulMod64(const uint64 & c) const
	{
		const size_t n = this->_n;
		const int ln = ilog2(uint32_t(n));

		if (this->_3primes)
		{
			size_t s = 1, lm = ln;
			for (; lm > 4; s *= 16, lm -= 4) { _engine.forward16x_P12(s, lm - 4); _engine.forward16x_P3(s, lm - 4); }
			if (lm > 2) _engine.forward4x_P123(s, lm - 2);
			if (lm % 2 != 0) _engine.mul2cond64xy_P123(c); else _engine.mul4cond64xy_P123(c);
			if (lm > 2) _engine.backward4x_P123(s, lm - 2);
			for (s /= 16, lm += 4; s > 0; s /= 16, lm += 4) { _engine.backward16x_P12(s, lm - 4); _engine.backward16x_P3(s, lm - 4); }
			_engine.normalize3ax();
			_engine.normalize3bx();
		}
		else
		{
			size_t s = 1, lm = ln;
			for (; lm > 4; s *= 16, lm -= 4) _engine.forward16x_P12(s, lm - 4);
			if (lm > 2) _engine.forward4x_P12(s, lm - 2);
			if (lm % 2 != 0) _engine.mul2cond64xy_P12(c); else _engine.mul4cond64xy_P12(c);
			if (lm > 2) _engine.backward4x_P12(s, lm - 2);
			for (s /= 16, lm += 4; s > 0; s /= 16, lm += 4) _engine.backward16x_P12(s, lm - 4);
			_engine.normalize2ax();
			_engine.normalize2bx();
		}
	}

private:
	void mulMod256(const uint64_4 & c) const
	{
		const size_t n = this->_n;
		const int ln = ilog2(uint32_t(n));

		if (this->_3primes)
		{
			size_t s = 1, lm = ln;
			for (; lm > 4; s *= 16, lm -= 4) { _engine.forward16x_P12(s, lm - 4); _engine.forward16x_P3(s, lm - 4); }
			if (lm > 2) _engine.forward4x_P123(s, lm - 2);
			if (lm % 2 != 0) _engine.mul2cond256xy_P123(c); else _engine.mul4cond256xy_P123(c);
			if (lm > 2) _engine.backward4x_P123(s, lm - 2);
			for (s /= 16, lm += 4; s > 0; s /= 16, lm += 4) { _engine.backward16x_P12(s, lm - 4); _engine.backward16x_P3(s, lm - 4); }
			_engine.normalize3ax();
			_engine.normalize3bx();
		}
		else
		{
			size_t s = 1, lm = ln;
			for (; lm > 4; s *= 16, lm -= 4) _engine.forward16x_P12(s, lm - 4);
			if (lm > 2) _engine.forward4x_P12(s, lm - 2);
			if (lm % 2 != 0) _engine.mul2cond256xy_P12(c); else _engine.mul4cond256xy_P12(c);
			if (lm > 2) _engine.backward4x_P12(s, lm - 2);
			for (s /= 16, lm += 4; s > 0; s /= 16, lm += 4) _engine.backward16x_P12(s, lm - 4);
			_engine.normalize2ax();
			_engine.normalize2bx();
		}
	}

private:
	void squareMod_4() const
	{
		const size_t n = this->_n;
		const int ln = ilog2(uint32_t(n));

		if (this->_3primes)
		{
			size_t s = 1, lm = ln;
			for (; lm > 2; s *= 4, lm -= 2) _engine.forward4x_P123(s, lm - 2);
			if (lm % 2 != 0) _engine.square2x_P123(); else _engine.square4x_P123();
			for (s /= 4, lm += 2; s > 0; s /= 4, lm += 2) _engine.backward4x_P123(s, lm - 2);
			_engine.normalize3ax();
			_engine.normalize3bx();
		}
		else
		{
			size_t s = 1, lm = ln;
			for (; lm > 2; s *= 4, lm -= 2) _engine.forward4x_P12(s, lm - 2);
			if (lm % 2 != 0) _engine.square2x_P12(); else _engine.square4x_P12();
			for (s /= 4, lm += 2; s > 0; s /= 4, lm += 2) _engine.backward4x_P12(s, lm - 2);
			_engine.normalize2ax();
			_engine.normalize2bx();
		}
	}

private:
	void mulMod64_4(const uint64 & c) const
	{
		const size_t n = this->_n;
		const int ln = ilog2(uint32_t(n));

		if (this->_3primes)
		{
			size_t s = 1, lm = ln;
			for (; lm > 2; s *= 4, lm -= 2) _engine.forward4x_P123(s, lm - 2);
			if (lm % 2 != 0) _engine.mul2cond64xy_P123(c); else _engine.mul4cond64xy_P123(c);
			for (s /= 4, lm += 2; s > 0; s /= 4, lm += 2) _engine.backward4x_P123(s, lm - 2);
			_engine.normalize3ax();
			_engine.normalize3bx();
		}
		else
		{
			size_t s = 1, lm = ln;
			for (; lm > 2; s *= 4, lm -= 2) _engine.forward4x_P12(s, lm - 2);
			if (lm % 2 != 0) _engine.mul2cond64xy_P12(c); else _engine.mul4cond64xy_P12(c);
			for (s /= 4, lm += 2; s > 0; s /= 4, lm += 2) _engine.backward4x_P12(s, lm - 2);
			_engine.normalize2ax();
			_engine.normalize2bx();
		}
	}

private:
	void mulMod256_4(const uint64_4 & c) const
	{
		const size_t n = this->_n;
		const int ln = ilog2(uint32_t(n));

		if (this->_3primes)
		{
			size_t s = 1, lm = ln;
			for (; lm > 2; s *= 4, lm -= 2) _engine.forward4x_P123(s, lm - 2);
			if (lm % 2 != 0) _engine.mul2cond256xy_P123(c); else _engine.mul4cond256xy_P123(c);
			for (s /= 4, lm += 2; s > 0; s /= 4, lm += 2) _engine.backward4x_P123(s, lm - 2);
			_engine.normalize3ax();
			_engine.normalize3bx();
		}
		else
		{
			size_t s = 1, lm = ln;
			for (; lm > 2; s *= 4, lm -= 2) _engine.forward4x_P12(s, lm - 2);
			if (lm % 2 != 0) _engine.mul2cond256xy_P12(c); else _engine.mul4cond256xy_P12(c);
			for (s /= 4, lm += 2; s > 0; s /= 4, lm += 2) _engine.backward4x_P12(s, lm - 2);
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

private:
	void initEngine() const
	{
		const size_t n = this->_n;
		const size_t vsize = this->_vsize, csize = this->_csize;

		std::stringstream src;
		src << "#define\tVSIZE\t" << vsize << std::endl;
		src << "#define\tCSIZE\t" << csize << std::endl << std::endl;
		src << "#define\tnorm1\t" << Zp1::norm(n).get() << "u" << std::endl;
		src << "#define\tnorm2\t" << Zp2::norm(n).get() << "u" << std::endl;
		src << "#define\tnorm3\t" << Zp3::norm(n).get() << "u" << std::endl;
		src << std::endl;

		// if xxx.cl file is not found then source is src_ocl_xxx string in src/ocl/xxx.h
		if (!readOpenCL("ocl/kernel.cl", "src/ocl/kernel.h", "src_ocl_kernel", src)) src << src_ocl_kernel;

		_engine.loadProgram(src.str());
		_engine.allocMemory(n, vsize, csize);
		_engine.createKernels();

		std::vector<uint32> wr1(n), wr2(n), wr3(n), wri1(n), wri2(n), wri3(n);
		create<Zp1>(n, wr1, wri1); create<Zp2>(n, wr2, wri2); create<Zp3>(n, wr3, wri3);

		std::vector<uint32_2> wr12(n), wri12(n);
		for (size_t i = 0; i < n; ++i) { wr12[i].s[0] = wr1[i]; wr12[i].s[1] = wr2[i]; }
		for (size_t i = 0; i < n; ++i) { wri12[i].s[0] = wri1[i]; wri12[i].s[1] = wri2[i]; }
		_engine.writeMemory_w(wr12.data(), wr3.data(), wri12.data(), wri3.data());
	}

private:
	void releaseEngine() const
	{
		_engine.releaseKernels();
		_engine.releaseMemory();
		_engine.clearProgram();
	}

public:
	transform(const size_t size, engine & engine, const bool isBoinc, const size_t vsize, const size_t csize, const bool radix16) :
		_n(size), _isBoinc(isBoinc), _engine(engine),
		_x(new uint32[VSIZE_MAX * size]), _gx(new uint32_2[VSIZE_MAX * size]), _gd(new uint32_2[VSIZE_MAX * size])
	{
		if (vsize == 0)
		{
			std::cout << " auto-tuning...\r";
			double bestTime = 1e100;
			bool bestRadix16 = true;
			size_t bestCsize = CSIZE_MIN, bestVsize = CSIZE_MIN;

			vint32 b; for (size_t i = 0; i < VSIZE_MAX; ++i) b[i] = 300000000 + 210 * i;

			engine.setProfiling(true);
			for (size_t r = 0; r < 2; ++r)
			{
				const bool radix16 = (r != 0);
				this->_radix16 = radix16;
				for (size_t csize = CSIZE_MIN; csize <= 64; csize *= 2)
				{
					this->_csize = csize;
					//for (size_t vsize = csize; vsize <= VSIZE_MAX; vsize *= 2)
					const size_t vsize = std::min(_engine.getMaxWorkGroupSize() * 4 / 16, size_t(VSIZE_MAX));
					{
						this->_vsize = vsize;

						initEngine();
						init(b, 2);
						engine.resetProfiles();
						const size_t m = 16;
						for (size_t i = 1; i < m; ++i)
						{
							powMod();
							if ((i & (m / 2 - 1)) == 0) gerbiczStep();
						}
						const double time = engine.getProfileTime() / double(vsize);
						powMod(); copyRes(); gerbiczLastStep();
						for (size_t j = 0; j < m / 2; ++j) powMod();
						saveRes();
						if (!gerbiczCheck(2)) throw std::runtime_error("Gerbicz failed");
						releaseEngine();

						std::cout << "radix-" << (radix16 ? 16 : 4) << ", csize = " << csize << ", vsize = " << vsize
								<< ", " << int64_t(time * size * 1e-6 / m) << " ms/b         ";

						if (time < bestTime)
						{
							bestTime = time;
							bestRadix16 = radix16;
							bestCsize = csize;
							bestVsize = vsize;
							std::cout << std::endl;
						}
						else
						{
							std::cout << "\r";
						}
					}
				}
			}

			this->_vsize = bestVsize;
			this->_csize = bestCsize;
			this->_radix16 = bestRadix16;
		}
		else
		{
			this->_vsize = vsize;
			this->_csize = csize;
			this->_radix16 = radix16;
		}

		std::ostringstream ss;
		ss << "Radix = " << (this->_radix16 ? 16 : 4) << ", vector size = " << this->_vsize << ", chunk size = " << this->_csize << "         " << std::endl << std::endl;
		pio::print(ss.str());

		engine.setProfiling(false);
		initEngine();
	}

public:
	virtual ~transform()
	{
		releaseEngine();

		delete[] _x;
		delete[] _gx;
		delete[] _gd;
	}

public:
	size_t getVsize() const { return this->_vsize; }
	size_t getCsize() const { return this->_csize; }
	bool getRadix16() const { return this->_radix16; }

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
		const size_t vsize = this->_vsize;

		const uint32_t bmax = b[vsize - 1];
		this->_3primes = (bmax * uint64_t(bmax) >= (P1P2 / (2 * n)));

		const int32 s = ilog2(bmax) - 1;
		this->_b_s = s;

		std::vector<uint32_2> bb_inv(vsize);
		for (size_t i = 0; i < vsize; ++i)
		{
			this->_b[i] = b[i];
			bb_inv[i].s[0] = b[i];
			bb_inv[i].s[1] = uint32((uint64(1) << (s + 32)) / b[i]);
		}

		uint64_4 * const c = this->_c;
		for (int j = s; j >= 0; --j)
		{
			uint64_4 cj; cj.s[0] = cj.s[1] = cj.s[2] = cj.s[3] = 0;
			for (size_t i = 0; i < vsize; ++i)
			{
				const uint64 ci = ((b[i] & (uint32(1) << j)) != 0) ? 1 : 0;
				cj.s[i / 64] |= ci << i;
			}
			c[j] = cj;
		}

		_engine.writeMemory_b(bb_inv.data());
		_engine.setParam_bs(s);

		if (this->_3primes) _engine.reset_P123(a);
		else _engine.reset_P12(a);
	}

public:
	void powMod() const
	{
		initMultiplicand();

		const bool radix16 = this->_radix16;
		const uint64_4 * const c = this->_c;

		if (radix16)
		{
			if (this->_vsize <= 64)
			{
				for (int j = this->_b_s; j >= 0; --j)
				{
					squareMod();
					const uint64 cj = c[j].s[0];
					if (cj != 0) mulMod64(cj);
				}
			}
			else
			{
				for (int j = this->_b_s; j >= 0; --j)
				{
					squareMod();
					const uint64 cj = c[j].s[0] | c[j].s[1] | c[j].s[2] | c[j].s[3];
					if (cj != 0) mulMod256(c[j]);
				}
			}
		}
		else
		{
			if (this->_vsize <= 64)
			{
				for (int j = this->_b_s; j >= 0; --j)
				{
					squareMod_4();
					const uint64 cj = c[j].s[0];
					if (cj != 0) mulMod64_4(cj);
				}
			}
			else
			{
				for (int j = this->_b_s; j >= 0; --j)
				{
					squareMod_4();
					const uint64 cj = c[j].s[0] | c[j].s[1] | c[j].s[2] | c[j].s[3];
					if (cj != 0) mulMod256_4(c[j]);
				}
			}
		}
	}

public:
	void gerbiczStep() const
	{
		const size_t n = this->_n;
		const int ln = ilog2(uint32_t(n));

		if (this->_3primes)
		{
			_engine.setxy_P123();
			for (size_t s = 1, lm = ln; lm > 4; s *= 16, lm -= 4) { _engine.forward16y_P12(s, lm - 4); _engine.forward16y_P3(s, lm - 4); }
			size_t s = 1, lm = ln;
			for (; lm > 4; s *= 16, lm -= 4) { _engine.forward16d_P12(s, lm - 4); _engine.forward16d_P3(s, lm - 4); }
			if (lm > 2) { _engine.forward4y_P123(s, lm - 2); _engine.forward4d_P123(s, lm - 2); }
			if (lm % 2 != 0) _engine.mul2dy_P123(); else _engine.mul4dy_P123();
			if (lm > 2) _engine.backward4d_P123(s, lm - 2);
			for (s /= 16, lm += 4; s > 0; s /= 16, lm += 4) { _engine.backward16d_P12(s, lm - 4); _engine.backward16d_P3(s, lm - 4); }
			_engine.normalize3ad();
			_engine.normalize3bd();
		}
		else
		{
			_engine.setxy_P12();
			for (size_t s = 1, lm = ln; lm > 4; s *= 16, lm -= 4) _engine.forward16y_P12(s, lm - 4);
			size_t s = 1, lm = ln;
			for (; lm > 4; s *= 16, lm -= 4) _engine.forward16d_P12(s, lm - 4);
			if (lm > 2) { _engine.forward4y_P12(s, lm - 2); _engine.forward4d_P12(s, lm - 2); }
			if (lm % 2 != 0) _engine.mul2dy_P12(); else _engine.mul4dy_P12();
			if (lm > 2) _engine.backward4d_P12(s, lm - 2);
			for (s /= 16, lm += 4; s > 0; s /= 16, lm += 4) _engine.backward16d_P12(s, lm - 4);
			_engine.normalize2ad();
			_engine.normalize2bd();
		}
	}

public:
	void gerbiczLastStep() const
	{
		const size_t n = this->_n;
		const int ln = ilog2(uint32_t(n));

		if (this->_3primes)
		{
			_engine.setdy_P123();
			for (size_t s = 1, lm = ln; lm > 4; s *= 16, lm -= 4) { _engine.forward16y_P12(s, lm - 4); _engine.forward16y_P3(s, lm - 4); }
			size_t s = 1, lm = ln;
			for (; lm > 4; s *= 16, lm -= 4) { _engine.forward16x_P12(s, lm - 4); _engine.forward16x_P3(s, lm - 4); }
			if (lm > 2) { _engine.forward4y_P123(s, lm - 2); _engine.forward4x_P123(s, lm - 2); }
			if (lm % 2 != 0) _engine.mul2xy_P123(); else _engine.mul4xy_P123();
			if (lm > 2) _engine.backward4x_P123(s, lm - 2);
			for (s /= 16, lm += 4; s > 0; s /= 16, lm += 4) { _engine.backward16x_P12(s, lm - 4); _engine.backward16x_P3(s, lm - 4); }
			_engine.swap_xd_P123();
			_engine.normalize3ad();
			_engine.normalize3bd();
		}
		else
		{
			_engine.setdy_P12();
			for (size_t s = 1, lm = ln; lm > 4; s *= 16, lm -= 4) _engine.forward16y_P12(s, lm - 4);
			size_t s = 1, lm = ln;
			for (; lm > 4; s *= 16, lm -= 4) _engine.forward16x_P12(s, lm - 4);
			if (lm > 2) { _engine.forward4y_P12(s, lm - 2); _engine.forward4x_P12(s, lm - 2); }
			if (lm % 2 != 0) _engine.mul2xy_P12(); else _engine.mul4xy_P12();
			if (lm > 2) _engine.backward4x_P12(s, lm - 2);
			for (s /= 16, lm += 4; s > 0; s /= 16, lm += 4) _engine.backward16x_P12(s, lm - 4);
			_engine.swap_xd_P12();
			_engine.normalize2ad();
			_engine.normalize2bd();
		}
	}

public:
	bool gerbiczCheck(const uint32_t a) const
	{
		const size_t n = this->_n;
		const size_t vsize = this->_vsize;
		const uint32_2 * const x = this->_gx;
		uint32_2 * const d = this->_gd;
		const uint32 * const b = this->_b;

		int32 f[VSIZE_MAX];
		for (size_t i = 0; i < vsize; ++i) f[i] = 0;

		for (size_t j = 0; j < n; ++j)
		{
			for (size_t i = 0; i < vsize; ++i)
			{
				const size_t k = j * vsize + i;
				const uint32 xk = x[k].s[0], dk = d[k].s[0];
				const uint32 cxk = (xk > P1 / 2) ? P1 : 0, cdk = (dk > P1 / 2) ? P1 : 0;
				const int32 ixk = int32(xk - cxk), idk = int32(dk - cdk);
				int64 e = f[i] + ixk * int64(a) - idk;
				d[k].s[0] = reduce64(e, b[i]);
				f[i] = int32(e);	// |e| <= 3
			}
		}

		for (size_t i = 0; i < vsize; ++i)
		{
			int32 e = f[i];
			while (e != 0)
			{
				e = -e;		// a_0 = -a_n
				for (size_t j = 0; j < n; ++j)
				{
					const size_t k = j * vsize + i;
					e += int32(d[k].s[0]);
					d[k].s[0] = reduce32(e, b[i]);
					if (e == 0) break;
				}
			}
		}

		for (size_t k = 0; k < vsize * n; ++k) if (d[k].s[0] != 0) return false;

		return true;
	}

public:
	void isPrime(bool prm[VSIZE_MAX], uint64_t res[VSIZE_MAX], uint64_t res64[VSIZE_MAX]) const
	{
		const size_t n = this->_n;
		const size_t vsize = this->_vsize;
		uint32 * const x = this->_x;
		const uint32 * const b = this->_b;

		int32 f[VSIZE_MAX];
		for (size_t i = 0; i < vsize; ++i) f[i] = 0;

		for (size_t j = 0; j < n; ++j)
		{
			for (size_t i = 0; i < vsize; ++i)
			{
				const size_t k = j * vsize + i;
				const uint32 xk = x[k];
				const int32 ixk = (xk > P1 / 2) ? int32(xk - P1) : int32(xk);
				int64 e = f[i] + int64(ixk);
				x[k] = reduce64(e, b[i]);
				f[i] = int32(e);
			}
		}

		for (size_t i = 0; i < vsize; ++i)
		{
			int32 e = f[i];
			while (e != 0)
			{
				e = -e;		// a_0 = -a_n
				for (size_t j = 0; j < n; ++j)
				{
					const size_t k = j * vsize + i;
					e += int32(x[k]);
					x[k] = reduce32(e, b[i]);
					if (e == 0) break;
				}
			}

			uint64_t r = 0;
			for (size_t j = 8; j > 0; --j)
			{
				r = (r << 8) | uint8_t(x[(n - j) * vsize + i]);
			}
			res[i] = r;

			prm[i] = (x[i] == 1);
		}

		uint64_t bi[VSIZE_MAX];
		for (size_t i = 0; i < vsize; ++i) bi[i] = b[i];

		for (size_t i = 0; i < vsize; ++i) res64[i] = x[i];

		for (size_t j = 1; j < n; ++j)
		{
			for (size_t i = 0; i < vsize; ++i)
			{
				const size_t k = j * vsize + i;
				const uint32 xk = x[k];
				prm[i] &= (xk == 0);
				res64[i] += xk * bi[i];
				bi[i] *= b[i];
			}
		}
	}
};
