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
#include <fstream>
#include <array>

#include "ocl/kernel.h"

typedef std::array<uint32, VSIZE> vint32;

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
		explicit Zp(const int32 i) : _n((i < 0) ? p - uint32(-i) : uint32(i)) {}

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
	const int32 _ln;
	const bool _isBoinc;
	engine & _engine;
	uint32 * const _x;
	size_t _csize = 0;
	vint32 _b;

private:
	static size_t bitRev(const size_t i, const size_t n)
	{
		size_t r = 0;
		for (size_t k = n, j = i; k > 1; k /= 2, j /= 2) r = (2 * r) | (j % 2);
		return r;
	}

private:
	template<typename Zp>
	static void create(const size_t n, uint32 * const wr, uint32 * const wri)
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

public:
	void set(const uint32 a) const { _engine.set(a); }
	void copy(const uint32 dst, const uint32 src) const { _engine.copy(dst, src); }
	void squareDup(const uint64 dup) const { _engine.squareDup(this->_ln, dup); }
	void mul() const { _engine.mul(this->_ln); }

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
		const size_t csize = this->_csize;

		std::stringstream src;
		src << "#define\tVSIZE\t" << VSIZE << std::endl;
		src << "#define\tNSIZE\t" << n << std::endl;
		src << "#define\tCSIZE\t" << csize << std::endl << std::endl;
		src << "#define\tNORM1\t" << Zp1::norm(n).get() << "u" << std::endl;
		src << "#define\tNORM2\t" << Zp2::norm(n).get() << "u" << std::endl;
		src << "#define\tNORM3\t" << Zp3::norm(n).get() << "u" << std::endl;
		src << "#define\tP1\t" << P1 << "u" << std::endl;
		src << "#define\tP2\t" << P2 << "u" << std::endl;
		src << "#define\tP3\t" << P3 << "u" << std::endl;
		src << "#define\tP1_INV\t" << P1_INV << "u" << std::endl;
		src << "#define\tP2_INV\t" << P2_INV << "u" << std::endl;
		src << "#define\tP3_INV\t" << P3_INV << "u" << std::endl;
		src << "#define\tInvP2_P1\t" << InvP2_P1 << "u" << std::endl;
		src << "#define\tInvP3_P1\t" << InvP3_P1 << "u" << std::endl;
		src << "#define\tInvP3_P2\t" << InvP3_P2 << "u" << std::endl;
		src << "#define\tP1P2P3l\t" << 15383592652180029441ull << "ul" << std::endl;
		src << "#define\tP1P2P3h\t" << 3942432002u << "u" << std::endl;
		src << "#define\tP1P2P3_2l\t" << 7691796326090014720ull << "ul" << std::endl;
		src << "#define\tP1P2P3_2h\t" << 1971216001u << "u" << std::endl;
		src << std::endl;

		// if xxx.cl file is not found then source is src_ocl_xxx string in src/ocl/xxx.h
		if (!readOpenCL("ocl/kernel.cl", "src/ocl/kernel.h", "src_ocl_kernel", src)) src << src_ocl_kernel;

		_engine.loadProgram(src.str());
		_engine.allocMemory(n, csize);
		_engine.createKernels();

		std::vector<uint32> wr1(n), wr2(n), wr3(n), wri1(n), wri2(n), wri3(n);
		create<Zp1>(n, wr1.data(), wri1.data()); create<Zp2>(n, wr2.data(), wri2.data()); create<Zp3>(n, wr3.data(), wri3.data());

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
	transform(const size_t size, engine & engine, const bool isBoinc, const size_t csize) :
		_n(size), _ln(ilog2_32(uint32_t(size))), _isBoinc(isBoinc), _engine(engine), _x(new uint32[3 * VSIZE * size])
	{
		if (csize == 0)
		{
			std::ostringstream ss; ss << " auto-tuning..." << std::endl;
			pio::display(ss.str());

			cl_ulong bestTime = cl_ulong(-1);
			size_t bestCsize = 8;

			vint32 b; for (uint32 i = 0; i < VSIZE; ++i) b[i] = (uint32(1) << 31) + 9699690 * i;

			engine.setProfiling(true);
			for (size_t csize = 8; csize <= 64; csize *= 2)
			{
				this->_csize = csize;
				initEngine();
				init(b);
				set(1);
				squareDup(uint64(-1));
				engine.resetProfiles();
				const size_t m = 16;
				uint64 e = 0; for (size_t j = 0; j < VSIZE; ++j) e |= uint64(j % 2) << j;
				for (size_t i = 0; i < m; ++i) squareDup(e);
				const cl_ulong time = engine.getProfileTime();
				releaseEngine();

				std::ostringstream ss; ss << "vsize = " << VSIZE << ", csize = " << csize << ", " << time * 1e-6 / m << " ms" << std::endl;
				pio::display(ss.str());

				if (time < bestTime)
				{
					bestTime = time;
					bestCsize = csize;
				}
			}

			this->_csize = bestCsize;
		}
		else
		{
			this->_csize = csize;
		}

		std::ostringstream ss; ss << "Chunk size = " << this->_csize << std::endl << std::endl;
		pio::print(ss.str());

		engine.setProfiling(false);
		initEngine();
	}

public:
	virtual ~transform()
	{
		releaseEngine();

		delete[] _x;
	}

public:
	size_t get_csize() const { return this->_csize; }

public:
	void init(const vint32 & b)
	{
		std::array<uint32_2, VSIZE> bb_inv;
		std::array<int32, VSIZE> bs;
		for (size_t i = 0; i < VSIZE; ++i)
		{
			const int s = ilog2_32(b[i]) - 1;
			this->_b[i] = b[i];
			bb_inv[i].s[0] = b[i];
			bb_inv[i].s[1] = uint32((uint64(1) << (s + 32)) / b[i]);
			bs[i] = s;
		}

		_engine.writeMemory_b(bb_inv.data(), bs.data());
	}

private:
	// f = f' * b + r such that 0 <= r < b
	static uint32 reduce(int64 & f, const uint32 b)
	{
		int64 d = 0;
		while (f < 0) { f += b; --d; }
		while (f >= b) { f -= b; ++d; }
		const uint32 r = uint32(f);
		f = d;
		return r;
	}

public:
	void getInt(const uint32 reg) const
	{
		const size_t n = this->_n;
		uint32 * const x = &this->_x[reg * VSIZE * n];
		const uint32 * const b = this->_b.data();

		for (size_t j = 0; j < n; ++j)
		{
			for (size_t i = 0; i < VSIZE; ++i)
			{
				const size_t k = j * VSIZE + i;
				x[k] = reg;
			}
		}

		_engine.readMemory_x3(x);

		std::array<int64, VSIZE> f; f.fill(0);

		for (size_t j = 0; j < n; ++j)
		{
			for (size_t i = 0; i < VSIZE; ++i)
			{
				const size_t k = j * VSIZE + i;
				const uint32 xk = x[k];
				const int32 ixk = (xk > P3 / 2) ? int32(xk - P3) : int32(xk);
				int64 e = f[i] + ixk;
				x[k] = reduce(e, b[i]);
				f[i] = e;
			}
		}

		for (size_t i = 0; i < VSIZE; ++i)
		{
			int64 e = f[i];
			while (e != 0)
			{
				e = -e;		// a_0 = -a_n
				for (size_t j = 0; j < n; ++j)
				{
					const size_t k = j * VSIZE + i;
					e += int32(x[k]);
					x[k] = reduce(e, b[i]);
					if (e == 0) break;
				}
			}
		}
	}

public:
	bool isPrime(bool * const prm, uint64_t * const res, uint64_t * const res64) const
	{
		const size_t n = this->_n;
		const uint32 * const x = &this->_x[0 * VSIZE * n];
		const uint32 * const b = this->_b.data();

		std::array<bool, VSIZE> err; err.fill(false);

		for (size_t i = 0; i < VSIZE; ++i)
		{
			uint64_t r = 0;
			for (size_t j = 8; j > 0; --j)
			{
				r = (r << 8) | uint8_t(x[(n - j) * VSIZE + i]);
			}
			res[i] = r;

			prm[i] = (x[i] == 1);
			err[i] = (x[i] == 0);
		}

		std::array<uint64_t, VSIZE> bi;
		for (size_t i = 0; i < VSIZE; ++i) bi[i] = b[i];

		for (size_t i = 0; i < VSIZE; ++i) res64[i] = x[i];

		for (size_t j = 1; j < n; ++j)
		{
			for (size_t i = 0; i < VSIZE; ++i)
			{
				const size_t k = j * VSIZE + i;
				const uint32 xk = x[k];
				prm[i] = (prm[i] && (xk == 0));
				err[i] = (err[i] && (xk == 0));
				res64[i] += xk * bi[i];
				bi[i] *= b[i];
			}
		}

		bool error = false;
		for (size_t i = 0; i < VSIZE; ++i) error |= err[i];

		return error;
	}

public:
	bool GerbiczLiCheck() const
	{
		const size_t n = this->_n;
		uint32 * const y = &this->_x[1 * VSIZE * n];
		uint32 * const z = &this->_x[2 * VSIZE * n];

		bool success = true;
		for (size_t j = 0; j < n; ++j)
		{
			for (size_t i = 0; i < VSIZE; ++i)
			{
				const size_t k = j * VSIZE + i;
				success &= (y[k] == z[k]);
			}
		}
		return success;
	}
};
