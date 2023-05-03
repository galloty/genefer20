/*
Copyright 2020, Yves Gallot

genefer20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include "ocl.h"
#include "transform.h"
#include "pio.h"
#include "timer.h"

#include <thread>
#include <chrono>

#include <gmp.h>

class genefer
{
private:
	struct deleter { void operator()(const genefer * const p) { delete p; } };

public:
	genefer() {}
	virtual ~genefer() {}

	static genefer & getInstance()
	{
		static std::unique_ptr<genefer, deleter> pInstance(new genefer());
		return *pInstance;
	}

public:
	void quit() { _quit = true; }
	void setBoinc(const bool isBoinc) { _isBoinc = isBoinc; }

protected:
	volatile bool _quit = false;
private:
	int _n = 0;
	engine * _engine = nullptr;
	bool _isBoinc = false;
	transform * _transform = nullptr;

private:
	static std::string res64String(const uint64_t res64, const bool uppercase = true)
	{
		std::stringstream ss;
		if (uppercase) ss << std::uppercase;
		ss << std::hex << std::setfill('0') << std::setw(16) << res64;
		return ss.str();
	}

private:
	static bool boincQuitRequest(const BOINC_STATUS & status)
	{
		if ((status.quit_request | status.abort_request | status.no_heartbeat) == 0) return false;

		std::ostringstream ss; ss << std::endl << "Terminating because BOINC ";
		if (status.quit_request != 0) ss << "requested that we should quit.";
		else if (status.abort_request != 0) ss << "requested that we should abort.";
		else if (status.no_heartbeat != 0) ss << "heartbeat was lost.";
		ss << std::endl;
		pio::print(ss.str());
		return true;
	}

public:
	void init(const uint32_t n, engine & eng, const bool isBoinc)
	{
		this->_n = n;
		this->_engine = &eng;
		this->_isBoinc = isBoinc;
	}

public:
	void release()
	{
		deleteTransform();
		this->_n = 0;
	}

private:
	void createTransform(const size_t csize)
	{
		this->_transform = new transform(size_t(1) << this->_n, *(this->_engine), this->_isBoinc, csize);
	}

private:
	void deleteTransform()
	{
		if (this->_transform != nullptr)
		{
			delete this->_transform;
			this->_transform = nullptr;
		}
	}

private:
	void check(const vint32 & b, const bool display) const
	{
		const int n = this->_n;
		transform * const t = this->_transform;

		std::array<mpz_t, VSIZE> exponent;
		int i0 = 0;
		for (size_t j = 0; j < VSIZE; ++j)
		{
			mpz_init(exponent[j]);
			mpz_ui_pow_ui(exponent[j], b[j], 1u << n);
			i0 = std::max(i0, int(mpz_sizeinbase(exponent[j], 2) - 1));
		}

		t->init(b);
		t->set(1);
		t->copy(1, 0);	// d(t)

		const size_t L = (size_t(2) << (ilog2_32(uint32_t(i0)) / 2));
		const int B_GL = int((i0 - 1) / L) + 1;

		for (int i = i0; i >= 0; --i)
		{
			uint64 e = 0; for (size_t j = 0; j < VSIZE; ++j) e |= ((mpz_tstbit(exponent[j], mp_bitcnt_t(i)) != 0) ? uint64(1) : uint64(0)) << j;
			t->squareDup(e);

			if ((i % B_GL == 0) && (i / B_GL != 0))
			{
				t->copy(2, 0);
				t->mul();	// d(t)
				t->copy(1, 0);
				t->copy(0, 2);
			}
		}

		t->getInt(0);

		// d(t + 1) = d(t) * result
		t->copy(2, 1);
		t->mul();
		t->copy(1, 2);
		t->copy(2, 0);

		// d(t)^{2^B}
		t->copy(0, 1);
		for (int i = B_GL - 1; i >= 0; --i)
		{
			t->squareDup(0);
		}
		t->copy(1, 0);

		std::array<mpz_t, VSIZE> res;
		i0 = 0;
		mpz_t e, tmp; mpz_init(e); mpz_init(tmp);
		for (size_t j = 0; j < VSIZE; ++j)
		{
			mpz_init_set_ui(res[j], 0);
			mpz_set(e, exponent[j]);
			while (mpz_sgn(e) != 0)
			{
				mpz_mod_2exp(tmp, e, B_GL);
				mpz_add(res[j], res[j], tmp);
				mpz_div_2exp(e, e, B_GL);
			}
			i0 = std::max(i0, int(mpz_sizeinbase(res[j], 2) - 1));
		}
		mpz_clear(e); mpz_clear(tmp);

		// 2^res
		t->set(1);
		for (int i = i0; i >= 0; --i)
		{
			uint64 e = 0; for (size_t j = 0; j < VSIZE; ++j) e |= ((mpz_tstbit(res[j], mp_bitcnt_t(i)) != 0) ? uint64(1) : uint64(0)) << j;
			t->squareDup(e);
		}

		for (size_t j = 0; j < VSIZE; ++j) mpz_clear(res[j]);
		for (size_t j = 0; j < VSIZE; ++j) mpz_clear(exponent[j]);

		// d(t)^{2^B} * 2^res
		t->mul();

		std::array<bool, VSIZE> isPrime;
		std::array<uint64_t, VSIZE> r, r64;
		const bool err = t->isPrime(isPrime.data(), r.data(), r64.data());

		if (err) throw std::runtime_error("Computation failed");

		// d(t)^{2^B} * 2^res ?= d(t + 1)
		t->getInt(1);
		// const uint64_t h1 = gi.gethash64();
		t->copy(0, 2);
		t->getInt(2);
		// const uint64_t h2 = gi.gethash64();

		const bool success = t->GerbiczLiCheck();
		if (!success) throw std::runtime_error("Gerbicz failed");

		std::ostringstream ssr;
		for (size_t i = 0; i < VSIZE; ++i)
		{
			if ((i > 0) && (b[i] == b[i - 1])) continue;

			ssr << b[i] << "^" << (1 << n) << " + 1 is ";
			if (isPrime[i])
			{
				ssr << "a probable prime";
			}
			else
			{
				const std::string res = res64String(r[i], false), res64 = res64String(r64[i]);
				ssr << "composite" << " [RES = " << res << ", RES64 = " << res64 << "]";
			}
			ssr << std::endl;
		}

		if (display) pio::display(std::string("\r") + ssr.str());
		pio::result(ssr.str());
	}

private:
	static void parseFile(const std::string & filename, std::vector<vint32> & bVec, const size_t vsize)
	{
		std::ifstream inFile(filename);
		if (!inFile.is_open()) throw std::runtime_error("cannot open input file");

		vint32 b;
		size_t i = 0;
		std::string line;
		while (std::getline(inFile, line))
		{
			const uint32_t u = std::stoi(line);
			b[i] = u;
			++i;
			if (i == vsize)
			{
				bVec.push_back(b);
				i = 0;
			}
		}
		if (i > 0)
		{
			while (i != vsize)
			{
				b[i] = b[i - 1];
				++i;
			}
			bVec.push_back(b);
		}

		inFile.close();
	}

public:
	bool checkFile(const std::string & filename, const bool display)
	{
		const std::string ctxFilename = filename + std::string(".ctx");
		size_t i0 = 0, csize = 0;
		std::ifstream ctxFile(ctxFilename);
		if (ctxFile.is_open())
		{
			ctxFile >> i0;
			size_t vsize; ctxFile >> vsize;
			ctxFile >> csize;
			int radix16; ctxFile >> radix16;
			ctxFile.close();

			std::cout << "Resuming from a checkpoint." << std::endl;
		}

		createTransform(csize);
		csize = _transform->get_csize();

		std::vector<vint32> bVec;
		parseFile(filename, bVec, VSIZE);
		const size_t n = bVec.size();

		std::ostringstream ss;
		ss << "Testing " << n * VSIZE << " candidates, starting at vector #" << i0 << std::endl << std::endl;
		pio::print(ss.str());

		if (_isBoinc) boinc_fraction_done(double(i0) / double(n));
		if (!display)
		{
			std::ostringstream ss; ss << std::setprecision(3) << " " << (i0 * 100.0 / n) << "% done    \r";
			std::cout << ss.str();
		}

		size_t i;
		for (i = i0; i < n; ++i)
		{
			check(bVec[i], display);

			std::ofstream ctxFile(ctxFilename);
			if (ctxFile.is_open())
			{
				ctxFile << i + 1 << " " << VSIZE << " " << csize << " 1" << std::endl;
				ctxFile.close();
			}

			if (_isBoinc) boinc_fraction_done(double(i + 1) / double(n));
			if (!display)
			{
				std::ostringstream ss; ss << std::setprecision(3) << " " << ((i + 1) * 100.0 / n) << "% done    \r";
				std::cout << ss.str();
			}

			if (_quit) break;
		}

		if (!display) std::cout << std::endl;

		if (i == n)
		{
			if (_isBoinc) boinc_fraction_done(1.0);
			return true;
		}
		return false;
	}
};
