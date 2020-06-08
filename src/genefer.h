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
	transform * _transform = nullptr;
	bool _isBoinc = false;

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

private:
	// static void printStatus(transform & t, const bool found, const uint32_t b, const uint32_t n)
	// {
	// 	std::ostringstream ss; ss << (found ? "Resuming from a checkpoint " : "Testing ");
	// 	const uint32_t m = uint32_t(1) << n;
	// 	ss << b << "^ " << m << " + 1, " << X.getDigits() << " digits, size = 2^" << arith::log2(X.getSize())
	// 		<< " x " << X.getDigitBit() << " bits, plan: " << X.getPlanString() << std::endl;
	// 	pio::print(ss.str());
	// }

private:
	// static void printProgress(chronometer & chrono, const uint32_t i, const uint32_t n, const uint32_t benchCnt)
	// {
	// 	const double elapsedTime = chrono.getBenchTime();
	// 	const double mulTime = elapsedTime / benchCnt, estimatedTime = mulTime * (n - i);
	// 	std::ostringstream ss; ss << std::setprecision(3) << " " << i * 100.0 / n << "% done, "
	// 		<< timer::formatTime(estimatedTime) << " remaining, " <<  mulTime * 1e3 << " ms/mul.        \r";
	// 	pio::display(ss.str());
	// 	chrono.resetBenchTime();
	// }

public:
	void init(const uint32_t n, engine & eng, const bool isBoinc)
	{
		this->_n = n;
		this->_transform = new transform(size_t(1) << n, eng, isBoinc);
		this->_isBoinc = isBoinc;
	}

public:
	void release()
	{
		this->_n = 0;
		if (this->_transform != nullptr)
		{
			delete this->_transform;
			this->_transform = nullptr;
		}
	}

private:
	void check_GPU(const vint32 & b, const uint32_t a) const
	{
		const int n = this->_n;
		const size_t m = size_t(1) << n;
		transform * const t = this->_transform;

		t->init(b, a);

		// Prp test is a^{b^{2^n}} ?= 1
		// Let L | 2^n and B = b^L. We have b^{2^n} = B^{{2^n}/L}.
		// Let d = a * a^B * a^{B^2} * ... * a^{B^{{2^n}/L - 1}}.
		// Gerbicz test is d * a^{B^{{2^n}/L}} ?= a * d^B.

		const size_t L = size_t(1) << (n / 2);

		for (size_t i = 1; i < m; ++i)
		{
			t->powMod();
			if ((i & (L - 1)) == 0) t->gerbiczStep();
		}

		t->powMod();
		t->copyRes();
		t->gerbiczLastStep();

		for (size_t j = 0; j < L; ++j) t->powMod();

		t->saveRes();
	}

private:
	void check_CPU(const vint32 & b, const uint32_t a, const bool display, const bool verif = false) const
	{
		const int n = this->_n;
		const size_t m = size_t(1) << n;
		transform * const t = this->_transform;
		const size_t vsize = t->getVsize();

		if (!t->gerbiczCheck(a)) throw std::runtime_error("Gerbicz failed");

		bool isPrime[VSIZE_MAX];
		uint64_t r[VSIZE_MAX], r64[VSIZE_MAX];
		t->isPrime(isPrime, r, r64);

		std::ostringstream ssr;
		for (size_t i = 0; i < vsize; ++i)
		{
			if ((i > 0) && (b[i] == b[i - 1])) continue;

			ssr << b[i] << "^" << m << " + 1 is ";
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

		if (verif)
		{
			std::ostringstream ss; ss << "verif_" << n << ".txt";
			std::ofstream file(ss.str(), std::ios::app);
			if (file.is_open())
			{
				for (size_t i = 0; i < vsize; ++i) file << res64String(r64[i]) << std::endl;
				file.close();
			}
		}
	}

private:
	void check(const vint32 & b, const uint32_t a, const bool display, const bool verif = false) const
	{
		check_GPU(b, a);
		check_CPU(b, a, display, verif);
	}

public:
	void bench()
	{
		const size_t vsize = this->_transform->getVsize();
		double elapsedTimeGPU = 0, elapsedTimeCPU = 0;
		uint32 bi = 300000000;
		for (size_t j = 1; true; ++j)
		{
			const timer::time t0 = timer::currentTime();
			vint32 b;
			for (size_t i = 0; i < vsize; ++i) { b[i] = bi; bi += 2; }
			check_GPU(b, 2);
			const timer::time t1 = timer::currentTime();
			elapsedTimeGPU += timer::diffTime(t1, t0);
			check_CPU(b, 2, false);
			elapsedTimeCPU += timer::diffTime(timer::currentTime(), t1);
			if (j % 1 == 0)
			{
				const double elapsedTime = elapsedTimeGPU + elapsedTimeCPU;
				std::cout << std::setprecision(3) << "GPU: " << 100 * elapsedTimeGPU / elapsedTime << "%, CPU: " << 100 * elapsedTimeCPU / elapsedTime
					<< "%, " << (j * vsize / elapsedTime) << " GFN-" << this->_n << "/sec" << std::endl;
				// return;
			}
			if (_quit) return;
		}
	}

public:
	void valid()
	{
		transform * const t = this->_transform;
		const size_t vsize = t->getVsize();
		vint32 b;
		uint32 bi = 300000000;
		for (size_t j = 0; j < 1024 / vsize; ++j)
		{
			for (size_t i = 0; i < vsize; ++i) { b[i] = bi; bi += 2; }
			check(b, 2, true, true);
		}
	}

private:
	static void parseFile(const std::string & filename, std::vector<vint32> & bVec, const size_t vsize)
	{
		std::ifstream inFile(filename);
		if (!inFile.is_open()) throw std::runtime_error("cannot open input file");

		vint32 b;
		size_t i = 0;
		int lgb = 0;
		std::string line;
		while (std::getline(inFile, line))
		{
			const uint32_t u = std::stoi(line);
			if (i == 0) lgb = ilog2(u);
			else if (ilog2(u) != lgb)
			{
				while (i != vsize)
				{
					b[i] = b[i - 1];
					++i;
				}
				bVec.push_back(b);
				i = 0;
				lgb = ilog2(u);
			}
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
	bool checkFile(const std::string & filename)
	{
		const size_t vsize = this->_transform->getVsize();
		std::vector<vint32> bVec;
		parseFile(filename, bVec, vsize);

		size_t i0 = 0, n = bVec.size();

		if (_isBoinc) boinc_fraction_done(double(i0) / double(n));

		for (size_t i = i0; i < n; ++i)
		{
			check(bVec[i], 2, true);

			if (_isBoinc)
			{
				BOINC_STATUS status;
				boinc_get_status(&status);
				bool quit = boincQuitRequest(status);
				if (quit || (status.suspended != 0))
				{
					// checkError(X);
					// X.saveContext(i, chrono.getElapsedTime(), "p");
				}
				if (quit) return false;
					
				if (status.suspended != 0)
				{
					std::ostringstream ss_s; ss_s << std::endl << "BOINC client is suspended." << std::endl;
					pio::print(ss_s.str());

					while (status.suspended != 0)
					{
						std::this_thread::sleep_for(std::chrono::seconds(1));
						boinc_get_status(&status);
						if (boincQuitRequest(status)) return false;
					}

					std::ostringstream ss_r; ss_r << "BOINC client is resumed." << std::endl;
					pio::print(ss_r.str());
				}

				if (boinc_time_to_checkpoint() != 0)
				{
					// checkError(X);
					// X.saveContext(i, chrono.getElapsedTime(), "p");
					boinc_checkpoint_completed();
				}
			}
			else
			{
				// const double elapsedTime = chrono.getRecordTime();
				// if (elapsedTime > 600)
				// {
				// 	checkError(X);
				// 	X.saveContext(i, chrono.getElapsedTime(), "p");
				// 	chrono.resetRecordTime();
				// }
			}

			if (_quit)
			{
				// X.saveContext(i, chrono.getElapsedTime());
				return false;
			}
		}

		if (_isBoinc) boinc_fraction_done(1.0);
		return true;
	}
};
