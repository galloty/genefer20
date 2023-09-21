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
#include "file.h"

#include <thread>
#include <sys/stat.h>

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
	bool _isBoinc = false;
	engine * _engine = nullptr;
	transform * _transform = nullptr;
	int _n = 0;
	int _print_range = 0, _print_i = 0;

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
	void init(const int n, engine & eng, const bool isBoinc)
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

	void deleteTransform()
	{
		if (this->_transform != nullptr)
		{
			delete this->_transform;
			this->_transform = nullptr;
		}
	}

private:
	void initPrintProgress(const int i0, const int i_start)
	{
		_print_range = i0; _print_i = i_start;
		if (_isBoinc) boinc_fraction_done((i0 > i_start) ? static_cast<double>(i0 - i_start) / i0 : 0.0);
	}

	void printProgress(const double displayTime, const int i)
	{
		if (_print_i == i) return;

		const double percent = static_cast<double>(_print_range - i) / _print_range;
		if (_isBoinc)
		{
			boinc_fraction_done(percent);
		}
		else
		{
			const double mulTime = displayTime / (_print_i - i); _print_i = i;
			const double estimatedTime = mulTime * i;
			std::ostringstream ss; ss << std::setprecision(3) << percent * 100.0 << "% done, " << timer::formatTime(estimatedTime)
									<< " remaining, " << mulTime * 1e3 << " ms/bit.        \r";
			pio::display(ss.str());
		}
	}

	static void clearline() { pio::display("                                                \r"); }

private:
	int _readContext(const std::string & filename, int & i, double & elapsedTime)
	{
		file contextFile(filename);
		if (!contextFile.exists()) return -1;

		int version = 0;
		if (!contextFile.read(reinterpret_cast<char *>(&version), sizeof(version))) return -2;
		if (version != 1) return -2;
		if (!contextFile.read(reinterpret_cast<char *>(&i), sizeof(i))) return -2;
		if (!contextFile.read(reinterpret_cast<char *>(&elapsedTime), sizeof(elapsedTime))) return -2;
		if (!_transform->readContext(contextFile)) return -2;
		if (!contextFile.check_crc32()) return -2;
		return 0;
	}

	bool readContext(const std::string & ctxFile, int & i, double & elapsedTime)
	{
		int error = _readContext(ctxFile, i, elapsedTime);
		if (error < -1)
		{
			std::ostringstream ss; ss << ctxFile << ": invalid context";
			pio::error(ss.str());
		}

		const std::string oldCtxFile = ctxFile + ".old";
		if (error < 0)
		{
			error = _readContext(oldCtxFile, i, elapsedTime);
			if (error < -1)
			{
				std::ostringstream ss; ss << oldCtxFile << ": invalid context";
				pio::error(ss.str());
			}
		}
		return (error == 0);
	}

	void saveContext(const std::string & ctxFile, const int i, const double elapsedTime) const
	{
		const std::string oldCtxFile = ctxFile + ".old", newCtxFile = ctxFile + ".new";

		{
			file contextFile(newCtxFile, "wb", false);
			int version = 1;
			if (!contextFile.write(reinterpret_cast<const char *>(&version), sizeof(version))) return;
			if (!contextFile.write(reinterpret_cast<const char *>(&i), sizeof(i))) return;
			if (!contextFile.write(reinterpret_cast<const char *>(&elapsedTime), sizeof(elapsedTime))) return;
			_transform->saveContext(contextFile);
			contextFile.write_crc32();
		}

		std::remove(oldCtxFile.c_str());

		struct stat s;
		if ((stat(ctxFile.c_str(), &s) == 0) && (std::rename(ctxFile.c_str(), oldCtxFile.c_str()) != 0))	// file exists and cannot rename it
		{
			pio::error("cannot save context");
			return;
		}

		if (std::rename(newCtxFile.c_str(), ctxFile.c_str()) != 0)
		{
			pio::error("cannot save context");
			return;
		}
	}

private:
	void boincMonitor()
	{
		BOINC_STATUS status; boinc_get_status(&status);
		if (boincQuitRequest(status)) { quit(); return; }

		while (status.suspended != 0)
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
			boinc_get_status(&status);
			if (boincQuitRequest(status)) { quit(); return; }
		}
	}

	void boincMonitor(const std::string & ctxFilename, const int i, watch & chrono)
	{
		BOINC_STATUS status; boinc_get_status(&status);
		if (boincQuitRequest(status)) { quit(); return; }

		if (status.suspended != 0)
		{
			saveContext(ctxFilename, i, chrono.getElapsedTime());
			while (status.suspended != 0)
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
				boinc_get_status(&status);
				if (boincQuitRequest(status)) { quit(); return; }
			}
		}

		if (boinc_time_to_checkpoint() != 0)
		{
			saveContext(ctxFilename, i, chrono.getElapsedTime());
			boinc_checkpoint_completed();
		}
	}

private:
	static void parseFile(const std::string & filename, vint32 & b)
	{
		std::ifstream inFile(filename);
		if (!inFile.is_open()) pio::error("cannot open input file", true);

		size_t i = 0;
		std::string line;
		while (std::getline(inFile, line))
		{
			b[i] = uint32_t(std::stoi(line));
			++i; if (i == VSIZE) break;
		}
		while (i < VSIZE)
		{
			b[i] = b[i - 1];
			++i;
		}

		inFile.close();
	}

private:
	inline static uint64 get_bitcnt(const size_t i, const mpz_t * const ze)
	{
		uint64 e = 0;
		for (size_t j = 0; j < VSIZE; ++j) e |= ((mpz_tstbit(ze[j], mp_bitcnt_t(i)) != 0) ? uint64(1) : uint64(0)) << j;
		return e;
	}

public:
	bool check(const std::string & filename, double & elapsedTime)
	{
		const std::string iniFilename = filename + std::string(".ini");
		size_t csize = 0;
		std::ifstream iniFile_i(iniFilename);
		if (iniFile_i.is_open())
		{
			iniFile_i >> csize;
			iniFile_i.close();
		}

		createTransform(csize);
		csize = _transform->get_csize();

		std::ofstream iniFile_o(iniFilename);
		if (iniFile_o.is_open())
		{
			iniFile_o << csize << std::endl;
			iniFile_o.close();
		}

		vint32 b; parseFile(filename, b);

		const int n = this->_n;
		transform * const pTransform = this->_transform;
		pTransform->init(b);

		std::array<mpz_t, VSIZE> exponent;
		int i0 = 0;
		for (size_t j = 0; j < VSIZE; ++j)
		{
			mpz_init(exponent[j]);
			mpz_ui_pow_ui(exponent[j], b[j], 1u << n);
			i0 = std::max(i0, int(mpz_sizeinbase(exponent[j], 2) - 1));
		}
		const int L = 2 << (ilog2_32(uint32_t(i0)) / 2), B_GL = int((i0 - 1) / L) + 1;

		const std::string ctxFilename = filename + std::string(".ctx");
		int ri = 0; double restoredTime = 0;
		const bool found = readContext(ctxFilename, ri, restoredTime);

		if (!found)
		{
			ri = 0; restoredTime = 0;
			pTransform->set(1);
			pTransform->copy(1, 0);	// d(t)
		}
		else
		{
			std::ostringstream ss; ss << "Resuming from a checkpoint." << std::endl;
			pio::print(ss.str());
		}

		watch chrono(found ? restoredTime : 0);
		const int i_start = found ? ri : i0;

		if (i_start >= 0)
		{
			initPrintProgress(i0, i_start);

			for (int i = i_start; i >= 0; --i)
			{
				if (_isBoinc) boincMonitor(ctxFilename, i, chrono);

				if (_quit)
				{
					saveContext(ctxFilename, i, chrono.getElapsedTime());
					return false;
				}

				if (i % B_GL == 0)
				{
					chrono.read(); const double displayTime = chrono.getDisplayTime();
					if (displayTime >= 1) { printProgress(displayTime, i); chrono.resetDisplayTime(); }
					if (!_isBoinc && (chrono.getRecordTime() > 600)) { saveContext(ctxFilename, i, chrono.getElapsedTime()); chrono.resetRecordTime(); }
				}

				// if (i == i0 / 2) e = e ^ 1;	// => invalid
				pTransform->squareDup(get_bitcnt(size_t(i), exponent.data()));

				if ((i % B_GL == 0) && (i / B_GL != 0))
				{
					pTransform->copy(2, 0);
					pTransform->mul();	// d(t)
					pTransform->copy(1, 0);
					pTransform->copy(0, 2);
				}
			}

			if (chrono.getElapsedTime() > 60) saveContext(ctxFilename, -1, chrono.getElapsedTime());
		}

		// get result
		pTransform->getInt(0);

		// Gerbicz-Li error checking
		clearline(); pio::display("Validating...\r");

		// d(t + 1) = d(t) * result
		pTransform->copy(2, 1);
		pTransform->mul();
		pTransform->copy(1, 2);
		pTransform->copy(2, 0);

		i0 = 0;
		mpz_t res, tmp; mpz_init(res); mpz_init(tmp);
		for (size_t j = 0; j < VSIZE; ++j)
		{
			mpz_init_set_ui(res, 0);
			mpz_t & e = exponent[j];
			while (mpz_sgn(e) != 0)
			{
				mpz_mod_2exp(tmp, e, mp_bitcnt_t(B_GL));
				mpz_add(res, res, tmp);
				mpz_div_2exp(e, e, mp_bitcnt_t(B_GL));
			}
			mpz_set(e, res);
			i0 = std::max(i0, int(mpz_sizeinbase(res, 2) - 1));
		}
		mpz_clear(res); mpz_clear(tmp);

		// 2^res * d(t)^{2^B}
		pTransform->set(1);
		for (int i = i0; i >= B_GL; --i)
		{
			if (_isBoinc) boincMonitor();
			if (_quit) return false;

			pTransform->squareDup(get_bitcnt(size_t(i), exponent.data()));
		}

		pTransform->mul();

		for (int i = B_GL - 1; i >= 0; --i)
		{
			if (_isBoinc) boincMonitor();
			if (_quit) return false;

			const uint64_t bitcnt = (i <= i0) ? get_bitcnt(size_t(i), exponent.data()) : 0;
			pTransform->squareDup(bitcnt);
		}

		for (size_t j = 0; j < VSIZE; ++j) mpz_clear(exponent[j]);

		std::array<bool, VSIZE> isPrime;
		std::array<uint64_t, VSIZE> r, r64;
		const bool err = pTransform->isPrime(isPrime.data(), r.data(), r64.data());
		if (err) pio::error("computation failed", true);

		// d(t)^{2^B} * 2^res ?= d(t + 1)
		pTransform->getInt(1);
		pTransform->copy(0, 2);
		pTransform->getInt(2);
		const bool success = pTransform->GerbiczLiCheck();
		if (!success) pio::error("Gerbicz failed", true);

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

		pio::result(ssr.str());

		ssr << std::endl;
		clearline();
		pio::display(ssr.str());

		if (_isBoinc) boinc_fraction_done(1.0);
		elapsedTime = chrono.getElapsedTime();
		return true;
	}
};
