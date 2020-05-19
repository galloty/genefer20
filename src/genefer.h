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
	bool _isBoinc = false;

private:
	static std::string res64String(const uint64_t res64)
	{
		std::stringstream ss; ss << std::uppercase << std::hex << std::setfill('0') << std::setw(16) << res64;
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

protected:
	static void checkError(transform & t)
	{
		const int err = t.getError();
		if (err != 0)
		{
			throw std::runtime_error("GPU error detected");
		}
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
	bool check(const uint32_t k, const uint32_t n, engine & engine, const bool checkRes = false, const uint64_t r64 = 0)
	{
		return true;
	}
};
