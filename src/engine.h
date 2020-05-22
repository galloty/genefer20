/*
Copyright 2020, Yves Gallot

genefer20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include "ocl.h"

typedef uint32_t	uint32;
typedef int32_t 	int32;
typedef uint64_t	uint64;
typedef int64_t 	int64;

#define VSIZE	8

inline uint32 mul_hi(const uint32 a, const uint32 b) { return uint32((uint64(a) * b) >> 32); }

struct uint96
{
	uint64 s0;
	uint32 s1;
};

struct int96
{
	uint64 s0;
	int32  s1;
};

inline int96 int96_set_si(const int64 n) { int96 r; r.s0 = uint64(n); r.s1 = (n < 0) ? -1 : 0; return r; }

inline uint96 uint96_set(const uint64 s0, const uint32 s1) { uint96 r; r.s0 = s0; r.s1 = s1; return r; }

inline int96 uint96_i(const uint96 x) { int96 r; r.s0 = x.s0; r.s1 = int32(x.s1); return r; }
inline uint96 int96_u(const int96 x) { uint96 r; r.s0 = x.s0; r.s1 = uint32(x.s1); return r; }

inline bool int96_is_neg(const int96 x) { return (x.s1 < 0); }

inline bool uint96_is_greater(const uint96 x, const uint96 y) { return (x.s1 > y.s1) || ((x.s1 == y.s1) && (x.s0 > y.s0)); }

inline int96 int96_neg(const int96 x)
{
	const int32 c = (x.s0 != 0) ? 1 : 0;
	int96 r; r.s0 = -x.s0; r.s1 = -x.s1 - c;
	return r;
}

inline int96 int96_add(const int96 x, const int96 y)
{
	const uint64 s0 = x.s0 + y.s0;
	const int32 c = (s0 < y.s0) ? 1 : 0;
	int96 r; r.s0 = s0; r.s1 = x.s1 + y.s1 + c;
	return r;
}

inline uint96 uint96_add_64(const uint96 x, const uint64 y)
{
	const uint64 s0 = x.s0 + y;
	const uint32 c = (s0 < y) ? 1 : 0;
	uint96 r; r.s0 = s0; r.s1 = x.s1 + c;
	return r;
}

inline int96 uint96_subi(const uint96 x, const uint96 y)
{
	const uint32 c = (x.s0 < y.s0) ? 1 : 0;
	int96 r; r.s0 = x.s0 - y.s0; r.s1 = int32(x.s1 - y.s1 - c);
	return r;
}

inline uint96 uint96_mul_64_32(const uint64 x, const uint32 y)
{
	const uint64 l = uint64(uint32(x)) * y, h = (x >> 32) * y + (l >> 32);
	uint96 r; r.s0 = (h << 32) | uint32(l); r.s1 = uint32(h >> 32);
	return r;
}

inline uint96 int96_abs(const int96 x)
{
	const int96 t = (int96_is_neg(x)) ? int96_neg(x) : x;
	return int96_u(t);
}

class engine : public ocl::device
{
private:
	struct Sp
	{
		cl_mem x, y, wr, wri, d;
		Sp() : x(nullptr), y(nullptr), wr(nullptr), wri(nullptr), d(nullptr) {}

		void alloc(ocl::device & device, const size_t n)
		{
			x = device._createBuffer(CL_MEM_READ_WRITE, sizeof(cl_uint) * VSIZE * n);
			y = device._createBuffer(CL_MEM_READ_WRITE, sizeof(cl_uint) * VSIZE * n);
			wr = device._createBuffer(CL_MEM_READ_ONLY, sizeof(cl_uint) * n);
			wri = device._createBuffer(CL_MEM_READ_ONLY, sizeof(cl_uint) * n);
			d = device._createBuffer(CL_MEM_READ_WRITE, sizeof(cl_uint) * VSIZE * n);
		}

		void release()
		{
			_releaseBuffer(x);
			_releaseBuffer(y);
			_releaseBuffer(wr);
			_releaseBuffer(wri);
			_releaseBuffer(d);
		}
	};

private:
	size_t _size = 0;
	Sp _z1, _z2, _z3;
	cl_kernel _square2_P1 = nullptr, _square2_P2 = nullptr, _square2_P3 = nullptr;

public:
	engine(const ocl::platform & platform, const size_t d) : ocl::device(platform, d) {}
	virtual ~engine() {}

public:
	void allocMemory(const size_t size)
	{
#if defined (ocl_debug)
		std::ostringstream ss; ss << "Alloc gpu memory." << std::endl;
		pio::display(ss.str());
#endif
		_size = size;
		_z1.alloc(*this, size);
		_z2.alloc(*this, size);
		_z3.alloc(*this, size);
	}

public:
	void releaseMemory()
	{
#if defined (ocl_debug)
		std::ostringstream ss; ss << "Free gpu memory." << std::endl;
		pio::display(ss.str());
#endif
		if (_size != 0)
		{
			_z1.release();
			_z2.release();
			_z3.release();
			_size = 0;
		}
	}

public:
	void createKernels()
	{
#if defined (ocl_debug)
		std::ostringstream ss; ss << "Create ocl kernels." << std::endl;
		pio::display(ss.str());
#endif

		_square2_P1 = _createKernel("square2_P1");
		_setKernelArg(_square2_P1, 0, sizeof(cl_mem), &_z1.x);
		_square2_P2 = _createKernel("square2_P2");
		_setKernelArg(_square2_P2, 0, sizeof(cl_mem), &_z2.x);
		_square2_P3 = _createKernel("square2_P3");
		_setKernelArg(_square2_P3, 0, sizeof(cl_mem), &_z3.x);
	}

public:
	void releaseKernels()
	{
#if defined (ocl_debug)
		std::ostringstream ss; ss << "Release ocl kernels." << std::endl;
		pio::display(ss.str());
#endif
		_releaseKernel(_square2_P1);
		_releaseKernel(_square2_P2);
		_releaseKernel(_square2_P3);
	}

public:
	void readMemory_x1(cl_uint * const ptr) { _readBuffer(_z1.x, ptr, sizeof(cl_uint) * VSIZE * _size); }
	void readMemory_x2(cl_uint * const ptr) { _readBuffer(_z2.x, ptr, sizeof(cl_uint) * VSIZE * _size); }
	void readMemory_x3(cl_uint * const ptr) { _readBuffer(_z3.x, ptr, sizeof(cl_uint) * VSIZE * _size); }

	void writeMemory_x1(const cl_uint * const ptr) { _writeBuffer(_z1.x, ptr, sizeof(cl_uint) * VSIZE * _size); }
	void writeMemory_x2(const cl_uint * const ptr) { _writeBuffer(_z2.x, ptr, sizeof(cl_uint) * VSIZE * _size); }
	void writeMemory_x3(const cl_uint * const ptr) { _writeBuffer(_z3.x, ptr, sizeof(cl_uint) * VSIZE * _size); }

public:
	void square2_P1() { _executeKernel(_square2_P1, VSIZE * _size); }
	void square2_P2() { _executeKernel(_square2_P2, VSIZE * _size); }
	void square2_P3() { _executeKernel(_square2_P3, VSIZE * _size); }
};
