/*
Copyright 2020, Yves Gallot

genefer20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include "ocl.h"

typedef cl_uint		uint32;
typedef cl_int		int32;
typedef cl_ulong	uint64;
typedef cl_long		int64;

#define VSIZE	8
#define	CSIZE	4

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
			x = device._createBuffer(CL_MEM_READ_WRITE, sizeof(uint32) * VSIZE * n);
			y = device._createBuffer(CL_MEM_READ_WRITE, sizeof(uint32) * VSIZE * n);
			wr = device._createBuffer(CL_MEM_READ_ONLY, sizeof(uint32) * n);
			wri = device._createBuffer(CL_MEM_READ_ONLY, sizeof(uint32) * n);
			d = device._createBuffer(CL_MEM_READ_WRITE, sizeof(uint32) * VSIZE * n);
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
	cl_kernel _set = nullptr, _swap = nullptr, _reset = nullptr;
	cl_kernel _square2_P1 = nullptr, _square2_P2 = nullptr, _square2_P3 = nullptr;
	cl_kernel _mul2_P1 = nullptr, _mul2_P2 = nullptr, _mul2_P3 = nullptr;
	cl_kernel _mul2cond_P1 = nullptr, _mul2cond_P2 = nullptr, _mul2cond_P3 = nullptr;
	cl_kernel  _forward2_P1 = nullptr, _forward2_P2 = nullptr, _forward2_P3 = nullptr;
	cl_kernel  _backward2_P1 = nullptr, _backward2_P2 = nullptr, _backward2_P3 = nullptr;

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

		_set = _createKernel("set");
		_swap = _createKernel("swap");
		_reset = _createKernel("reset");

		_square2_P1 = _createKernel("square2_P1");
		_setKernelArg(_square2_P1, 0, sizeof(cl_mem), &_z1.x);
		_square2_P2 = _createKernel("square2_P2");
		_setKernelArg(_square2_P2, 0, sizeof(cl_mem), &_z2.x);
		_square2_P3 = _createKernel("square2_P3");
		_setKernelArg(_square2_P3, 0, sizeof(cl_mem), &_z3.x);

		_mul2_P1 = _createKernel("mul2_P1");
		_setKernelArg(_mul2_P1, 0, sizeof(cl_mem), &_z1.y);
		_mul2_P2 = _createKernel("mul2_P2");
		_setKernelArg(_mul2_P2, 0, sizeof(cl_mem), &_z2.y);
		_mul2_P3 = _createKernel("mul2_P3");
		_setKernelArg(_mul2_P3, 0, sizeof(cl_mem), &_z3.y);

		_mul2cond_P1 = _createKernel("mul2cond_P1");
		_setKernelArg(_mul2cond_P1, 0, sizeof(cl_mem), &_z1.y);
		_setKernelArg(_mul2cond_P1, 1, sizeof(cl_mem), &_z1.x);
		_mul2cond_P2 = _createKernel("mul2cond_P2");
		_setKernelArg(_mul2cond_P2, 0, sizeof(cl_mem), &_z2.y);
		_setKernelArg(_mul2cond_P2, 1, sizeof(cl_mem), &_z2.x);
		_mul2cond_P3 = _createKernel("mul2cond_P3");
		_setKernelArg(_mul2cond_P3, 0, sizeof(cl_mem), &_z3.y);
		_setKernelArg(_mul2cond_P3, 1, sizeof(cl_mem), &_z3.x);

		_forward2_P1 = _createKernel("forward2_P1");
		_setKernelArg(_forward2_P1, 0, sizeof(cl_mem), &_z1.wr);
		_forward2_P2 = _createKernel("forward2_P2");
		_setKernelArg(_forward2_P2, 0, sizeof(cl_mem), &_z2.wr);
		_forward2_P3 = _createKernel("forward2_P3");
		_setKernelArg(_forward2_P3, 0, sizeof(cl_mem), &_z3.wr);

		_backward2_P1 = _createKernel("backward2_P1");
		_setKernelArg(_backward2_P1, 0, sizeof(cl_mem), &_z1.wri);
		_backward2_P2 = _createKernel("backward2_P2");
		_setKernelArg(_backward2_P2, 0, sizeof(cl_mem), &_z2.wri);
		_backward2_P3 = _createKernel("backward2_P3");
		_setKernelArg(_backward2_P3, 0, sizeof(cl_mem), &_z3.wri);
	}

public:
	void releaseKernels()
	{
#if defined (ocl_debug)
		std::ostringstream ss; ss << "Release ocl kernels." << std::endl;
		pio::display(ss.str());
#endif
		_releaseKernel(_set); _releaseKernel(_swap); _releaseKernel(_reset);
		_releaseKernel(_square2_P1); _releaseKernel(_square2_P2); _releaseKernel(_square2_P3);
		_releaseKernel(_mul2_P1); _releaseKernel(_mul2_P2); _releaseKernel(_mul2_P3);
		_releaseKernel(_mul2cond_P1); _releaseKernel(_mul2cond_P2); _releaseKernel(_mul2cond_P3);
		_releaseKernel(_forward2_P1); _releaseKernel(_forward2_P2); _releaseKernel(_forward2_P3);
		_releaseKernel(_backward2_P1); _releaseKernel(_backward2_P2); _releaseKernel(_backward2_P3);
	}

public:
	void writeMemory_w1(const uint32 * const wr1, const uint32 * const wri1)
	{
		_writeBuffer(_z1.wr, wr1, sizeof(uint32) * _size);
		_writeBuffer(_z1.wri, wri1, sizeof(uint32) * _size);
	}
	void writeMemory_w2(const uint32 * const wr2, const uint32 * const wri2)
	{
		_writeBuffer(_z2.wr, wr2, sizeof(uint32) * _size);
		_writeBuffer(_z2.wri, wri2, sizeof(uint32) * _size);
	}
	void writeMemory_w3(const uint32 * const wr3, const uint32 * const wri3)
	{
		_writeBuffer(_z3.wr, wr3, sizeof(uint32) * _size);
		_writeBuffer(_z3.wri, wri3, sizeof(uint32) * _size);
	}

public:
	void readMemory_x1(uint32 * const x1) { _readBuffer(_z1.x, x1, sizeof(uint32) * VSIZE * _size); }
	void readMemory_x2(uint32 * const x2) { _readBuffer(_z2.x, x2, sizeof(uint32) * VSIZE * _size); }
	void readMemory_x3(uint32 * const x3) { _readBuffer(_z3.x, x3, sizeof(uint32) * VSIZE * _size); }

public:
	void readMemory_y1(uint32 * const y1) { _readBuffer(_z1.y, y1, sizeof(uint32) * VSIZE * _size); }
	void readMemory_y2(uint32 * const y2) { _readBuffer(_z2.y, y2, sizeof(uint32) * VSIZE * _size); }
	void readMemory_y3(uint32 * const y3) { _readBuffer(_z3.y, y3, sizeof(uint32) * VSIZE * _size); }

public:
	void readMemory_d1(uint32 * const d1) { _readBuffer(_z1.d, d1, sizeof(uint32) * VSIZE * _size); }
	void readMemory_d2(uint32 * const d2) { _readBuffer(_z2.d, d2, sizeof(uint32) * VSIZE * _size); }
	void readMemory_d3(uint32 * const d3) { _readBuffer(_z3.d, d3, sizeof(uint32) * VSIZE * _size); }

public:
	void writeMemory_x1(const uint32 * const x1) { _writeBuffer(_z1.x, x1, sizeof(uint32) * VSIZE * _size); }
	void writeMemory_x2(const uint32 * const x2) { _writeBuffer(_z2.x, x2, sizeof(uint32) * VSIZE * _size); }
	void writeMemory_x3(const uint32 * const x3) { _writeBuffer(_z3.x, x3, sizeof(uint32) * VSIZE * _size); }

public:
	void writeMemory_y1(const uint32 * const y1) { _writeBuffer(_z1.y, y1, sizeof(uint32) * VSIZE * _size); }
	void writeMemory_y2(const uint32 * const y2) { _writeBuffer(_z2.y, y2, sizeof(uint32) * VSIZE * _size); }
	void writeMemory_y3(const uint32 * const y3) { _writeBuffer(_z3.y, y3, sizeof(uint32) * VSIZE * _size); }

public:
	void writeMemory_d1(const uint32 * const d1) { _writeBuffer(_z1.d, d1, sizeof(uint32) * VSIZE * _size); }
	void writeMemory_d2(const uint32 * const d2) { _writeBuffer(_z2.d, d2, sizeof(uint32) * VSIZE * _size); }
	void writeMemory_d3(const uint32 * const d3) { _writeBuffer(_z3.d, d3, sizeof(uint32) * VSIZE * _size); }

public:
	void setxy_P1()
	{
		_setKernelArg(_set, 0, sizeof(cl_mem), &_z1.x);
		_setKernelArg(_set, 1, sizeof(cl_mem), &_z1.y);
		_executeKernel(_set, VSIZE * _size);
	}
	void setxy_P2()
	{
		_setKernelArg(_set, 0, sizeof(cl_mem), &_z2.x);
		_setKernelArg(_set, 1, sizeof(cl_mem), &_z2.y);
		_executeKernel(_set, VSIZE * _size);
	}
	void setxy_P3()
	{
		_setKernelArg(_set, 0, sizeof(cl_mem), &_z3.x);
		_setKernelArg(_set, 1, sizeof(cl_mem), &_z3.y);
		_executeKernel(_set, VSIZE * _size);
	}

public:
	void setdy_P1()
	{
		_setKernelArg(_set, 0, sizeof(cl_mem), &_z1.d);
		_setKernelArg(_set, 1, sizeof(cl_mem), &_z1.y);
		_executeKernel(_set, VSIZE * _size);
	}
	void setdy_P2()
	{
		_setKernelArg(_set, 0, sizeof(cl_mem), &_z2.d);
		_setKernelArg(_set, 1, sizeof(cl_mem), &_z2.y);
		_executeKernel(_set, VSIZE * _size);
	}
	void setdy_P3()
	{
		_setKernelArg(_set, 0, sizeof(cl_mem), &_z3.d);
		_setKernelArg(_set, 1, sizeof(cl_mem), &_z3.y);
		_executeKernel(_set, VSIZE * _size);
	}

public:
	void swap_xd_P1()
	{
		_setKernelArg(_swap, 0, sizeof(cl_mem), &_z1.x);
		_setKernelArg(_swap, 1, sizeof(cl_mem), &_z1.d);
		_executeKernel(_swap, VSIZE * _size);
	}
	void swap_xd_P2()
	{
		_setKernelArg(_swap, 0, sizeof(cl_mem), &_z2.x);
		_setKernelArg(_swap, 1, sizeof(cl_mem), &_z2.d);
		_executeKernel(_swap, VSIZE * _size);
	}
	void swap_xd_P3()
	{
		_setKernelArg(_swap, 0, sizeof(cl_mem), &_z3.x);
		_setKernelArg(_swap, 1, sizeof(cl_mem), &_z3.d);
		_executeKernel(_swap, VSIZE * _size);
	}

public:
	void reset_P1(const uint32 a)
	{
		_setKernelArg(_reset, 0, sizeof(cl_mem), &_z1.x);
		_setKernelArg(_reset, 1, sizeof(uint32), &a);
		_executeKernel(_reset, VSIZE * _size);
		_setKernelArg(_reset, 0, sizeof(cl_mem), &_z1.d);
		_executeKernel(_reset, VSIZE * _size);
	}
public:
	void reset_P2(const uint32 a)
	{
		_setKernelArg(_reset, 0, sizeof(cl_mem), &_z2.x);
		_setKernelArg(_reset, 1, sizeof(uint32), &a);
		_executeKernel(_reset, VSIZE * _size);
		_setKernelArg(_reset, 0, sizeof(cl_mem), &_z2.d);
		_executeKernel(_reset, VSIZE * _size);
	}
public:
	void reset_P3(const uint32 a)
	{
		_setKernelArg(_reset, 0, sizeof(cl_mem), &_z3.x);
		_setKernelArg(_reset, 1, sizeof(uint32), &a);
		_executeKernel(_reset, VSIZE * _size);
		_setKernelArg(_reset, 0, sizeof(cl_mem), &_z3.d);
		_executeKernel(_reset, VSIZE * _size);
	}

public:
	void square2_P1() { _executeKernel(_square2_P1, VSIZE * _size); }
	void square2_P2() { _executeKernel(_square2_P2, VSIZE * _size); }
	void square2_P3() { _executeKernel(_square2_P3, VSIZE * _size); }

public:
	void mul2dy_P1()
	{
		_setKernelArg(_mul2_P1, 1, sizeof(cl_mem), &_z1.d);
		_executeKernel(_mul2_P1, VSIZE * _size);
	}
	void mul2dy_P2()
	{
		_setKernelArg(_mul2_P2, 1, sizeof(cl_mem), &_z2.d);
		_executeKernel(_mul2_P2, VSIZE * _size);
	}
	void mul2dy_P3()
	{
		_setKernelArg(_mul2_P3, 1, sizeof(cl_mem), &_z3.d);
		_executeKernel(_mul2_P3, VSIZE * _size);
	}

public:
	void mul2xy_P1()
	{
		_setKernelArg(_mul2_P1, 1, sizeof(cl_mem), &_z1.x);
		_executeKernel(_mul2_P1, VSIZE * _size);
	}
	void mul2xy_P2()
	{
		_setKernelArg(_mul2_P2, 1, sizeof(cl_mem), &_z2.x);
		_executeKernel(_mul2_P2, VSIZE * _size);
	}
	void mul2xy_P3()
	{
		_setKernelArg(_mul2_P3, 1, sizeof(cl_mem), &_z3.x);
		_executeKernel(_mul2_P3, VSIZE * _size);
	}

public:
	void mul2condxy_P1(const uint64 c)
	{
		_setKernelArg(_mul2cond_P1, 2, sizeof(cl_ulong), &c);
		_executeKernel(_mul2cond_P1, VSIZE * _size);
	}
	void mul2condxy_P2(const uint64 c)
	{
		_setKernelArg(_mul2cond_P2, 2, sizeof(cl_ulong), &c);
		_executeKernel(_mul2cond_P2, VSIZE * _size);
	}
	void mul2condxy_P3(const uint64 c)
	{
		_setKernelArg(_mul2cond_P3, 2, sizeof(cl_ulong), &c);
		_executeKernel(_mul2cond_P3, VSIZE * _size);
	}

private:
	void forward(cl_kernel kernel, const uint32 s, const int32 lm)
	{
		const uint32 m = uint32(1) << lm;
		_setKernelArg(kernel, 2, sizeof(uint32), &s);
		_setKernelArg(kernel, 3, sizeof(uint32), &m);
		_setKernelArg(kernel, 4, sizeof(int32), &lm);
		_executeKernel(kernel, VSIZE * _size / 2);
	}

	void backward(cl_kernel kernel, const uint32 s, const int32 lm)
	{
		const uint32 m = uint32(1) << lm;
		_setKernelArg(kernel, 2, sizeof(uint32), &s);
		_setKernelArg(kernel, 3, sizeof(uint32), &m);
		_setKernelArg(kernel, 4, sizeof(int32), &lm);
		_executeKernel(kernel, VSIZE * _size / 2);
	}
	
public:
	void forward2x_P1(const uint32 s, const int32 lm)
	{
		_setKernelArg(_forward2_P1, 1, sizeof(cl_mem), &_z1.x);
		forward(_forward2_P1, s, lm);
	}
	void forward2x_P2(const uint32 s, const int32 lm)
	{
		_setKernelArg(_forward2_P2, 1, sizeof(cl_mem), &_z2.x);
		forward(_forward2_P2, s, lm);
	}
	void forward2x_P3(const uint32 s, const int32 lm)
	{
		_setKernelArg(_forward2_P3, 1, sizeof(cl_mem), &_z3.x);
		forward(_forward2_P3, s, lm);
	}

public:
	void forward2y_P1(const uint32 s, const int32 lm)
	{
		_setKernelArg(_forward2_P1, 1, sizeof(cl_mem), &_z1.y);
		forward(_forward2_P1, s, lm);
	}
	void forward2y_P2(const uint32 s, const int32 lm)
	{
		_setKernelArg(_forward2_P2, 1, sizeof(cl_mem), &_z2.y);
		forward(_forward2_P2, s, lm);
	}
	void forward2y_P3(const uint32 s, const int32 lm)
	{
		_setKernelArg(_forward2_P3, 1, sizeof(cl_mem), &_z3.y);
		forward(_forward2_P3, s, lm);
	}

public:
	void forward2d_P1(const uint32 s, const int32 lm)
	{
		_setKernelArg(_forward2_P1, 1, sizeof(cl_mem), &_z1.d);
		forward(_forward2_P1, s, lm);
	}
	void forward2d_P2(const uint32 s, const int32 lm)
	{
		_setKernelArg(_forward2_P2, 1, sizeof(cl_mem), &_z2.d);
		forward(_forward2_P2, s, lm);
	}
	void forward2d_P3(const uint32 s, const int32 lm)
	{
		_setKernelArg(_forward2_P3, 1, sizeof(cl_mem), &_z3.d);
		forward(_forward2_P3, s, lm);
	}

public:
	void backward2x_P1(const uint32 s, const int32 lm)
	{
		_setKernelArg(_backward2_P1, 1, sizeof(cl_mem), &_z1.x);
		backward(_backward2_P1, s, lm);
	}
	void backward2x_P2(const uint32 s, const int32 lm)
	{
		_setKernelArg(_backward2_P2, 1, sizeof(cl_mem), &_z2.x);
		backward(_backward2_P2, s, lm);
	}
	void backward2x_P3(const uint32 s, const int32 lm)
	{
		_setKernelArg(_backward2_P3, 1, sizeof(cl_mem), &_z3.x);
		backward(_backward2_P3, s, lm);
	}

public:
	void backward2d_P1(const uint32 s, const int32 lm)
	{
		_setKernelArg(_backward2_P1, 1, sizeof(cl_mem), &_z1.d);
		backward(_backward2_P1, s, lm);
	}
	void backward2d_P2(const uint32 s, const int32 lm)
	{
		_setKernelArg(_backward2_P2, 1, sizeof(cl_mem), &_z2.d);
		backward(_backward2_P2, s, lm);
	}
	void backward2d_P3(const uint32 s, const int32 lm)
	{
		_setKernelArg(_backward2_P3, 1, sizeof(cl_mem), &_z3.d);
		backward(_backward2_P3, s, lm);
	}
};
