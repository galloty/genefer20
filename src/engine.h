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
typedef cl_uint2	uint32_2;

#define VSIZE	64
#define	CSIZE	4

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
	cl_mem _res, _bb_inv, _f;
	cl_kernel _set = nullptr, _swap = nullptr, _reset = nullptr;
	cl_kernel _square2_P1 = nullptr, _square2_P2 = nullptr, _square2_P3 = nullptr;
	cl_kernel _mul2_P1 = nullptr, _mul2_P2 = nullptr, _mul2_P3 = nullptr;
	cl_kernel _mul2cond_P1 = nullptr, _mul2cond_P2 = nullptr, _mul2cond_P3 = nullptr;
	cl_kernel  _forward2_P1 = nullptr, _forward2_P2 = nullptr, _forward2_P3 = nullptr;
	cl_kernel  _backward2_P1 = nullptr, _backward2_P2 = nullptr, _backward2_P3 = nullptr;
	cl_kernel _normalize2a = nullptr, _normalize2b = nullptr, _normalize3a = nullptr, _normalize3b = nullptr;

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
		_res = _createBuffer(CL_MEM_READ_WRITE, sizeof(uint32) * VSIZE * size);
		_bb_inv = _createBuffer(CL_MEM_READ_ONLY, sizeof(uint32_2) * VSIZE);
		_f = _createBuffer(CL_MEM_READ_WRITE, sizeof(int64) * VSIZE / CSIZE * size);
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
			_releaseBuffer(_res);
			_releaseBuffer(_bb_inv);
			_releaseBuffer(_f);
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

		_normalize2a = _createKernel("normalize2a");
		_setKernelArg(_normalize2a, 0, sizeof(cl_mem), &_bb_inv);
		_setKernelArg(_normalize2a, 1, sizeof(cl_mem), &_f);

		_normalize2b = _createKernel("normalize2b");
		_setKernelArg(_normalize2b, 0, sizeof(cl_mem), &_bb_inv);
		_setKernelArg(_normalize2b, 1, sizeof(cl_mem), &_f);

		_normalize3a = _createKernel("normalize3a");
		_setKernelArg(_normalize3a, 0, sizeof(cl_mem), &_bb_inv);
		_setKernelArg(_normalize3a, 1, sizeof(cl_mem), &_f);

		_normalize3b = _createKernel("normalize3b");
		_setKernelArg(_normalize3b, 0, sizeof(cl_mem), &_bb_inv);
		_setKernelArg(_normalize3b, 1, sizeof(cl_mem), &_f);
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
		_releaseKernel(_normalize2a); _releaseKernel(_normalize2b); _releaseKernel(_normalize3a); _releaseKernel(_normalize3b);
	}

public:
	void setParam_bs(const int32 s)
	{
		_setKernelArg(_normalize2a, 4, sizeof(int32), &s);
		_setKernelArg(_normalize2b, 4, sizeof(int32), &s);
		_setKernelArg(_normalize3a, 5, sizeof(int32), &s);
		_setKernelArg(_normalize3b, 5, sizeof(int32), &s);
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

	void writeMemory_b(const uint32_2 * const bb_inv) { _writeBuffer(_bb_inv, bb_inv, sizeof(uint32_2) * VSIZE); }

public:
	void readMemory_x1(uint32 * const x1) { _readBuffer(_z1.x, x1, sizeof(uint32) * VSIZE * _size); }
	void readMemory_d1(uint32 * const d1) { _readBuffer(_z1.d, d1, sizeof(uint32) * VSIZE * _size); }
	void readMemory_res(uint32 * const res) { _readBuffer(_res, res, sizeof(uint32) * VSIZE * _size); }

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
	void setxres_P1()
	{
		_setKernelArg(_set, 0, sizeof(cl_mem), &_z1.x);
		_setKernelArg(_set, 1, sizeof(cl_mem), &_res);
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

public:
	void normalize2ax()
	{
		_setKernelArg(_normalize2a, 2, sizeof(cl_mem), &_z1.x);
		_setKernelArg(_normalize2a, 3, sizeof(cl_mem), &_z2.x);
		_executeKernel(_normalize2a, VSIZE / CSIZE * _size);
	}
	void normalize2bx()
	{
		_setKernelArg(_normalize2b, 2, sizeof(cl_mem), &_z1.x);
		_setKernelArg(_normalize2b, 3, sizeof(cl_mem), &_z2.x);
		_executeKernel(_normalize2b, VSIZE / CSIZE * _size);
	}
	void normalize2ad()
	{
		_setKernelArg(_normalize2a, 2, sizeof(cl_mem), &_z1.d);
		_setKernelArg(_normalize2a, 3, sizeof(cl_mem), &_z2.d);
		_executeKernel(_normalize2a, VSIZE / CSIZE * _size);
	}
	void normalize2bd()
	{
		_setKernelArg(_normalize2b, 2, sizeof(cl_mem), &_z1.d);
		_setKernelArg(_normalize2b, 3, sizeof(cl_mem), &_z2.d);
		_executeKernel(_normalize2b, VSIZE / CSIZE * _size);
	}

public:
	void normalize3ax()
	{
		_setKernelArg(_normalize3a, 2, sizeof(cl_mem), &_z1.x);
		_setKernelArg(_normalize3a, 3, sizeof(cl_mem), &_z2.x);
		_setKernelArg(_normalize3a, 4, sizeof(cl_mem), &_z3.x);
		_executeKernel(_normalize3a, VSIZE / CSIZE * _size);
	}
	void normalize3bx()
	{
		_setKernelArg(_normalize3b, 2, sizeof(cl_mem), &_z1.x);
		_setKernelArg(_normalize3b, 3, sizeof(cl_mem), &_z2.x);
		_setKernelArg(_normalize3b, 4, sizeof(cl_mem), &_z3.x);
		_executeKernel(_normalize3b, VSIZE / CSIZE * _size);
	}
	void normalize3ad()
	{
		_setKernelArg(_normalize3a, 2, sizeof(cl_mem), &_z1.d);
		_setKernelArg(_normalize3a, 3, sizeof(cl_mem), &_z2.d);
		_setKernelArg(_normalize3a, 4, sizeof(cl_mem), &_z3.d);
		_executeKernel(_normalize3a, VSIZE / CSIZE * _size);
	}
	void normalize3bd()
	{
		_setKernelArg(_normalize3b, 2, sizeof(cl_mem), &_z1.d);
		_setKernelArg(_normalize3b, 3, sizeof(cl_mem), &_z2.d);
		_setKernelArg(_normalize3b, 4, sizeof(cl_mem), &_z3.d);
		_executeKernel(_normalize3b, VSIZE / CSIZE * _size);
	}
};
