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

#define VSIZE	32
#define	CSIZE	8

class engine : public ocl::device
{
private:
	size_t _size = 0;
	cl_mem _x12 = nullptr, _x3 = nullptr, _y12 = nullptr, _y3 = nullptr, _d12 = nullptr, _d3 = nullptr;
	cl_mem _wr12 = nullptr, _wr3 = nullptr, _wri12 = nullptr, _wri3 = nullptr, _res = nullptr, _bb_inv = nullptr, _f = nullptr;
	cl_kernel _set_P1 = nullptr, _setxy_P12 = nullptr, _setdy_P12 = nullptr, _setxy_P123 = nullptr, _setdy_P123 = nullptr;
	cl_kernel _swap_P12 = nullptr, _swap_P123 = nullptr, _reset_P12 = nullptr, _reset_P3 = nullptr;
	cl_kernel _square2_P12 = nullptr, _square2_P123 = nullptr, _square4_P12 = nullptr, _square4_P123 = nullptr;
	cl_kernel _mul2cond_P12 = nullptr, _mul2cond_P123 = nullptr, _mul4cond_P12 = nullptr, _mul4cond_P123 = nullptr;
	cl_kernel _mul2_P12 = nullptr, _mul2_P3 = nullptr;
	cl_kernel _forward2_P12 = nullptr, _forward2_P3 = nullptr, _forward4_P12 = nullptr, _forward4_P123 = nullptr, _backward2_P12 = nullptr, _backward2_P3 = nullptr, _backward4_P12 = nullptr, _backward4_P123 = nullptr;
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
		_x12 = _createBuffer(CL_MEM_READ_WRITE, sizeof(uint32_2) * VSIZE * size);
		_x3 = _createBuffer(CL_MEM_READ_WRITE, sizeof(uint32) * VSIZE * size);
		_y12 = _createBuffer(CL_MEM_READ_WRITE, sizeof(uint32_2) * VSIZE * size);
		_y3 = _createBuffer(CL_MEM_READ_WRITE, sizeof(uint32) * VSIZE * size);
		_d12 = _createBuffer(CL_MEM_READ_WRITE, sizeof(uint32_2) * VSIZE * size);
		_d3 = _createBuffer(CL_MEM_READ_WRITE, sizeof(uint32) * VSIZE * size);
		_wr12 = _createBuffer(CL_MEM_READ_ONLY, sizeof(uint32_2) * size);
		_wr3 = _createBuffer(CL_MEM_READ_ONLY, sizeof(uint32) * size);
		_wri12 = _createBuffer(CL_MEM_READ_ONLY, sizeof(uint32_2) * size);
		_wri3 = _createBuffer(CL_MEM_READ_ONLY, sizeof(uint32) * size);
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
			_releaseBuffer(_x12); _releaseBuffer(_x3);
			_releaseBuffer(_y12); _releaseBuffer(_y3);
			_releaseBuffer(_d12); _releaseBuffer(_d3);
			_releaseBuffer(_wr12); _releaseBuffer(_wr3); _releaseBuffer(_wri12); _releaseBuffer(_wri3);
			_releaseBuffer(_res); _releaseBuffer(_bb_inv); _releaseBuffer(_f);
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
		_set_P1 = _createKernel("set_P1");
		_setKernelArg(_set_P1, 0, sizeof(cl_mem), &_x12);
		_setKernelArg(_set_P1, 1, sizeof(cl_mem), &_res);

		_setxy_P12 = _createKernel("set_P12");
		_setKernelArg(_setxy_P12, 0, sizeof(cl_mem), &_x12);
		_setKernelArg(_setxy_P12, 1, sizeof(cl_mem), &_y12);

		_setdy_P12 = _createKernel("set_P12");
		_setKernelArg(_setdy_P12, 0, sizeof(cl_mem), &_d12);
		_setKernelArg(_setdy_P12, 1, sizeof(cl_mem), &_y12);

		_setxy_P123 = _createKernel("set_P123");
		_setKernelArg(_setxy_P123, 0, sizeof(cl_mem), &_x12);
		_setKernelArg(_setxy_P123, 1, sizeof(cl_mem), &_x3);
		_setKernelArg(_setxy_P123, 2, sizeof(cl_mem), &_y12);
		_setKernelArg(_setxy_P123, 3, sizeof(cl_mem), &_y3);

		_setdy_P123 = _createKernel("set_P123");
		_setKernelArg(_setdy_P123, 0, sizeof(cl_mem), &_d12);
		_setKernelArg(_setdy_P123, 1, sizeof(cl_mem), &_d3);
		_setKernelArg(_setdy_P123, 2, sizeof(cl_mem), &_y12);
		_setKernelArg(_setdy_P123, 3, sizeof(cl_mem), &_y3);

		_swap_P12 = _createKernel("swap_P12");
		_swap_P123 = _createKernel("swap_P123");
		_reset_P12 = _createKernel("reset_P12");
		_reset_P3 = _createKernel("reset_P3");

		_square2_P12 = _createKernel("square2_P12");
		_setKernelArg(_square2_P12, 0, sizeof(cl_mem), &_wr12);
		_setKernelArg(_square2_P12, 1, sizeof(cl_mem), &_wri12);
		_square2_P123 = _createKernel("square2_P123");
		_setKernelArg(_square2_P123, 0, sizeof(cl_mem), &_wr12);
		_setKernelArg(_square2_P123, 1, sizeof(cl_mem), &_wr3);
		_setKernelArg(_square2_P123, 2, sizeof(cl_mem), &_wri12);
		_setKernelArg(_square2_P123, 3, sizeof(cl_mem), &_wri3);
		_square4_P12 = _createKernel("square4_P12");
		_setKernelArg(_square4_P12, 0, sizeof(cl_mem), &_wr12);
		_setKernelArg(_square4_P12, 1, sizeof(cl_mem), &_wri12);
		_square4_P123 = _createKernel("square4_P123");
		_setKernelArg(_square4_P123, 0, sizeof(cl_mem), &_wr12);
		_setKernelArg(_square4_P123, 1, sizeof(cl_mem), &_wr3);
		_setKernelArg(_square4_P123, 2, sizeof(cl_mem), &_wri12);
		_setKernelArg(_square4_P123, 3, sizeof(cl_mem), &_wri3);

		_mul2cond_P12 = _createKernel("mul2cond_P12");
		_setKernelArg(_mul2cond_P12, 0, sizeof(cl_mem), &_wr12);
		_setKernelArg(_mul2cond_P12, 1, sizeof(cl_mem), &_wri12);
		_mul2cond_P123 = _createKernel("mul2cond_P123");
		_setKernelArg(_mul2cond_P123, 0, sizeof(cl_mem), &_wr12);
		_setKernelArg(_mul2cond_P123, 1, sizeof(cl_mem), &_wr3);
		_setKernelArg(_mul2cond_P123, 2, sizeof(cl_mem), &_wri12);
		_setKernelArg(_mul2cond_P123, 3, sizeof(cl_mem), &_wri3);
		_mul4cond_P12 = _createKernel("mul4cond_P12");
		_setKernelArg(_mul4cond_P12, 0, sizeof(cl_mem), &_wr12);
		_setKernelArg(_mul4cond_P12, 1, sizeof(cl_mem), &_wri12);
		_mul4cond_P123 = _createKernel("mul4cond_P123");
		_setKernelArg(_mul4cond_P123, 0, sizeof(cl_mem), &_wr12);
		_setKernelArg(_mul4cond_P123, 1, sizeof(cl_mem), &_wr3);
		_setKernelArg(_mul4cond_P123, 2, sizeof(cl_mem), &_wri12);
		_setKernelArg(_mul4cond_P123, 3, sizeof(cl_mem), &_wri3);

		_mul2_P12 = _createKernel("mul2_P12");
		_setKernelArg(_mul2_P12, 0, sizeof(cl_mem), &_y12);
		_mul2_P3 = _createKernel("mul2_P3");
		_setKernelArg(_mul2_P3, 0, sizeof(cl_mem), &_y3);

		_forward2_P12 = _createKernel("forward2_P12");
		_setKernelArg(_forward2_P12, 0, sizeof(cl_mem), &_wr12);
		_forward2_P3 = _createKernel("forward2_P3");
		_setKernelArg(_forward2_P3, 0, sizeof(cl_mem), &_wr3);
		_forward4_P12 = _createKernel("forward4_P12");
		_setKernelArg(_forward4_P12, 0, sizeof(cl_mem), &_wr12);
		_forward4_P123 = _createKernel("forward4_P123");
		_setKernelArg(_forward4_P123, 0, sizeof(cl_mem), &_wr12);
		_setKernelArg(_forward4_P123, 1, sizeof(cl_mem), &_wr3);

		_backward2_P12 = _createKernel("backward2_P12");
		_setKernelArg(_backward2_P12, 0, sizeof(cl_mem), &_wri12);
		_backward2_P3 = _createKernel("backward2_P3");
		_setKernelArg(_backward2_P3, 0, sizeof(cl_mem), &_wri3);
		_backward4_P12 = _createKernel("backward4_P12");
		_setKernelArg(_backward4_P12, 0, sizeof(cl_mem), &_wri12);
		_backward4_P123 = _createKernel("backward4_P123");
		_setKernelArg(_backward4_P123, 0, sizeof(cl_mem), &_wri12);
		_setKernelArg(_backward4_P123, 1, sizeof(cl_mem), &_wri3);

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
		_releaseKernel(_set_P1); _releaseKernel(_setxy_P12); _releaseKernel(_setdy_P12); _releaseKernel(_setxy_P123); _releaseKernel(_setdy_P123);
		_releaseKernel(_swap_P12); _releaseKernel(_swap_P123); _releaseKernel(_reset_P12); _releaseKernel(_reset_P3);
		_releaseKernel(_square2_P12); _releaseKernel(_square2_P123); _releaseKernel(_square4_P12); _releaseKernel(_square4_P123);
		_releaseKernel(_mul2cond_P12); _releaseKernel(_mul2cond_P123); _releaseKernel(_mul4cond_P12); _releaseKernel(_mul4cond_P123);
		_releaseKernel(_mul2_P12); _releaseKernel(_mul2_P3);
		_releaseKernel(_forward2_P12); _releaseKernel(_forward2_P3); _releaseKernel(_forward4_P12); _releaseKernel(_forward4_P123); _releaseKernel(_backward2_P12); _releaseKernel(_backward2_P3); _releaseKernel(_backward4_P12); _releaseKernel(_backward4_P123);
		_releaseKernel(_normalize2a); _releaseKernel(_normalize2b); _releaseKernel(_normalize3a); _releaseKernel(_normalize3b);
	}

public:
	void setParam_bs(const int32 s)
	{
		_setKernelArg(_normalize2a, 3, sizeof(int32), &s);
		_setKernelArg(_normalize2b, 3, sizeof(int32), &s);
		_setKernelArg(_normalize3a, 4, sizeof(int32), &s);
		_setKernelArg(_normalize3b, 4, sizeof(int32), &s);
	}

public:
	void writeMemory_w(const uint32_2 * const wr12, const uint32 * const wr3, const uint32_2 * const wri12, const uint32 * const wri3)
	{
		_writeBuffer(_wr12, wr12, sizeof(uint32_2) * _size);
		_writeBuffer(_wr3, wr3, sizeof(uint32) * _size);
		_writeBuffer(_wri12, wri12, sizeof(uint32_2) * _size);
		_writeBuffer(_wri3, wri3, sizeof(uint32) * _size);
	}

	void writeMemory_b(const uint32_2 * const bb_inv) { _writeBuffer(_bb_inv, bb_inv, sizeof(uint32_2) * VSIZE); }

public:
	void readMemory_x12(uint32_2 * const x12) { _readBuffer(_x12, x12, sizeof(uint32_2) * VSIZE * _size); }
	void readMemory_d12(uint32_2 * const d12) { _readBuffer(_d12, d12, sizeof(uint32_2) * VSIZE * _size); }
	void readMemory_res(uint32 * const res) { _readBuffer(_res, res, sizeof(uint32) * VSIZE * _size); }

public:
	void setxy_P12() { _executeKernel(_setxy_P12, VSIZE * _size); }
	void setxy_P123() { _executeKernel(_setxy_P123, VSIZE * _size); }
	void setxres_P1() { _executeKernel(_set_P1, VSIZE * _size); }
	void setdy_P12() { _executeKernel(_setdy_P12, VSIZE * _size); }
	void setdy_P123() { _executeKernel(_setdy_P123, VSIZE * _size); }

public:
	void swap_xd_P12()
	{
		_setKernelArg(_swap_P12, 0, sizeof(cl_mem), &_x12);
		_setKernelArg(_swap_P12, 1, sizeof(cl_mem), &_d12);
		_executeKernel(_swap_P12, VSIZE * _size);
	}
	void swap_xd_P123()
	{
		_setKernelArg(_swap_P123, 0, sizeof(cl_mem), &_x12);
		_setKernelArg(_swap_P123, 1, sizeof(cl_mem), &_x3);
		_setKernelArg(_swap_P123, 2, sizeof(cl_mem), &_d12);
		_setKernelArg(_swap_P123, 3, sizeof(cl_mem), &_d3);
		_executeKernel(_swap_P123, VSIZE * _size);
	}

public:
	void reset_P12(const uint32 a)
	{
		_setKernelArg(_reset_P12, 0, sizeof(cl_mem), &_x12);
		_setKernelArg(_reset_P12, 1, sizeof(uint32), &a);
		_executeKernel(_reset_P12, VSIZE * _size);
		_setKernelArg(_reset_P12, 0, sizeof(cl_mem), &_d12);
		_executeKernel(_reset_P12, VSIZE * _size);
	}
public:
	void reset_P3(const uint32 a)
	{
		_setKernelArg(_reset_P3, 0, sizeof(cl_mem), &_x3);
		_setKernelArg(_reset_P3, 1, sizeof(uint32), &a);
		_executeKernel(_reset_P3, VSIZE * _size);
		_setKernelArg(_reset_P3, 0, sizeof(cl_mem), &_d3);
		_executeKernel(_reset_P3, VSIZE * _size);
	}

public:
	void square2x_P12()
	{
		_setKernelArg(_square2_P12, 2, sizeof(cl_mem), &_x12);
		_executeKernel(_square2_P12, VSIZE * _size / 2);
	}
	void square2x_P123()
	{
		_setKernelArg(_square2_P123, 4, sizeof(cl_mem), &_x12);
		_setKernelArg(_square2_P123, 5, sizeof(cl_mem), &_x3);
		_executeKernel(_square2_P123, VSIZE * _size / 2);
	}
	void square4x_P12()
	{
		_setKernelArg(_square4_P12, 2, sizeof(cl_mem), &_x12);
		_executeKernel(_square4_P12, VSIZE * _size / 4);
	}
	void square4x_P123()
	{
		_setKernelArg(_square4_P123, 4, sizeof(cl_mem), &_x12);
		_setKernelArg(_square4_P123, 5, sizeof(cl_mem), &_x3);
		_executeKernel(_square4_P123, VSIZE * _size / 4);
	}

public:
	void mul2condxy_P12(const uint64 c)
	{
		_setKernelArg(_mul2cond_P12, 2, sizeof(cl_mem), &_y12);
		_setKernelArg(_mul2cond_P12, 3, sizeof(cl_mem), &_x12);
		_setKernelArg(_mul2cond_P12, 4, sizeof(cl_ulong), &c);
		_executeKernel(_mul2cond_P12, VSIZE * _size / 2);
	}
	void mul2condxy_P123(const uint64 c)
	{
		_setKernelArg(_mul2cond_P123, 4, sizeof(cl_mem), &_y12);
		_setKernelArg(_mul2cond_P123, 5, sizeof(cl_mem), &_y3);
		_setKernelArg(_mul2cond_P123, 6, sizeof(cl_mem), &_x12);
		_setKernelArg(_mul2cond_P123, 7, sizeof(cl_mem), &_x3);
		_setKernelArg(_mul2cond_P123, 8, sizeof(cl_ulong), &c);
		_executeKernel(_mul2cond_P123, VSIZE * _size / 2);
	}
	void mul4condxy_P12(const uint64 c)
	{
		_setKernelArg(_mul4cond_P12, 2, sizeof(cl_mem), &_y12);
		_setKernelArg(_mul4cond_P12, 3, sizeof(cl_mem), &_x12);
		_setKernelArg(_mul4cond_P12, 4, sizeof(cl_ulong), &c);
		_executeKernel(_mul4cond_P12, VSIZE * _size / 4);
	}
	void mul4condxy_P123(const uint64 c)
	{
		_setKernelArg(_mul4cond_P123, 4, sizeof(cl_mem), &_y12);
		_setKernelArg(_mul4cond_P123, 5, sizeof(cl_mem), &_y3);
		_setKernelArg(_mul4cond_P123, 6, sizeof(cl_mem), &_x12);
		_setKernelArg(_mul4cond_P123, 7, sizeof(cl_mem), &_x3);
		_setKernelArg(_mul4cond_P123, 8, sizeof(cl_ulong), &c);
		_executeKernel(_mul4cond_P123, VSIZE * _size / 4);
	}

public:
	void mul2dy_P12()
	{
		_setKernelArg(_mul2_P12, 1, sizeof(cl_mem), &_d12);
		_executeKernel(_mul2_P12, VSIZE * _size);
	}
	void mul2dy_P3()
	{
		_setKernelArg(_mul2_P3, 1, sizeof(cl_mem), &_d3);
		_executeKernel(_mul2_P3, VSIZE * _size);
	}

public:
	void mul2xy_P12()
	{
		_setKernelArg(_mul2_P12, 1, sizeof(cl_mem), &_x12);
		_executeKernel(_mul2_P12, VSIZE * _size);
	}
	void mul2xy_P3()
	{
		_setKernelArg(_mul2_P3, 1, sizeof(cl_mem), &_x3);
		_executeKernel(_mul2_P3, VSIZE * _size);
	}

private:
	void forward2_P12(cl_mem x12, const uint32 s, const int32 lm)
	{
		const uint32 m = uint32(1) << lm;
		_setKernelArg(_forward2_P12, 1, sizeof(cl_mem), &x12);
		_setKernelArg(_forward2_P12, 2, sizeof(uint32), &s);
		_setKernelArg(_forward2_P12, 3, sizeof(uint32), &m);
		_setKernelArg(_forward2_P12, 4, sizeof(int32), &lm);
		_executeKernel(_forward2_P12, VSIZE * _size / 2);
	}
	void forward2_P3(cl_mem x3, const uint32 s, const int32 lm)
	{
		const uint32 m = uint32(1) << lm;
		_setKernelArg(_forward2_P3, 1, sizeof(cl_mem), &x3);
		_setKernelArg(_forward2_P3, 2, sizeof(uint32), &s);
		_setKernelArg(_forward2_P3, 3, sizeof(uint32), &m);
		_setKernelArg(_forward2_P3, 4, sizeof(int32), &lm);
		_executeKernel(_forward2_P3, VSIZE * _size / 2);
	}
	void forward4_P12(cl_mem x12, const uint32 s, const int32 lm)
	{
		const uint32 m = uint32(1) << lm;
		_setKernelArg(_forward4_P12, 1, sizeof(cl_mem), &x12);
		_setKernelArg(_forward4_P12, 2, sizeof(uint32), &s);
		_setKernelArg(_forward4_P12, 3, sizeof(uint32), &m);
		_setKernelArg(_forward4_P12, 4, sizeof(int32), &lm);
		_executeKernel(_forward4_P12, VSIZE * _size / 4);
	}
	void forward4_P123(cl_mem x12, cl_mem x3, const uint32 s, const int32 lm)
	{
		const uint32 m = uint32(1) << lm;
		_setKernelArg(_forward4_P123, 2, sizeof(cl_mem), &x12);
		_setKernelArg(_forward4_P123, 3, sizeof(cl_mem), &x3);
		_setKernelArg(_forward4_P123, 4, sizeof(uint32), &s);
		_setKernelArg(_forward4_P123, 5, sizeof(uint32), &m);
		_setKernelArg(_forward4_P123, 6, sizeof(int32), &lm);
		_executeKernel(_forward4_P123, VSIZE * _size / 4);
	}

private:
	void backward2_P12(cl_mem x12, const uint32 s, const int32 lm)
	{
		const uint32 m = uint32(1) << lm;
		_setKernelArg(_backward2_P12, 1, sizeof(cl_mem), &x12);
		_setKernelArg(_backward2_P12, 2, sizeof(uint32), &s);
		_setKernelArg(_backward2_P12, 3, sizeof(uint32), &m);
		_setKernelArg(_backward2_P12, 4, sizeof(int32), &lm);
		_executeKernel(_backward2_P12, VSIZE * _size / 2);
	}
	void backward2_P3(cl_mem x3, const uint32 s, const int32 lm)
	{
		const uint32 m = uint32(1) << lm;
		_setKernelArg(_backward2_P3, 1, sizeof(cl_mem), &x3);
		_setKernelArg(_backward2_P3, 2, sizeof(uint32), &s);
		_setKernelArg(_backward2_P3, 3, sizeof(uint32), &m);
		_setKernelArg(_backward2_P3, 4, sizeof(int32), &lm);
		_executeKernel(_backward2_P3, VSIZE * _size / 2);
	}
	void backward4_P12(cl_mem x12, const uint32 s, const int32 lm)
	{
		const uint32 m = uint32(1) << lm;
		_setKernelArg(_backward4_P12, 1, sizeof(cl_mem), &x12);
		_setKernelArg(_backward4_P12, 2, sizeof(uint32), &s);
		_setKernelArg(_backward4_P12, 3, sizeof(uint32), &m);
		_setKernelArg(_backward4_P12, 4, sizeof(int32), &lm);
		_executeKernel(_backward4_P12, VSIZE * _size / 4);
	}
	void backward4_P123(cl_mem x12, cl_mem x3, const uint32 s, const int32 lm)
	{
		const uint32 m = uint32(1) << lm;
		_setKernelArg(_backward4_P123, 2, sizeof(cl_mem), &x12);
		_setKernelArg(_backward4_P123, 3, sizeof(cl_mem), &x3);
		_setKernelArg(_backward4_P123, 4, sizeof(uint32), &s);
		_setKernelArg(_backward4_P123, 5, sizeof(uint32), &m);
		_setKernelArg(_backward4_P123, 6, sizeof(int32), &lm);
		_executeKernel(_backward4_P123, VSIZE * _size / 4);
	}

public:
	void forward2x_P12(const uint32 s, const int32 lm) { forward2_P12(_x12, s, lm); }
	void forward2x_P3(const uint32 s, const int32 lm) { forward2_P3(_x3, s, lm); }
	void forward2y_P12(const uint32 s, const int32 lm) { forward2_P12(_y12, s, lm); }
	void forward2y_P3(const uint32 s, const int32 lm) { forward2_P3(_y3, s, lm); }
	void forward2d_P12(const uint32 s, const int32 lm) { forward2_P12(_d12, s, lm); }
	void forward2d_P3(const uint32 s, const int32 lm) { forward2_P3(_d3, s, lm); }
	void forward4x_P12(const uint32 s, const int32 lm) { forward4_P12(_x12, s, lm); }
	void forward4x_P123(const uint32 s, const int32 lm) { forward4_P123(_x12, _x3, s, lm); }
	void forward4y_P12(const uint32 s, const int32 lm) { forward4_P12(_y12, s, lm); }
	void forward4y_P123(const uint32 s, const int32 lm) { forward4_P123(_y12, _y3, s, lm); }
	void forward4d_P12(const uint32 s, const int32 lm) { forward4_P12(_d12, s, lm); }
	void forward4d_P123(const uint32 s, const int32 lm) { forward4_P123(_d12, _d3, s, lm); }

public:
	void backward2x_P12(const uint32 s, const int32 lm) { backward2_P12(_x12, s, lm); }
	void backward2x_P3(const uint32 s, const int32 lm) { backward2_P3(_x3, s, lm); }
	void backward2d_P12(const uint32 s, const int32 lm) { backward2_P12(_d12, s, lm); }
	void backward2d_P3(const uint32 s, const int32 lm) { backward2_P3(_d3, s, lm); }
	void backward4x_P12(const uint32 s, const int32 lm) { backward4_P12(_x12, s, lm); }
	void backward4x_P123(const uint32 s, const int32 lm) { backward4_P123(_x12, _x3, s, lm); }
	void backward4d_P12(const uint32 s, const int32 lm) { backward4_P12(_d12, s, lm); }
	void backward4d_P123(const uint32 s, const int32 lm) { backward4_P123(_d12, _d3, s, lm); }

public:
	void normalize2ax()
	{
		_setKernelArg(_normalize2a, 2, sizeof(cl_mem), &_x12);
		_executeKernel(_normalize2a, VSIZE / CSIZE * _size);
	}
	void normalize2bx()
	{
		_setKernelArg(_normalize2b, 2, sizeof(cl_mem), &_x12);
		_executeKernel(_normalize2b, VSIZE / CSIZE * _size);
	}
	void normalize2ad()
	{
		_setKernelArg(_normalize2a, 2, sizeof(cl_mem), &_d12);
		_executeKernel(_normalize2a, VSIZE / CSIZE * _size);
	}
	void normalize2bd()
	{
		_setKernelArg(_normalize2b, 2, sizeof(cl_mem), &_d12);
		_executeKernel(_normalize2b, VSIZE / CSIZE * _size);
	}

public:
	void normalize3ax()
	{
		_setKernelArg(_normalize3a, 2, sizeof(cl_mem), &_x12);
		_setKernelArg(_normalize3a, 3, sizeof(cl_mem), &_x3);
		_executeKernel(_normalize3a, VSIZE / CSIZE * _size);
	}
	void normalize3bx()
	{
		_setKernelArg(_normalize3b, 2, sizeof(cl_mem), &_x12);
		_setKernelArg(_normalize3b, 3, sizeof(cl_mem), &_x3);
		_executeKernel(_normalize3b, VSIZE / CSIZE * _size);
	}
	void normalize3ad()
	{
		_setKernelArg(_normalize3a, 2, sizeof(cl_mem), &_d12);
		_setKernelArg(_normalize3a, 3, sizeof(cl_mem), &_d3);
		_executeKernel(_normalize3a, VSIZE / CSIZE * _size);
	}
	void normalize3bd()
	{
		_setKernelArg(_normalize3b, 2, sizeof(cl_mem), &_d12);
		_setKernelArg(_normalize3b, 3, sizeof(cl_mem), &_d3);
		_executeKernel(_normalize3b, VSIZE / CSIZE * _size);
	}
};
