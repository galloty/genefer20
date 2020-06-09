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
typedef cl_ulong4	uint64_4;

#define VSIZE_MAX	256
#define	CSIZE		8

class engine : public ocl::device
{
private:
	size_t _nsize = 0, _vsize = 0, _vnsize = 0;
	cl_mem _x12 = nullptr, _x3 = nullptr, _y12 = nullptr, _y3 = nullptr, _d12 = nullptr, _d3 = nullptr;
	cl_mem _wr12 = nullptr, _wr3 = nullptr, _wri12 = nullptr, _wri3 = nullptr, _res = nullptr, _bb_inv = nullptr, _f = nullptr;
	cl_kernel _set_P1 = nullptr, _setxy_P12 = nullptr, _setxy_P123 = nullptr, _setdy_P12 = nullptr, _setdy_P123 = nullptr;
	cl_kernel _swap_P12 = nullptr, _swap_P123 = nullptr, _reset_P12 = nullptr, _reset_P123 = nullptr;
	cl_kernel _square2_P12 = nullptr, _square2_P123 = nullptr, _square4_P12 = nullptr, _square4_P123 = nullptr;
	cl_kernel _square8_P12 = nullptr, _square8_P3 = nullptr, _square16_P12 = nullptr, _square16_P3 = nullptr;
	cl_kernel _mul2cond64_P12 = nullptr, _mul2cond64_P123 = nullptr, _mul4cond64_P12 = nullptr, _mul4cond64_P123 = nullptr;
	cl_kernel _mul2cond256_P12 = nullptr, _mul2cond256_P123 = nullptr, _mul4cond256_P12 = nullptr, _mul4cond256_P123 = nullptr;
	cl_kernel _mul2_P12 = nullptr, _mul2_P123 = nullptr, _mul4_P12 = nullptr, _mul4_P123 = nullptr;
	cl_kernel _forward2_P12 = nullptr, _forward2_P123 = nullptr, _forward4_P12 = nullptr, _forward4_P123 = nullptr;
	cl_kernel _forward16_P12 = nullptr, _forward16_P3 = nullptr, _backward16_P12 = nullptr, _backward16_P3 = nullptr;
	cl_kernel _backward2_P12 = nullptr, _backward2_P123 = nullptr, _backward4_P12 = nullptr, _backward4_P123 = nullptr;
	cl_kernel _normalize2a = nullptr, _normalize2b = nullptr, _normalize3a = nullptr, _normalize3b = nullptr;

public:
	engine(const ocl::platform & platform, const size_t d) : ocl::device(platform, d) {}
	virtual ~engine() {}

public:
	void allocMemory(const size_t size, const size_t vsize)
	{
#if defined (ocl_debug)
		std::ostringstream ss; ss << "Alloc gpu memory." << std::endl;
		pio::display(ss.str());
#endif
		const size_t vnsize = vsize * size;
		_nsize = size;
		_vsize = vsize;
		_vnsize = vnsize;
		_x12 = _createBuffer(CL_MEM_READ_WRITE, sizeof(uint32_2) * vnsize);
		_x3 = _createBuffer(CL_MEM_READ_WRITE, sizeof(uint32) * vnsize);
		_y12 = _createBuffer(CL_MEM_READ_WRITE, sizeof(uint32_2) * vnsize);
		_y3 = _createBuffer(CL_MEM_READ_WRITE, sizeof(uint32) * vnsize);
		_d12 = _createBuffer(CL_MEM_READ_WRITE, sizeof(uint32_2) * vnsize);
		_d3 = _createBuffer(CL_MEM_READ_WRITE, sizeof(uint32) * vnsize);
		_wr12 = _createBuffer(CL_MEM_READ_ONLY, sizeof(uint32_2) * size);
		_wr3 = _createBuffer(CL_MEM_READ_ONLY, sizeof(uint32) * size);
		_wri12 = _createBuffer(CL_MEM_READ_ONLY, sizeof(uint32_2) * size);
		_wri3 = _createBuffer(CL_MEM_READ_ONLY, sizeof(uint32) * size);
		_res = _createBuffer(CL_MEM_READ_WRITE, sizeof(uint32) * vnsize);
		_bb_inv = _createBuffer(CL_MEM_READ_ONLY, sizeof(uint32_2) * vsize);
		_f = _createBuffer(CL_MEM_READ_WRITE, sizeof(int64) * vnsize / CSIZE);
	}

public:
	void releaseMemory()
	{
#if defined (ocl_debug)
		std::ostringstream ss; ss << "Free gpu memory." << std::endl;
		pio::display(ss.str());
#endif
		if (_nsize != 0)
		{
			_releaseBuffer(_x12); _releaseBuffer(_x3);
			_releaseBuffer(_y12); _releaseBuffer(_y3);
			_releaseBuffer(_d12); _releaseBuffer(_d3);
			_releaseBuffer(_wr12); _releaseBuffer(_wr3); _releaseBuffer(_wri12); _releaseBuffer(_wri3);
			_releaseBuffer(_res); _releaseBuffer(_bb_inv); _releaseBuffer(_f);
			_vnsize = _vsize = _nsize = 0;
		}
	}

private:
	void createKernel_square_P12(cl_kernel & kernel, const char * const name)
	{
		kernel = _createKernel(name);
		_setKernelArg(kernel, 0, sizeof(cl_mem), &_wr12);
		_setKernelArg(kernel, 1, sizeof(cl_mem), &_wri12);
		_setKernelArg(kernel, 2, sizeof(cl_mem), &_x12);
	}
	void createKernel_square_P3(cl_kernel & kernel, const char * const name)
	{
		kernel = _createKernel(name);
		_setKernelArg(kernel, 0, sizeof(cl_mem), &_wr3);
		_setKernelArg(kernel, 1, sizeof(cl_mem), &_wri3);
		_setKernelArg(kernel, 2, sizeof(cl_mem), &_x3);
	}
	void createKernel_square_P123(cl_kernel & kernel, const char * const name)
	{
		kernel = _createKernel(name);
		_setKernelArg(kernel, 0, sizeof(cl_mem), &_wr12);
		_setKernelArg(kernel, 1, sizeof(cl_mem), &_wr3);
		_setKernelArg(kernel, 2, sizeof(cl_mem), &_wri12);
		_setKernelArg(kernel, 3, sizeof(cl_mem), &_wri3);
		_setKernelArg(kernel, 4, sizeof(cl_mem), &_x12);
		_setKernelArg(kernel, 5, sizeof(cl_mem), &_x3);
	}

private:
	void createKernel_mulcond_P12(cl_kernel & kernel, const char * const name)
	{
		kernel = _createKernel(name);
		_setKernelArg(kernel, 0, sizeof(cl_mem), &_wr12);
		_setKernelArg(kernel, 1, sizeof(cl_mem), &_wri12);
		_setKernelArg(kernel, 2, sizeof(cl_mem), &_y12);
		_setKernelArg(kernel, 3, sizeof(cl_mem), &_x12);
	}
	void createKernel_mulcond_P123(cl_kernel & kernel, const char * const name)
	{
		kernel = _createKernel(name);
		_setKernelArg(kernel, 0, sizeof(cl_mem), &_wr12);
		_setKernelArg(kernel, 1, sizeof(cl_mem), &_wr3);
		_setKernelArg(kernel, 2, sizeof(cl_mem), &_wri12);
		_setKernelArg(kernel, 3, sizeof(cl_mem), &_wri3);
		_setKernelArg(kernel, 4, sizeof(cl_mem), &_y12);
		_setKernelArg(kernel, 5, sizeof(cl_mem), &_y3);
		_setKernelArg(kernel, 6, sizeof(cl_mem), &_x12);
		_setKernelArg(kernel, 7, sizeof(cl_mem), &_x3);
	}

private:
	void createKernel_mul_P12(cl_kernel & kernel, const char * const name)
	{
		kernel = _createKernel(name);
		_setKernelArg(kernel, 0, sizeof(cl_mem), &_wr12);
		_setKernelArg(kernel, 1, sizeof(cl_mem), &_wri12);
		_setKernelArg(kernel, 2, sizeof(cl_mem), &_y12);
	}
	void createKernel_mul_P123(cl_kernel & kernel, const char * const name)
	{
		kernel = _createKernel(name);
		_setKernelArg(kernel, 0, sizeof(cl_mem), &_wr12);
		_setKernelArg(kernel, 1, sizeof(cl_mem), &_wr3);
		_setKernelArg(kernel, 2, sizeof(cl_mem), &_wri12);
		_setKernelArg(kernel, 3, sizeof(cl_mem), &_wri3);
		_setKernelArg(kernel, 4, sizeof(cl_mem), &_y12);
		_setKernelArg(kernel, 5, sizeof(cl_mem), &_y3);
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
		_setxy_P123 = _createKernel("set_P123");
		_setKernelArg(_setxy_P123, 0, sizeof(cl_mem), &_x12);
		_setKernelArg(_setxy_P123, 1, sizeof(cl_mem), &_x3);
		_setKernelArg(_setxy_P123, 2, sizeof(cl_mem), &_y12);
		_setKernelArg(_setxy_P123, 3, sizeof(cl_mem), &_y3);

		_setdy_P12 = _createKernel("set_P12");
		_setKernelArg(_setdy_P12, 0, sizeof(cl_mem), &_d12);
		_setKernelArg(_setdy_P12, 1, sizeof(cl_mem), &_y12);
		_setdy_P123 = _createKernel("set_P123");
		_setKernelArg(_setdy_P123, 0, sizeof(cl_mem), &_d12);
		_setKernelArg(_setdy_P123, 1, sizeof(cl_mem), &_d3);
		_setKernelArg(_setdy_P123, 2, sizeof(cl_mem), &_y12);
		_setKernelArg(_setdy_P123, 3, sizeof(cl_mem), &_y3);

		_swap_P12 = _createKernel("swap_P12");
		_setKernelArg(_swap_P12, 0, sizeof(cl_mem), &_x12);
		_setKernelArg(_swap_P12, 1, sizeof(cl_mem), &_d12);
		_swap_P123 = _createKernel("swap_P123");
		_setKernelArg(_swap_P123, 0, sizeof(cl_mem), &_x12);
		_setKernelArg(_swap_P123, 1, sizeof(cl_mem), &_x3);
		_setKernelArg(_swap_P123, 2, sizeof(cl_mem), &_d12);
		_setKernelArg(_swap_P123, 3, sizeof(cl_mem), &_d3);
		_reset_P12 = _createKernel("reset_P12");
		_reset_P123 = _createKernel("reset_P123");

		createKernel_square_P12(_square2_P12, "square2_P12");
		createKernel_square_P123(_square2_P123, "square2_P123");
		createKernel_square_P12(_square4_P12, "square4_P12");
		createKernel_square_P123(_square4_P123, "square4_P123");

		createKernel_square_P12(_square8_P12, "square8_P12");
		createKernel_square_P3(_square8_P3, "square8_P3");
		createKernel_square_P12(_square16_P12, "square16_P12");
		createKernel_square_P3(_square16_P3, "square16_P3");

		createKernel_mulcond_P12(_mul2cond64_P12, "mul2cond64_P12");
		createKernel_mulcond_P123(_mul2cond64_P123, "mul2cond64_P123");
		createKernel_mulcond_P12(_mul4cond64_P12, "mul4cond64_P12");
		createKernel_mulcond_P123(_mul4cond64_P123, "mul4cond64_P123");
		createKernel_mulcond_P12(_mul2cond256_P12, "mul2cond256_P12");
		createKernel_mulcond_P123(_mul2cond256_P123, "mul2cond256_P123");
		createKernel_mulcond_P12(_mul4cond256_P12, "mul4cond256_P12");
		createKernel_mulcond_P123(_mul4cond256_P123, "mul4cond256_P123");

		createKernel_mul_P12(_mul2_P12, "mul2_P12");
		createKernel_mul_P123(_mul2_P123, "mul2_P123");
		createKernel_mul_P12(_mul4_P12, "mul4_P12");
		createKernel_mul_P123(_mul4_P123, "mul4_P123");

		_forward2_P12 = _createKernel("forward2_P12");
		_setKernelArg(_forward2_P12, 0, sizeof(cl_mem), &_wr12);
		_forward2_P123 = _createKernel("forward2_P123");
		_setKernelArg(_forward2_P123, 0, sizeof(cl_mem), &_wr12);
		_setKernelArg(_forward2_P123, 1, sizeof(cl_mem), &_wr3);
		_forward4_P12 = _createKernel("forward4_P12");
		_setKernelArg(_forward4_P12, 0, sizeof(cl_mem), &_wr12);
		_forward4_P123 = _createKernel("forward4_P123");
		_setKernelArg(_forward4_P123, 0, sizeof(cl_mem), &_wr12);
		_setKernelArg(_forward4_P123, 1, sizeof(cl_mem), &_wr3);
		_forward16_P12 = _createKernel("forward16_P12");
		_setKernelArg(_forward16_P12, 0, sizeof(cl_mem), &_wr12);
		_forward16_P3 = _createKernel("forward16_P3");
		_setKernelArg(_forward16_P3, 0, sizeof(cl_mem), &_wr3);

		_backward2_P12 = _createKernel("backward2_P12");
		_setKernelArg(_backward2_P12, 0, sizeof(cl_mem), &_wri12);
		_backward2_P123 = _createKernel("backward2_P123");
		_setKernelArg(_backward2_P123, 0, sizeof(cl_mem), &_wri12);
		_setKernelArg(_backward2_P123, 1, sizeof(cl_mem), &_wri3);
		_backward4_P12 = _createKernel("backward4_P12");
		_setKernelArg(_backward4_P12, 0, sizeof(cl_mem), &_wri12);
		_backward4_P123 = _createKernel("backward4_P123");
		_setKernelArg(_backward4_P123, 0, sizeof(cl_mem), &_wri12);
		_setKernelArg(_backward4_P123, 1, sizeof(cl_mem), &_wri3);
		_backward16_P12 = _createKernel("backward16_P12");
		_setKernelArg(_backward16_P12, 0, sizeof(cl_mem), &_wri12);
		_backward16_P3 = _createKernel("backward16_P3");
		_setKernelArg(_backward16_P3, 0, sizeof(cl_mem), &_wri3);

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
		_releaseKernel(_swap_P12); _releaseKernel(_swap_P123); _releaseKernel(_reset_P12); _releaseKernel(_reset_P123);
		_releaseKernel(_square2_P12); _releaseKernel(_square2_P123); _releaseKernel(_square4_P12); _releaseKernel(_square4_P123);
		_releaseKernel(_square8_P12); _releaseKernel(_square8_P3); _releaseKernel(_square16_P12); _releaseKernel(_square16_P3);
		_releaseKernel(_mul2cond64_P12); _releaseKernel(_mul2cond64_P123); _releaseKernel(_mul4cond64_P12); _releaseKernel(_mul4cond64_P123);
		_releaseKernel(_mul2cond256_P12); _releaseKernel(_mul2cond256_P123); _releaseKernel(_mul4cond256_P12); _releaseKernel(_mul4cond256_P123);
		_releaseKernel(_mul2_P12); _releaseKernel(_mul2_P123); _releaseKernel(_mul4_P12); _releaseKernel(_mul4_P123);
		_releaseKernel(_forward2_P12); _releaseKernel(_forward2_P123); _releaseKernel(_forward4_P12); _releaseKernel(_forward4_P123);
		_releaseKernel(_forward16_P12); _releaseKernel(_forward16_P3); _releaseKernel(_backward16_P12); _releaseKernel(_backward16_P3);
		_releaseKernel(_backward2_P12); _releaseKernel(_backward2_P123); _releaseKernel(_backward4_P12); _releaseKernel(_backward4_P123);
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
		const size_t nsize = this->_nsize;
		_writeBuffer(_wr12, wr12, sizeof(uint32_2) * nsize);
		_writeBuffer(_wr3, wr3, sizeof(uint32) * nsize);
		_writeBuffer(_wri12, wri12, sizeof(uint32_2) * nsize);
		_writeBuffer(_wri3, wri3, sizeof(uint32) * nsize);
	}

	void writeMemory_b(const uint32_2 * const bb_inv) { _writeBuffer(_bb_inv, bb_inv, sizeof(uint32_2) * this->_vsize); }

public:
	void readMemory_x12(uint32_2 * const x12) { _readBuffer(_x12, x12, sizeof(uint32_2) * this->_vnsize); }
	void readMemory_d12(uint32_2 * const d12) { _readBuffer(_d12, d12, sizeof(uint32_2) * this->_vnsize); }
	void readMemory_res(uint32 * const res) { _readBuffer(_res, res, sizeof(uint32) * this->_vnsize); }

public:
	void setxy_P12() { _executeKernel(_setxy_P12, this->_vnsize); }
	void setxy_P123() { _executeKernel(_setxy_P123, this->_vnsize); }
	void setxres_P1() { _executeKernel(_set_P1, this->_vnsize); }
	void setdy_P12() { _executeKernel(_setdy_P12, this->_vnsize); }
	void setdy_P123() { _executeKernel(_setdy_P123, this->_vnsize); }

public:
	void swap_xd_P12() { _executeKernel(_swap_P12, this->_vnsize); }
	void swap_xd_P123() { _executeKernel(_swap_P123, this->_vnsize); }

public:
	void reset_P12(const uint32 a)
	{
		_setKernelArg(_reset_P12, 0, sizeof(cl_mem), &_x12);
		_setKernelArg(_reset_P12, 1, sizeof(uint32), &a);
		_executeKernel(_reset_P12, this->_vnsize);
		_setKernelArg(_reset_P12, 0, sizeof(cl_mem), &_d12);
		_executeKernel(_reset_P12, this->_vnsize);
	}
	void reset_P123(const uint32 a)
	{
		_setKernelArg(_reset_P123, 0, sizeof(cl_mem), &_x12);
		_setKernelArg(_reset_P123, 1, sizeof(cl_mem), &_x3);
		_setKernelArg(_reset_P123, 2, sizeof(uint32), &a);
		_executeKernel(_reset_P123, this->_vnsize);
		_setKernelArg(_reset_P123, 0, sizeof(cl_mem), &_d12);
		_setKernelArg(_reset_P123, 1, sizeof(cl_mem), &_d3);
		_executeKernel(_reset_P123, this->_vnsize);
	}

public:
	void square2x_P12() { _executeKernel(_square2_P12, this->_vnsize / 2); }
	void square2x_P123() { _executeKernel(_square2_P123, this->_vnsize / 2); }
	void square4x_P12() { _executeKernel(_square4_P12, this->_vnsize / 4); }
	void square4x_P123() { _executeKernel(_square4_P123, this->_vnsize / 4); }

public:
	void square8x_P12() { _executeKernel(_square8_P12, this->_vnsize / 4, this->_vsize * 8 / 4); }
	void square8x_P3() { _executeKernel(_square8_P3, this->_vnsize / 4, this->_vsize * 8 / 4); }
	void square16x_P12() { _executeKernel(_square16_P12, this->_vnsize / 4, this->_vsize * 16 / 4); }
	void square16x_P3() { _executeKernel(_square16_P3, this->_vnsize / 4, this->_vsize * 16 / 4); }

public:
	void mul2cond64xy_P12(const uint64 c)
	{
		_setKernelArg(_mul2cond64_P12, 4, sizeof(uint64), &c);
		_executeKernel(_mul2cond64_P12, this->_vnsize / 2);
	}
	void mul2cond64xy_P123(const uint64 c)
	{
		_setKernelArg(_mul2cond64_P123, 8, sizeof(uint64), &c);
		_executeKernel(_mul2cond64_P123, this->_vnsize / 2);
	}
	void mul4cond64xy_P12(const uint64 c)
	{
		_setKernelArg(_mul4cond64_P12, 4, sizeof(uint64), &c);
		_executeKernel(_mul4cond64_P12, this->_vnsize / 4);
	}
	void mul4cond64xy_P123(const uint64 c)
	{
		_setKernelArg(_mul4cond64_P123, 8, sizeof(uint64), &c);
		_executeKernel(_mul4cond64_P123, this->_vnsize / 4);
	}

public:
	void mul2cond256xy_P12(const uint64_4 & c)
	{
		_setKernelArg(_mul2cond256_P12, 4, sizeof(uint64_4), &c);
		_executeKernel(_mul2cond256_P12, this->_vnsize / 2);
	}
	void mul2cond256xy_P123(const uint64_4 & c)
	{
		_setKernelArg(_mul2cond256_P123, 8, sizeof(uint64_4), &c);
		_executeKernel(_mul2cond256_P123, this->_vnsize / 2);
	}
	void mul4cond256xy_P12(const uint64_4 & c)
	{
		_setKernelArg(_mul4cond256_P12, 4, sizeof(uint64_4), &c);
		_executeKernel(_mul4cond256_P12, this->_vnsize / 4);
	}
	void mul4cond256xy_P123(const uint64_4 & c)
	{
		_setKernelArg(_mul4cond256_P123, 8, sizeof(uint64_4), &c);
		_executeKernel(_mul4cond256_P123, this->_vnsize / 4);
	}

public:
	void mul2dy_P12()
	{
		_setKernelArg(_mul2_P12, 3, sizeof(cl_mem), &_d12);
		_executeKernel(_mul2_P12, this->_vnsize / 2);
	}
	void mul2dy_P123()
	{
		_setKernelArg(_mul2_P123, 6, sizeof(cl_mem), &_d12);
		_setKernelArg(_mul2_P123, 7, sizeof(cl_mem), &_d3);
		_executeKernel(_mul2_P123, this->_vnsize / 2);
	}
	void mul4dy_P12()
	{
		_setKernelArg(_mul4_P12, 3, sizeof(cl_mem), &_d12);
		_executeKernel(_mul4_P12, this->_vnsize / 4);
	}
	void mul4dy_P123()
	{
		_setKernelArg(_mul4_P123, 6, sizeof(cl_mem), &_d12);
		_setKernelArg(_mul4_P123, 7, sizeof(cl_mem), &_d3);
		_executeKernel(_mul4_P123, this->_vnsize / 4);
	}

public:
	void mul2xy_P12()
	{
		_setKernelArg(_mul2_P12, 3, sizeof(cl_mem), &_x12);
		_executeKernel(_mul2_P12, this->_vnsize / 2);
	}
	void mul2xy_P123()
	{
		_setKernelArg(_mul2_P123, 6, sizeof(cl_mem), &_x12);
		_setKernelArg(_mul2_P123, 7, sizeof(cl_mem), &_x3);
		_executeKernel(_mul2_P123, this->_vnsize / 2);
	}
	void mul4xy_P12()
	{
		_setKernelArg(_mul4_P12, 3, sizeof(cl_mem), &_x12);
		_executeKernel(_mul4_P12, this->_vnsize / 4);
	}
	void mul4xy_P123()
	{
		_setKernelArg(_mul4_P123, 6, sizeof(cl_mem), &_x12);
		_setKernelArg(_mul4_P123, 7, sizeof(cl_mem), &_x3);
		_executeKernel(_mul4_P123, this->_vnsize / 4);
	}

private:
	void setP12args(cl_kernel kernel, cl_mem x12, const uint32 s, const int32 lm)
	{
		const uint32 m = uint32(1) << lm;
		_setKernelArg(kernel, 1, sizeof(cl_mem), &x12);
		_setKernelArg(kernel, 2, sizeof(uint32), &s);
		_setKernelArg(kernel, 3, sizeof(uint32), &m);
		_setKernelArg(kernel, 4, sizeof(int32), &lm);
	}
	void setP123args(cl_kernel kernel, cl_mem x12, cl_mem x3, const uint32 s, const int32 lm)
	{
		const uint32 m = uint32(1) << lm;
		_setKernelArg(kernel, 2, sizeof(cl_mem), &x12);
		_setKernelArg(kernel, 3, sizeof(cl_mem), &x3);
		_setKernelArg(kernel, 4, sizeof(uint32), &s);
		_setKernelArg(kernel, 5, sizeof(uint32), &m);
		_setKernelArg(kernel, 6, sizeof(int32), &lm);
	}

private:
	void forward2_P12(cl_mem x12, const uint32 s, const int32 lm)
	{
		setP12args(_forward2_P12, x12, s, lm);
		_executeKernel(_forward2_P12, this->_vnsize / 2);
	}
	void forward2_P123(cl_mem x12, cl_mem x3, const uint32 s, const int32 lm)
	{
		setP123args(_forward2_P123, x12, x3, s, lm);
		_executeKernel(_forward2_P123, this->_vnsize / 2);
	}
	void forward4_P12(cl_mem x12, const uint32 s, const int32 lm)
	{
		setP12args(_forward4_P12, x12, s, lm);
		_executeKernel(_forward4_P12, this->_vnsize / 4);
	}
	void forward4_P123(cl_mem x12, cl_mem x3, const uint32 s, const int32 lm)
	{
		setP123args(_forward4_P123, x12, x3, s, lm);
		_executeKernel(_forward4_P123, this->_vnsize / 4);
	}
	void forward16_P12(cl_mem x12, const uint32 s, const int32 lm)
	{
		setP12args(_forward16_P12, x12, s, lm);
		_executeKernel(_forward16_P12, this->_vnsize / 4, this->_vsize * 16 / 4);
	}
	void forward16_P3(cl_mem x3, const uint32 s, const int32 lm)
	{
		setP12args(_forward16_P3, x3, s, lm);
		_executeKernel(_forward16_P3, this->_vnsize / 4, this->_vsize * 16 / 4);
	}

private:
	void backward2_P12(cl_mem x12, const uint32 s, const int32 lm)
	{
		setP12args(_backward2_P12, x12, s, lm);
		_executeKernel(_backward2_P12, this->_vnsize / 2);
	}
	void backward2_P123(cl_mem x12, cl_mem x3, const uint32 s, const int32 lm)
	{
		setP123args(_backward2_P123, x12, x3, s, lm);
		_executeKernel(_backward2_P123, this->_vnsize / 2);
	}
	void backward4_P12(cl_mem x12, const uint32 s, const int32 lm)
	{
		setP12args(_backward4_P12, x12, s, lm);
		_executeKernel(_backward4_P12, this->_vnsize / 4);
	}
	void backward4_P123(cl_mem x12, cl_mem x3, const uint32 s, const int32 lm)
	{
		setP123args(_backward4_P123, x12, x3, s, lm);
		_executeKernel(_backward4_P123, this->_vnsize / 4);
	}
	void backward16_P12(cl_mem x12, const uint32 s, const int32 lm)
	{
		setP12args(_backward16_P12, x12, s, lm);
		_executeKernel(_backward16_P12, this->_vnsize / 4, this->_vsize * 16 / 4);
	}
	void backward16_P3(cl_mem x3, const uint32 s, const int32 lm)
	{
		setP12args(_backward16_P3, x3, s, lm);
		_executeKernel(_backward16_P3, this->_vnsize / 4, this->_vsize * 16 / 4);
	}

public:
	void forward2y_P12(const uint32 s, const int32 lm) { forward2_P12(_y12, s, lm); }
	void forward2y_P123(const uint32 s, const int32 lm) { forward2_P123(_y12, _y3, s, lm); }

public:
	void forward4x_P12(const uint32 s, const int32 lm) { forward4_P12(_x12, s, lm); }
	void forward4x_P123(const uint32 s, const int32 lm) { forward4_P123(_x12, _x3, s, lm); }
	void forward4y_P12(const uint32 s, const int32 lm) { forward4_P12(_y12, s, lm); }
	void forward4y_P123(const uint32 s, const int32 lm) { forward4_P123(_y12, _y3, s, lm); }
	void forward4d_P12(const uint32 s, const int32 lm) { forward4_P12(_d12, s, lm); }
	void forward4d_P123(const uint32 s, const int32 lm) { forward4_P123(_d12, _d3, s, lm); }

public:
	void forward16x_P12(const uint32 s, const int32 lm) { forward16_P12(_x12, s, lm); }
	void forward16x_P3(const uint32 s, const int32 lm) { forward16_P3(_x3, s, lm); }
	void forward16y_P12(const uint32 s, const int32 lm) { forward16_P12(_y12, s, lm); }
	void forward16y_P3(const uint32 s, const int32 lm) { forward16_P3(_y3, s, lm); }
	void forward16d_P12(const uint32 s, const int32 lm) { forward16_P12(_d12, s, lm); }
	void forward16d_P3(const uint32 s, const int32 lm) { forward16_P3(_d3, s, lm); }

public:
	void backward4x_P12(const uint32 s, const int32 lm) { backward4_P12(_x12, s, lm); }
	void backward4x_P123(const uint32 s, const int32 lm) { backward4_P123(_x12, _x3, s, lm); }
	void backward4d_P12(const uint32 s, const int32 lm) { backward4_P12(_d12, s, lm); }
	void backward4d_P123(const uint32 s, const int32 lm) { backward4_P123(_d12, _d3, s, lm); }

public:
	void backward16x_P12(const uint32 s, const int32 lm) { backward16_P12(_x12, s, lm); }
	void backward16x_P3(const uint32 s, const int32 lm) { backward16_P3(_x3, s, lm); }
	void backward16d_P12(const uint32 s, const int32 lm) { backward16_P12(_d12, s, lm); }
	void backward16d_P3(const uint32 s, const int32 lm) { backward16_P3(_d3, s, lm); }

public:
	void normalize2ax()
	{
		_setKernelArg(_normalize2a, 2, sizeof(cl_mem), &_x12);
		_executeKernel(_normalize2a, this->_vnsize / CSIZE);
	}
	void normalize2bx()
	{
		_setKernelArg(_normalize2b, 2, sizeof(cl_mem), &_x12);
		_executeKernel(_normalize2b, this->_vnsize / CSIZE);
	}
	void normalize2ad()
	{
		_setKernelArg(_normalize2a, 2, sizeof(cl_mem), &_d12);
		_executeKernel(_normalize2a, this->_vnsize / CSIZE);
	}
	void normalize2bd()
	{
		_setKernelArg(_normalize2b, 2, sizeof(cl_mem), &_d12);
		_executeKernel(_normalize2b, this->_vnsize / CSIZE);
	}

public:
	void normalize3ax()
	{
		_setKernelArg(_normalize3a, 2, sizeof(cl_mem), &_x12);
		_setKernelArg(_normalize3a, 3, sizeof(cl_mem), &_x3);
		_executeKernel(_normalize3a, this->_vnsize / CSIZE);
	}
	void normalize3bx()
	{
		_setKernelArg(_normalize3b, 2, sizeof(cl_mem), &_x12);
		_setKernelArg(_normalize3b, 3, sizeof(cl_mem), &_x3);
		_executeKernel(_normalize3b, this->_vnsize / CSIZE);
	}
	void normalize3ad()
	{
		_setKernelArg(_normalize3a, 2, sizeof(cl_mem), &_d12);
		_setKernelArg(_normalize3a, 3, sizeof(cl_mem), &_d3);
		_executeKernel(_normalize3a, this->_vnsize / CSIZE);
	}
	void normalize3bd()
	{
		_setKernelArg(_normalize3b, 2, sizeof(cl_mem), &_d12);
		_setKernelArg(_normalize3b, 3, sizeof(cl_mem), &_d3);
		_executeKernel(_normalize3b, this->_vnsize / CSIZE);
	}
};
