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

#define VSIZE_MAX	64
#define	CSIZE_MIN	8

class engine : public ocl::device
{
private:
	size_t _nsize = 0, _vsize = 0, _vnsize = 0, _vn_csize = 0;
	cl_mem _x12 = nullptr, _x3 = nullptr, _y12 = nullptr, _y3 = nullptr, _d12 = nullptr, _d3 = nullptr;
	cl_mem _wr12 = nullptr, _wr3 = nullptr, _wri12 = nullptr, _wri3 = nullptr, _res = nullptr, _bb_inv = nullptr, _f = nullptr;
	cl_kernel _set_P1 = nullptr, _setxy = nullptr, _setdy = nullptr, _swap = nullptr, _reset = nullptr;
	cl_kernel _square2 = nullptr, _square4 = nullptr, _square8 = nullptr, _square16 = nullptr;
	cl_kernel _mul2cond = nullptr, _mul4cond = nullptr, _mul2 = nullptr, _mul4 = nullptr;
	cl_kernel _forward2 = nullptr, _forward4 = nullptr, _forward16 = nullptr;
	cl_kernel _backward2 = nullptr, _backward4 = nullptr, _backward16 = nullptr;
	cl_kernel _normalize_1 = nullptr, _normalize_2 = nullptr;

public:
	engine(const ocl::platform & platform, const size_t d) : ocl::device(platform, d) {}
	virtual ~engine() {}

public:
	void allocMemory(const size_t size, const size_t vsize, const size_t csize)
	{
#if defined (ocl_debug)
		std::ostringstream ss; ss << "Alloc gpu memory." << std::endl;
		pio::display(ss.str());
#endif
		const size_t vnsize = vsize * size;
		_nsize = size;
		_vsize = vsize;
		_vnsize = vnsize;
		_vn_csize = vnsize / csize;
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
		_f = _createBuffer(CL_MEM_READ_WRITE, sizeof(int64) * _vn_csize);
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
			_vn_csize = _vnsize = _vsize = _nsize = 0;
		}
	}

private:
	void createKernel_square(cl_kernel & kernel, const char * const name)
	{
		kernel = _createKernel(name);
		_setKernelArg(kernel, 0, sizeof(cl_mem), &_wr12);
		_setKernelArg(kernel, 1, sizeof(cl_mem), &_wr3);
		_setKernelArg(kernel, 2, sizeof(cl_mem), &_wri12);
		_setKernelArg(kernel, 3, sizeof(cl_mem), &_wri3);
		_setKernelArg(kernel, 4, sizeof(cl_mem), &_x12);
		_setKernelArg(kernel, 5, sizeof(cl_mem), &_x3);
	}

	void createKernel_mulcond(cl_kernel & kernel, const char * const name)
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

	void createKernel_mul(cl_kernel & kernel, const char * const name)
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

		_setxy = _createKernel("set");
		_setKernelArg(_setxy, 0, sizeof(cl_mem), &_x12);
		_setKernelArg(_setxy, 1, sizeof(cl_mem), &_x3);
		_setKernelArg(_setxy, 2, sizeof(cl_mem), &_y12);
		_setKernelArg(_setxy, 3, sizeof(cl_mem), &_y3);

		_setdy = _createKernel("set");
		_setKernelArg(_setdy, 0, sizeof(cl_mem), &_d12);
		_setKernelArg(_setdy, 1, sizeof(cl_mem), &_d3);
		_setKernelArg(_setdy, 2, sizeof(cl_mem), &_y12);
		_setKernelArg(_setdy, 3, sizeof(cl_mem), &_y3);

		_swap = _createKernel("swap");
		_setKernelArg(_swap, 0, sizeof(cl_mem), &_x12);
		_setKernelArg(_swap, 1, sizeof(cl_mem), &_x3);
		_setKernelArg(_swap, 2, sizeof(cl_mem), &_d12);
		_setKernelArg(_swap, 3, sizeof(cl_mem), &_d3);

		_reset = _createKernel("reset");

		createKernel_square(_square2, "square2");
		createKernel_square(_square4, "square4");
		createKernel_square(_square8, "square8");
		createKernel_square(_square16, "square16");

		createKernel_mulcond(_mul2cond, "mul2cond");
		createKernel_mulcond(_mul4cond, "mul4cond");
		createKernel_mul(_mul2, "mul2");
		createKernel_mul(_mul4, "mul4");

		_forward2 = _createKernel("forward2");
		_setKernelArg(_forward2, 0, sizeof(cl_mem), &_wr12);
		_setKernelArg(_forward2, 1, sizeof(cl_mem), &_wr3);
		_forward4 = _createKernel("forward4");
		_setKernelArg(_forward4, 0, sizeof(cl_mem), &_wr12);
		_setKernelArg(_forward4, 1, sizeof(cl_mem), &_wr3);
		_forward16 = _createKernel("forward16");
		_setKernelArg(_forward16, 0, sizeof(cl_mem), &_wr12);
		_setKernelArg(_forward16, 1, sizeof(cl_mem), &_wr3);

		_backward2 = _createKernel("backward2");
		_setKernelArg(_backward2, 0, sizeof(cl_mem), &_wri12);
		_setKernelArg(_backward2, 1, sizeof(cl_mem), &_wri3);
		_backward4 = _createKernel("backward4");
		_setKernelArg(_backward4, 0, sizeof(cl_mem), &_wri12);
		_setKernelArg(_backward4, 1, sizeof(cl_mem), &_wri3);
		_backward16 = _createKernel("backward16");
		_setKernelArg(_backward16, 0, sizeof(cl_mem), &_wri12);
		_setKernelArg(_backward16, 1, sizeof(cl_mem), &_wri3);

		_normalize_1 = _createKernel("normalize_1");
		_setKernelArg(_normalize_1, 0, sizeof(cl_mem), &_bb_inv);
		_setKernelArg(_normalize_1, 1, sizeof(cl_mem), &_f);

		_normalize_2 = _createKernel("normalize_2");
		_setKernelArg(_normalize_2, 0, sizeof(cl_mem), &_bb_inv);
		_setKernelArg(_normalize_2, 1, sizeof(cl_mem), &_f);
	}

public:
	void releaseKernels()
	{
#if defined (ocl_debug)
		std::ostringstream ss; ss << "Release ocl kernels." << std::endl;
		pio::display(ss.str());
#endif
		_releaseKernel(_set_P1); _releaseKernel(_setxy); _releaseKernel(_setdy); _releaseKernel(_swap); _releaseKernel(_reset);
		_releaseKernel(_square2); _releaseKernel(_square4); _releaseKernel(_square8); _releaseKernel(_square16);
		_releaseKernel(_mul2cond); _releaseKernel(_mul4cond); _releaseKernel(_mul2); _releaseKernel(_mul4);
		_releaseKernel(_forward2); _releaseKernel(_forward4); _releaseKernel(_forward16);
		_releaseKernel(_backward2); _releaseKernel(_backward4); _releaseKernel(_backward16);
		_releaseKernel(_normalize_1); _releaseKernel(_normalize_2);
	}

public:
	void setParam_bs(const int32 s)
	{
		_setKernelArg(_normalize_1, 4, sizeof(int32), &s);
		_setKernelArg(_normalize_2, 4, sizeof(int32), &s);
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
	void setxy() { _executeKernel(_setxy, this->_vnsize); }
	void setxres_P1() { _executeKernel(_set_P1, this->_vnsize); }
	void setdy() { _executeKernel(_setdy, this->_vnsize); }

	void swap_xd() { _executeKernel(_swap, this->_vnsize); }

public:
	void reset(const uint32 a)
	{
		_setKernelArg(_reset, 0, sizeof(cl_mem), &_x12);
		_setKernelArg(_reset, 1, sizeof(cl_mem), &_x3);
		_setKernelArg(_reset, 2, sizeof(uint32), &a);
		_executeKernel(_reset, this->_vnsize);
		_setKernelArg(_reset, 0, sizeof(cl_mem), &_d12);
		_setKernelArg(_reset, 1, sizeof(cl_mem), &_d3);
		_executeKernel(_reset, this->_vnsize);
	}

public:
	void square2x() { _executeKernel(_square2, this->_vnsize / 2); }
	void square4x() { _executeKernel(_square4, this->_vnsize / 4); }
	void square8x() { _executeKernel(_square8, this->_vnsize / 4, this->_vsize * 8 / 4); }
	void square16x() { _executeKernel(_square16, this->_vnsize / 4, this->_vsize * 16 / 4); }

public:
	void mul2condxy(const uint64 c)
	{
		_setKernelArg(_mul2cond, 8, sizeof(uint64), &c);
		_executeKernel(_mul2cond, this->_vnsize / 2);
	}
	void mul4condxy(const uint64 c)
	{
		_setKernelArg(_mul4cond, 8, sizeof(uint64), &c);
		_executeKernel(_mul4cond, this->_vnsize / 4);
	}

public:
	void mul2dy()
	{
		_setKernelArg(_mul2, 6, sizeof(cl_mem), &_d12);
		_setKernelArg(_mul2, 7, sizeof(cl_mem), &_d3);
		_executeKernel(_mul2, this->_vnsize / 2);
	}
	void mul4dy()
	{
		_setKernelArg(_mul4, 6, sizeof(cl_mem), &_d12);
		_setKernelArg(_mul4, 7, sizeof(cl_mem), &_d3);
		_executeKernel(_mul4, this->_vnsize / 4);
	}

public:
	void mul2xy()
	{
		_setKernelArg(_mul2, 6, sizeof(cl_mem), &_x12);
		_setKernelArg(_mul2, 7, sizeof(cl_mem), &_x3);
		_executeKernel(_mul2, this->_vnsize / 2);
	}
	void mul4xy()
	{
		_setKernelArg(_mul4, 6, sizeof(cl_mem), &_x12);
		_setKernelArg(_mul4, 7, sizeof(cl_mem), &_x3);
		_executeKernel(_mul4, this->_vnsize / 4);
	}

private:
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
	void forward2(cl_mem x12, cl_mem x3, const uint32 s, const int32 lm)
	{
		setP123args(_forward2, x12, x3, s, lm);
		_executeKernel(_forward2, this->_vnsize / 2);
	}
	void forward4(cl_mem x12, cl_mem x3, const uint32 s, const int32 lm)
	{
		setP123args(_forward4, x12, x3, s, lm);
		_executeKernel(_forward4, this->_vnsize / 4);
	}
	void forward16(cl_mem x12, cl_mem x3, const uint32 s, const int32 lm)
	{
		setP123args(_forward16, x12, x3, s, lm);
		_executeKernel(_forward16, this->_vnsize / 4, this->_vsize * 16 / 4);
	}

private:
	void backward2(cl_mem x12, cl_mem x3, const uint32 s, const int32 lm)
	{
		setP123args(_backward2, x12, x3, s, lm);
		_executeKernel(_backward2, this->_vnsize / 2);
	}
	void backward4(cl_mem x12, cl_mem x3, const uint32 s, const int32 lm)
	{
		setP123args(_backward4, x12, x3, s, lm);
		_executeKernel(_backward4, this->_vnsize / 4);
	}
	void backward16(cl_mem x12, cl_mem x3, const uint32 s, const int32 lm)
	{
		setP123args(_backward16, x12, x3, s, lm);
		_executeKernel(_backward16, this->_vnsize / 4, this->_vsize * 16 / 4);
	}

public:
	void forward2y(const uint32 s, const int32 lm) { forward2(_y12, _y3, s, lm); }

public:
	void forward4x(const uint32 s, const int32 lm) { forward4(_x12, _x3, s, lm); }
	void forward4y(const uint32 s, const int32 lm) { forward4(_y12, _y3, s, lm); }
	void forward4d(const uint32 s, const int32 lm) { forward4(_d12, _d3, s, lm); }

public:
	void forward16x(const uint32 s, const int32 lm) { forward16(_x12, _x3, s, lm); }
	void forward16y(const uint32 s, const int32 lm) { forward16(_y12, _y3, s, lm); }
	void forward16d(const uint32 s, const int32 lm) { forward16(_d12, _d3, s, lm); }

public:
	void backward4x(const uint32 s, const int32 lm) { backward4(_x12, _x3, s, lm); }
	void backward4d(const uint32 s, const int32 lm) { backward4(_d12, _d3, s, lm); }

public:
	void backward16x(const uint32 s, const int32 lm) { backward16(_x12, _x3, s, lm); }
	void backward16d(const uint32 s, const int32 lm) { backward16(_d12, _d3, s, lm); }

public:
	void normalize_1x()
	{
		_setKernelArg(_normalize_1, 2, sizeof(cl_mem), &_x12);
		_setKernelArg(_normalize_1, 3, sizeof(cl_mem), &_x3);
		_executeKernel(_normalize_1, this->_vn_csize);
	}
	void normalize_2x()
	{
		_setKernelArg(_normalize_2, 2, sizeof(cl_mem), &_x12);
		_setKernelArg(_normalize_2, 3, sizeof(cl_mem), &_x3);
		_executeKernel(_normalize_2, this->_vn_csize);
	}
	void normalize_1d()
	{
		_setKernelArg(_normalize_1, 2, sizeof(cl_mem), &_d12);
		_setKernelArg(_normalize_1, 3, sizeof(cl_mem), &_d3);
		_executeKernel(_normalize_1, this->_vn_csize);
	}
	void normalize_2d()
	{
		_setKernelArg(_normalize_2, 2, sizeof(cl_mem), &_d12);
		_setKernelArg(_normalize_2, 3, sizeof(cl_mem), &_d3);
		_executeKernel(_normalize_2, this->_vn_csize);
	}
};
