/*
Copyright 2020, Yves Gallot

genefer20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include "engine.h"
#include "pio.h"

#include <cstdint>
#include <cmath>
#include <sstream>
#include <vector>

#include "ocl/kernel.h"

class transform
{
private:
	static const uint32_t P1 = 2130706433u;		// 127 * 2^24 + 1 = 2^31 - 2^24 + 1
	static const uint32_t P2 = 2013265921u;		//  15 * 2^27 + 1 = 2^31 - 2^27 + 1

private:
	const size_t _size;
	const bool _isBoinc;
	engine & _engine;

private:
	template <uint32_t p> class Zp
	{
	private:
		uint32_t n;
	};

private:
	bool readOpenCL(const char * const clFileName, const char * const headerFileName, const char * const varName, std::stringstream & src) const
	{
		if (_isBoinc) return false;

		std::ifstream clFile(clFileName);
		if (!clFile.is_open()) return false;
		
		// if .cl file exists then generate header file
		std::ofstream hFile(headerFileName, std::ios::binary);	// binary: don't convert line endings to `CRLF` 
		if (!hFile.is_open()) throw std::runtime_error("cannot write openCL header file");

		hFile << "/*" << std::endl;
		hFile << "Copyright 2020, Yves Gallot" << std::endl << std::endl;
		hFile << "genefer20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it." << std::endl;
		hFile << "Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful." << std::endl;
		hFile << "*/" << std::endl << std::endl;

		hFile << "#pragma once" << std::endl << std::endl;
		hFile << "#include <cstdint>" << std::endl << std::endl;

		hFile << "static const char * const " << varName << " = \\" << std::endl;

		std::string line;
		while (std::getline(clFile, line))
		{
			hFile << "\"";
			for (char c : line)
			{
				if ((c == '\\') || (c == '\"')) hFile << '\\';
				hFile << c;
			}
			hFile << "\\n\" \\" << std::endl;

			src << line << std::endl;
		}
		hFile << "\"\";" << std::endl;

		hFile.close();
		clFile.close();
		return true;
	}

private:
	void _initEngine()
	{
		const size_t size = _size;

		std::stringstream src;
		src << "#define\txxx\t" << 1 << std::endl << std::endl;
		src << std::endl;

		// if xxx.cl file is not found then source is src_ocl_xxx string in src/ocl/xxx.h
		if (!readOpenCL("ocl/kernel.cl", "src/ocl/kernel.h", "src_ocl_kernel", src)) src << src_ocl_kernel;

		_engine.loadProgram(src.str());

		_engine.allocMemory();
		_engine.createKernels();
	}

private:
	void _clearEngine()
	{
		_engine.releaseKernels();
		_engine.releaseMemory();
		_engine.clearProgram();
	}

public:
	transform(const uint32_t n, engine & engine, const bool isBoinc) :
		_size(n), _isBoinc(isBoinc), _engine(engine)
	{
		const size_t size = _size;

		_initEngine();
	}

public:
	virtual ~transform()
	{
		_clearEngine();
	}

public:
	int getError() const
	{
		return 0;
	}

private:
	static bool _writeContext(FILE * const cFile, const char * const ptr, const size_t size)
	{
		const size_t ret = std::fwrite(ptr , sizeof(char), size, cFile);
		if (ret == size * sizeof(char)) return true;
		std::fclose(cFile);
		return false;
	}

private:
	static bool _readContext(FILE * const cFile, char * const ptr, const size_t size)
	{
		const size_t ret = std::fread(ptr , sizeof(char), size, cFile);
		if (ret == size * sizeof(char)) return true;
		std::fclose(cFile);
		return false;
	}

private:
	static std::string _filename(const char * const ext)
	{
		return std::string("genefer_") + std::string(ext) + std::string(".ctx");
	}

public:
	bool saveContext(const uint32_t i, const double elapsedTime, const char * const ext)
	{
		FILE * const cFile = pio::open(_filename(ext).c_str(), "wb");
		if (cFile == nullptr)
		{
			std::ostringstream ss; ss << "cannot write 'genefer.ctx' file " << std::endl;
			pio::error(ss.str());
			return false;
		}

		const size_t size = _size;

		const uint32_t version = 0;
		if (!_writeContext(cFile, reinterpret_cast<const char *>(&version), sizeof(version))) return false;

		std::fclose(cFile);
		return true;
	}

public:
	bool restoreContext(uint32_t & i, double & elapsedTime, const char * const ext, const bool restore_uv = true)
	{
		FILE * const cFile = pio::open(_filename(ext).c_str(), "rb");
		if (cFile == nullptr) return false;

		const size_t size = _size;

		uint32_t version = 0;
		if (!_readContext(cFile, reinterpret_cast<char *>(&version), sizeof(version))) return false;
		if (version != 0) return false;

		std::fclose(cFile);
		return true;
	}
};
