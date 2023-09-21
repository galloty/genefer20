/*
Copyright 2020, Yves Gallot

genefer20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <iomanip>

#include "pio.h"

class file
{
private:
	const std::string _filename;
	FILE * const _cFile;
	const bool _fatal;
	uint32_t _crc32 = 0;

public:
	file(const std::string & filename, const char * const mode, const bool fatal)
		: _filename(filename), _cFile(pio::open(filename.c_str(), mode)), _fatal(fatal), _crc32(0)
	{
		if (_cFile == nullptr) error("cannot open file");
	}

	file(const std::string & filename)
		: _filename(filename), _cFile(pio::open(filename.c_str(), "rb")), _fatal(false), _crc32(0)
	{
		// _cFile may be null
	}

	virtual ~file()
	{
		if (_cFile != nullptr)
		{
			if (std::fclose(_cFile) != 0) error("cannot close file");
		}
	}

	void error(const std::string & str) const
	{
		std::ostringstream ss; ss << _filename << ": " << str;
		pio::error(ss.str(), _fatal);
	}

	bool exists() const { return (_cFile != nullptr); }

	uint32_t crc32() const { return _crc32; }

	// Rosetta Code, CRC-32, C
	static uint32_t rc_crc32(const uint32_t crc32, const char * const buf, const size_t len)
	{
		static uint32_t table[256];
		static bool have_table = false;
	
		// This check is not thread safe; there is no mutex
		if (!have_table)
		{
			// Calculate CRC table
			for (size_t i = 0; i < 256; ++i)
			{
				uint32_t rem = static_cast<uint32_t>(i);  // remainder from polynomial division
				for (size_t j = 0; j < 8; ++j)
				{
					if (rem & 1)
					{
						rem >>= 1;
						rem ^= 0xedb88320;
					}
					else rem >>= 1;
				}
				table[i] = rem;
			}
			have_table = true;
		}

		uint32_t crc = ~crc32;
		for (size_t i = 0; i < len; ++i)
		{
			const uint8_t octet = static_cast<uint8_t>(buf[i]);  // Cast to unsigned octet
			crc = (crc >> 8) ^ table[(crc & 0xff) ^ octet];
		}
		return ~crc;
	}

	bool read(char * const ptr, const size_t size)
	{
		const size_t ret = std::fread(ptr , sizeof(char), size, _cFile);
		_crc32 = rc_crc32(_crc32, ptr, size);
		if (ret == size * sizeof(char)) return true;
		error("failure of a read operation");
		return false;
	}

	bool write(const char * const ptr, const size_t size)
	{
		const size_t ret = std::fwrite(ptr , sizeof(char), size, _cFile);
		_crc32 = rc_crc32(_crc32, ptr, size);
		if (ret == size * sizeof(char)) return true;
		error("failure of a write operation");
		return false;
	}

	void write_crc32()
	{
		uint32_t crc32 = ~_crc32 ^ 0xa23777ac;
		write(reinterpret_cast<const char *>(&crc32), sizeof(crc32));
	}

	bool check_crc32()
	{
		uint32_t crc32 = 0, ocrc32 = ~_crc32 ^ 0xa23777ac;	// before the read operation
		read(reinterpret_cast<char *>(&crc32), sizeof(crc32));
		const bool success = (crc32 == ocrc32);
		if (!success) error("bad file (crc32)");
		return success;
	}
};
