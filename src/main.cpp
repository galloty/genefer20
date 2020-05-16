/*
Copyright 2020, Yves Gallot

genefer20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <cstdlib>
#include <stdexcept>
#include <iostream>

int main(int argc, char * argv[])
{
	try
	{
	}
	catch (const std::runtime_error & e)
	{
		std::cerr << std::endl << "error: " << e.what() << "." << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
