# genefer20
Generalized Fermat Prime search program

## About

**genefer20** is an [OpenCL™](https://www.khronos.org/opencl/) application.  

It performs a fast probable primality test for numbers of the form *b*<sup>2<sup>*n*</sup></sup> + 1 with [Fermat test](https://en.wikipedia.org/wiki/Fermat_primality_test).  

[genefer](https://github.com/galloty/genefer22) was created by Yves Gallot in 2001. It has been extensively used by [PrimeGrid](https://www.primegrid.com/forum_forum.php?id=75) computing project. genefer is dedicated to the search for large primes (*n* &ge; 15).  
genefer20 is a highly optimised GPU application, created in 2020. It is dedicated to the search for GFN primes in the range 8 &le; *n* &le; 14. [Efficient Modular Exponentiation Proof Scheme](https://arxiv.org/abs/2209.15623) discovered by Darren Li is implemented and the tests are validated with [Gerbicz - Li](https://www.mersenneforum.org/showthread.php?t=22510) error checking.  

Any number of the form *b*<sup>2<sup>*n*</sup></sup> + 1 such that 2 &le; *b* < 2,000,000,000 and 8 &le; *n* &le; 15 can be tested.  

## Build

Binaries are generated using *Ubuntu 18.04 amd64, gcc 7.5* and *Windows - MSYS2, gcc 12.2*.  

OpenCL SDK is not required. OpenCL header files are included in the project and the application can be linked with the dynamic OpenCL library of the OS.
