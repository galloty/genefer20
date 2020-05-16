# genefer20
Generalized Fermat Prime search program

## About

**genefer20** is an [OpenCL™](https://www.khronos.org/opencl/) application.  
It performs a fast probable primality test for numbers of the form *b*<sup>2<sup>*n*</sup></sup> + 1 with [Fermat test](https://en.wikipedia.org/wiki/Fermat_primality_test).  
[genefer](https://primes.utm.edu/bios/page.php?id=2740) was created by Yves Gallot in 2001. It has been extensively used by [PrimeGrid](https://www.primegrid.com/forum_forum.php?id=75) computing project. genefer is dedicated to the search for large primes (*n* &ge; 15).  
genefer20 is a new highly optimised GPU application, created in 2020. It is dedicated to the search for GFN primes in the range 10 &le; *n* &le; 14.

## Build

genefer20 is under development.  
Any number of the form *b*<sup>2<sup>*n*</sup></sup> + 1 such that 2 &le; *b* < 2,000,000,000 and 10 &le; *n* &le; 14 can be tested.  
*This version was compiled with gcc and tested on Windows and Linux (Ubuntu).*  
An OpenCL SDK is not required. OpenCL header files are included in the project and the application can be linked with the dynamic OpenCL library of the OS.

## TODO

- ...
