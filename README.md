# genefer&nbsp;20
Generalized Fermat Prime search program

## About

**genefer** *version 20* is an [OpenCL™](https://www.khronos.org/opencl/) application.  

It performs a fast probable primality test for numbers of the form *b*<sup>2<sup>*n*</sup></sup>&nbsp;+&nbsp;1 ([Generalized Fermat Numbers](https://genefer.great-site.net/)) using a [Fermat test](https://en.wikipedia.org/wiki/Fermat_primality_test).  

[genefer](https://github.com/galloty/genefer22) was created by Yves Gallot in 2001. It has been extensively used by [PrimeGrid](https://www.primegrid.com/forum_forum.php?id=75) computing project. genefer is dedicated to the search for large primes (*n* &ge; 16).  

genefer 20 is a highly optimised GPU application, created in 2020. It is dedicated to the search for GFN primes in the range 8 &le; *n* &le; 16. The search for *b* < 2,000,000,000 and *n* &le; 14 is now complete thanks to the [PRIVATE GFN SERVER](http://boincvm.proxyma.ru:30080/test4vm/index.php). Statistics are available at [Generalized Fermat Numbers](https://genefer.great-site.net/#search) and data at [GFN Prime Search Status and History](https://www.primegrid.com/gfn_history.php).  

[Efficient Modular Exponentiation Proof Scheme](https://arxiv.org/abs/2209.15623) discovered by Darren Li is implemented and the tests are validated with [Gerbicz - Li](https://www.mersenneforum.org/showthread.php?t=22510) error checking.  

Any number of the form *b*<sup>2<sup>*n*</sup></sup> + 1 such that 2 &le; *b* &le; 2,000,000,000 and 8 &le; *n* &le; 17 can be tested.  

genefer *version 20* is deprecated. It is replaced with **genefer** [version 26](https://github.com/galloty/genefer26).
