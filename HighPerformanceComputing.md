---
name: HighPerformanceComputing
topic: High-Performance and Parallel Computing with R
maintainer: Dirk Eddelbuettel
email: Dirk.Eddelbuettel@R-project.org
version: 2022-01-17
source: https://github.com/cran-task-views/HighPerformanceComputing/
---

This CRAN Task View contains a list of packages, grouped by topic, that are useful for
high-performance computing (HPC) with R. In this context, we are defining 'high-performance
computing' rather loosely as just about anything related to pushing R a little further: using
compiled code, parallel computing (in both explicit and implicit modes), working with large objects
as well as profiling.

Unless otherwise mentioned, all packages presented with hyperlinks are available from the
[Comprehensive R Archive Network (CRAN)](https://cran.r-project.org).

Several of the areas discussed in this Task View are undergoing rapid change. Please send
suggestions for additions and extensions for this task view via e-mail to the maintainer or submit
an issue or pull request in the GitHub repository linked above.  See the [Contributing
page](https://github.com/cran-task-views/ctv/blob/main/Contributing.md) in the [CRAN Task
Views](https://github.com/cran-task-views) repo for details.

Suggestions and corrections by Achim Zeileis, Markus Schmidberger, Martin Morgan, Max Kuhn, Tomas
Radivoyevitch, Jochen Knaus, Tobias Verbeke, Hao Yu, David Rosenberg, Marco Enea, Ivo Welch, Jay
Emerson, Wei-Chen Chen, Bill Cleveland, Ross Boylan, Ramon Diaz-Uriarte, Mark Zeligman, Kevin Ushey,
Graham Jeffries, Will Landau, Tim Flutre, Reza Mohammadi, Ralf Stubner, Bob Jansen, Matt Fidler,
Brent Brewington and Ben Bolder (as well as others I may have forgotten to add here) are gratefully
acknowledged.

The `ctv` package supports these Task Views. Its functions `install.views` and `update.views` allow,
respectively, installation or update of packages from a given Task View; the option `coreOnly` can
restrict operations to packages labeled as *core* below.

Direct support in R started with release 2.14.0 which includes a new package **parallel**
incorporating (slightly revised) copies of packages multicore and `r pkg("snow", priority =
"core")`. Some types of clusters are not handled directly by the base package 'parallel'.  However,
and as explained in the package vignette, the parts of parallel which provide `r pkg("snow")` -like
functions will accept `r pkg("snow")` clusters including MPI clusters. Use `vignette("parallel")` to
view the package vignette.\ The **parallel** package also contains support for multiple RNG streams
following L'Ecuyer et al (2002), with support for both mclapply and snow clusters.\ The version
released for R 2.14.0 contains base functionality: higher-level convenience functions are planned
for later R releases.

### Parallel computing: Explicit parallelism

-   Several packages provide the communications layer required for parallel computing. The first
    package in this area was rpvm by Li and Rossini which uses the PVM (Parallel Virtual Machine)
    standard and libraries. rpvm is no longer actively maintained, but available from its CRAN
    archive directory.
-   In recent years, the alternative MPI (Message Passing Interface) standard has become the de
    facto standard in parallel computing. It is supported in R via the `r pkg("Rmpi", priority =
    "core")` by Yu.  `r pkg("Rmpi")` package is mature yet actively maintained and offers access to
    numerous functions from the MPI API, as well as a number of R-specific extensions.
    `r pkg("Rmpi")` can be used with the LAM/MPI, MPICH / MPICH2, Open MPI, and Deino MPI
    implementations. It should be noted that LAM/MPI is now in maintenance mode, and new development
    is focused on Open MPI.
-   The `r pkg("pbdMPI")` package provides S4 classes to directly interface MPI in order to support
    the Single Program/Multiple Data (SPMD) parallel programming style which is particularly useful
    for batch parallel execution.
-   The `r pkg("snow")` (Simple Network of Workstations) package by Tierney et al. can use PVM, MPI,
    NWS as well as direct networking sockets. It provides an abstraction layer by hiding the
    communications details. The `r pkg("snowFT")` package provides fault-tolerance extensions to
    `r pkg("snow")`.
-   The `r pkg("snowfall")` package by Knaus provides a more recent alternative to
    `r pkg("snow")`. Functions can be used in sequential or parallel mode.
-   The `r pkg("foreach")` package allows general iteration over elements in a collection without
    the use of an explicit loop counter. Using foreach without side effects also facilitates
    executing the loop in parallel which is possible via the `r pkg("doMC")` (using
    parallel/multicore on single workstations), `r pkg("doSNOW")` (using `r pkg("snow")`, see
    above), `r pkg("doMPI")` (using `r pkg("Rmpi")`) packages, and `r pkg("doFuture")` (using
    `r pkg("future")`) packages.
-   The `r pkg("future")` package allows for synchronous (sequential) and asynchronous (parallel)
    evaluations via abstraction of futures, either via function calls or implicitly via promises.
    Global variables are automatically identified. Iteration over elements in a collection is
    supported.
-   The `r pkg("Rborist")` package employs OpenMP pragmas to exploit predictor-level parallelism in
    the Random Forest algorithm which promotes efficient use of multicore hardware in restaging data
    and in determining splitting criteria, both of which are performance bottlenecks in the
    algorithm.
-   The `r pkg("h2o")` package connects to the h2o open source machine learning environment which
    has scalable implementations of random forests, GBM, GLM (with elastic net regularization), and
    deep learning.
-   The `r pkg("randomForestSRC")` package can use both OpenMP as well as MPI for random forest
    extensions suitable for survival analysis, competing risks analysis, classification as well as
    regression
-   The `r pkg("parSim")` package can perform simulation studies using one or multiple cores, both
    locally and on HPC clusters.
-   The `r pkg("qsub")` package can submit commands to run on gridengine clusters.

### Parallel computing: Implicit parallelism

-   The pnmath package by Tierney ( [link](http://www.stat.uiowa.edu/~luke/R/experimental/) ) uses
    the OpenMP parallel processing directives of recent compilers (such gcc 4.2 or later) for
    implicit parallelism by replacing a number of internal R functions with replacements that can
    make use of multiple cores \-\-- without any explicit requests from the user. The alternate
    pnmath0 package offers the same functionality using Pthreads for environments in which the newer
    compilers are not available. Similar functionality is expected to become integrated into R
    'eventually'.
-   The romp package by Jamitzky was presented at useR! 2008 (
    [slides](http://www.statistik.tu-dortmund.de/useR-2008/slides/Jamitzky.pdf) ) and offers another
    interface to OpenMP using Fortran. The code is still pre-alpha and available from the Google
    Code project `r gcode("romp")`. An R-Forge project `r rforge("romp")` was initiated but there is
    no package, yet.
-   The `r pkg("Rdsm")` package provides a threads-like parallel computing environment, both on
    multicore machine and across the network by providing facilities inspired from distributed
    shared memory programming.
-   The `r pkg("RhpcBLASctl")` package detects the number of available BLAS cores, and permits
    explicit selection of the number of cores.
-   The `r pkg("targets")` package and its predecessor `r pkg("drake")` are R-focused pipeline
    toolkits similar to [Make](https://www.gnu.org/software/make) . Each constructs a directed
    acyclic graph representation of the workflow and orchestrates distributed computing across
    `clustermq` and `future` workers.
-   The `r pkg("flexiblas")` package manages BLAS/LAPACK libraries by loading and possibly switching
    them if FlexiBLAS ( [link](https://www.mpi-magdeburg.mpg.de/projects/flexiblas) ) is used.

### Parallel computing: Grid computing

-   The multiR package by Grose was presented at useR! 2008 but has not been released. It may offer
    a snow-style framework on a grid computing platform.
-   The `r rforge("biocep-distrib")` project by Chine offers a Java-based framework for local, Grid,
    or Cloud computing. It is under active development.

### Parallel computing: Hadoop

-   The `r github("saptarshiguha/RHIPE")` package, started by Saptarshi Guha, provides an interface
    between R and Hadoop for analysis of large complex data wholly from within R using the Divide
    and Recombine approach to big data.
-   The rmr package by Revolution Analytics also provides an interface between R and Hadoop for a
    Map/Reduce programming framework. (
    [link](https://github.com/RevolutionAnalytics/RHadoop/wiki/rmr) )
-   A related package, segue package by Long, permits easy execution of embarassingly parallel task
    on Elastic Map Reduce (EMR) at Amazon. ( [link](http://code.google.com/p/segue/) )
-   The `r pkg("RProtoBuf")` package provides an interface to Google's language-neutral,
    platform-neutral, extensible mechanism for serializing structured data. This package can be used
    in R code to read data streams from other systems in a distributed MapReduce setting where data
    is serialized and passed back and forth between tasks.
-   The `r pkg("HistogramTools")` package provides a number of routines useful for the construction,
    aggregation, manipulation, and plotting of large numbers of histograms such as those created by
    mappers in a MapReduce application.

### Parallel computing: Random numbers

-   Random-number generators for parallel computing are available via the `r pkg("rlecuyer")`
    package, the `r pkg("rstream")` package, the `r pkg("sitmo")` package as well as the
    `r pkg("dqrng")` package.
-   The `r pkg("doRNG")` package provides functions to perform reproducible parallel foreach loops,
    using independent random streams as generated by the package rstream, suitable for the different
    foreach backends.

### Parallel computing: Resource managers and batch schedulers

-   Job-scheduling toolkits permit management of parallel computing resources and tasks. The slurm
    (Simple Linux Utility for Resource Management) set of programs works well with MPI and slurm
    jobs can be submitted from R using the `r pkg("rslurm")` package.  (
    [link](http://slurm.schedmd.com/) )
-   The Condor toolkit ( [link](http://www.cs.wisc.edu/condor/) ) from the University of
    Wisconsin-Madison has been used with R as described in this [R News
    article](http://www.r-project.org/doc/Rnews/Rnews_2005-2.pdf) .
-   The sfCluster package by Knaus can be used with `r pkg("snowfall")`. (
    [link](http://www.imbi.uni-freiburg.de/parallel/) ) but is currently limited to LAM/MPI.
-   The `r pkg("batch")` package by Hoffmann can launch parallel computing requests onto a cluster
    and gather results.
-   The `r pkg("BatchJobs")` package provides Map, Reduce and Filter variants to manage R jobs and
    their results on batch computing systems like PBS/Torque, LSF and Sun Grid Engine.  Multicore
    and SSH systems are also supported. The `r pkg("BatchExperiments")` package extends it with an
    abstraction layer for running statistical experiments. Package `r pkg("batchtools")` is a
    successor / extension to both.
-   The `r pkg("flowr")` package offers a scatter-gather approach to submit jobs lists (including
    dependencies) to the computing cluster via simple data.frames as inputs. It supports LSF, SGE,
    Torque and SLURM.
-   The `r pkg("clustermq")` package sends function calls as jobs on LSF, SGE and SLURM via a single
    line of code without using network-mounted storage. It also supports use of remote clusters via
    SSH.

### Parallel computing: Applications

-   The `r pkg("caret")` package by Kuhn can use various frameworks (MPI, NWS etc) to parallelized
    cross-validation and bootstrap characterizations of predictive models.
-   The `r bioc("maanova")` package on Bioconductor by Wu can use `r pkg("snow")` and
    `r pkg("Rmpi")` for the analysis of micro-array experiments.
-   The `r pkg("pvclust")` package by Suzuki and Shimodaira can use `r pkg("snow")` and
    `r pkg("Rmpi")` for hierarchical clustering via multiscale bootstraps.
-   The `r pkg("tm")` package by Feinerer can use `r pkg("snow")` and `r pkg("Rmpi")` for
    parallelized text mining.
-   The `r pkg("varSelRF")` package by Diaz-Uriarte can use `r pkg("snow")` and `r pkg("Rmpi")` for
    parallelized use of variable selection via random forests.
-   The `r pkg("bcp")` package by Erdman and Emerson for the Bayesian analysis of change points can
    use `r pkg("foreach")` for parallelized operations.
-   The `r bioc("multtest")` package by Pollard et al. on Bioconductor can use `r pkg("snow")`,
    `r pkg("Rmpi")` or rpvm for resampling-based testing of multiple hypothesis.
-   The `r pkg("Matching")` package by Sekhon for multivariate and propensity score matching, the
    `r pkg("STAR")` package by Pouzat for spike train analysis, the `r pkg("bnlearn")` package by
    Scutari for bayesian network structure learning, the `r pkg("latentnet")` package by Krivitsky
    and Handcock for latent position and cluster models, the `r pkg("peperr")` package by Porzelius
    and Binder for parallelised estimation of prediction error, the `r pkg("orloca")` package by
    Fernandez-Palacin and Munoz-Marquez for operations research locational analysis, the
    `r pkg("rgenoud")` package by Mebane and Sekhon for genetic optimization using derivatives the
    `r bioc("affyPara")` package by Schmidberger, Vicedo and Mansmann for parallel normalization of
    Affymetrix microarrays, and the `r bioc("puma")` package by Pearson et al. which propagates
    uncertainty into standard microarray analyses such as differential expression all can use
    `r pkg("snow")` for parallelized operations using either one of the MPI, PVM, NWS or socket
    protocols supported by `r pkg("snow")`.
-   The `r gcode("bugsparallel")` package uses `r pkg("Rmpi")` for distributed computing of multiple
    MCMC chains using WinBUGS.
-   The `r pkg("xgboost")` package by Chen et al. is an optimized distributed gradient boosting
    library designed to be highly efficient, flexible and portable. The same code runs on major
    distributed environment, such as Hadoop, SGE, and MPI.
-   The `r pkg("dclone")` package provides a global optimization approach and a variant of simulated
    annealing which exploits Bayesian MCMC tools to get MLE point estimates and standard errors
    using low level functions for implementing maximum likelihood estimating procedures for complex
    models using data cloning and Bayesian Markov chain Monte Carlo methods with support for JAGS,
    WinBUGS and OpenBUGS; parallel computing is supported via the `r pkg("snow")` package.
-   Nowadays, many packages can use the facilities offered by the **parallel** package. One example
    is `r pkg("pls")`.
-   The `r pkg("pbapply")` package offers a progress bar for vectorized R functions in the `\*apply`
    family, and supports several backends.
-   The `r pkg("Sim.DiffProc")` package simulates and estimates multidimensional Itô and
    Stratonovich stochastic differential equations in parallel.
-   The `r pkg("keras")` package by by Allaire et al.  provides a high-level neural networks API. It
    was developed with a focus on enabling fast experimentation for convolutional networks,
    recurrent networks, any combination of both, and custom neural network architectures.
-   The `r pkg("mvnfast")` uses the sumo random number generator to generate multivariate and normal
    distribtuions in parallel.

### Parallel computing: GPUs

-   The rgpu package (see below for link) aims to speed up bioinformatics analysis by using the GPU.
-   The `r pkg("gcbd")` package implements a benchmarking framework for BLAS and GPUs.
-   The `r pkg("OpenCL")` package provides an interface from R to OpenCL permitting hardware- and
    vendor neutral interfaces to GPU programming.
-   The `r pkg("tensorflow")` package by by Allaire et al.  provides access to the complete
    TensorFlow API from within R that enables numerical computation using data flow graphs. The
    flexible architecture allows users to deploy computation to one or more CPUs or GPUs in a
    desktop, server, or mobile device with a single API.
-   The `r pkg("tfestimators")` package by by Tang et al.  offers a high-level API that provides
    implementations of many different model types including linear models and deep neural
    networks. It also provides a flexible framework for defining arbitrary new model types as custom
    estimators with the distributed power of TensorFlow for free.
-   The `r pkg("BDgraph")` package provides statistical tools for Bayesian structure learning in
    undirected graphical models for multivariate continuous, discrete, and mixed data using parallel
    sampling algorithms implemented using OpenMP and C++.
-   The `r pkg("ssgraph")` package offers Bayesian inference in undirected graphical models using
    spike-and-slab priors for multivariate continuous, discrete, and mixed data. Computationally
    intensive tasks of the package are using OpenMP via C++.

### Large memory and out-of-memory data

-   The `r pkg("biglm")` package by Lumley uses incremental computations to offer `lm()` and `glm()`
    functionality to data sets stored outside of R's main memory.
-   The `r pkg("ff")` package by Adler et al. offers file-based access to data sets that are too
    large to be loaded into memory, along with a number of higher-level functions.
-   The `r pkg("bigmemory")` package by Kane and Emerson permits storing large objects such as
    matrices in memory (as well as via files) and uses external pointer objects to refer to
    them. This permits transparent access from R without bumping against R's internal memory
    limits. Several R processes on the same computer can also share big memory objects.
-   A large number of database packages, and database-alike packages (such as `r pkg("sqldf")` by
    Grothendieck and `r pkg("data.table")` by Dowle) are also of potential interest but not reviewed
    here.
-   The `r pkg("HadoopStreaming")` package provides a framework for writing map/reduce scripts for
    use in Hadoop Streaming; it also facilitates operating on data in a streaming fashion which does
    not require Hadoop.
-   The `r pkg("speedglm")` package permits to fit (generalised) linear models to large data. For
    in-memory data sets, speedlm() or speedglm() can be used along with update.speedlm() which can
    update fitted models with new data. For out-of-memory data sets, shglm() is available; it works
    in the presence of factors and can check for singular matrices.
-   The `r pkg("MonetDB.R")` package allows R to access the MonetDB column-oriented, open source
    database system as a backend.
-   The `r pkg("ffbase")` package by de Jonge et al adds basic statistical functionality to the
    `r pkg("ff")` package.
-   The `r pkg("LaF")` package provides methods for fast access to large ASCII files in csv or
    fixed-width format.
-   The `r pkg("bigstatsr")` package also operates on file-backed large matrices via memory-mapped
    access, and offeres several matrix operationc, PCA, sparse methods and more..
-   The `r pkg("disk.frame")` package leverages several other packages to provide efficient access
    and manipulation operations for data sets that are larger than RAM.

### Easier interfaces for Compiled code

-   The `r pkg("inline")` package by Sklyar et al eases adding code in C, C++ or Fortran to R. It
    takes care of the compilation, linking and loading of embedded code segments that are stored as
    R strings.
-   The `r pkg("Rcpp")` package by Eddelbuettel and Francois offers a number of C++ classes that
    makes transferring R objects to C++ functions (and back) easier, and the `r pkg("RInside")`
    package by the same authors allows easy embedding of R itself into C++ applications for faster
    and more direct data transfer.
-   The `r pkg("RcppParallel")` package by Allaire et al.  bundles the [Intel Threading Building
    Blocks](https://www.threadingbuildingblocks.org) and
    [TinyThread](http://tinythreadpp.bitsnbites.eu) libraries. Together with `r pkg("Rcpp")`,
    RcppParallel makes it easy to write safe, performant, concurrently-executing C++ code, and use
    that code within R and R packages.
-   The `r pkg("rJava")` package by Urbanek provides a low-level interface to Java similar to the
    `.Call()` interface for C and C++.
-   The `r pkg("reticulate")` package by Allaire provides interface to Python modules, classes, and
    functions. It allows R users to access many high-performance Python packages such as
    `r pkg("tensorflow")` and `r pkg("tfestimators")` within R.

### Profiling tools

Packages `r pkg("profvis")`, `r pkg("proffer")`, `r pkg("profmem")`, `r pkg("GUIProfiler")`,
`r pkg("proftools")`, and `r pkg("aprof")` summarize and visualize output from the `Rprof` interface
for profiling.  The `r pkg("profile")` package reads and writes profiling data and converts among
file formats such as [`pprof`](https://github.com/google/pprof) by Google and `Rprof`. The
[`xrprof`](https://github.com/atheriel/xrprof) command-line tool implements profile sampling for a
given R process on Linux or Windows, and it can profile R code alongside compiled code.


### Links
-   [HPC computing notes by Luke Tierney for HPC class at University of Iowa](http://www.stat.uiowa.edu/~luke/classes/295-hpc/)
-   [Mailing List: R Special Interest Group High Performance Computing](https://stat.ethz.ch/mailman/listinfo/r-sig-hpc/)
-   [Schmidberger, Morgan, Eddelbuettel, Yu, Tierney and Mansmann (2009) paper on 'State of the Art in Parallel Computing with R'](http://www.jstatsoft.org/v31/i01/)
-   [Luke Tierney's code directory for pnmath and pnmath0](http://www.stat.uiowa.edu/~luke/R/experimental/)
-   [Slurm open-source workload manager](http://slurm.schedmd.com/)
-   [Condor project at University of Wisconsin-Madison](http://www.cs.wisc.edu/condor/)
-   [Parallel Computing in R with sfCluster/snowfall](http://www.imbi.uni-freiburg.de/parallel/)
-   [Wikipedia: Message Passing Interface (MPI)](http://en.wikipedia.org/wiki/Message_Passing_Interface)
-   [Wikipedia: Parallel Virtual Machine (PVM)](http://en.wikipedia.org/wiki/Parallel_Virtual_Machine)
-   [Slides from Introduction to High-Performance Computing with R tutorial help in Nov 2009 at the Institute for Statistical Mathematics, Tokyo, Japan](http://dirk.eddelbuettel.com/papers/ismNov2009introHPCwithR.pdf)
-   [rgpu project at nbic.nl](https://gforge.nbic.nl/projects/rgpu/)
-   [Magma: Matrix Algebra on GPU and Multicore architectures](http://icl.cs.utk.edu/magma/)
-   [Parallel R: Data Analysis in the Distributed World](http://shop.oreilly.com/product/0636920021421.do)
-   [High Performance Statistical Computing for Data Intensive Research](https://snoweye.github.io/hpsc/)
-   [Rth: Parallel R through Thrust](http://heather.cs.ucdavis.edu/~matloff/rth.html)
-   [Programming with Big Data in R](http://r-pbd.org)
