## CRAN Task View: High-Performance and Parallel Computing with R

                                                                             
--------------- ----------------------------------------------------------   
**Maintainer:** Dirk Eddelbuettel                                            
**Contact:**    Dirk.Eddelbuettel at R-project.org                           
**Version:**    2021-11-08                                                   
**URL:**        <https://CRAN.R-project.org/view=HighPerformanceComputing>   

<div>

This CRAN task view contains a list of packages, grouped by topic, that
are useful for high-performance computing (HPC) with R. In this context,
we are defining 'high-performance computing' rather loosely as just
about anything related to pushing R a little further: using compiled
code, parallel computing (in both explicit and implicit modes), working
with large objects as well as profiling.

Unless otherwise mentioned, all packages presented with hyperlinks are
available from CRAN, the Comprehensive R Archive Network.

Several of the areas discussed in this Task View are undergoing rapid
change. Please send suggestions for additions and extensions for this
task view to the [task view
maintainer](mailto:Dirk.Eddelbuettel@R-project.org) .

Suggestions and corrections by Achim Zeileis, Markus Schmidberger,
Martin Morgan, Max Kuhn, Tomas Radivoyevitch, Jochen Knaus, Tobias
Verbeke, Hao Yu, David Rosenberg, Marco Enea, Ivo Welch, Jay Emerson,
Wei-Chen Chen, Bill Cleveland, Ross Boylan, Ramon Diaz-Uriarte, Mark
Zeligman, Kevin Ushey, Graham Jeffries, Will Landau, Tim Flutre, Reza
Mohammadi, Ralf Stubner, Bob Jansen, Matt Fidler, Brent Brewington and
Ben Bolder (as well as others I may have forgotten to add here) are
gratefully acknowledged.

Contributions are always welcome, and encouraged. Since the start of
this CRAN task view in October 2008, most contributions have arrived as
email suggestions. The source file for this particular task view file
now also reside in a GitHub repository (see below) so that pull requests
are also possible.

The `ctv` package supports these Task Views. Its functions
`install.views` and `update.views` allow, respectively, installation or
update of packages from a given Task View; the option `coreOnly` can
restrict operations to packages labeled as *core* below.

**Direct support in R started with release 2.14.0** which includes a new
package **parallel** incorporating (slightly revised) copies of packages
multicore and [snow](https://cran.r-project.org/package=snow). Some types of
clusters are not handled directly by the base package 'parallel'.
However, and as explained in the package vignette, the parts of parallel
which provide [snow](https://cran.r-project.org/package=snow) -like functions will
accept [snow](https://cran.r-project.org/package=snow) clusters including MPI
clusters. Use `vignette("parallel")` to view the package vignette.  
The **parallel** package also contains support for multiple RNG streams
following L'Ecuyer et al (2002), with support for both mclapply and snow
clusters.  
The version released for R 2.14.0 contains base functionality:
higher-level convenience functions are planned for later R releases.

**Parallel computing: Explicit parallelism**

  - Several packages provide the communications layer required for
    parallel computing. The first package in this area was rpvm by Li
    and Rossini which uses the PVM (Parallel Virtual Machine) standard
    and libraries. rpvm is no longer actively maintained, but available
    from its CRAN archive directory.
  - In recent years, the alternative MPI (Message Passing Interface)
    standard has become the de facto standard in parallel computing. It
    is supported in R via the [Rmpi](https://cran.r-project.org/package=Rmpi) by Yu.
    [Rmpi](https://cran.r-project.org/package=Rmpi) package is mature yet actively
    maintained and offers access to numerous functions from the MPI API,
    as well as a number of R-specific extensions.
    [Rmpi](https://cran.r-project.org/package=Rmpi) can be used with the LAM/MPI,
    MPICH / MPICH2, Open MPI, and Deino MPI implementations. It should
    be noted that LAM/MPI is now in maintenance mode, and new
    development is focused on Open MPI.
  - The [pbdMPI](https://cran.r-project.org/package=pbdMPI) package provides S4
    classes to directly interface MPI in order to support the Single
    Program/Multiple Data (SPMD) parallel programming style which is
    particularly useful for batch parallel execution.
  - The [snow](https://cran.r-project.org/package=snow) (Simple Network of
    Workstations) package by Tierney et al. can use PVM, MPI, NWS as
    well as direct networking sockets. It provides an abstraction layer
    by hiding the communications details. The
    [snowFT](https://cran.r-project.org/package=snowFT) package provides
    fault-tolerance extensions to [snow](https://cran.r-project.org/package=snow).
  - The [snowfall](https://cran.r-project.org/package=snowfall) package by Knaus
    provides a more recent alternative to
    [snow](https://cran.r-project.org/package=snow). Functions can be used in
    sequential or parallel mode.
  - The [foreach](https://cran.r-project.org/package=foreach) package allows general
    iteration over elements in a collection without the use of an
    explicit loop counter. Using foreach without side effects also
    facilitates executing the loop in parallel which is possible via the
    [doMC](https://cran.r-project.org/package=doMC) (using parallel/multicore on
    single workstations), [doSNOW](https://cran.r-project.org/package=doSNOW) (using
    [snow](https://cran.r-project.org/package=snow), see above),
    [doMPI](https://cran.r-project.org/package=doMPI) (using
    [Rmpi](https://cran.r-project.org/package=Rmpi)) packages, and
    [doFuture](https://cran.r-project.org/package=doFuture) (using
    [future](https://cran.r-project.org/package=future)) packages.
  - The [future](https://cran.r-project.org/package=future) package allows for
    synchronous (sequential) and asynchronous (parallel) evaluations via
    abstraction of futures, either via function calls or implicitly via
    promises. Global variables are automatically identified. Iteration
    over elements in a collection is supported.
  - The [Rborist](https://cran.r-project.org/package=Rborist) package employs OpenMP
    pragmas to exploit predictor-level parallelism in the Random Forest
    algorithm which promotes efficient use of multicore hardware in
    restaging data and in determining splitting criteria, both of which
    are performance bottlenecks in the algorithm.
  - The [h2o](https://cran.r-project.org/package=h2o) package connects to the h2o
    open source machine learning environment which has scalable
    implementations of random forests, GBM, GLM (with elastic net
    regularization), and deep learning.
  - The [randomForestSRC](https://cran.r-project.org/package=randomForestSRC)
    package can use both OpenMP as well as MPI for random forest
    extensions suitable for survival analysis, competing risks analysis,
    classification as well as regression
  - The [parSim](https://cran.r-project.org/package=parSim) package can perform
    simulation studies using one or multiple cores, both locally and on
    HPC clusters.
  - The [qsub](https://cran.r-project.org/package=qsub) package can submit commands
    to run on gridengine clusters.

**Parallel computing: Implicit parallelism**

  - The pnmath package by Tierney (
    [link](http://www.stat.uiowa.edu/~luke/R/experimental/) ) uses the
    OpenMP parallel processing directives of recent compilers (such gcc
    4.2 or later) for implicit parallelism by replacing a number of
    internal R functions with replacements that can make use of multiple
    cores --- without any explicit requests from the user. The alternate
    pnmath0 package offers the same functionality using Pthreads for
    environments in which the newer compilers are not available. Similar
    functionality is expected to become integrated into R 'eventually'.
  - The romp package by Jamitzky was presented at useR\! 2008 (
    [slides](http://www.statistik.tu-dortmund.de/useR-2008/slides/Jamitzky.pdf)
    ) and offers another interface to OpenMP using Fortran. The code is
    still pre-alpha and available from the Google Code project
    [<span class="Gcode">romp</span>](https://code.google.com/archive/p/romp/).
    An R-Forge project
    [<span class="Rforge">romp</span>](https://R-Forge.R-project.org/projects/romp/)
    was initiated but there is no package, yet.
  - The [Rdsm](https://cran.r-project.org/package=Rdsm) package provides a
    threads-like parallel computing environment, both on multicore
    machine and across the network by providing facilities inspired from
    distributed shared memory programming.
  - The [RhpcBLASctl](https://cran.r-project.org/package=RhpcBLASctl) package
    detects the number of available BLAS cores, and permits explicit
    selection of the number of cores.
  - The [targets](https://cran.r-project.org/package=targets) package and its
    predecessor [drake](https://cran.r-project.org/package=drake) are R-focused
    pipeline toolkits similar to
    [Make](https://www.gnu.org/software/make) . Each constructs a
    directed acyclic graph representation of the workflow and
    orchestrates distributed computing across `clustermq` and `future`
    workers.
  - The [flexiblas](https://cran.r-project.org/package=flexiblas) package manages
    BLAS/LAPACK libraries by loading and possibly switching them if
    FlexiBLAS (
    [link](https://www.mpi-magdeburg.mpg.de/projects/flexiblas) ) is
    used.

**Parallel computing: Grid computing**

  - The multiR package by Grose was presented at useR\! 2008 but has not
    been released. It may offer a snow-style framework on a grid
    computing platform.
  - The
    [<span class="Rforge">biocep-distrib</span>](https://R-Forge.R-project.org/projects/biocep-distrib/)
    project by Chine offers a Java-based framework for local, Grid, or
    Cloud computing. It is under active development.

**Parallel computing: Hadoop**

  - The
    [<span class="GitHub">RHIPE</span>](https://github.com/saptarshiguha/RHIPE/)
    package, started by Saptarshi Guha, provides an interface between R
    and Hadoop for analysis of large complex data wholly from within R
    using the Divide and Recombine approach to big data.
  - The rmr package by Revolution Analytics also provides an interface
    between R and Hadoop for a Map/Reduce programming framework. (
    [link](https://github.com/RevolutionAnalytics/RHadoop/wiki/rmr) )
  - A related package, segue package by Long, permits easy execution of
    embarassingly parallel task on Elastic Map Reduce (EMR) at Amazon. (
    [link](http://code.google.com/p/segue/) )
  - The [RProtoBuf](https://cran.r-project.org/package=RProtoBuf) package provides
    an interface to Google's language-neutral, platform-neutral,
    extensible mechanism for serializing structured data. This package
    can be used in R code to read data streams from other systems in a
    distributed MapReduce setting where data is serialized and passed
    back and forth between tasks.
  - The [HistogramTools](https://cran.r-project.org/package=HistogramTools) package
    provides a number of routines useful for the construction,
    aggregation, manipulation, and plotting of large numbers of
    histograms such as those created by mappers in a MapReduce
    application.

**Parallel computing: Random numbers**

  - Random-number generators for parallel computing are available via
    the [rlecuyer](https://cran.r-project.org/package=rlecuyer) package, the
    [rstream](https://cran.r-project.org/package=rstream) package, the
    [sitmo](https://cran.r-project.org/package=sitmo) package as well as the
    [dqrng](https://cran.r-project.org/package=dqrng) package.
  - The [doRNG](https://cran.r-project.org/package=doRNG) package provides functions
    to perform reproducible parallel foreach loops, using independent
    random streams as generated by the package rstream, suitable for the
    different foreach backends.

**Parallel computing: Resource managers and batch schedulers**

  - Job-scheduling toolkits permit management of parallel computing
    resources and tasks. The slurm (Simple Linux Utility for Resource
    Management) set of programs works well with MPI and slurm jobs can
    be submitted from R using the
    [rslurm](https://cran.r-project.org/package=rslurm) package. (
    [link](http://slurm.schedmd.com/) )
  - The Condor toolkit ([link](http://www.cs.wisc.edu/condor/) ) from
    the University of Wisconsin-Madison has been used with R as
    described in this [R News
    article](http://www.r-project.org/doc/Rnews/Rnews_2005-2.pdf) .
  - The sfCluster package by Knaus can be used with
    [snowfall](https://cran.r-project.org/package=snowfall). (
    [link](http://www.imbi.uni-freiburg.de/parallel/) ) but is currently
    limited to LAM/MPI.
  - The [batch](https://cran.r-project.org/package=batch) package by Hoffmann can
    launch parallel computing requests onto a cluster and gather
    results.
  - The [BatchJobs](https://cran.r-project.org/package=BatchJobs) package provides
    Map, Reduce and Filter variants to manage R jobs and their results
    on batch computing systems like PBS/Torque, LSF and Sun Grid Engine.
    Multicore and SSH systems are also supported. The
    [BatchExperiments](https://cran.r-project.org/package=BatchExperiments) package
    extends it with an abstraction layer for running statistical
    experiments. Package [batchtools](https://cran.r-project.org/package=batchtools)
    is a successor / extension to both.
  - The [flowr](https://cran.r-project.org/package=flowr) package offers a
    scatter-gather approach to submit jobs lists (including
    dependencies) to the computing cluster via simple data.frames as
    inputs. It supports LSF, SGE, Torque and SLURM.
  - The [clustermq](https://cran.r-project.org/package=clustermq) package sends
    function calls as jobs on LSF, SGE and SLURM via a single line of
    code without using network-mounted storage. It also supports use of
    remote clusters via SSH.

**Parallel computing: Applications**

  - The [caret](https://cran.r-project.org/package=caret) package by Kuhn can use
    various frameworks (MPI, NWS etc) to parallelized cross-validation
    and bootstrap characterizations of predictive models.
  - The
    [<span class="BioC">maanova</span>](https://www.Bioconductor.ohttps://cran.r-project.org/package=release/bioc/html/maanova.html)
    package on Bioconductor by Wu can use
    [snow](https://cran.r-project.org/package=snow) and
    [Rmpi](https://cran.r-project.org/package=Rmpi) for the analysis of micro-array
    experiments.
  - The [pvclust](https://cran.r-project.org/package=pvclust) package by Suzuki and
    Shimodaira can use [snow](https://cran.r-project.org/package=snow) and
    [Rmpi](https://cran.r-project.org/package=Rmpi) for hierarchical clustering via
    multiscale bootstraps.
  - The [tm](https://cran.r-project.org/package=tm) package by Feinerer can use
    [snow](https://cran.r-project.org/package=snow) and
    [Rmpi](https://cran.r-project.org/package=Rmpi) for parallelized text mining.
  - The [varSelRF](https://cran.r-project.org/package=varSelRF) package by
    Diaz-Uriarte can use [snow](https://cran.r-project.org/package=snow) and
    [Rmpi](https://cran.r-project.org/package=Rmpi) for parallelized use of variable
    selection via random forests.
  - The [bcp](https://cran.r-project.org/package=bcp) package by Erdman and Emerson
    for the Bayesian analysis of change points can use
    [foreach](https://cran.r-project.org/package=foreach) for parallelized
    operations.
  - The
    [<span class="BioC">multtest</span>](https://www.Bioconductor.ohttps://cran.r-project.org/package=release/bioc/html/multtest.html)
    package by Pollard et al. on Bioconductor can use
    [snow](https://cran.r-project.org/package=snow),
    [Rmpi](https://cran.r-project.org/package=Rmpi) or rpvm for resampling-based
    testing of multiple hypothesis.
  - The [Matching](https://cran.r-project.org/package=Matching) package by Sekhon
    for multivariate and propensity score matching, the
    [STAR](https://cran.r-project.org/package=STAR) package by Pouzat for spike
    train analysis, the [bnlearn](https://cran.r-project.org/package=bnlearn)
    package by Scutari for bayesian network structure learning, the
    [latentnet](https://cran.r-project.org/package=latentnet) package by Krivitsky
    and Handcock for latent position and cluster models, the
    [peperr](https://cran.r-project.org/package=peperr) package by Porzelius and
    Binder for parallelised estimation of prediction error, the
    [orloca](https://cran.r-project.org/package=orloca) package by Fernandez-Palacin
    and Munoz-Marquez for operations research locational analysis, the
    [rgenoud](https://cran.r-project.org/package=rgenoud) package by Mebane and
    Sekhon for genetic optimization using derivatives the
    [<span class="BioC">affyPara</span>](https://www.Bioconductor.ohttps://cran.r-project.org/package=release/bioc/html/affyPara.html)
    package by Schmidberger, Vicedo and Mansmann for parallel
    normalization of Affymetrix microarrays, and the
    [<span class="BioC">puma</span>](https://www.Bioconductor.ohttps://cran.r-project.org/package=release/bioc/html/puma.html)
    package by Pearson et al. which propagates uncertainty into standard
    microarray analyses such as differential expression all can use
    [snow](https://cran.r-project.org/package=snow) for parallelized operations
    using either one of the MPI, PVM, NWS or socket protocols supported
    by [snow](https://cran.r-project.org/package=snow).
  - The
    [<span class="Gcode">bugsparallel</span>](https://code.google.com/archive/p/bugsparallel/)
    package uses [Rmpi](https://cran.r-project.org/package=Rmpi) for distributed
    computing of multiple MCMC chains using WinBUGS.
  - The [xgboost](https://cran.r-project.org/package=xgboost) package by Chen et al.
    is an optimized distributed gradient boosting library designed to be
    highly efficient, flexible and portable. The same code runs on major
    distributed environment, such as Hadoop, SGE, and MPI.
  - The [dclone](https://cran.r-project.org/package=dclone) package provides a
    global optimization approach and a variant of simulated annealing
    which exploits Bayesian MCMC tools to get MLE point estimates and
    standard errors using low level functions for implementing maximum
    likelihood estimating procedures for complex models using data
    cloning and Bayesian Markov chain Monte Carlo methods with support
    for JAGS, WinBUGS and OpenBUGS; parallel computing is supported via
    the [snow](https://cran.r-project.org/package=snow) package.
  - Nowadays, many packages can use the facilities offered by the
    **parallel** package. One example is
    [pls](https://cran.r-project.org/package=pls).
  - The [pbapply](https://cran.r-project.org/package=pbapply) package offers a
    progress bar for vectorized R functions in the \`\*apply\` family,
    and supports several backends.
  - The [Sim.DiffProc](https://cran.r-project.org/package=Sim.DiffProc) package
    simulates and estimates multidimensional Itô and Stratonovich
    stochastic differential equations in parallel.
  - The [keras](https://cran.r-project.org/package=keras) package by by Allaire et
    al. provides a high-level neural networks API. It was developed with
    a focus on enabling fast experimentation for convolutional networks,
    recurrent networks, any combination of both, and custom neural
    network architectures.
  - The [mvnfast](https://cran.r-project.org/package=mvnfast) uses the sumo random
    number generator to generate multivariate and normal distribtuions
    in parallel.

**Parallel computing: GPUs**

  - The rgpu package (see below for link) aims to speed up
    bioinformatics analysis by using the GPU.
  - The [gcbd](https://cran.r-project.org/package=gcbd) package implements a
    benchmarking framework for BLAS and GPUs.
  - The [OpenCL](https://cran.r-project.org/package=OpenCL) package provides an
    interface from R to OpenCL permitting hardware- and vendor neutral
    interfaces to GPU programming.
  - The [tensorflow](https://cran.r-project.org/package=tensorflow) package by by
    Allaire et al. provides access to the complete TensorFlow API from
    within R that enables numerical computation using data flow graphs.
    The flexible architecture allows users to deploy computation to one
    or more CPUs or GPUs in a desktop, server, or mobile device with a
    single API.
  - The [tfestimators](https://cran.r-project.org/package=tfestimators) package by
    by Tang et al. offers a high-level API that provides implementations
    of many different model types including linear models and deep
    neural networks. It also provides a flexible framework for defining
    arbitrary new model types as custom estimators with the distributed
    power of TensorFlow for free.
  - The [BDgraph](https://cran.r-project.org/package=BDgraph) package provides
    statistical tools for Bayesian structure learning in undirected
    graphical models for multivariate continuous, discrete, and mixed
    data using parallel sampling algorithms implemented using OpenMP and
    C++.
  - The [ssgraph](https://cran.r-project.org/package=ssgraph) package offers
    Bayesian inference in undirected graphical models using
    spike-and-slab priors for multivariate continuous, discrete, and
    mixed data. Computationally intensive tasks of the package are using
    OpenMP via C++.

**Large memory and out-of-memory data**

  - The [biglm](https://cran.r-project.org/package=biglm) package by Lumley uses
    incremental computations to offer `lm()` and `glm()` functionality
    to data sets stored outside of R's main memory.
  - The [ff](https://cran.r-project.org/package=ff) package by Adler et al. offers
    file-based access to data sets that are too large to be loaded into
    memory, along with a number of higher-level functions.
  - The [bigmemory](https://cran.r-project.org/package=bigmemory) package by Kane
    and Emerson permits storing large objects such as matrices in memory
    (as well as via files) and uses external pointer objects to refer to
    them. This permits transparent access from R without bumping against
    R's internal memory limits. Several R processes on the same computer
    can also share big memory objects.
  - A large number of database packages, and database-alike packages
    (such as [sqldf](https://cran.r-project.org/package=sqldf) by Grothendieck and
    [data.table](https://cran.r-project.org/package=data.table) by Dowle) are also
    of potential interest but not reviewed here.
  - The [HadoopStreaming](https://cran.r-project.org/package=HadoopStreaming)
    package provides a framework for writing map/reduce scripts for use
    in Hadoop Streaming; it also facilitates operating on data in a
    streaming fashion which does not require Hadoop.
  - The [speedglm](https://cran.r-project.org/package=speedglm) package permits to
    fit (generalised) linear models to large data. For in-memory data
    sets, speedlm() or speedglm() can be used along with
    update.speedlm() which can update fitted models with new data. For
    out-of-memory data sets, shglm() is available; it works in the
    presence of factors and can check for singular matrices.
  - The [MonetDB.R](https://cran.r-project.org/package=MonetDB.R) package allows R
    to access the MonetDB column-oriented, open source database system
    as a backend.
  - The [ffbase](https://cran.r-project.org/package=ffbase) package by de Jonge et
    al adds basic statistical functionality to the
    [ff](https://cran.r-project.org/package=ff) package.
  - The [LaF](https://cran.r-project.org/package=LaF) package provides methods for
    fast access to large ASCII files in csv or fixed-width format.
  - The [bigstatsr](https://cran.r-project.org/package=bigstatsr) package also
    operates on file-backed large matrices via memory-mapped access, and
    offeres several matrix operationc, PCA, sparse methods and more..
  - The [disk.frame](https://cran.r-project.org/package=disk.frame) package
    leverages several other packages to provide efficient access and
    manipulation operations for data sets that are larger than RAM.

**Easier interfaces for Compiled code**

  - The [inline](https://cran.r-project.org/package=inline) package by Sklyar et al
    eases adding code in C, C++ or Fortran to R. It takes care of the
    compilation, linking and loading of embedded code segments that are
    stored as R strings.
  - The [Rcpp](https://cran.r-project.org/package=Rcpp) package by Eddelbuettel and
    Francois offers a number of C++ classes that makes transferring R
    objects to C++ functions (and back) easier, and the
    [RInside](https://cran.r-project.org/package=RInside) package by the same
    authors allows easy embedding of R itself into C++ applications for
    faster and more direct data transfer.
  - The [RcppParallel](https://cran.r-project.org/package=RcppParallel) package by
    Allaire et al. bundles the [Intel Threading Building
    Blocks](https://www.threadingbuildingblocks.org) and
    [TinyThread](http://tinythreadpp.bitsnbites.eu) libraries. Together
    with [Rcpp](https://cran.r-project.org/package=Rcpp), RcppParallel makes it easy
    to write safe, performant, concurrently-executing C++ code, and use
    that code within R and R packages.
  - The [rJava](https://cran.r-project.org/package=rJava) package by Urbanek
    provides a low-level interface to Java similar to the `.Call()`
    interface for C and C++.
  - The [reticulate](https://cran.r-project.org/package=reticulate) package by
    Allaire provides interface to Python modules, classes, and
    functions. It allows R users to access many high-performance Python
    packages such as [tensorflow](https://cran.r-project.org/package=tensorflow) and
    [tfestimators](https://cran.r-project.org/package=tfestimators) within R.

**Profiling tools**

Packages [profvis](https://cran.r-project.org/package=profvis),
[proffer](https://cran.r-project.org/package=proffer),
[profmem](https://cran.r-project.org/package=profmem),
[GUIProfiler](https://cran.r-project.org/package=GUIProfiler),
[proftools](https://cran.r-project.org/package=proftools), and
[aprof](https://cran.r-project.org/package=aprof) summarize and visualize output
from the `Rprof` interface for profiling. The
[profile](https://cran.r-project.org/package=profile) package reads and writes
profiling data and converts among file formats such as
[`pprof`](https://github.com/google/pprof) by Google and `Rprof`. The
[`xrprof`](https://github.com/atheriel/xrprof) command-line tool
implements profile sampling for a given R process on Linux or Windows,
and it can profile R code alongside compiled code.

</div>

### CRAN packages:

  - [aprof](https://cran.r-project.org/package=aprof)
  - [batch](https://cran.r-project.org/package=batch)
  - [BatchExperiments](https://cran.r-project.org/package=BatchExperiments)
  - [BatchJobs](https://cran.r-project.org/package=BatchJobs)
  - [batchtools](https://cran.r-project.org/package=batchtools)
  - [bcp](https://cran.r-project.org/package=bcp)
  - [BDgraph](https://cran.r-project.org/package=BDgraph)
  - [biglm](https://cran.r-project.org/package=biglm)
  - [bigmemory](https://cran.r-project.org/package=bigmemory)
  - [bigstatsr](https://cran.r-project.org/package=bigstatsr)
  - [bnlearn](https://cran.r-project.org/package=bnlearn)
  - [caret](https://cran.r-project.org/package=caret)
  - [clustermq](https://cran.r-project.org/package=clustermq)
  - [data.table](https://cran.r-project.org/package=data.table)
  - [dclone](https://cran.r-project.org/package=dclone)
  - [disk.frame](https://cran.r-project.org/package=disk.frame)
  - [doFuture](https://cran.r-project.org/package=doFuture)
  - [doMC](https://cran.r-project.org/package=doMC)
  - [doMPI](https://cran.r-project.org/package=doMPI)
  - [doRNG](https://cran.r-project.org/package=doRNG)
  - [doSNOW](https://cran.r-project.org/package=doSNOW)
  - [dqrng](https://cran.r-project.org/package=dqrng)
  - [drake](https://cran.r-project.org/package=drake)
  - [ff](https://cran.r-project.org/package=ff)
  - [ffbase](https://cran.r-project.org/package=ffbase)
  - [flexiblas](https://cran.r-project.org/package=flexiblas)
  - [flowr](https://cran.r-project.org/package=flowr)
  - [foreach](https://cran.r-project.org/package=foreach)
  - [future](https://cran.r-project.org/package=future)
  - [gcbd](https://cran.r-project.org/package=gcbd)
  - [GUIProfiler](https://cran.r-project.org/package=GUIProfiler)
  - [h2o](https://cran.r-project.org/package=h2o)
  - [HadoopStreaming](https://cran.r-project.org/package=HadoopStreaming)
  - [HistogramTools](https://cran.r-project.org/package=HistogramTools)
  - [inline](https://cran.r-project.org/package=inline)
  - [keras](https://cran.r-project.org/package=keras)
  - [LaF](https://cran.r-project.org/package=LaF)
  - [latentnet](https://cran.r-project.org/package=latentnet)
  - [Matching](https://cran.r-project.org/package=Matching)
  - [MonetDB.R](https://cran.r-project.org/package=MonetDB.R)
  - [mvnfast](https://cran.r-project.org/package=mvnfast)
  - [OpenCL](https://cran.r-project.org/package=OpenCL)
  - [orloca](https://cran.r-project.org/package=orloca)
  - [parSim](https://cran.r-project.org/package=parSim)
  - [pbapply](https://cran.r-project.org/package=pbapply)
  - [pbdMPI](https://cran.r-project.org/package=pbdMPI)
  - [peperr](https://cran.r-project.org/package=peperr)
  - [pls](https://cran.r-project.org/package=pls)
  - [proffer](https://cran.r-project.org/package=proffer)
  - [profile](https://cran.r-project.org/package=profile)
  - [profmem](https://cran.r-project.org/package=profmem)
  - [profr](https://cran.r-project.org/package=profr)
  - [proftools](https://cran.r-project.org/package=proftools)
  - [profvis](https://cran.r-project.org/package=profvis)
  - [pvclust](https://cran.r-project.org/package=pvclust)
  - [qsub](https://cran.r-project.org/package=qsub)
  - [randomForestSRC](https://cran.r-project.org/package=randomForestSRC)
  - [Rborist](https://cran.r-project.org/package=Rborist)
  - [Rcpp](https://cran.r-project.org/package=Rcpp)
  - [RcppParallel](https://cran.r-project.org/package=RcppParallel)
  - [Rdsm](https://cran.r-project.org/package=Rdsm)
  - [reticulate](https://cran.r-project.org/package=reticulate)
  - [rgenoud](https://cran.r-project.org/package=rgenoud)
  - [RhpcBLASctl](https://cran.r-project.org/package=RhpcBLASctl)
  - [RInside](https://cran.r-project.org/package=RInside)
  - [rJava](https://cran.r-project.org/package=rJava)
  - [rlecuyer](https://cran.r-project.org/package=rlecuyer)
  - [Rmpi](https://cran.r-project.org/package=Rmpi) (core)
  - [RProtoBuf](https://cran.r-project.org/package=RProtoBuf)
  - [rredis](https://cran.r-project.org/package=rredis)
  - [rslurm](https://cran.r-project.org/package=rslurm)
  - [rstream](https://cran.r-project.org/package=rstream)
  - [Sim.DiffProc](https://cran.r-project.org/package=Sim.DiffProc)
  - [sitmo](https://cran.r-project.org/package=sitmo)
  - [snow](https://cran.r-project.org/package=snow) (core)
  - [snowfall](https://cran.r-project.org/package=snowfall)
  - [snowFT](https://cran.r-project.org/package=snowFT)
  - [speedglm](https://cran.r-project.org/package=speedglm)
  - [sqldf](https://cran.r-project.org/package=sqldf)
  - [ssgraph](https://cran.r-project.org/package=ssgraph)
  - [STAR](https://cran.r-project.org/package=STAR)
  - [targets](https://cran.r-project.org/package=targets)
  - [tensorflow](https://cran.r-project.org/package=tensorflow)
  - [tfestimators](https://cran.r-project.org/package=tfestimators)
  - [tm](https://cran.r-project.org/package=tm)
  - [varSelRF](https://cran.r-project.org/package=varSelRF)
  - [xgboost](https://cran.r-project.org/package=xgboost)

### Related links:

  - [HPC computing notes by Luke Tierney for HPC class at University of
    Iowa](http://www.stat.uiowa.edu/~luke/classes/295-hpc/)
  - [Mailing List: R Special Interest Group High Performance
    Computing](https://stat.ethz.ch/mailman/listinfo/r-sig-hpc/)
  - [Schmidberger, Morgan, Eddelbuettel, Yu, Tierney and Mansmann (2009)
    paper on 'State of the Art in Parallel Computing with
    R'](http://www.jstatsoft.org/v31/i01/)
  - [Luke Tierney's code directory for pnmath and
    pnmath0](http://www.stat.uiowa.edu/~luke/R/experimental/)
  - R-Forge Project:
    [<span class="Rforge">biocep-distrib</span>](https://R-Forge.R-project.org/projects/biocep-distrib/)
  - Bioconductor Package:
    [<span class="BioC">affyPara</span>](https://www.Bioconductor.ohttps://cran.r-project.org/package=release/bioc/html/affyPara.html)
  - Bioconductor Package:
    [<span class="BioC">maanova</span>](https://www.Bioconductor.ohttps://cran.r-project.org/package=release/bioc/html/maanova.html)
  - Bioconductor Package:
    [<span class="BioC">multtest</span>](https://www.Bioconductor.ohttps://cran.r-project.org/package=release/bioc/html/multtest.html)
  - Bioconductor Package:
    [<span class="BioC">puma</span>](https://www.Bioconductor.ohttps://cran.r-project.org/package=release/bioc/html/puma.html)
  - Google Code Project:
    [<span class="Gcode">romp</span>](https://code.google.com/archive/p/romp/)
  - Google Code Project:
    [<span class="Gcode">bugsparallel</span>](https://code.google.com/archive/p/bugsparallel/)
  - [Slurm open-source workload manager](http://slurm.schedmd.com/)
  - [Condor project at University of
    Wisconsin-Madison](http://www.cs.wisc.edu/condor/)
  - [Parallel Computing in R with
    sfCluster/snowfall](http://www.imbi.uni-freiburg.de/parallel/)
  - [Wikipedia: Message Passing Interface
    (MPI)](http://en.wikipedia.org/wiki/Message_Passing_Interface)
  - [Wikipedia: Parallel Virtual Machine
    (PVM)](http://en.wikipedia.org/wiki/Parallel_Virtual_Machine)
  - [Slides from Introduction to High-Performance Computing with R
    tutorial help in Nov 2009 at the Institute for Statistical
    Mathematics, Tokyo,
    Japan](http://dirk.eddelbuettel.com/papers/ismNov2009introHPCwithR.pdf)
  - [rgpu project at nbic.nl](https://gforge.nbic.nl/projects/rgpu/)
  - [Magma: Matrix Algebra on GPU and Multicore
    architectures](http://icl.cs.utk.edu/magma/)
  - [Parallel R: Data Analysis in the Distributed
    World](http://shop.oreilly.com/product/0636920021421.do)
  - [High Performance Statistical Computing for Data Intensive
    Research](https://snoweye.github.io/hpsc/)
  - [Rth: Parallel R through
    Thrust](http://heather.cs.ucdavis.edu/~matloff/rth.html)
  - [Programming with Big Data in R](http://r-pbd.org)
  - GitHub Project:
    [<span class="GitHub">RHIPE</span>](https://github.com/saptarshiguha/RHIPE/)
  - GitHub Project:
    [<span class="GitHub">beyond-single-core-R</span>](https://github.com/ljdursi/beyond-single-core-R/)
  - GitHub Project:
    [<span class="GitHub">pprof</span>](https://github.com/google/pprof/)
  - GitHub Project:
    [<span class="GitHub">xrprof</span>](https://github.com/atheriel/xrprof/)
  - [GitHub repository for this Task
    View](https://github.com/eddelbuettel/ctv-hpc)
