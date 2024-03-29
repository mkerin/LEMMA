# BGEN reference implementation

This repository contains a reference implementation of the [BGEN
format](http://www.bgenformat.org), written in C++. The library can be used as the basis for BGEN
support in other software, or as a reference for developers writing their own implementations of
the BGEN format.

### What's included?

This repository contains the library itself, a set of [example data files](/dir?name=example), and a number
of example programs (e.g. [bgen\_to\_vcf](/finfo?name=example/bgen_to_vcf.cpp)) that demonstrate the use of the
library API.

In addition, a number of utilities built using the library are also included in this repository:

* [bgenix](/doc/trunk/doc/wiki/bgenix.md) - a tool to index and efficiently retrieve subsets of a BGEN file. 
* [cat-bgen](/doc/trunk/doc/wiki/cat-bgen.md) - a tool to efficiently concatenate BGEN files.
* [edit-bgen](/doc/trunk/doc/wiki/edit-bgen.md) - a tool to edit BGEN file metadata.
* An R package called [rbgen](/doc/trunk/doc/wiki/rbgen.md) is also constructed in the build directory.
See the [rbgen wiki page](/doc/trunk/doc/wiki/rbgen.md) for more information on using this package.

### Citing BGEN

If you make use of the BGEN library, its tools or example programs, please cite:

Band, G. and Marchini, J., "*BGEN: a binary file format for imputed genotype and haplotype data*",
bioArxiv 308296; doi: <https://doi.org/10.1101/308296>

Thanks!

### License

This BGEN implementation is released under the Boost Software License v1.0. This is a relatively
permissive open-source license that is compatible with many other open-source licenses. See [this
page](http://www.boost.org/users/license.html) and the file
[LICENSE_1_0.txt](/artifact/12bafc460efc829b) for full details.

This repository also contains code from the [sqlite](www.sqlite.org), [boost](www.boost.org), and
[zstandard](http://www.zstd.net) libraries, which comes with their own respective licenses.
(Respectively, [public domain](http://www.sqlite.org/copyright.html), the boost software license,
and the [BSD license](https://github.com/facebook/zstd/blob/dev/LICENSE)). These libraries are not
used in the core BGEN implementation, but are used in the applications, example programs, and
`rbgen` R package.

### Note on UK Biobank data

A particularly important dataset released in BGEN is the imputed genotype data released by the UK
Biobank. See [the relevant wiki page](/wiki/?name=BGEN+in+the+UK+Biobank) for details.

---

# Obtaining and installing BGEN

### In brief

The following commands (typed into a UNIX shell - the dollar symbol indicates the prompt, and shouldn't be typed in)
should perform a basic download and install of the BGEN library, example data and tools:

```bash
# get it
wget http://code.enkre.net/bgen/tarball/release/bgen.tgz
cd bgen
# compile it
./waf configure
./waf
# test it
./build/test/unit/test_bgen
./build/apps/bgenix -g example/example.16bits.bgen -list
```

The following sections contains more information on this process.

### Download

A tarball of the latest release branch is available here: <http://code.enkre.net/bgen/tarball/release>

Alternatively, use [fossil](https://fossil-scm.org) to download the master branch as follows:

```
mkdir bgen
cd bgen
fossil clone https://code.enkre.net/bgen bgen.fossil
fossil open bgen.fossil release
```

(This command can take a while.)

Additionally, pre-built version of the bgen utilities may be available from [this
page](http://www.well.ox.ac.uk/~gav/resources/). **Note**: the recommended use is to download and
compile bgenix for your platform; these binaries are provided for convenience in getting started
quickly.

### Compilation

To compile the code, use the supplied waf build tool:
```
./waf configure
./waf
```
Results will appear under the `build/` directory.  

Note: a full build requires a compiler that supports C++11, e.g. gcc v4.7 or above.  To specify the compiler used, set the `CXX` environment variable during the configure step.  For example (if your shell is `bash`):
```
CXX=/path/to/g++ ./waf configure
./waf
```

The sqlite and zstd libraries are written in C; to specify the C compiler you can additionally add
`CC=/path/to/gcc`. We have tested compilation on gcc 4.9.3 and 5.4.0, and using clang, among others.

If you don't have access to a compiler with C++11 support, you can still build the core bgen
implementation, but won't be able to build the applications or example programs. See [the
wiki](/wiki?name=Troubleshooting_compilation) for more information.

### Testing

BGEN's tests can be run by typing 

```
./build/test/test_bgen
```

or, for more recent versions:

```
./build/test/unit/test_bgen
```

If all goes well a message like `All tests passed` should be printed.

If you have [Robot Test Framework](http://robotframework.org/) installed, you can instead run the
full suite of unit and functional tests like so:

```
./test/functional/run_tests.sh
```

Test results will be placed in the directory `build/test/functional/test-reports`.

### Trying an example

The example program `bgen_to_vcf` reads a bgen file (v1.1 or v1.2) and outputs it as a VCF file to stdout.  You can try running it
by typing

```
./build/example/bgen_to_vcf example/example.8bits.bgen
```

which should output vcf-formatted data to stdout.  We've provided further example bgen files in the `example/` subdirectory.

### Installation

The command

```
./waf install
```

will install the applications listed above into a specified system or user directory.  By default this is `/usr/local`.  To change it, specify the prefix at the configure step:

```
./waf configure --prefix=/path/to/installation/directory
./waf install
```

The programs listed above will be installed into a folder called `bin/` under the prefix dir, e.g.
`bgenix` will be installed as `/path/to/installation/directory/bin/bgenix` etc.

Note that in many cases there's no need for installation; the executables are self-contained. The
install step simply copies them into the destination directory.

(The installation prefix need not be a system-wide directory. For example, I typically specify an
installation directory within my home dir, e.g. `~gav/projects/software/`.

### Branches

This repo follows the branch naming practice in which `release` represents the most up-to-date code
considered in a 'releasable' state. If you are interested in using bgen code in your own project,
we therefore recommend cloning the `release` branch. Code development takes place in the `trunk`
branch and/or in feature branches branched from the `trunk` branch. The command given above
downloads the release branch, which is what most people will want.

### More information

See the [source code](/dir?ci=release), 
BGEN [releases](/wiki?name=Releases),
or the [Wiki](/wiki?name=Home) for more information.
