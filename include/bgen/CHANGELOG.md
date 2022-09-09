History
====

27 June 2020
----
- Transition repository to code.enkre.net/bgen

13 March 2020
----
v1.1.6 release.  Changes are:

- use PRAGMA synchronous=OFF when connecting to the index db.  This speeds up indexing
operations (`bgenix -index`) substantially on some network filesystems.

15 January 2020
----
v1.1.5 release.  Changes are:

- incorporate fix from Maarten Kooyman to build using Python 3.
- fix issue #39 <https://bitbucket.org/gavinband/bgen/issues/39/rbgen-segfault-when-samples-are-given-in>

7 August 2018
----
v1.1.4 release.  Update to fix sample subset issue with BGEN v1.1.

2 May 2015
-----
v1.1.3 release.  The main changes are:

- The rbgen R package, which gets data from indexed BGEN files into R, is has several improvements - it's easier to install, and has additional features (see below).
- New, improved bgenix vcf output - now up to 50X faster.
- Further performance improvements and resolution of a number of issues across the library.

To accompany this we have written a paper which is now available on bioArxiv: https://doi.org/10.1101/308296. 

13 July 2017
----
v1.0 release

7 July 2016
----

* Updates to bgenix to handle UK biobank interim files and to avoid extra index tables in the index file.

21 March 2016
----

* BGEN spec and implementation updated to alter probability order for unphased data when the number of alleles (K) or the ploidy is greater than two.
This order now better matches the order of VCF GP fields and as a simple enumeration scheme.

10 Nov 2015
----

Major changes in revision ff11254f9505:

1. I've implemented two new tools
    - cat-bgen, which can be used to concatenate BGEN files.
    - bgenix, which can be used to index BGEN files and efficiently retrieve specified data.

2. For this purpose I've imported several extra pieces of code
    - appcontext/ and db/ sublibs from qctool
    - sqlite3 3.9.2
    - boost 1.55.0

Note: these changes were erroneously applied first to the master branch (they were intended for default first).

6 Nov 2015
----
Major changes in revision 392429affc42:

1. I’ve changed the behaviour of BGEN v1.2 with respect to samples with missing data: they are now stored with dummy zero probabilities.  The spec is now in 'beta' which means I don’t have any other planned changes to make; unless major issues are uncovered this will be the final version of the format.

2. I’ve revamped the setter api of parse_probability_data somewhat.  It is documented in the code and here [on the wiki](https://bitbucket.org/gavinband/bgen/wiki/The_Setter_API).  The main breaking changes are:
- Renamed operator() to set_value(), and given it an index argument; I think these make the API more consistent.
- Added an initial ploidy argument to set_number_of_entries() as requested.  (The type of data - phased or unphased - is already reported in the order_type argument so I don’t think another argument is needed).
- Added two new method calls, which are optional: set_min_max_ploidy() (useful for setting storage) and finalise().  See the docs for info.

3. I’ve also got rid of the max_id_size option to write_snp_identifying_data().  (This is now not needed because writing BGEN v1.0 files is no longer supported.)

4. I’ve also added some test code (using the [catch framework](https://github.com/philsquared/catch), which seems pretty good).  Tests are not exhaustive but hopefully a start.
:q
5. I've removed some code warnings - thanks to Robert V. Baron of [Mega2](https://watson.hgen.pitt.edu/docs/mega2_html/mega2.html) for testing this code.

23 Sep 2015
----
First version, based on qctool implementation.
