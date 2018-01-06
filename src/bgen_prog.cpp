// Adapted from example/bgen_to_vcf.cpp by Gavin Band
// From project at http://bitbucket.org/gavinband/bgen/get/master.tar.gz

#include <boost/algorithm/string/predicate.hpp>
#include <boost/lexical_cast.hpp>
#include <iostream>
#include <fstream>
#include <cassert>
#include <cstdlib>
#include <stdexcept>
#include <memory>
#include "parse_arguments.hpp"
#include "class.h"
#include "data.hpp"
#include "genfile/bgen/bgen.hpp"
#include "bgen_parser.hpp"
#include "version.h"

// TODO: implement maf, info filters
// TODO: read in covars
// TODO: regress out covars
// TODO: copy argument_sanity()

// This example program reads data from a bgen file specified as the first argument
// and outputs it as a VCF file.
int main( int argc, char** argv ) {
	parameters p;

	try {
		parse_arguments(p, argc, argv);
		data Data( p.bgen_file );
		Data.params = p;
		Data.output_init();
		Data.n_samples = Data.bgenParser.number_of_samples();

		// Output header
		Data.outf << "##fileformat=VCFv4.2\n"
			<< "FORMAT=<ID=GP,Type=Float,Number=G,Description=\"Genotype call probabilities\">\n"
			<< "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT" ;
		Data.bgenParser.get_sample_ids(
			[&Data]( std::string const& id ) { Data.outf << "\t" << id ; }
		) ;
		Data.outf << "\n" ;

		// Reading in from bgen file
		while (Data.read_bgen_chunk()) {
			std::cout << "Top of the bgen chunk while-loop" << std::endl;
			for (int jj = 0; jj < Data.G_ncol; jj++ ) {
				Data.outf << Data.chromosome[jj] << '\t'
					<< Data.position[jj] << '\t'
					<< Data.rsid[jj] << '\t' ;
				Data.outf << Data.alleles[jj][0] << '\t' ;
				for( std::size_t i = 1; i < Data.alleles[jj].size(); ++i ) {
					Data.outf << ( i > 1 ? "," : "" ) << Data.alleles[jj][i] ;
				}
				Data.outf << "\t.\t.\t.\tGP" ;

				for (int ii = 0; ii < Data.n_samples; ii++ ) {
					Data.outf << '\t' ;
					Data.outf << Data.G(ii,jj);
				}
				Data.outf << "\n" ;
			}
		}
		return 0 ;
	}
	catch( genfile::bgen::BGenError const& e ) {
		std::cerr << "!! Uh-oh, error parsing bgen file.\n" ;
		return -1 ;
	}
}
