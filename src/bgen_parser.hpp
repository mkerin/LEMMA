// File of BgenParser class for use with src/bgen_prog.cpp
#ifndef BGEN_PARSER_HPP
#define BGEN_PARSER_HPP

#include <boost/algorithm/string/predicate.hpp>
#include <boost/lexical_cast.hpp>
#include <iostream>
#include "genfile/bgen/bgen.hpp"
#include "typedefs.hpp"

#include <fstream>
#include <cassert>
#include <stdexcept>
#include <memory>

// ProbSetter is a callback object appropriate for passing to bgen::read_genotype_data_block() or
// the synonymous method of genfile::bgen::View. See the comment in bgen.hpp above
// bgen::read_genotype_data_block(), or the bgen wiki for a description of the API.
// The purpose of this object is to store genotype probability values in the desired
// data structure (which here is a vector of vectors of doubles).
struct ProbSetter {
	typedef std::vector< std::vector< double > > Data ;
	ProbSetter( Data* result ):
		m_result( result ),
		m_sample_i(0)
	{}
		
	// Called once allowing us to set storage.
	void initialise( std::size_t number_of_samples, std::size_t number_of_alleles ) {
		m_result->clear() ;
		m_result->resize( number_of_samples ) ;
	}
	
	// If present with this signature, called once after initialise()
	// to set the minimum and maximum ploidy and numbers of probabilities among samples in the data.
	// This enables us to set up storage for the data ahead of time.
	void set_min_max_ploidy( uint32_t min_ploidy, uint32_t max_ploidy, uint32_t min_entries, uint32_t max_entries ) {
		for( std::size_t i = 0; i < m_result->size(); ++i ) {
			m_result->at( i ).reserve( max_entries ) ;
		}
	}
	
	// Called once per sample to determine whether we want data for this sample
	bool set_sample( std::size_t i ) {
		m_sample_i = i ;
		// Yes, here we want info for all samples.
		return true ;
	}
	
	// Called once per sample to set the number of probabilities that are present.
	void set_number_of_entries(
		std::size_t ploidy,
		std::size_t number_of_entries,
		genfile::OrderType order_type,
		genfile::ValueType value_type
	) {
		assert( value_type == genfile::eProbability ) ;
		m_result->at( m_sample_i ).resize( number_of_entries ) ;
		m_entry_i = 0 ;
	}

	// Called once for each genotype (or haplotype) probability per sample.
	void set_value( uint32_t, double value ) {
		m_result->at( m_sample_i ).at( m_entry_i++ ) = value ;
	}

	// Ditto, but called if data is missing for this sample.
	void set_value( uint32_t, genfile::MissingValue value ) {
		// Here we encode missing probabilities with -1
		m_result->at( m_sample_i ).at( m_entry_i++ ) = -1 ;
	}

	// If present with this signature, called once after all data has been set.
	void finalise() {
		// nothing to do in this implementation.
	}

private:
	Data* m_result ;
	std::size_t m_sample_i ;
	std::size_t m_entry_i ;
} ;

// ProbSetter_v2 is a callback object appropriate for passing to bgen::read_genotype_data_block() or
// the synonymous method of genfile::bgen::View. See the comment in bgen.hpp above
// bgen::read_genotype_data_block(), or the bgen wiki for a description of the API.
// The purpose of this object is to store genotype probability values in the desired
// data structure (which here is a vector of vectors of doubles).
struct ProbSetter_v2 {
	typedef EigenDataVector Data;

	ProbSetter_v2(std::set<long> invalid_sample_ids):
			m_invalid_sample_indexes(invalid_sample_ids)
	{}

	// Called once allowing us to set storage.
	void initialise( std::size_t number_of_samples, std::size_t number_of_alleles ) {
		m_dosage.resize(number_of_samples - m_invalid_sample_indexes.size());
		m_samples_skipped = 0;
		m_missing_entries.clear();

		m_sum_eij = 0;
		m_sum_eij2 = 0;
		m_sum_fij_minus_eij2 = 0;
		m_fij = 0;
		m_eij = 0;
	}

	// If present with this signature, called once after initialise()
	// to set the minimum and maximum ploidy and numbers of probabilities among samples in the data.
	// This enables us to set up storage for the data ahead of time.
	void set_min_max_ploidy( uint32_t min_ploidy, uint32_t max_ploidy, uint32_t min_entries, uint32_t max_entries ) {
		// pass
	}

	// Called once per sample to determine whether we want data for this sample
	bool set_sample( std::size_t i ) {
		// Only want data from samples with complete data across pheno/covar/env
		if (m_invalid_sample_indexes.find(i) != m_invalid_sample_indexes.end()){
			m_samples_skipped++;
			return false;
		} else {
			m_sample_i = i - m_samples_skipped;
			return true;
		}
	}

	// Called once per sample to set the number of probabilities that are present.
	void set_number_of_entries(
			std::size_t ploidy,
			std::size_t number_of_entries,
			genfile::OrderType order_type,
			genfile::ValueType value_type
	) {
		assert( value_type == genfile::eProbability );
		assert(number_of_entries == 3);
	}

	// Called once for each genotype (or haplotype) probability per sample.
	void set_value(const uint32_t& gg, const double& value ) {
		if(gg == 0){
			m_dosage(m_sample_i) = 0;
		} else {
			m_dosage(m_sample_i) += gg * value;

			m_eij += gg * value;
			m_fij += gg * gg * value;
			if(gg == 2){
				m_sum_fij_minus_eij2 += m_fij - m_eij * m_eij;
				m_sum_eij += m_eij;
				m_sum_eij2 += m_eij * m_eij;

				m_fij = 0;
				m_eij = 0;
			}	
		}
	}

	// Ditto, but called if data is missing for this sample.
	void set_value(const uint32_t&, const genfile::MissingValue& value){
		m_missing_entries.push_back(m_sample_i);
	}

	// If present with this signature, called once after all data has been set.
	void finalise() {
		double Nvalid = m_dosage.rows() - m_missing_entries.size();
		m_missingness = (double) m_missing_entries.size() / (double) m_dosage.rows();

		m_maf = 0.5 * m_sum_eij / Nvalid;
		m_info = 1.0;
		if(m_maf > 1e-10 && m_maf < 0.9999999999) {
			m_info -= m_sum_fij_minus_eij2 / (2.0 * Nvalid * m_maf * (1.0 - m_maf));
		}

		m_mean = m_sum_eij / Nvalid;
		m_sigma2 = m_sum_eij2 - m_mean * m_mean * Nvalid;
		m_sigma2 /= (Nvalid - 1);

		// Fill in missing values with mean
		for (auto ii : m_missing_entries){
			m_dosage(ii) = m_mean;
		}
		assert(m_samples_skipped == m_invalid_sample_indexes.size());
	}

	Data m_dosage;
	double m_missingness;
	double m_maf;
	double m_info;
	double m_sigma2;
	double m_mean;
	double m_sum_eij;

private:
	std::set<long> m_invalid_sample_indexes;
	std::size_t m_samples_skipped;
	std::size_t m_sample_i;

	std::vector<long> m_missing_entries;
	double m_sum_eij2;
	double m_sum_fij_minus_eij2;
	double m_eij;
	double m_fij;
};

// BgenParser is a thin wrapper around the core functions in genfile/bgen/bgen.hpp.
// This class tracks file state and handles passing the right callbacks.
struct BgenParser {
	
	BgenParser( std::string const& filename ):
		m_filename( filename ),
		m_state( e_NotOpen ),
		m_have_sample_ids( false )
	{
		// Open the stream
		m_stream.reset(
			new std::ifstream( filename, std::ifstream::binary )
		) ;
		if( !*m_stream ) {
			throw std::invalid_argument( filename ) ;
		}
		m_state = e_Open ;

		// Read the offset, header, and sample IDs if present.
		genfile::bgen::read_offset( *m_stream, &m_offset ) ;
		genfile::bgen::read_header_block( *m_stream, &m_context ) ;
		if( m_context.flags & genfile::bgen::e_SampleIdentifiers ) {
			genfile::bgen::read_sample_identifier_block(
				*m_stream, m_context,
				[this]( std::string id ) { m_sample_ids.push_back( id ) ; }
			) ;
			m_have_sample_ids = true ;
		}
		
		// Jump to the first variant data block.
		m_stream->seekg( m_offset + 4 ) ;

		// We keep track of state (though it's not really needed for this implementation.)
		m_state = e_ReadyForVariant ;
	}

	std::ostream& summarise( std::ostream& o ) const {
		o << "BgenParser: bgen file ("
			<< ( m_context.flags & genfile::bgen::e_Layout2 ? "v1.2 layout" : "v1.1 layout" )
			<< ", "
			<< ( m_context.flags & genfile::bgen::e_CompressedSNPBlocks ? "compressed" : "uncompressed" ) << ")"
			<< " with " 
			<< m_context.number_of_samples << " " << ( m_have_sample_ids ? "named" : "anonymous" ) << " samples and "
			<< m_context.number_of_variants << " variants.\n" ;
		return o ;
	}

	std::size_t number_of_samples() const {
		return m_context.number_of_samples ;
	}

	// Report the sample IDs in the file using the given setter object
	// (If there are no sample IDs in the file, we report a dummy identifier).
	template< typename Setter >
	void get_sample_ids( Setter setter ) {
		if( m_have_sample_ids ) {
			for( std::size_t i = 0; i < m_context.number_of_samples; ++i ) {
				setter( m_sample_ids[i] ) ;
			}
		} else {
			for( std::size_t i = 0; i < m_context.number_of_samples; ++i ) {
				setter( "(unknown_sample_" + std::to_string( i+1 ) + ")" ) ;
			}
		}
	}

	// Attempt to read identifying information about a variant from the bgen file, returning
	// it in the given fields.
	// If this method returns true, data was successfully read, and it should be safe to call read_probs()
	// or ignore_probs().
	// If this method returns false, data was not successfully read indicating the end of the file.
	bool read_variant(
		std::string* chromosome,
		uint32_t* position,
		std::string* rsid,
		std::vector< std::string >* alleles
	) {
		assert( m_state == e_ReadyForVariant ) ;
		std::string SNPID ; // read but ignored in this toy implementation
		
		if(
			genfile::bgen::read_snp_identifying_data(
				*m_stream, m_context,
				&SNPID, rsid, chromosome, position,
				[&alleles]( std::size_t n ) { alleles->resize( n ) ; },
				[&alleles]( std::size_t i, std::string const& allele ) { alleles->at(i) = allele ; }
			)
		) {
			m_state = e_ReadyForProbs ;
			return true ;
		} else {
			return false ;
		}
	}
	
	// Read genotype probability data for the SNP just read using read_variant()
	// After calling this method it should be safe to call read_variant() to fetch
	// the next variant from the file.
	void read_probs( std::vector< std::vector< double > >* probs ) {
		assert( m_state == e_ReadyForProbs ) ;
		ProbSetter setter( probs ) ;
		genfile::bgen::read_and_parse_genotype_data_block< ProbSetter >(
			*m_stream,
			m_context,
			setter,
			&m_buffer1,
			&m_buffer2
		) ;
		m_state = e_ReadyForVariant ;
	}

	// Ignore genotype probability data for the SNP just read using read_variant()
	// After calling this method it should be safe to call read_variant()
	// to fetch the next variant from the file.
	void ignore_probs() {
		genfile::bgen::ignore_genotype_data_block( *m_stream, m_context ) ;
		m_state = e_ReadyForVariant ;
	}

private:
	std::string const m_filename ;
	std::unique_ptr< std::istream > m_stream ;

	// bgen::Context object holds information from the header block,
	// including bgen flags
	genfile::bgen::Context m_context ;

	// offset byte from top of bgen file.
	uint32_t m_offset ;

	// We keep track of our state in the file.
	// Not strictly necessary for this implentation but makes it clear that
	// calls must be read_variant() followed by read_probs() (or ignore_probs())
	// repeatedly.
	enum State { e_NotOpen = 0, e_Open = 1, e_ReadyForVariant = 2, e_ReadyForProbs = 3, eComplete = 4 } ;
	State m_state ;

	// If the BGEN file contains samples ids, they will be read here.
	bool m_have_sample_ids ;
	std::vector< std::string > m_sample_ids ;
	
	// Buffers, these are used as working space by bgen implementation.
	std::vector< genfile::byte_t > m_buffer1, m_buffer2 ;
} ;

#endif
