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
#include <set>
#include <unordered_map>
#include <unordered_set>

// DosageSetter is a callback object appropriate for passing to bgen::read_genotype_data_block() or
// the synonymous method of genfile::bgen::View. See the comment in bgen.hpp above
// bgen::read_genotype_data_block(), or the bgen wiki for a description of the API.
// The purpose of this object is to store genotype probability values in the desired
// data structure (which here is a vector of vectors of doubles).
struct DosageSetter {
	typedef EigenDataVector Data;

	DosageSetter(std::unordered_map<long, bool> invalid_sample_ids, long nInvalid) :
		m_sample_is_invalid(invalid_sample_ids), m_nInvalid(nInvalid)
	{
	}

	// Called once allowing us to set storage.
	void initialise( std::size_t number_of_samples, std::size_t number_of_alleles ) {
		m_dosage.resize(number_of_samples - m_nInvalid);
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
		if (m_sample_is_invalid[i]){
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
		if(gg == 0) {
			m_dosage(m_sample_i) = 0;
		} else {
			m_dosage(m_sample_i) += gg * value;

			m_eij += gg * value;
			m_fij += gg * gg * value;
			if(gg == 2) {
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
		m_missing_entries.insert(m_sample_i);
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
		for (const auto& ii : m_missing_entries) {
			m_dosage(ii) = m_mean;
		}
		assert(m_samples_skipped == m_nInvalid);
	}

	Data m_dosage;
	double m_missingness;
	double m_maf;
	double m_info;
	double m_sigma2;
	double m_mean;
	double m_sum_eij;

private:
	std::unordered_map<long, bool> m_sample_is_invalid;
	std::size_t m_samples_skipped;
	std::size_t m_sample_i;
	long m_nInvalid;

	std::unordered_set<long> m_missing_entries;
	double m_sum_eij2;
	double m_sum_fij_minus_eij2;
	double m_eij;
	double m_fij;
};

#endif
