//
// Created by kerin on 12/11/2018.
//

#ifndef BGEN_WRAPPER_HPP
#define BGEN_WRAPPER_HPP

#include <cstddef>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <string>
#include <vector>

#include "typedefs.hpp"

namespace bgenWrapper
{
	class ViewImpl;
	class IndexQueryImpl;

	class IndexQuery
	{
	public:
		IndexQuery(std::string const& filename);
		~IndexQuery();
		void include_range(std::string range_chr, std::uint32_t start, std::uint32_t end);
		void include_rsids(std::vector<std::string> rsid_list);
		void initialise();
		friend class View;
		
	private:
		std::unique_ptr<IndexQueryImpl> m_pImpl;
		const IndexQueryImpl *Pimpl() const { return m_pImpl.get(); }
		IndexQueryImpl *Pimpl() { return m_pImpl.get(); }
	};

	class View
	{
	public:
		View();
		View(std::string const& filename);
		~View();
		std::size_t number_of_samples() const;
		std::uint32_t number_of_variants() const;
		std::ostream& summarise( std::ostream& o ) const ;
		std::vector<std::string> get_sample_ids() const ;

		void set_query(IndexQuery& query);

		struct SnpStats {
			double maf,info,mu,missingness,sigma,d1;
		};

		SnpStats read_genotype_data_block(const std::unordered_map<long, bool> &sample_is_invalid,
                                const long &n_samples,EigenRefDataVector m_dosage) const;
		void ignore_genotype_data_block() const;
		bool read_variant(
				std::string* SNPID,
				std::string* rsid,
				std::string* chromosome,
				uint32_t* position,
				std::vector< std::string >* alleles
			) const;

		// movable:
		View(View && rhs) noexcept;   
		View& operator=(View && rhs) noexcept;

		// and copyable
		View(const View& rhs);
		View& operator=(const View& rhs);

	private:
		std::unique_ptr<ViewImpl> m_pImpl;

		const ViewImpl *Pimpl() const { return m_pImpl.get(); }
		ViewImpl *Pimpl() { return m_pImpl.get(); }
	};

	void check_file_exists(std::string filename, std::string extension);
}

#endif // BGEN_WRAPPER_HPP