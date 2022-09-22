

#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include <boost/filesystem.hpp>

#include "genfile/bgen/bgen.hpp"
#include "genfile/bgen/View.hpp"

#include "bgen_wrapper.hpp"
#include "bgen_parser.hpp"
#include "typedefs.hpp"

namespace fs = boost::filesystem;

namespace bgenWrapper
{
    class IndexQueryImpl
    {
    public:
        IndexQueryImpl(std::string const& filename) {
            check_file_exists(filename,".bgi");
            query = genfile::bgen::IndexQuery::create(filename);
        }
        genfile::bgen::IndexQuery::UniquePtr query;
        ~IndexQueryImpl() = default;
    };

    IndexQuery::IndexQuery(std::string const& filename) : m_pImpl(new IndexQueryImpl(filename)) {
    }

    IndexQuery::~IndexQuery() = default;

    void IndexQuery::include_range(std::string range_chr, std::uint32_t start, std::uint32_t end){
        genfile::bgen::IndexQuery::GenomicRange rr1(range_chr, start, end);
        m_pImpl->query->include_range( rr1 );
    }

    void IndexQuery::include_rsids(std::vector<std::string> rsid_list) {
        m_pImpl->query->include_rsids( rsid_list );
    }

    void IndexQuery::initialise() {
        m_pImpl->query->initialise();
    }

    class ViewImpl
    {
    public:
        genfile::bgen::View::UniquePtr bgenView;

        ViewImpl() = default;
        ViewImpl(std::string const& filename){
            check_file_exists(filename,".bgen");
            bgenView = genfile::bgen::View::create(filename);
        }
        ~ViewImpl() = default;
    };

    View::View() : m_pImpl(new ViewImpl()) {
    }

    View::~View() = default;


    /* Wrappers around BGEN-Lib methods */
    View::View(std::string const& filename) : m_pImpl(new ViewImpl(filename)) {
    }

    std::size_t View::number_of_samples() const {
        return m_pImpl->bgenView->number_of_samples();
    }

    std::uint32_t View::number_of_variants() const {
        return m_pImpl->bgenView->number_of_variants();
    }

	void View::set_query(IndexQuery& query) {
        m_pImpl->bgenView->set_query(query.m_pImpl->query);
    }
	
    std::ostream& View::summarise( std::ostream& o ) const {
        return m_pImpl->bgenView->summarise(o);
    }

    void View::ignore_genotype_data_block() const {
        m_pImpl->bgenView->ignore_genotype_data_block();
    }

    View::SnpStats View::read_genotype_data_block(const std::unordered_map<long, bool> &sample_is_invalid,
                                const long &n_samples,
                                EigenRefDataVector m_dosage) const {
// MK TODO: use a struct here to output named summary statistics
// MK TODO: DosageSetter can now live entirely in here
        long nInvalid = sample_is_invalid.size() - n_samples;
	    DosageSetter setter(sample_is_invalid, nInvalid);
        m_pImpl->bgenView->read_genotype_data_block(setter);
        SnpStats stats;
        stats.d1   = setter.m_sum_eij;
		stats.maf  = setter.m_maf;
		stats.info = setter.m_info;
		stats.mu   = setter.m_mean;
		stats.missingness    = setter.m_missingness;
		stats.sigma = std::sqrt(setter.m_sigma2);
        m_dosage = setter.m_dosage;
        return stats;
    }

    bool View::read_variant(
				std::string* SNPID,
				std::string* rsid,
				std::string* chromosome,
				uint32_t* position,
				std::vector< std::string >* alleles
			) const {
        return m_pImpl->bgenView->read_variant(SNPID,rsid,chromosome,position,alleles);
    }

    std::vector<std::string> View::get_sample_ids() const {
        std::vector<std::string> sample_ids;
        m_pImpl->bgenView->get_sample_ids(
            [&]( std::string const& id ) {
				sample_ids.push_back(id);
        });
        return sample_ids;
    }

    // moveable
    View::View(View &&) noexcept = default;
    View& View::operator=(View &&) noexcept = default;

    // copyable
    View::View(const View& rhs)
        : m_pImpl(new ViewImpl(*rhs.m_pImpl))
    {}

    View& View::operator=(const View& rhs) {
        if (this != &rhs) 
            m_pImpl.reset(new ViewImpl(*rhs.m_pImpl));

        return *this;
    }

    void check_file_exists(std::string filename, std::string extension){
        fs::path path(filename.begin(),filename.end());
        if (!fs::exists(path)){
            std::cout << "Throwing error: File '"+filename+"' does not exist" << std::endl;
            throw std::invalid_argument("File '"+filename+"' does not exist");
        }
        if (path.extension() != extension){
            std::cout << "Throwing error: Expecting file '"+filename+"' to have extension '"+extension+"'" << std::endl;
            throw std::invalid_argument("Expecting file '"+filename+"' to have extension '"+extension+"'");
        }
    }
}
