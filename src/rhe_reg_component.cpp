
#include "parameters.hpp"
#include "eigen_utils.hpp"
#include "mpi_utils.hpp"
#include "eigen_utils.hpp"
#include "rhe_reg_component.hpp"

#include "tools/eigen3.3/Dense"

#include <boost/iostreams/filtering_stream.hpp>

#include <random>


RHEreg_Component::RHEreg_Component(const parameters &myparams, Eigen::VectorXd &myY, Eigen::MatrixXd &myzz,
                                   Eigen::MatrixXd &myWzz, Eigen::MatrixXd &myC, Eigen::MatrixXd &myCtC_inv,
                                   const long &myNJacknifeLocal) : params(myparams), Y(myY),
	zz(myzz), Wzz(myWzz), C(myC), CtC_inv(myCtC_inv), n_jacknife_local(myNJacknifeLocal) {
	assert(n_jacknife_local > 0);
	n_covar = C.cols();
	n_samples = zz.rows();
	n_draws = zz.cols();

	n_env = 0;
	label = "";
	is_active = true;
	rm_jacknife_block = -1;

	ytXXty = 0;
	ytXXtys.resize(n_jacknife_local, 0);

	n_var_local = 0;
	n_vars_local.resize(n_jacknife_local, 0);

	_XXtzs.resize(n_jacknife_local);
	_XXtWzs.resize(n_jacknife_local);
	for(long ii = 0; ii < n_jacknife_local; ii++) {
		_XXtzs[ii] = Eigen::MatrixXd::Zero(n_samples, n_draws);
		_XXtWzs[ii] = Eigen::MatrixXd::Zero(n_samples, n_draws);
	}
}

void RHEreg_Component::set_eta(Eigen::Ref <Eigen::VectorXd> myeta) {
	assert(is_active);
	n_env = 1;
	eta = myeta;
	eta = myeta;
	Y.array() *= eta.array();
	zz.array().colwise() *= eta.array();
	Wzz.array().colwise() *= eta.array();
}

void RHEreg_Component::set_inactive() {
	// The inactive component corresponds to sigma_e
	// Ie the 'noise' component
	assert(n_env == 0);
	is_active = false;
	_XXtz = zz;
	_XXtWz = Wzz;
	n_var_local = 1;
	ytXXty = mpiUtils::mpiReduce_inplace(Y.squaredNorm());
//	_XXtWzs.clear();
//	_XXtzs.clear();
}

void RHEreg_Component::add_to_trace_estimator(Eigen::Ref <Eigen::MatrixXd> X, long jacknife_index) {
	assert(jacknife_index < n_jacknife_local);
	if(is_active) {
		Eigen::MatrixXd Xty = X.transpose() * Y;
		Xty = mpiUtils::mpiReduce_inplace(Xty);
		ytXXtys[jacknife_index] += Xty.squaredNorm();
		Eigen::MatrixXd Xtz = X.transpose() * zz;
		Xtz = mpiUtils::mpiReduce_inplace(Xtz);
		_XXtzs[jacknife_index] += X * Xtz;
		if(n_covar > 0) {
			Eigen::MatrixXd XtWz = X.transpose() * Wzz;
			XtWz = mpiUtils::mpiReduce_inplace(XtWz);
			_XXtWzs[jacknife_index] += X * XtWz;
		}
		n_vars_local[jacknife_index] += X.cols();
	}
}

void RHEreg_Component::finalise() {
	// Sum over the different jacknife blocks;
	if(is_active) {
		if(n_env > 0) {
			for (auto& mm : _XXtzs) {
				mm.array().colwise() *= eta.array();
			}
			for (auto& mm : _XXtWzs) {
				mm.array().colwise() *= eta.array();
			}
		}

		_XXtz = Eigen::MatrixXd::Zero(n_samples, n_draws);
		_XXtWz = Eigen::MatrixXd::Zero(n_samples, n_draws);
		for (auto& mm : _XXtzs) {
			_XXtz += mm;
		}
		for (auto& mm : _XXtWzs) {
			_XXtWz += mm;
		}

		n_var_local = std::accumulate(n_vars_local.begin(), n_vars_local.end(), 0.0);
		ytXXty = std::accumulate(ytXXtys.begin(), ytXXtys.end(), 0.0);
	}
}

Eigen::MatrixXd RHEreg_Component::getXXtz() const {
	if(rm_jacknife_block >= 0) {
		return (_XXtz - _XXtzs[rm_jacknife_block]);
	} else {
		return _XXtz;
	}
}

Eigen::MatrixXd RHEreg_Component::getXXtWz() const {
	if(rm_jacknife_block >= 0) {
		return (_XXtWz - _XXtWzs[rm_jacknife_block]);
	} else {
		return _XXtWz;
	}
}

double RHEreg_Component::get_bb_trace() const {
	if(rm_jacknife_block >= 0) {
		return (ytXXty - ytXXtys[rm_jacknife_block]) / get_n_var_local();
	} else {
		return ytXXty / get_n_var_local();
	}
}

double RHEreg_Component::get_n_var_local() const {
	if(rm_jacknife_block >= 0) {
		return n_var_local - n_vars_local[rm_jacknife_block];
	} else {
		return n_var_local;
	}
}

double RHEreg_Component::operator*(const RHEreg_Component &other) const {
	double res;
	if(n_covar == 0) {
		res = getXXtz().cwiseProduct(other.getXXtz()).sum();
	} else if (n_covar > 0) {
		if(label == "noise" || other.label == "noise") {
			res = getXXtz().cwiseProduct(other.getXXtWz()).sum();
		} else {
			Eigen::MatrixXd XXtz = getXXtz();
			Eigen::MatrixXd WXXtz = project_out_covars(XXtz);
			res = WXXtz.cwiseProduct(other.getXXtWz()).sum();
		}
	} else {
		throw std::runtime_error("Error in PVE_Component");
	}
	res = mpiUtils::mpiReduce_inplace(res);
	return res / get_n_var_local() / other.get_n_var_local() / (double) n_draws;
}

Eigen::MatrixXd RHEreg_Component::project_out_covars(Eigen::Ref <Eigen::MatrixXd> rhs) const {
	return EigenUtils::project_out_covars(rhs, C, CtC_inv, params.mode_debug);
}
