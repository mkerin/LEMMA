
#include "parameters.hpp"
#include "eigen_utils.hpp"
#include "mpi_utils.hpp"
#include "eigen_utils.hpp"
#include "rhe_reg_component.hpp"

#include "tools/eigen3.3/Dense"

#include <boost/iostreams/filtering_stream.hpp>

#include <random>

RHEreg_Component::RHEreg_Component(const parameters &myparams, const Eigen::VectorXd &myY,
                                   const Eigen::MatrixXd &myWzz, const Eigen::MatrixXd &myC, const Eigen::MatrixXd &myCtC_inv,
                                   const long &myNJacknifeLocal) : params(myparams), Y(myY),
	zz(myWzz), C(myC), CtC_inv(myCtC_inv), n_jacknife_local(myNJacknifeLocal) {
	assert(n_jacknife_local > 0);
	n_covar = C.cols();
	n_samples = zz.rows();
	n_draws = zz.cols();

	label = "";
	is_gxe = false;
	is_active = true;
	is_finalised = false;
	rm_jacknife_block = -1;

	ytXXty = 0;
	ytXXtys.resize(n_jacknife_local, 0);

	n_var_local = 0;
	n_vars_local.resize(n_jacknife_local, 0);

	_XXtzs.resize(n_jacknife_local);
	for(long ii = 0; ii < n_jacknife_local; ii++) {
		_XXtzs[ii] = Eigen::MatrixXd::Zero(n_samples, n_draws);
	}
}

void RHEreg_Component::set_env_var(const Eigen::Ref<const Eigen::VectorXd>& my_env_var) {
	assert(is_active);
	is_gxe = true;
	env_var = my_env_var;
	Y.array() *= env_var.array();
	if(zz.rows() > 0) {
		zz.array().colwise() *= env_var.array();
	}
}

void RHEreg_Component::change_env_var(const Eigen::Ref<const Eigen::VectorXd>& new_env_var) {
	assert(is_active);
	assert(is_gxe);
	Y.array() /= env_var.array();
	Y.array() *= new_env_var.array();
	if(zz.rows() > 0) {
		zz.array().colwise() /= env_var.array();
		zz.array().colwise() *= new_env_var.array();
	}
	env_var = new_env_var;
}

void RHEreg_Component::set_inactive() {
	// The inactive component corresponds to sigma_e
	// Ie the 'noise' component
	assert(!is_gxe);
	is_active = false;
	_XXtWz = zz;
	n_var_local = 1;
	ytXXty = mpiUtils::mpiReduce_inplace(Y.squaredNorm());
}

void RHEreg_Component::add_to_trace_estimator(Eigen::Ref <Eigen::MatrixXd> X, long jacknife_index) {
	assert(jacknife_index < n_jacknife_local);
	if(is_active) {
		Eigen::MatrixXd Xty = X.transpose() * Y;
		Xty = mpiUtils::mpiReduce_inplace(Xty);
		ytXXtys[jacknife_index] += Xty.squaredNorm();
		if(n_covar > 0) {
			Eigen::MatrixXd XtWz = X.transpose() * zz;
			XtWz = mpiUtils::mpiReduce_inplace(XtWz);
			_XXtzs[jacknife_index] += X * XtWz;
		}
		n_vars_local[jacknife_index] += X.cols();
	}
}

void RHEreg_Component::finalise() {
	// Sum over the different jacknife blocks;
	if(is_active) {
		_XXtWz = Eigen::MatrixXd::Zero(n_samples, n_draws);
		for (auto& mm : _XXtzs) {
			_XXtWz += mm;
		}

		n_var_local = std::accumulate(n_vars_local.begin(), n_vars_local.end(), 0.0);
		ytXXty = std::accumulate(ytXXtys.begin(), ytXXtys.end(), 0.0);
		is_finalised = true;
	}
}

Eigen::MatrixXd RHEreg_Component::getXXtz() const {
	Eigen::MatrixXd res;
	if(rm_jacknife_block >= 0) {
		res = _XXtWz - _XXtzs[rm_jacknife_block];
	} else {
		res = _XXtWz;
	}
	if(is_gxe){
		res.array().colwise() *= env_var.array();
	}
	return res;
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
			res = getXXtz().cwiseProduct(other.getXXtz()).sum();
		} else {
			Eigen::MatrixXd XXtWz = getXXtz();
			Eigen::MatrixXd WXXtWz = project_out_covars(XXtWz);
			res = WXXtWz.cwiseProduct(other.getXXtz()).sum();
		}
	} else {
		throw std::runtime_error("Error in PVE_Component");
	}
	res = mpiUtils::mpiReduce_inplace(res);
	return res / get_n_var_local() / other.get_n_var_local() / (double) n_draws;
}

Eigen::MatrixXd RHEreg_Component::project_out_covars(Eigen::Ref <Eigen::MatrixXd> rhs) const {
	return EigenUtils::project_out_covars(rhs, C, CtC_inv);
}

void get_GxE_collapsed_component(const std::vector<RHEreg_Component> &vec_of_components, RHEreg_Component &new_comp,
								 const Eigen::Ref<const Eigen::MatrixXd> &E,
								 const Eigen::Ref<const Eigen::VectorXd> &env_weights,
								 const Eigen::Ref<const Eigen::MatrixXd> &ytEXXtEy) {
	long n_components = vec_of_components.size();
	long n_samples = vec_of_components[0].n_samples;
	long n_draws = vec_of_components[0].n_draws;

	Eigen::VectorXd eta = E * env_weights;
	Eigen::MatrixXd sum_WlVbl = Eigen::MatrixXd::Zero(n_samples, n_draws);
	long GxE_comp_index;
	for (int ii = 0; ii < n_components; ii++) {
		if (vec_of_components[ii].effect_type == "GxE") {
			assert(vec_of_components[ii].rm_jacknife_block == -1);
			GxE_comp_index = ii;
			long ll = vec_of_components[ii].env_var_index;
			sum_WlVbl += env_weights[ll] * E.col(ll).asDiagonal().inverse() * vec_of_components[ii].getXXtz();
		}
	}
	new_comp._XXtWz = sum_WlVbl;
	new_comp.ytXXty = env_weights.transpose() * ytEXXtEy * env_weights;
	new_comp.is_finalised = true;

	new_comp.set_env_var(eta);
	new_comp.n_var_local = vec_of_components[GxE_comp_index].n_var_local;
	new_comp.n_draws = vec_of_components[GxE_comp_index].n_draws;
	new_comp.n_covar = vec_of_components[GxE_comp_index].n_covar;
	new_comp.Y = vec_of_components[GxE_comp_index].Y;
	new_comp.C = vec_of_components[GxE_comp_index].C;
	new_comp.CtC_inv = vec_of_components[GxE_comp_index].CtC_inv;
	new_comp.label = "GxE";
	new_comp.effect_type = "GxE";
}

void get_GxE_collapsed_system(const std::vector<RHEreg_Component> &vec_of_components,
							  std::vector<RHEreg_Component> &new_components,
							  const Eigen::Ref<const Eigen::MatrixXd> &E,
							  const Eigen::Ref<const Eigen::VectorXd> &env_weights,
							  const Eigen::Ref<const Eigen::MatrixXd> &ytEXXtEy){

	long n_components = vec_of_components.size();
	long n_samples = vec_of_components[0].n_samples;
	long n_draws = vec_of_components[0].n_draws;

	new_components.clear();
	new_components.reserve(3);

	// Get main and noise components
	for (int ii = 0; ii < n_components; ii++) {
		if(vec_of_components[ii].effect_type == "G") {
			new_components.push_back(vec_of_components[ii]);
		} else if (vec_of_components[ii].effect_type == "noise") {
			new_components.push_back(vec_of_components[ii]);
		}
	}
	assert(new_components.size() == 2);

	// Create component for \diag{Ew} \left( \sum_l w_l \bm{v}_{b, l} \right)
	Eigen::MatrixXd placeholder = Eigen::MatrixXd::Zero(n_samples, n_draws);
	RHEreg_Component combined_comp(vec_of_components[0].params, vec_of_components[0].Y,
			vec_of_components[0].C, vec_of_components[0].CtC_inv,
			placeholder);
	get_GxE_collapsed_component(vec_of_components, combined_comp, E, env_weights, ytEXXtEy);

	new_components.push_back(std::move(combined_comp));
}
