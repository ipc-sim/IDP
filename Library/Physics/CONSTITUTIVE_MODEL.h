#pragma once

#include <Storage/prelude.hpp>
#include <Utils/PROFILER.h>
#include <Physics/EQUATION_OF_STATE.h>
#include <Physics/FIXED_COROTATED.h>
#include <Physics/NEOHOOKEAN.h>
#include <Physics/LINEAR_COROTATED.h>
#include <Physics/NEOHOOKEAN_BORDEN.h>
#include <Physics/SYMMETRIC_DIRICHLET.h>
#include <Physics/STVK_HENCKY.h>
#include <Physics/NACC_PLASTICITY.h>
#include <Physics/PLASTICITY.h>

namespace py = pybind11;
namespace JGSL {

template <class MODEL, class T, int dim>
std::unique_ptr<MODEL> Create_Constitutive_Model(const VECTOR<T, dim>& val) {
    return std::make_unique<MODEL>();
}

//TEST FEM material switch
template <class ELASTICITY_TYPE, class T, int dim>
void Compute_First_PiolaKirchoff_Stress_Derivative(void) {
    // ELASTICITY_TYPE::template Compute_Psi<T, dim>();
}

template <class MODEL, class T, int dim>
py::module Export_Model_Submodule(py::module& m, const std::string &name, bool export_storage = true) {

    // The storage type might already been exported, so we add an option
    if (export_storage) {
        std::string storage_name = std::string(name).append("_STORAGE_" + std::to_string(dim));
        py::class_<typename MODEL::STORAGE>(m, storage_name.c_str(), py::module_local()).def(py::init<>());
    }

    // Export the module
    std::string module_name = std::string(name).append("_" + std::to_string(dim));
    py::module module = m.def_submodule(module_name.c_str());
    
    // Define member functions
    module.def("Create", &MODEL::Create);
    module.def("Append", &MODEL::Append);
    return module;
}

template <class T, int dim>
void Export_Constitutive_Model(py::module& m) {

    using FCR_MODEL = FIXED_COROTATED_FUNCTOR<T, dim>;
    auto fcr_mod = Export_Model_Submodule<FCR_MODEL, T, dim>(m, "FIXED_COROTATED");
    fcr_mod.def("Append_FEM", &FCR_MODEL::Append_FEM);
    fcr_mod.def("Append_All_FEM", &FCR_MODEL::Append_All_FEM);
    fcr_mod.def("All_Append_FEM", &FCR_MODEL::All_Append_FEM);

    using EOS_MODEL = EQUATION_OF_STATE_FUNCTOR<T, dim>;
    auto eos_mod = Export_Model_Submodule<EOS_MODEL, T, dim>(m, "EQUATION_OF_STATE" , dim == 2);

    using LCR_MODEL = FIXED_COROTATED_FUNCTOR<T, dim>;
    auto lcr_mod = Export_Model_Submodule<LCR_MODEL, T, dim>(m, "LINEAR_COROTATED", false);

    using NHB_MODEL = NEOHOOKEAN_BORDEN_FUNCTOR<T, dim>;
    auto nhb_mod = Export_Model_Submodule<NHB_MODEL, T, dim>(m, "NEOHOOKEAN_BORDEN");

    // TODO: Do Project Strain NACC
    // m.def("Project_Strain_NACC", &Project_Strain_NACC<double, 2>);

    m.def("Von_Mises_Project_Strain", Von_Mises_Project_Strain<T, dim>);
}

}