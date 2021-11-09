#pragma once

#include <FEM/DEFORMATION_GRADIENT.h>

namespace JGSL {

template <class T, int dim>
void Von_Mises_Project_Strain(
    MESH_ELEM<dim>& Elem,
    MESH_NODE<T, dim>& X,
    MESH_ELEM_ATTR<T, dim>& elemAttr,
    FIXED_COROTATED<T, dim>& elasticityAttr
) {
    Compute_Deformation_Gradient(X, Elem, elemAttr, elasticityAttr);

    T E = PARAMETER::Get("Von_Mises_E", (T)1e4);
    T nu = PARAMETER::Get("Von_Mises_nu", (T)0.4);
    T lambda = E * nu / (((T)1 + nu) * ((T)1 - (T)2 * nu));
    T mu = E / ((T)2 * ((T)1 + nu));
    T yield_stress = PARAMETER::Get("Von_Mises_yield_stress", (T)0);
    T xi = PARAMETER::Get("Von_Mises_xi", (T)0);
    T fail_stress = PARAMETER::Get("Von_Mises_fail_stress", (T)0);
    elemAttr.Join(elasticityAttr).Par_Each([&](int id, auto data) {
        auto& [IP, B, strain, _, __, ___] = data;

        MATRIX<T, dim> oldStrain = strain;
        MATRIX<T, dim> U(1), V(1);
        VECTOR<T, dim> sigma;
        Singular_Value_Decomposition(strain, U, sigma, V);
        //TV epsilon = sigma.array().log();
        //TV epsilon = sigma.array().max(1e-4).log();
        VECTOR<T, dim> clamped_sigma = sigma;
        for (int d = 0; d < dim; ++d)
            clamped_sigma(d) = std::max(clamped_sigma(d), 1e-4);
        VECTOR<T, dim> epsilon = clamped_sigma.log(); //TODO: need the max part?
        T trace_epsilon = epsilon.Sum();
        VECTOR<T, dim> epsilon_hat = epsilon - (trace_epsilon / (T)dim) * VECTOR<T, dim>::Ones_Vector();
        T epsilon_hat_squared_norm = epsilon_hat.length2();
        T epsilon_hat_norm = std::sqrt(epsilon_hat_squared_norm);
        T delta_gamma = epsilon_hat_norm - yield_stress / (2 * mu);
        if (delta_gamma <= 0) // case I
        {
            //do nothing
        }
        else {
            //hardening
            yield_stress -= xi * delta_gamma; //supposed to only increase yield_stress
            //yield_stress = std::max((T)0, yield_stress);

            VECTOR<T, dim> H = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat; // case II
            T tau_0 = 2 * mu * H(0) + lambda * H.Sum();
            VECTOR<T, dim> exp_H = H.exp();
            strain = U * MATRIX<T, dim>(exp_H) * V.transpose();
        }
        IP = IP * oldStrain.inverse() * strain;
    });
}

}
