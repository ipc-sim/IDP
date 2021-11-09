#pragma once

#include <Math/VECTOR.h>
#include <Utils/MESHIO.h>

namespace py = pybind11;
namespace JGSL {

template <class T, int dim>
T Get_F_coeff(MESH_NODE<T, dim>& X, VECTOR<int, dim + 1> elemVInd)
{
    if constexpr (dim == 3) {
        const VECTOR<T, dim>& X0 = std::get<0>(X.Get_Unchecked_Const(elemVInd(0)));
        const VECTOR<T, dim>& X1 = std::get<0>(X.Get_Unchecked_Const(elemVInd(1)));
        const VECTOR<T, dim>& X2 = std::get<0>(X.Get_Unchecked_Const(elemVInd(2)));
        const VECTOR<T, dim>& X3 = std::get<0>(X.Get_Unchecked_Const(elemVInd(3)));
        if (PARAMETER::Get("Init_F_script", std::string("")) == std::string("bunny_inside_box")) {
            if (X0.min() < -0.25 || X0.max() > 0.25)
                return (T)1.0;
            else
                return PARAMETER::Get("bunny_inside_box_scale", (T)1.0);
        }
        if (PARAMETER::Get("Init_F_script", std::string("")) == std::string("two")) {
            return 2.0;
        }
        if (PARAMETER::Get("Init_F_script", std::string("")) == std::string("armadillo")) {
            T radius = PARAMETER::Get("Init_F_script_armadillo_radius", T(20));
            T scale = PARAMETER::Get("Init_F_script_armadillo_scale", T(5));
            VECTOR<T, dim> c0(52.8, 76, -35);
            T dist0 = std::max(std::max((X0 - c0).length(), (X1 - c0).length()), std::max((X2 - c0).length(), (X3 - c0).length()));
            if (dist0 < radius) {
                return 1 + (1 - dist0 / radius) * scale;
            }
            VECTOR<T, dim> c1(-57.2, 72, -47);
            T dist1 = std::max(std::max((X0 - c1).length(), (X1 - c1).length()), std::max((X2 - c1).length(), (X3 - c1).length()));
            if (dist1 < radius) {
                return 1 + (1 - dist1 / radius) * scale;
            }
        }
    }
    return (T)1;
}

template <class T, int dim>
void Compute_Vol_And_Inv_Basis(
    MESH_NODE<T, dim>& X,
    MESH_ELEM<dim>& Elem,
    SCALAR_STORAGE<T>& vol, 
    MESH_ELEM_ATTR<T, dim>& elemAttr)
{
    vol.Reserve(Elem.size);
    elemAttr.Reserve(Elem.size);
    Elem.Each([&](int id, auto data) {
        auto &[elemVInd] = data;
        T F_coeff = Get_F_coeff<T, dim>(X, elemVInd);
        const VECTOR<T, dim>& X0 = std::get<0>(X.Get_Unchecked_Const(elemVInd(0))) * F_coeff;
        const VECTOR<T, dim>& X1 = std::get<0>(X.Get_Unchecked_Const(elemVInd(1))) * F_coeff;
        const VECTOR<T, dim>& X2 = std::get<0>(X.Get_Unchecked_Const(elemVInd(2))) * F_coeff;

        MATRIX<T, dim> IB;
        IB(0, 0) = X1(0) - X0(0);
        IB(1, 0) = X1(1) - X0(1);
        IB(0, 1) = X2(0) - X0(0);
        IB(1, 1) = X2(1) - X0(1);

        if constexpr (dim == 3) {
            const VECTOR<T, dim>& X3 = std::get<0>(X.Get_Unchecked_Const(elemVInd(3))) * F_coeff;
            IB(2, 0) = X1(2) - X0(2);
            IB(2, 1) = X2(2) - X0(2);
            IB(0, 2) = X3(0) - X0(0);
            IB(1, 2) = X3(1) - X0(1);
            IB(2, 2) = X3(2) - X0(2);
            vol.Append(IB.determinant() / 6.0);
        }
        else {
            vol.Append(IB.determinant() / 2.0);
        }

        IB.invert();
        elemAttr.Append(IB, MATRIX<T, dim>());
    });
}

template <class T, int dim = 2>
void Compute_Vol_And_Inv_Basis_Tex(
    MESH_NODE<T, dim + 1>& X,
    MESH_ELEM<dim>& Elem,
    SCALAR_STORAGE<T>& vol, 
    MESH_ELEM_ATTR<T, dim>& elemAttr)
{
    vol.Reserve(Elem.size);
    elemAttr.Reserve(Elem.size);
    Elem.Each([&](int id, auto data) {
        auto &[elemVInd] = data;
        const VECTOR<T, dim + 1>& X1 = std::get<0>(X.Get_Unchecked_Const(elemVInd(0)));
        const VECTOR<T, dim + 1>& X2 = std::get<0>(X.Get_Unchecked_Const(elemVInd(1)));
        const VECTOR<T, dim + 1>& X3 = std::get<0>(X.Get_Unchecked_Const(elemVInd(2)));

        MATRIX<T, dim> IB;
        IB(0, 0) = (X2 - X1).norm();
        IB(1, 0) = 0;
        IB(0, 1) = (X2 - X1).dot(X3 - X1) / IB(0, 0);
        IB(1, 1) = cross(X2 - X1, X3 - X1).norm() / IB(0, 0);

        vol.Append(IB.determinant() / 2.0);

        IB.invert();
        elemAttr.Append(IB, MATRIX<T, dim>());
    });
}

template <class T, int dim>
void Compute_Deformation_Gradient(
    const MESH_NODE<T, dim>& X,
    MESH_ELEM<dim>& Elem,
    const MESH_ELEM_ATTR<T, dim>& elemAttr,
    FIXED_COROTATED<T, dim>& elasticityAttr)
{
    TIMER_FLAG("computeDG");
    Elem.Par_Each([&](int id, auto data) {
        auto &[elemVInd] = data;
        const VECTOR<T, dim>& X0 = std::get<0>(X.Get_Unchecked_Const(elemVInd(0)));
        const VECTOR<T, dim>& X1 = std::get<0>(X.Get_Unchecked_Const(elemVInd(1)));
        const VECTOR<T, dim>& X2 = std::get<0>(X.Get_Unchecked_Const(elemVInd(2)));
        const MATRIX<T, dim>& IB = std::get<FIELDS<MESH_ELEM_ATTR<T, dim>>::IB>(elemAttr.Get_Unchecked_Const(id));
        MATRIX<T, dim>& F = std::get<FIELDS<FIXED_COROTATED<T, dim>>::F>(elasticityAttr.Get_Unchecked(id));

        F(0, 0) = X1(0) - X0(0);
        F(1, 0) = X1(1) - X0(1);
        F(0, 1) = X2(0) - X0(0);
        F(1, 1) = X2(1) - X0(1);
        if constexpr (dim == 3) {
            const VECTOR<T, dim>& X3 = std::get<0>(X.Get_Unchecked_Const(elemVInd(3)));
            F(2, 0) = X1(2) - X0(2);
            F(2, 1) = X2(2) - X0(2);
            F(0, 2) = X3(0) - X0(0);
            F(1, 2) = X3(1) - X0(1);
            F(2, 2) = X3(2) - X0(2);
        }

        F = F * IB;
    });
}

void Export_Deformation_Gradient(py::module& m) {
    m.def("Compute_Vol_And_Inv_Basis", &Compute_Vol_And_Inv_Basis<double, 2>);
    m.def("Compute_Vol_And_Inv_Basis", &Compute_Vol_And_Inv_Basis<double, 3>);
    m.def("Compute_Vol_And_Inv_Basis_Tex", &Compute_Vol_And_Inv_Basis_Tex<double, 2>);
}

}
