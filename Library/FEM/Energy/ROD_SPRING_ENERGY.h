#pragma once
#include <pybind11/pybind11.h>
#include <functional>
#include <FEM/Shell/Rod/MASS_SPRING_DERIVATIVES.h>

namespace JGSL {

template <class T, int dim>
class ABSTRACT_ENERGY;

std::vector<VECTOR<int, 2>> rod;
std::vector<VECTOR<double, 3>> rodInfo;

template <class T, int dim>
class ROD_SPRING_ENERGY : public ABSTRACT_ENERGY<T, dim> {
public:
    void Compute_IncPotential(
        MESH_ELEM<dim>& Elem,
        const VECTOR<T, dim>& gravity,
        T h, MESH_NODE<T, dim>& X,
        MESH_NODE<T, dim>& Xtilde,
        MESH_NODE_ATTR<T, dim>& nodeAttr,
        MESH_ELEM_ATTR<T, dim>& elemAttr,
        FIXED_COROTATED<T, dim>& elasticityAttr,
        std::vector<VECTOR<int, dim + 1>>& constraintSet,
        T dHat2, T kappa[],
        double& value
    ) {
        int i = 0;
        for (const auto& segI : rod) {
            const VECTOR<T, dim>& x0 = std::get<0>(X.Get_Unchecked(segI[0]));
            const VECTOR<T, dim>& x1 = std::get<0>(X.Get_Unchecked(segI[1]));

            value += h * h * M_PI * rodInfo[i][2] * rodInfo[i][2] / 4 * rodInfo[i][1] * rodInfo[i][0] / 2 *
                 std::pow((x0 - x1).length() / rodInfo[i][1] - 1, 2);

            ++i;
        }
    }
    void Compute_IncPotential_Gradient(
        MESH_ELEM<dim>& Elem,
        const VECTOR<T, dim>& gravity,
        T h, MESH_NODE<T, dim>& X,
        MESH_NODE<T, dim>& Xtilde,
        MESH_NODE_ATTR<T, dim>& nodeAttr,
        MESH_ELEM_ATTR<T, dim>& elemAttr,
        FIXED_COROTATED<T, dim>& elasticityAttr,
        std::vector<VECTOR<int, dim + 1>>& constraintSet,
        T dHat2, T kappa[]
    ) {
        int i = 0;
        for (const auto& segI : rod) {
            const VECTOR<T, dim>& x0 = std::get<0>(X.Get_Unchecked(segI[0]));
            const VECTOR<T, dim>& x1 = std::get<0>(X.Get_Unchecked(segI[1]));

            T g[6];
            g_MS(rodInfo[i][0], rodInfo[i][1],
                 x0[0], x0[1], x0[2], x1[0], x1[1], x1[2], g);

            T w = h * h * M_PI * rodInfo[i][2] * rodInfo[i][2] / 4;
            for (int endI = 0; endI < 2; ++endI) {
                VECTOR<T, dim>& grad = std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(nodeAttr.Get_Unchecked(segI[endI]));
                for (int dimI = 0; dimI < dim; ++dimI) {
                    grad[dimI] += w * g[endI * dim + dimI];
                }
            }

            ++i;
        }
    }
    void Compute_IncPotential_Hessian(
        MESH_ELEM<dim>& Elem,
        T h, MESH_NODE<T, dim>& X,
        MESH_NODE_ATTR<T, dim>& nodeAttr,
        MESH_ELEM_ATTR<T, dim>& elemAttr,
        FIXED_COROTATED<T, dim>& elasticityAttr,
        std::vector<VECTOR<int, dim + 1>>& constraintSet,
        T dHat2, T kappa[],
        std::vector<Eigen::Triplet<T>>& triplets
    ) {
        BASE_STORAGE<int> threads(rod.size());
        for (int i = 0; i < rod.size(); ++i) {
            threads.Append(triplets.size() + i * 36);
        }

        triplets.resize(triplets.size() + rod.size() * 36);
        threads.Par_Each([&](int i, auto data) {
          const auto& [tripletStartInd] = data;
          const VECTOR<int, 2>& segI = rod[i];
          const VECTOR<T, dim>& x0 = std::get<0>(X.Get_Unchecked(segI[0]));
          const VECTOR<T, dim>& x1 = std::get<0>(X.Get_Unchecked(segI[1]));

          Eigen::Matrix<T, 6, 6> hessian;
          H_MS(rodInfo[i][0], rodInfo[i][1],
               x0[0], x0[1], x0[2], x1[0], x1[1], x1[2], hessian.data());

          if (true) {
              makePD(hessian);
          }

          int globalInd[6] = {
              segI[0] * dim,
              segI[0] * dim + 1,
              segI[0] * dim + 2,
              segI[1] * dim,
              segI[1] * dim + 1,
              segI[1] * dim + 2,
          };
          T w = h * h * M_PI * rodInfo[i][2] * rodInfo[i][2] / 4;
          for (int rowI = 0; rowI < 6; ++rowI) {
              for (int colI = 0; colI < 6; ++colI) {
                  triplets[tripletStartInd + rowI * 6 + colI] = Eigen::Triplet<T>(
                      globalInd[rowI], globalInd[colI], w * hessian(rowI, colI)
                  );
              }
          }
        });
    }
};

}