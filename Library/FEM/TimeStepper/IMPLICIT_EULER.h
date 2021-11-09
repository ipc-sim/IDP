#pragma once

#include <Physics/FIXED_COROTATED.h>
#include <FEM/DEFORMATION_GRADIENT.h>
#include <FEM/ELEM_TO_NODE.h>
#include <FEM/IPC.h>
#include <FEM/FRICTION.h>
#include <Math/CSR_MATRIX.h>
#include <Math/DIRECT_SOLVER.h>
#include <FEM/Energy/ENERGY.h>
#include <FEM/Energy/BOUNDED_DISTORTION.h>
#include <Utils/PARAMETER.h>

namespace py = pybind11;
namespace JGSL {

template <class T, int dim>
void Initialize_Elastic_IPC(
    MESH_NODE<T, dim>& X0,
    MESH_ELEM<dim>& Elem,
    T h, T E, T nu, T dHat2,
    VECTOR<T, 3>& kappa)
{
    std::vector<int> boundaryNode;
    std::vector<VECTOR<int, 2>> boundaryEdge;
    std::vector<VECTOR<int, 3>> boundaryTri;
    if constexpr (dim == 2) {
        Find_Boundary_Edge_And_Node(X0.size, Elem, boundaryNode, boundaryEdge);
        //TODO
    }
    else {
        BASE_STORAGE<int> TriVI2TetVI;
        BASE_STORAGE<VECTOR<int, 3>> Tri;
        Find_Surface_TriMesh<T, false>(X0, Elem, TriVI2TetVI, Tri);
        Find_Surface_Primitives(X0.size, Tri, boundaryNode, boundaryEdge, boundaryTri);

        T areaSum = 0;
        Tri.Each([&](int id, auto data) {
            auto &[ind] = data;
            const VECTOR<T, dim>& X1 = std::get<0>(X0.Get_Unchecked(ind[0]));
            const VECTOR<T, dim>& X2 = std::get<0>(X0.Get_Unchecked(ind[1]));
            const VECTOR<T, dim>& X3 = std::get<0>(X0.Get_Unchecked(ind[2]));
            areaSum += 0.5 * cross(X2 - X1, X3 - X1).length();
        });

        const T h2vol = h * h * areaSum / (Tri.size * 3) * sqrt(dHat2);
        const T lambda = E * nu / ((T)1 - nu * nu);
        const T mu = E / ((T)2 * ((T)1 + nu));
        kappa[0] = h2vol * mu;
        kappa[1] = h2vol * lambda;
        kappa[2] = nu;
    }
}

template <class T, int dim, bool shell = false, bool elasticIPC = false>
int Advance_One_Step_IE(MESH_ELEM<dim>& Elem,
    VECTOR_STORAGE<T, dim + 1>& DBC,
    const VECTOR<T, dim>& gravity, T h,
    T NewtonTol, bool withCollision,
    T dHat2, VECTOR<T, 3>& kappaVec, //TODO: dHat as input and relative to bbox, adapt kappa
    T mu, T epsv2, T bdstiff, T K,
    bool staticSolve, bool withShapeMatching,
    std::string output_folder, int current_frame,
    MESH_NODE<T, dim>& X,
    MESH_NODE<T, dim>& X0,
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    MESH_ELEM_ATTR<T, dim>& elemAttr,
    FIXED_COROTATED<T, dim>& elasticityAttr,
    MESH_ELEM<2>& Extra_Mesh)
{
    Eigen::setNbThreads(1);
    std::string name = "implicitEuler";
    name += (PARAMETER::Get("With_dRdx", true)) ? "_drdx" : "_nodrdx";
    name += withCollision ? "_ipc" : "_noipc";
    TIMER_ANALYZE(name.c_str());

    if (PARAMETER::Get("Target_Psi", std::string("")) != std::string("NO")) {
        T targetPsi = std::stod(PARAMETER::Get("Target_Psi", std::string("")));
        T Psi = 0;
        Compute_Deformation_Gradient(X, Elem, elemAttr, elasticityAttr);
        SYMMETRIC_DIRICHLET_FUNCTOR<T, dim>::Compute_Psi(elasticityAttr, 1, elemAttr, Psi);
        printf("curPsi = %le\n", Psi);
        if (Psi <= targetPsi) {
            printf("curPsi %le <= targetPsi %le\n", Psi, targetPsi);
            return 0;
        }
    }

    if (PARAMETER::Get("Progressively_Update_F_scale", (T)0) > 0) {
        T scale = PARAMETER::Get("Progressively_Update_F_scale", (T)0);
        int frame = PARAMETER::Get("Progressively_Update_F_frame", 0);
        static int times = 0;
        ++times;
        if (times <= frame) {
            elemAttr.Par_Each([&](int id, auto data) {
                auto &[IB, P] = data;
                IB *= (1 + scale * (times - 1)) / (1 + scale * times);
            });
        }
    }

    {
        TIMER_FLAG("Compute Total Volume");
        T total_vol = 0;
        Elem.Each([&](int id, auto data) {
          auto &[elemVInd] = data;
          const VECTOR<T, dim>& X0 = std::get<0>(X.Get_Unchecked_Const(elemVInd(0)));
          const VECTOR<T, dim>& X1 = std::get<0>(X.Get_Unchecked_Const(elemVInd(1)));
          const VECTOR<T, dim>& X2 = std::get<0>(X.Get_Unchecked_Const(elemVInd(2)));

          MATRIX<T, dim> IB;
          IB(0, 0) = X1(0) - X0(0);
          IB(1, 0) = X1(1) - X0(1);
          IB(0, 1) = X2(0) - X0(0);
          IB(1, 1) = X2(1) - X0(1);

          if constexpr (dim == 3) {
              const VECTOR<T, dim>& X3 = std::get<0>(X.Get_Unchecked_Const(elemVInd(3)));
              IB(2, 0) = X1(2) - X0(2);
              IB(2, 1) = X2(2) - X0(2);
              IB(0, 2) = X3(0) - X0(0);
              IB(1, 2) = X3(1) - X0(1);
              IB(2, 2) = X3(2) - X0(2);
              total_vol += IB.determinant() / 6.0;
          }
          else {
              total_vol += IB.determinant() / 2.0;
          }
        });
        JGSL_FILE("TotalVol", total_vol);
    }

    T kappa[] = {kappaVec[0], kappaVec[1], kappaVec[2]}; // dumb pybind does not support c array

    ENERGY<T, dim> energy;
    energy.Add(std::make_shared<ELASTICITY_ENERGY<T,dim>>());
    if (!staticSolve) {
        energy.Add(std::make_shared<INERTIA_ENERGY<T,dim>>());
    }
    if (withCollision) {
        energy.Add(std::make_shared<IPC_ENERGY<T,dim,elasticIPC>>());
    }
    if (withShapeMatching) {
        energy.Add(std::make_shared<SHAPE_MATCHING_ENERGY<T,dim>>());
        std::dynamic_pointer_cast<SHAPE_MATCHING_ENERGY<T, dim>>(energy.energies.back())->Precompute(Elem, X, X0, elasticityAttr);
    }

    // record Xn and compute predictive pos Xtilde
    MESH_NODE<T, dim> Xn, Xtilde;
    Append_Attribute(X, Xn);
    Append_Attribute(X, Xtilde);
    {
        TIMER_FLAG("Compute XTilde");
        Xtilde.Join(nodeAttr).Par_Each([&](int id, auto data) {
            auto &[x, x0, v, g, m] = data;
            x += h * v + h * h * gravity;
        });
    }

    CSR_MATRIX<T> sysMtr;
    std::vector<T> rhs(X.size * dim), sol(X.size * dim);

    //TODO: only once
    // compute contact primitives
    std::vector<int> boundaryNode;
    std::vector<VECTOR<int, 2>> boundaryEdge;
    std::vector<VECTOR<int, 3>> boundaryTri;
    std::vector<T> BNArea, BEArea, BTArea;
    if (withCollision) {
        TIMER_FLAG("Find Object Boundary");
        if constexpr (dim == 2) {
            Find_Boundary_Edge_And_Node(X.size, Elem, boundaryNode, boundaryEdge);
        }
        else {
            BASE_STORAGE<int> TriVI2TetVI;
            BASE_STORAGE<VECTOR<int, 3>> Tri;
            Find_Surface_TriMesh_With_Extra_Mesh<T, false>(X, Elem, TriVI2TetVI, Tri, Extra_Mesh);
            Find_Surface_Primitives_And_Compute_Area(X, Tri, boundaryNode, boundaryEdge, boundaryTri,
                BNArea, BEArea, BTArea);
        }
    }

    T DBCAlpha = 1;
    std::vector<bool> DBCb(X.size, false); // this mask does not change with whether augmented Lagrangian is turned on
    std::vector<bool> DBCb_fixed(X.size, false); // this masks nodes that are fixed (DBC with 0 velocity)
    std::vector<T> DBCDisp(X.size * dim, T(0));
    DBC.Each([&](int id, auto data) {
      auto &[dbcI] = data;
      int vI = dbcI(0);
      const VECTOR<T, dim> &x = std::get<0>(X.Get_Unchecked(vI));
      DBCDisp[vI * dim] = dbcI(1) - x(0);
      DBCDisp[vI * dim + 1] = dbcI(2) - x(1);
      if constexpr (dim == 3) {
          DBCDisp[vI * dim + 2] = dbcI(3) - x(2);
          if (!(DBCDisp[vI * dim] || DBCDisp[vI * dim + 1] || DBCDisp[vI * dim + 2])) {
              DBCb_fixed[vI] = true;
          }
      }
      else {
          if (!(DBCDisp[vI * dim] || DBCDisp[vI * dim + 1])) {
              DBCb_fixed[vI] = true;
          }
      }
      DBCb[dbcI(0)] = true; // bool array cannot be written in parallel by entries
    });
    if (PARAMETER::Get("Elasticity_model", std::string("")) != std::string("FCR")) {
        Compute_Inversion_Free_StepSize<T, dim>(X, Elem, DBCDisp, DBCAlpha);
        printf("DBCAlpha under inversion free step size = %le\n", DBCAlpha);
    }
    if (bdstiff) {
        T E;
        bool valid = false;
        do {
            DBC.Each([&](int id, auto data) {
                auto &[dbcI] = data;
                VECTOR<T, dim> &x = std::get<0>(X.Get_Unchecked(dbcI(0)));
                const VECTOR<T, dim> &xn = std::get<0>(Xn.Get_Unchecked(dbcI(0)));

                x(0) = xn[0] + DBCAlpha * (dbcI(1) - xn[0]);
                x(1) = xn[1] + DBCAlpha * (dbcI(2) - xn[1]);
                if constexpr (dim == 3) {
                    x(2) = xn[2] + DBCAlpha * (dbcI(3) - xn[2]);
                }
            });
            valid = Compute_BD_Energy(Elem, h, K, bdstiff, DBCb, X, nodeAttr, elemAttr, elasticityAttr, E);
            if (!valid) {
                DBCAlpha /= 2.0;
            }
            printf("backtracking DBCAlpha for bounded distortion: %le\n", DBCAlpha);
        } while (!valid);
        printf("DBCAlpha under bounded distortion: %le\n", DBCAlpha);
        Xn.deep_copy_to(X);
    }
    if (withCollision) {
        Compute_Intersection_Free_StepSize<T, dim, false, elasticIPC>(X, boundaryNode, boundaryEdge, boundaryTri,
            std::vector<int>(), std::vector<VECTOR<int, 2>>(), std::map<int, std::set<int>>(), 
            VECTOR<int, 2>(boundaryNode.size(), boundaryNode.size()),
            DBCb, DBCDisp, T(0), DBCAlpha); // CCD
        printf("DBCAlpha under contact: %le\n", DBCAlpha);
    }
    DBC.Each([&](int id, auto data) {
        auto &[dbcI] = data;
        VECTOR<T, dim> &x = std::get<0>(X.Get_Unchecked(dbcI(0)));
        const VECTOR<T, dim> &xn = std::get<0>(Xn.Get_Unchecked(dbcI(0)));

        x(0) = xn[0] + DBCAlpha * (dbcI(1) - xn[0]);
        x(1) = xn[1] + DBCAlpha * (dbcI(2) - xn[1]);
        if constexpr (dim == 3) {
            x(2) = xn[2] + DBCAlpha * (dbcI(3) - xn[2]);
        }
    });
    T DBCStiff = 0, DBCPenaltyXn = 0;
    if (DBCAlpha == 1) {
        DBC.Each([&](int id, auto data) {
          auto &[dbcI] = data;
          VECTOR<T, dim> &x = std::get<0>(X.Get_Unchecked(dbcI(0)));
          x(0) = dbcI(1);
          x(1) = dbcI(2);
          if constexpr (dim == 3) {
              x(2) = dbcI(3);
          }
        });
        printf("DBC handled\n");
    }
    else {
        //TODO: haven't moved yet
        printf("moved DBC by %le, turn on Augmented Lagrangian\n", DBCAlpha);
        DBCStiff = 1e6;
        Compute_DBC_Dist2(Xn, DBC, DBCPenaltyXn);
    }

    // Newton loop
    int PNIter = 0;
    T infNorm = 0.0;
    bool useGD = false;
    // compute deformation gradient, constraint set, and energy
    Compute_Deformation_Gradient(X, Elem, elemAttr, elasticityAttr);
    std::vector<VECTOR<int, dim + 1>> constraintSet_prev;
    std::vector<T> dist2_prev;
    std::vector<VECTOR<int, dim + 1>> constraintSet;
    std::vector<VECTOR<int, 2>> constraintSetPTEE;
    std::vector<VECTOR<T, 2>> stencilInfo;
    // friction:
    std::vector<VECTOR<int, dim + 1>> fricConstraintSet;
    std::vector<Eigen::Matrix<T, dim - 1, 1>> closestPoint;
    std::vector<Eigen::Matrix<T, dim, dim - 1>> tanBasis;
    std::vector<T> normalForce;
    if (withCollision) {
        Compute_Constraint_Set<T, dim, shell, elasticIPC>(X, nodeAttr, boundaryNode, boundaryEdge, boundaryTri,
            std::vector<int>(), std::vector<VECTOR<int, 2>>(), std::map<int, std::set<int>>(), BNArea, BEArea, BTArea,
            VECTOR<int, 2>(boundaryNode.size(), boundaryNode.size()),
            DBCb, dHat2, T(0), false, constraintSet, constraintSetPTEE, stencilInfo);
        if (mu > 0) {
            Compute_Friction_Basis<T, dim, elasticIPC>(X, constraintSet, stencilInfo, fricConstraintSet, closestPoint, tanBasis, normalForce, dHat2, kappa, T(0));
        }
    }

    if constexpr (dim == 3) {
        if (PARAMETER::Get("dumpling_on_plate", (T)0)) {
            T stiffness = PARAMETER::Get("dumpling_on_plate", (T)0);
            T thickness = PARAMETER::Get("dumpling_on_plate_thickness", (T)0.001);
            int threshold = PARAMETER::Get("dumpling_on_plate_threshold", 0);
            if (rod.empty()) {
                for (int cI = 0; cI < constraintSet.size(); ++cI) {
                    const VECTOR<int, 4>& cIVInd = constraintSet[cI];
                    assert(cIVInd[1] >= 0);
                    if (cIVInd[0] >= 0) {
                        // EE
                        if (cIVInd[3] >= 0 && cIVInd[2] >= 0) {
                            // ++++ EE, no mollification
                            int a0 = cIVInd[0];
                            int a1 = cIVInd[1];
                            int b0 = cIVInd[2];
                            int b1 = cIVInd[3];
                            const VECTOR<T, 3>& Xea0 = std::get<0>(X.Get_Unchecked(cIVInd[0]));
                            const VECTOR<T, 3>& Xea1 = std::get<0>(X.Get_Unchecked(cIVInd[1]));
                            const VECTOR<T, 3>& Xeb0 = std::get<0>(X.Get_Unchecked(cIVInd[2]));
                            const VECTOR<T, 3>& Xeb1 = std::get<0>(X.Get_Unchecked(cIVInd[3]));
                            if (a0 < threshold || a1 < threshold || b0 < threshold || b1 < threshold)
                                continue;
                            rod.emplace_back(a0, b0);
                            rod.emplace_back(a0, b1);
                            rod.emplace_back(a1, b0);
                            rod.emplace_back(a1, b1);
                            rodInfo.emplace_back(stiffness, (Xea0 - Xeb0).length(), thickness);
                            rodInfo.emplace_back(stiffness, (Xea0 - Xeb1).length(), thickness);
                            rodInfo.emplace_back(stiffness, (Xea1 - Xeb0).length(), thickness);
                            rodInfo.emplace_back(stiffness, (Xea1 - Xeb1).length(), thickness);
                        }
                        else {
                            // EE, PE, or PP with mollification
                            std::array<int, 4> edgeVInd;
                            if (cIVInd[3] >= 0) {
                                // ++-+ EE with mollification
                                edgeVInd = {cIVInd[0], cIVInd[1], -cIVInd[2] - 1, cIVInd[3]};
                                int a0 = cIVInd[0];
                                int a1 = cIVInd[1];
                                int b0 = -cIVInd[2] - 1;
                                int b1 = cIVInd[3];
                                if (a0 < threshold || a1 < threshold || b0 < threshold || b1 < threshold)
                                    continue;
                                const VECTOR<T, 3>& Xea0 = std::get<0>(X.Get_Unchecked(edgeVInd[0]));
                                const VECTOR<T, 3>& Xea1 = std::get<0>(X.Get_Unchecked(edgeVInd[1]));
                                const VECTOR<T, 3>& Xeb0 = std::get<0>(X.Get_Unchecked(edgeVInd[2]));
                                const VECTOR<T, 3>& Xeb1 = std::get<0>(X.Get_Unchecked(edgeVInd[3]));
                                rod.emplace_back(a0, b0);
                                rod.emplace_back(a0, b1);
                                rod.emplace_back(a1, b0);
                                rod.emplace_back(a1, b1);
                                rodInfo.emplace_back(stiffness, (Xea0 - Xeb0).length(), thickness);
                                rodInfo.emplace_back(stiffness, (Xea0 - Xeb1).length(), thickness);
                                rodInfo.emplace_back(stiffness, (Xea1 - Xeb0).length(), thickness);
                                rodInfo.emplace_back(stiffness, (Xea1 - Xeb1).length(), thickness);
                            }
                            else if (cIVInd[2] >= 0) {
                                // +++- PE with mollification, multiplicity 1
                                edgeVInd = {cIVInd[0], -cIVInd[3] - 1, cIVInd[1], cIVInd[2]};
                                int a0 = cIVInd[0];
                                int a1 = -cIVInd[3] - 1;
                                int b0 = cIVInd[1];
                                int b1 = cIVInd[2];
                                if (a0 < threshold || b0 < threshold || b1 < threshold)
                                    continue;
                                const VECTOR<T, 3>& Xea0 = std::get<0>(X.Get_Unchecked(edgeVInd[0]));
                                const VECTOR<T, 3>& Xea1 = std::get<0>(X.Get_Unchecked(edgeVInd[1]));
                                const VECTOR<T, 3>& Xeb0 = std::get<0>(X.Get_Unchecked(edgeVInd[2]));
                                const VECTOR<T, 3>& Xeb1 = std::get<0>(X.Get_Unchecked(edgeVInd[3]));
                                rod.emplace_back(a0, b0);
                                rod.emplace_back(a0, b1);
                                rodInfo.emplace_back(stiffness, (Xea0 - Xeb0).length(), thickness);
                                rodInfo.emplace_back(stiffness, (Xea0 - Xeb1).length(), thickness);
                            }
                            else {
                                // ++-- PP with mollification, multiplicity 1
                                edgeVInd = {cIVInd[0], -cIVInd[2] - 1, cIVInd[1], -cIVInd[3] - 1};
                                int a0 = cIVInd[0];
                                int a1 = -cIVInd[2] - 1;
                                int b0 = cIVInd[1];
                                int b1 = -cIVInd[3] - 1;
                                if (a0 < threshold || b0 < threshold)
                                    continue;
                                const VECTOR<T, 3>& Xea0 = std::get<0>(X.Get_Unchecked(edgeVInd[0]));
                                const VECTOR<T, 3>& Xea1 = std::get<0>(X.Get_Unchecked(edgeVInd[1]));
                                const VECTOR<T, 3>& Xeb0 = std::get<0>(X.Get_Unchecked(edgeVInd[2]));
                                const VECTOR<T, 3>& Xeb1 = std::get<0>(X.Get_Unchecked(edgeVInd[3]));
                                rod.emplace_back(a0, b0);
                                rodInfo.emplace_back(stiffness, (Xea0 - Xeb0).length(), thickness);
                            }
                        }
                    }
                    else {
                        // PT, PE, and PP
                        if (cIVInd[3] >= 0) {
                            // -+++ PT
                            assert(cIVInd[2] >= 0);
                            int p = -cIVInd[0] - 1;
                            int t0 = cIVInd[1];
                            int t1 = cIVInd[2];
                            int t2 = cIVInd[3];
                            if (p < threshold || t0 < threshold || t1 < threshold || t2 < threshold)
                                continue;
                            const VECTOR<T, 3>& Xp = std::get<0>(X.Get_Unchecked(-cIVInd[0] - 1));
                            const VECTOR<T, 3>& Xt0 = std::get<0>(X.Get_Unchecked(cIVInd[1]));
                            const VECTOR<T, 3>& Xt1 = std::get<0>(X.Get_Unchecked(cIVInd[2]));
                            const VECTOR<T, 3>& Xt2 = std::get<0>(X.Get_Unchecked(cIVInd[3]));
                            rod.emplace_back(p, t0);
                            rod.emplace_back(p, t1);
                            rod.emplace_back(p, t2);
                            rodInfo.emplace_back(stiffness, (Xp - Xt0).length(), thickness);
                            rodInfo.emplace_back(stiffness, (Xp - Xt1).length(), thickness);
                            rodInfo.emplace_back(stiffness, (Xp - Xt2).length(), thickness);
                        }
                        else if (cIVInd[2] >= 0) {
                            // -++[-] PE, last digit stores muliplicity
                            int p = -cIVInd[0] - 1;
                            int e0 = cIVInd[1];
                            int e1 = cIVInd[2];
                            if (p < threshold || e0 < threshold || e1 < threshold)
                                continue;
                            const VECTOR<T, 3>& Xp = std::get<0>(X.Get_Unchecked(-cIVInd[0] - 1));
                            const VECTOR<T, 3>& Xe0 = std::get<0>(X.Get_Unchecked(cIVInd[1]));
                            const VECTOR<T, 3>& Xe1 = std::get<0>(X.Get_Unchecked(cIVInd[2]));
                            rod.emplace_back(p, e0);
                            rod.emplace_back(p, e1);
                            rodInfo.emplace_back(stiffness, (Xp - Xe0).length(), thickness);
                            rodInfo.emplace_back(stiffness, (Xp - Xe1).length(), thickness);
                        }
                        else {
                            // -+-[-] PP, last digit stores muliplicity
                            int p0 = -cIVInd[0] - 1;
                            int p1 = cIVInd[1];
                            if (p0 < threshold || p1 < threshold)
                                continue;
                            const VECTOR<T, 3>& Xp0 = std::get<0>(X.Get_Unchecked(-cIVInd[0] - 1));
                            const VECTOR<T, 3>& Xp1 = std::get<0>(X.Get_Unchecked(cIVInd[1]));
                            rod.emplace_back(p0, p1);
                            rodInfo.emplace_back(stiffness, (Xp0 - Xp1).length(), thickness);
                        }
                    }
                }
            }
            energy.Add(std::make_shared<ROD_SPRING_ENERGY<T, dim>>());
        }
    }

    T Eprev;
    energy.Compute_IncPotential(Elem, gravity, h, X, Xtilde, nodeAttr, elemAttr, elasticityAttr,
        constraintSet, dHat2, kappa, Eprev);
    if (withCollision && mu > 0) {
        Compute_Friction_Potential(X, Xn, fricConstraintSet, closestPoint, tanBasis, normalForce, epsv2 * h * h, mu, Eprev);
    }
    if (bdstiff) {
        Compute_BD_Energy(Elem, h, K, bdstiff, DBCb, X, nodeAttr, elemAttr, elasticityAttr, Eprev);
    }
    if (DBCStiff) {
        Compute_DBC_Energy(X, nodeAttr, DBC, DBCStiff, Eprev);
    }

    std::deque<T> MDBCProgressI;
    int currentIter = 0;
    do {
        if (PARAMETER::Get("Max_Newton_Iter", -1) >= 0 && currentIter >= PARAMETER::Get("Max_Newton_Iter", -1))
            break;
        ++currentIter;
        // Check_BD_Gradient(Elem, 1.0, K, 1.0, DBCb, X, nodeAttr, elemAttr, elasticityAttr);
        // Check_BD_Hessian(Elem, 1.0, K, 1.0, DBCb, X, nodeAttr, elemAttr, elasticityAttr);

        // compute gradient
        energy.Compute_IncPotential_Gradient(Elem, gravity, h, X, Xtilde, nodeAttr, elemAttr, elasticityAttr,
            constraintSet, dHat2, kappa);
        if (withCollision && mu > 0) {
            Compute_Friction_Gradient(X, Xn, fricConstraintSet, closestPoint, tanBasis, normalForce, epsv2 * h * h, mu, nodeAttr);
        }
        if (bdstiff) {
            Compute_BD_Gradient(Elem, h, K, bdstiff, DBCb, X, nodeAttr, elemAttr, elasticityAttr);
        }
        // project rhs for Dirichlet boundary condition
        if (DBCStiff) {
            Compute_DBC_Gradient(X, nodeAttr, DBC, DBCStiff);
            for (int vI = 0; vI < DBCb_fixed.size(); ++vI) {
                if (DBCb_fixed[vI]) {
                    std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(nodeAttr.Get_Unchecked(vI)).setZero();
                }
            }
        }
        else {
            // project rhs for Dirichlet boundary condition
            DBC.Par_Each([&](int id, auto data) {
              auto &[dbcI] = data;
              std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::g>(nodeAttr.Get_Unchecked(dbcI(0))).setZero();
            });
            std::cout << "project rhs for Dirichlet boundary condition " << DBC.size << std::endl;
        }
        nodeAttr.Par_Each([&](int id, auto data) {
            auto &[x0, v, g, m] = data;
            rhs[id * dim] = -g[0];
            rhs[id * dim + 1] = -g[1];
            if constexpr (dim == 3) {
                rhs[id * dim + 2] = -g[2];
            }
        });

        // compute Hessian
        if (!useGD) {
            std::vector<Eigen::Triplet<T>> triplets;
            energy.Compute_IncPotential_Hessian(Elem, h, X, nodeAttr, elemAttr, elasticityAttr,
                constraintSet, dHat2, kappa, triplets);
            if (withCollision && mu > 0) {
                Compute_Friction_Hessian(X, Xn, fricConstraintSet, closestPoint, tanBasis, normalForce, epsv2 * h * h, mu, true, triplets);
            }
            if (bdstiff) {
                Compute_BD_Hessian(Elem, h, K, bdstiff, DBCb, X, nodeAttr, elemAttr, elasticityAttr, true, triplets);
            }
            if (DBCStiff) {
                Compute_DBC_Hessian(X, nodeAttr, DBC, DBCStiff, triplets);
            }
            sysMtr.Construct_From_Triplet(X.size * dim, X.size * dim, triplets);
            // project Matrix for Dirichlet boundary condition
            if (DBCStiff) {
                sysMtr.Project_DBC(DBCb_fixed, dim);
            }
            else {
                sysMtr.Project_DBC(DBCb, dim);
            }
        }

        // compute search direction
        {
            TIMER_FLAG("linearSolve");

            if (useGD) {
                printf("use gradient descent\n");
                std::memcpy(sol.data(), rhs.data(), sizeof(T) * rhs.size());
            }
            else {
                // direct factorization
                if(!Solve_Direct(sysMtr, rhs, sol)) {
                    useGD = true;
                    printf("use gradient descent\n");
                    std::memcpy(sol.data(), rhs.data(), sizeof(T) * rhs.size());
                }
            }
        }

        // line search
        MESH_NODE<T, dim> Xprev;
        Append_Attribute(X, Xprev);
        T alpha = 1.0, E;
        if (PARAMETER::Get("Elasticity_model", std::string("")) != std::string("FCR")) {
            Compute_Inversion_Free_StepSize<T, dim>(X, Elem, sol, alpha);
            printf("inversion free step size = %le\n", alpha);
        }
        if (bdstiff) {
            MESH_NODE<T, dim> Xprev;
            Append_Attribute(X, Xprev);
            T E;
            bool valid = false; // whether the current iterate x is feasible
            do {
                X.Join(Xprev).Par_Each([&](int id, auto data) {
                    auto &[x, xprev] = data;
                    x[0] = xprev[0] + alpha * sol[id * dim];
                    x[1] = xprev[1] + alpha * sol[id * dim + 1];
                    if constexpr (dim == 3) {
                        x[2] = xprev[2] + alpha * sol[id * dim + 2];
                    }
                });
                valid = Compute_BD_Energy(Elem, h, K, bdstiff, DBCb, X, nodeAttr, elemAttr, elasticityAttr, E);
                if (!valid) {
                    alpha /= 2.0;
                }
                printf("backtracking alpha for bounded distortion: %le\n", alpha);
            } while (!valid);
            Xprev.deep_copy_to(X);
            printf("bounded distortion feasible alpha = %le\n", alpha);
        }
        if (withCollision) {
            Compute_Intersection_Free_StepSize<T, dim, shell, elasticIPC>(X, boundaryNode, boundaryEdge, boundaryTri, 
                std::vector<int>(), std::vector<VECTOR<int, 2>>(), std::map<int, std::set<int>>(), 
                VECTOR<int, 2>(boundaryNode.size(), boundaryNode.size()),
                DBCb, sol, T(0), alpha); // CCD
            printf("intersection free step size = %le\n", alpha);
        }
        T alpha_feasible = alpha;

        do {
            X.Join(Xprev).Par_Each([&](int id, auto data) {
                auto &[x, xprev] = data;
                x[0] = xprev[0] + alpha * sol[id * dim];
                x[1] = xprev[1] + alpha * sol[id * dim + 1];
                if constexpr (dim == 3) {
                    x[2] = xprev[2] + alpha * sol[id * dim + 2];
                }
            });
            Compute_Deformation_Gradient(X, Elem, elemAttr, elasticityAttr);
            if (withCollision) {
                Compute_Constraint_Set<T, dim, shell, elasticIPC>(X, nodeAttr, boundaryNode, boundaryEdge, boundaryTri, 
                    std::vector<int>(), std::vector<VECTOR<int, 2>>(), std::map<int, std::set<int>>(), BNArea, BEArea, BTArea,
                    VECTOR<int, 2>(boundaryNode.size(), boundaryNode.size()),
                    DBCb, dHat2, T(0), false, constraintSet, constraintSetPTEE, stencilInfo);
            }
            energy.Compute_IncPotential(Elem, gravity, h, X, Xtilde, nodeAttr, elemAttr, elasticityAttr,
                constraintSet, dHat2, kappa, E);
            if (withCollision && mu > 0) {
                Compute_Friction_Potential(X, Xn, fricConstraintSet, closestPoint, tanBasis, normalForce, epsv2 * h * h, mu, E);
            }
            if (bdstiff) {
                Compute_BD_Energy(Elem, h, K, bdstiff, DBCb, X, nodeAttr, elemAttr, elasticityAttr, E);
            }
            if (DBCStiff) {
                Compute_DBC_Energy(X, nodeAttr, DBC, DBCStiff, E);
            }
            alpha /= 2.0;
            printf("E = %le, Eprev = %le, alpha = %le\n", E, Eprev, alpha * 2.0);
        } while (E > Eprev);
        printf("alpha = %le\n", alpha * 2.0);
        Eprev = E;

        if constexpr (!elasticIPC) {
            if (constraintSet_prev.size()) {
                T minDist2;
                std::vector<T> curdist2_prev;
                Compute_Min_Dist2<T, dim, elasticIPC>(X, constraintSet_prev, (T)0, curdist2_prev, minDist2);
                bool updateKappa = false;
                for (int i = 0; i < curdist2_prev.size(); ++i) {
                    if (dist2_prev[i] < 1e-18 && curdist2_prev[i] < 1e-18) {
                        updateKappa = true;
                        break;
                    }
                }
                if (updateKappa && kappa[0] < kappa[1]) {
                    kappa[0] *= 2;
                    kappaVec[0] *= 2;
                    energy.Compute_IncPotential(Elem, gravity, h, X, Xtilde, nodeAttr, elemAttr, elasticityAttr,
                                                constraintSet, dHat2, kappa, Eprev);
                    if (withCollision && mu > 0) {
                        Compute_Friction_Potential(X, Xn, fricConstraintSet, closestPoint, tanBasis, normalForce, epsv2 * h * h, mu, Eprev);
                    }
                    if (bdstiff) {
                        Compute_BD_Energy(Elem, h, K, bdstiff, DBCb, X, nodeAttr, elemAttr, elasticityAttr, Eprev);
                    }
                    if (DBCStiff) {
                        Compute_DBC_Energy(X, nodeAttr, DBC, DBCStiff, Eprev);
                    }
                }
            }
        }

        std::vector<T> dist2;
        if (constraintSet.size()) {
            T minDist2;
            Compute_Min_Dist2<T, dim, elasticIPC>(X, constraintSet, T(0), dist2, minDist2);
            printf("minDist2 = %le, kappa = %le, %le\n", minDist2, kappa[0], kappa[1]);
        }
        constraintSet_prev = constraintSet;
        dist2_prev = dist2;

        // stopping criteria
        infNorm = 0.0;
        for (int i = 0; i < sol.size(); ++i) {
            if (infNorm < std::abs(sol[i])) {
                infNorm = std::abs(sol[i]);
            }
        }
        printf("PNIter%d: Newton res = %le, tol = %le\n", PNIter++, infNorm / h, NewtonTol);

        if (useGD) {
            infNorm = NewtonTol * h * 10; // ensures not exit Newton loop
        }

        if (alpha * 2 < 1e-6 && alpha_feasible > 1e-6) {
            if (!useGD) {
                useGD = true;
                Eigen::VectorXd pe(sol.size()), mge(rhs.size());
                std::memcpy(pe.data(), sol.data(), sizeof(T) * sol.size());
                std::memcpy(mge.data(), rhs.data(), sizeof(T) * rhs.size());
                printf("-gdotp = %le, -gpcos = %le\n", mge.dot(pe),
                    mge.dot(pe) / std::sqrt(mge.squaredNorm() * pe.squaredNorm()));
                printf("linear solve relErr = %le\n",
                    std::sqrt((sysMtr.Get_Matrix() * pe - mge).squaredNorm() / mge.squaredNorm()));
            }
            else {
                printf("GD tiny step size!\n");
            }
        }
        else {
            useGD = false;
        }

        if (DBCStiff) {
            TIMER_FLAG("DBCStiff");
            T penaltyCur = 0;
            Compute_DBC_Dist2(X, DBC, penaltyCur);
            T progress = 1 - std::sqrt(penaltyCur / DBCPenaltyXn);
            printf("MDBC progress: %le, DBCStiff %le\n", progress, DBCStiff);
            MDBCProgressI.emplace_back(progress);
            if (MDBCProgressI.size() > 4) {
                MDBCProgressI.pop_front();
            }
            if(progress < 0.99) {
                //TODO: update Augmented Lagrangian parameters if necessary
                if (infNorm / h < NewtonTol * 10) {
                    if (DBCStiff < 1e8) {
                        DBCStiff *= 2;
                        energy.Compute_IncPotential(Elem, gravity, h, X, Xtilde, nodeAttr, elemAttr, elasticityAttr,
                                                    constraintSet, dHat2, kappa, Eprev);
                        if (withCollision && mu > 0) {
                            Compute_Friction_Potential(X, Xn, fricConstraintSet, closestPoint, tanBasis, normalForce, epsv2 * h * h, mu, Eprev);
                        }
                        if (bdstiff) {
                            Compute_BD_Energy(Elem, h, K, bdstiff, DBCb, X, nodeAttr, elemAttr, elasticityAttr, Eprev);
                        }
                        if (DBCStiff) {
                            Compute_DBC_Energy(X, nodeAttr, DBC, DBCStiff, Eprev);
                        }
                        printf("updated DBCStiff to %le\n", DBCStiff);
                    }
                }
                infNorm = NewtonTol * h * 10; // ensures not exit Newton loop
            }
            else {
                DBCStiff = 0;
                energy.Compute_IncPotential(Elem, gravity, h, X, Xtilde, nodeAttr, elemAttr, elasticityAttr,
                                            constraintSet, dHat2, kappa, Eprev);
                if (withCollision && mu > 0) {
                    Compute_Friction_Potential(X, Xn, fricConstraintSet, closestPoint, tanBasis, normalForce, epsv2 * h * h, mu, Eprev);
                }
                if (bdstiff) {
                    Compute_BD_Energy(Elem, h, K, bdstiff, DBCb, X, nodeAttr, elemAttr, elasticityAttr, Eprev);
                }
                printf("DBC moved to target, turn off Augmented Lagrangian\n");
            }
        }

        //TODO: DBC criteria, augLag updates, update Eprev
        //if (augLag_DBC) {
        //}
        if (infNorm / h < NewtonTol) {
            const auto energyPtr = std::dynamic_pointer_cast<SHAPE_MATCHING_ENERGY<T, dim>>(energy.energies.back());
            if (!energyPtr || !energyPtr->Keep_Going(X)) {
                break;
            } else {
                energy.Compute_IncPotential(Elem, gravity, h, X, Xtilde, nodeAttr, elemAttr, elasticityAttr,
                                            constraintSet, dHat2, kappa, Eprev);
            }
        }
    } while (1); //TODO: newtonTol relative to bbox

    if (withCollision) {
        printf("contact #: %lu\n", constraintSet.size());
        JGSL_FILE("ContactNum", constraintSet.size());
        PARAMETER::Set("Contact_Num", (int)constraintSet.size());
    }

    // update velocity
    X.Join(nodeAttr).Par_Each([&](int id, auto data) {
        auto &[x, x0, v, g, m] = data;
        v = (x - std::get<0>(Xn.Get_Unchecked(id))) / h;
    });

    const auto energyPtr = std::dynamic_pointer_cast<SHAPE_MATCHING_ENERGY<T, dim>>(energy.energies.back());
    if (energyPtr) {
        energyPtr->Print(Elem, X, output_folder, current_frame);
    }

    T v_inf = 0;
    X.Join(nodeAttr).Each([&](int id, auto data) {
        auto &[x, x0, v, g, m] = data;
        v_inf = std::max(v_inf, v.abs().max());
    });
    printf("velocity inf norm: %le\n", v_inf);

    if (PARAMETER::Get("Zero_velocity", false)) {
        X.Join(nodeAttr).Par_Each([&](int id, auto data) {
          auto &[x, x0, v, g, m] = data;
          v = VECTOR<T, dim>();
        });
    }

    if constexpr (dim == 2) {
        if (PARAMETER::Get("Output_Psi", std::string("")) == std::string("YES") &&
            PARAMETER::Get("Elasticity_model", std::string("")).substr(0, 2) == std::string("SD"))
        {
            std::string filePath = output_folder + "/distortion" + std::to_string(current_frame) + ".obj";
            SYMMETRIC_DIRICHLET_FUNCTOR<T, dim>::Output_Psi(filePath, X, Elem, elasticityAttr);
        }
    }

    JGSL_FILE("iteration", PNIter);
    return PNIter;
}

template <class T, int dim>
void multiply(std::vector<Eigen::Triplet<T>>& triplets, MESH_NODE<T, dim>& src, MESH_NODE_ATTR<T, dim>& dst) {
    dst.Each([&](int id, auto data) {
        auto &[x0, v, g, m] = data;
        g = VECTOR<T, dim>();
    });
    for (auto& tri : triplets) {
        int r = tri.row();
        int c = tri.col();
        T v = tri.value();
        std::get<2>(dst.Get_Unchecked(r / dim))(r % dim) += v * std::get<0>(src.Get_Unchecked(c / dim))(c % dim);
    }
}

template <class T, int dim = 3>
void Write_COM(
    MESH_NODE<T, dim>& X,
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    const std::string& filePath)
{
    VECTOR<T, dim> com;
    com.setZero();
    T m_sum = 0.0;
    X.Join(nodeAttr).Each([&](int id, auto data) {
        if (id >= 800) {
            auto &[x, x0, v, g, m] = data;
            com += x * m;
            m_sum += m;
        }
    });

    FILE *out = fopen(filePath.c_str(), "a+");
    if constexpr (dim == 3) {
        fprintf(out, "%.20lf %.20lf %.20lf\n", com[0] / m_sum, com[1] / m_sum, com[2] / m_sum);
    }
    else {
        //TODO
    }
    fclose(out);
}

template <class T, int dim>
void Check_Gradient_FEM(
    MESH_ELEM<dim>& Elem,
    const VECTOR_STORAGE<T, dim + 1>& DBC,
    const VECTOR<T, dim>& gravity,
    T dt,
    MESH_NODE<T, dim>& X,
    MESH_NODE<T, dim>& X0,
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    MESH_ELEM_ATTR<T, dim>& elemAttr,
    FIXED_COROTATED<T, dim>& elasticityAttr)
{
    ENERGY<T, dim> energy;
//    energy.Add(std::make_shared<ELASTICITY_ENERGY<T,dim>>());
//    energy.Add(std::make_shared<INERTIA_ENERGY<T,dim>>());
//    energy.Add(std::make_shared<IPC_ENERGY<T,dim>>());
    energy.Add(std::make_shared<SHAPE_MATCHING_ENERGY<T,dim>>());
    std::dynamic_pointer_cast<SHAPE_MATCHING_ENERGY<T, dim>>(energy.energies.back())->Precompute(Elem, X, X0, elasticityAttr, false);

    std::vector<VECTOR<int, dim + 1>> constraintSet;
    MESH_NODE<T, dim> Xtilde;
    Append_Attribute(X, Xtilde);
    Xtilde.Join(nodeAttr).Par_Each([&](int id, auto data) {
      auto &[x, x0, v, g, m] = data;
      x += dt * v + dt * dt * gravity;
    });
    T dHat2 = 0.0001, kappa[] = {1, 0}; //TODO: dHat as input and relative to bbox, adapt kappa

    MESH_NODE<T, dim> step;
    X.deep_copy_to(step);
    T l2 = 0;
    for (int i = 0; i < X.size; ++i) {
        for (int d = 0; d < dim; ++d)
            std::get<0>(step.Get_Unchecked(i))[d] = (T)rand() / RAND_MAX;
        l2 += std::get<0>(step.Get_Unchecked(i)).length2();
    }
    T l = std::sqrt(l2);
    for (int i = 0; i < X.size; ++i) {
        for (int d = 0; d < dim; ++d)
            std::get<0>(step.Get_Unchecked(i))[d] /= l;
    }

    T e0 = 0;
    Compute_Deformation_Gradient(X, Elem, elemAttr, elasticityAttr);
    energy.Compute_IncPotential(Elem, gravity, dt, X, Xtilde, nodeAttr, elemAttr, elasticityAttr, constraintSet, dHat2, kappa, e0);
    MESH_NODE_ATTR<T, dim> nodeAttr0;
    nodeAttr.deep_copy_to(nodeAttr0);
    energy.Compute_IncPotential_Gradient(Elem, gravity, dt, X, Xtilde, nodeAttr0, elemAttr, elasticityAttr, constraintSet, dHat2, kappa);
    std::vector<Eigen::Triplet<T>> triplets;
    energy.Compute_IncPotential_Hessian(Elem, dt, X, nodeAttr, elemAttr, elasticityAttr, constraintSet, dHat2, kappa, triplets);
    MESH_NODE_ATTR<T, dim> dnodeAttr0;
    nodeAttr.deep_copy_to(dnodeAttr0);
    multiply(triplets, step, dnodeAttr0);

    const int DIFF_SIZE = 10;
    std::vector<T> energy_difference(DIFF_SIZE);
    std::vector<T> energy_differential(DIFF_SIZE);
    std::vector<T> energy_err(DIFF_SIZE);
    std::vector<T> energy_log_err(DIFF_SIZE);
    std::vector<T> force_difference_norm(DIFF_SIZE);
    std::vector<T> force_differential_norm(DIFF_SIZE);
    std::vector<T> force_err(DIFF_SIZE);
    std::vector<T> force_log_err(DIFF_SIZE);
    std::setprecision(12);
    std::cout << "e0\t=" << std::setw(20) << e0 << std::endl;
    for (int i = 1; i <= 10; ++i) {
        T h = std::pow((T)(2), -i);
        MESH_NODE<T, dim> Xperturb(X.size);
        X.deep_copy_to(Xperturb);
        for (int idx = 0; idx < X.size; ++idx)
            for (int d = 0; d < dim; ++d)
                std::get<0>(Xperturb.Get_Unchecked(idx))[d] += std::get<0>(step.Get_Unchecked(idx))[d] * h;
        T e1 = 0;
        Compute_Deformation_Gradient(Xperturb, Elem, elemAttr, elasticityAttr);
        energy.Compute_IncPotential(Elem, gravity, dt, Xperturb, Xtilde, nodeAttr, elemAttr, elasticityAttr, constraintSet, dHat2, kappa, e1);
        std::cout << "e1\t=" << std::setw(20) << e1 << "\th = " << h << std::endl;
        MESH_NODE_ATTR<T, dim> nodeAttr1;
        nodeAttr.deep_copy_to(nodeAttr1);
        energy.Compute_IncPotential_Gradient(Elem, gravity, dt, Xperturb, Xtilde, nodeAttr1, elemAttr, elasticityAttr, constraintSet, dHat2, kappa);
        T difference = (e0 - e1) / h;
        T differential = 0;
        nodeAttr0.Join(nodeAttr1).Each([&](int id, auto data) {
            auto &[x0, v0, g0, m0, x0_, v1, g1, m1] = data;
            differential += -(g0 + g1).dot(std::get<0>(step.Get_Unchecked(id))) / (T)2;
        });

        T err = (difference - differential);
        T log_err = std::log(std::abs(err));
        energy_difference[i - 1] = difference;
        energy_differential[i - 1] = differential;
        energy_err[i - 1] = err;
        energy_log_err[i - 1] = log_err;

        energy.Compute_IncPotential_Hessian(Elem, dt, Xperturb, nodeAttr, elemAttr, elasticityAttr, constraintSet, dHat2, kappa, triplets);
        MESH_NODE_ATTR<T, dim> dnodeAttr1;
        nodeAttr.deep_copy_to(dnodeAttr1);
        multiply(triplets, step, dnodeAttr1);

        T force_difference = 0;
        T force_differential = 0;
        T err_force = 0;
        nodeAttr0.Join(nodeAttr1).Each([&](int id, auto data) {
            auto &[x0, v0, g0, m0, x0_, v1, g1, m1] = data;
            force_difference += (-(g0 - g1) / h).length2();
        });
        dnodeAttr0.Join(dnodeAttr1).Each([&](int id, auto data) {
            auto &[x0, v0, g0, m0, x0_, v1, g1, m1] = data;
            force_differential += ((g0 + g1) / 2).length2();
            auto& gg0 = std::get<2>(nodeAttr0.Get_Unchecked(id));
            auto& gg1 = std::get<2>(nodeAttr1.Get_Unchecked(id));
            err_force += (-(gg0 - gg1) / h - (g0 + g1) / 2).length2();
        });
        force_difference = std::sqrt(force_difference);
        force_differential = std::sqrt(force_differential);
        err_force = std::sqrt(err_force);
        T log_err_force = std::log(std::abs(err_force));
        force_difference_norm[i - 1] = force_difference;
        force_differential_norm[i - 1] = force_differential;
        force_err[i - 1] = err_force;
        force_log_err[i - 1] = log_err_force;
    }
    std::cout << std::setprecision(12) << "energy["
              << "i"
              << "] = " << std::setw(20) << "difference"
              << std::setw(20) << "differential"
              << std::setw(20) << "err"
              << std::setw(20) << "log_err"
              << std::endl;
    for (int i = 0; i < DIFF_SIZE; ++i) {
        std::cout << std::setprecision(12) << "energy[" << i << "] = " << std::setw(20) << energy_difference[i]
                  << std::setw(20) << energy_differential[i]
                  << std::setw(20) << energy_err[i]
                  << std::setw(20) << energy_log_err[i]
                  << std::endl;
    }
    std::cout << std::endl;
    std::cout << std::setprecision(12) << "force["
              << "i"
              << "] = " << std::setw(20) << "difference_norm"
              << std::setw(20) << "differential_norm"
              << std::setw(20) << "err"
              << std::setw(20) << "log_err"
              << std::endl;
    for (int i = 0; i < DIFF_SIZE; ++i) {
        std::cout << std::setprecision(12) << "force[" << i << "] = " << std::setw(20) << force_difference_norm[i]
                  << std::setw(20) << force_differential_norm[i]
                  << std::setw(20) << force_err[i]
                  << std::setw(20) << force_log_err[i]
                  << std::endl;
    }
    std::cin.get();
}

}
