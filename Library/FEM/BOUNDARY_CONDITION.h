#pragma once
//#####################################################################
// Function Init_Dirichlet
//#####################################################################
#include <Utils/MESHIO.h>

namespace py = pybind11;
namespace JGSL {

template<class T, int dim>
using DBC_MOTION = BASE_STORAGE<VECTOR<int, 2>, VECTOR<T, dim>, VECTOR<T, dim>, VECTOR<T, dim>, T, T>; // range, v, rotCenter, rotAxis, angVelDeg, scaleVel

template <std::size_t OFFSET, class T, int dim>
struct FIELDS_WITH_OFFSET<OFFSET, DBC_MOTION<T, dim>> {
    enum INDICES { range = OFFSET, v, rotCenter, rotAxis, angVelDeg, scaleVel };
};

//#####################################################################
// Function Init_Dirichlet
//#####################################################################
template <class T, int dim>
void Init_Dirichlet(MESH_NODE<T, dim>& X,
    const VECTOR<T, dim>& relBoxMin,
    const VECTOR<T, dim>& relBoxMax,
    const VECTOR<T, dim>& v,
    const VECTOR<T, dim>& rotCenter,
    const VECTOR<T, dim>& rotAxis,
    T angVelDeg,
    T scaleVel,
    VECTOR_STORAGE<T, dim + 1>& DBC,
    DBC_MOTION<T, dim>& DBCMotion,
    const VECTOR<int, 4>& vIndRange = VECTOR<int, 4>(0, 0, __INT_MAX__, -1))
{
    if (!X.size) {
        puts("no nodes in the model!");
        exit(-1);
    }

    VECTOR<T, dim> bboxMin;
    VECTOR<T, dim> bboxMax;
    X.Each([&](int id, auto data) {
        if (id >= vIndRange[0] && id < vIndRange[2]) {
            auto &[x] = data;
            if (id == vIndRange[0]) {
                bboxMin = x;
                bboxMax = x;
            }
            else {
                for (int dimI = 0; dimI < dim; ++dimI) {
                    if (bboxMax(dimI) < x(dimI)) {
                        bboxMax(dimI) = x(dimI);
                    }
                    if (bboxMin(dimI) > x(dimI)) {
                        bboxMin(dimI) = x(dimI);
                    }
                }
            }
        }
    });

    VECTOR<T, dim> rangeMin = relBoxMin;
    VECTOR<T, dim> rangeMax = relBoxMax;
    VECTOR<T, dim> rangeMid;
    for (int dimI = 0; dimI < dim; ++dimI) {
        rangeMin(dimI) *= bboxMax(dimI) - bboxMin(dimI);
        rangeMin(dimI) += bboxMin(dimI);
        rangeMax(dimI) *= bboxMax(dimI) - bboxMin(dimI);
        rangeMax(dimI) += bboxMin(dimI);
    }
    rangeMid = (rangeMax + rangeMin) * 0.5;

    std::cout << "DBC node inds: ";
    VECTOR<int, 2> range;
    range[0] = DBC.size;
    range[1] = -1;
    // special treatment for the same set
    DBC.Each([&](int id, auto data) {
        auto &[x] = data;
        if constexpr (dim == 3) {
            if (x(1) >= rangeMin(0) && x(1) <= rangeMax(0) &&
                x(2) >= rangeMin(1) && x(2) <= rangeMax(1) &&
                x(3) >= rangeMin(2) && x(3) <= rangeMax(2))
            {
                range[0] = std::min(range[0], id);
                range[1] = std::max(range[1], id + 1);
            }
        } else {
            if (x(1) >= rangeMin(0) && x(1) <= rangeMax(0) &&
                x(2) >= rangeMin(1) && x(2) <= rangeMax(1))
            {
                range[0] = std::min(range[0], id);
                range[1] = std::max(range[1], id + 1);
            }
        }
    });
    if (range[1] == -1) {
        int DBCCount = DBC.size;
        X.Each([&](int id, auto data) {
            if (id >= vIndRange[0] && id < vIndRange[2]) {
                auto &[x] = data;
                if constexpr (dim == 3) {
                    if (x(0) >= rangeMin(0) && x(0) <= rangeMax(0) &&
                        x(1) >= rangeMin(1) && x(1) <= rangeMax(1) &&
                        x(2) >= rangeMin(2) && x(2) <= rangeMax(2))
                    {
                        T r = (x - rangeMid).length();
                        if ((PARAMETER::Get("Dirichlet_circle", (T)-1) < 0 || r <= PARAMETER::Get("Dirichlet_circle", (T)-1)) &&
                            (PARAMETER::Get("Dirichlet_zero_ring", (T)-1) < 0 || x.length() >= PARAMETER::Get("Dirichlet_zero_ring", (T)-1)))
                            DBC.Insert(DBCCount++, VECTOR<T, dim + 1>(id, x(0), x(1), x(2)));
                        std::cout << " " << id;
                    }
                }
                else {
                    if (x(0) >= rangeMin(0) && x(0) <= rangeMax(0) &&
                        x(1) >= rangeMin(1) && x(1) <= rangeMax(1))
                    {
                        T r = (x - rangeMid).length();
                        if ((PARAMETER::Get("Dirichlet_circle", (T)-1) < 0 || r <= PARAMETER::Get("Dirichlet_circle", (T)-1)) &&
                            (PARAMETER::Get("Dirichlet_zero_ring", (T)-1) < 0 || x.length() >= PARAMETER::Get("Dirichlet_zero_ring", (T)-1)))
                            DBC.Insert(DBCCount++, VECTOR<T, dim + 1>(id, x(0), x(1)));
                        std::cout << " " << id;
                    }
                }
            }
        });
        range[1] = DBCCount;
    }
    DBCMotion.Insert(DBCMotion.size, range, v, rotCenter, rotAxis, angVelDeg, scaleVel);
    if constexpr (dim == 3) {
        printf("\nvelocity %le %le %le, rotCenter %le %le %le, rotAxis %le %le %le, angVelDeg %le\n",
            v[0], v[1], v[2], rotCenter[0], rotCenter[1], rotCenter[2], 
            rotAxis[0], rotAxis[1], rotAxis[2], angVelDeg);
    }
    else {
        printf("\nvelocity %le %le, rotCenter %le %le, rotAxis %le %le, angVelDeg %le\n",
            v[0], v[1], rotCenter[0], rotCenter[1], rotAxis[0], rotAxis[1], angVelDeg);
    }
}

template <class T, int dim>
void Step_Dirichlet(
    DBC_MOTION<T, dim>& DBCMotion,
    T h, VECTOR_STORAGE<T, dim + 1>& DBC)
{
    static VECTOR_STORAGE<T, dim + 1> rest_space;
    if (!rest_space.size) {
        DBC.deep_copy_to(rest_space);
    }
    T min_rx = std::numeric_limits<T>::max();
    int min_id = -1;

    DBCMotion.Each([&](int id, auto data) {
        auto &[range, v, rotCenter, rotAxis, angVelDeg, scaleVel] = data;

        if (PARAMETER::Get("rose_base_v", (T)0)) {
            T v = PARAMETER::Get("rose_base_v", (T)0);
            T r = PARAMETER::Get("rose_base_r", (T)0.005);
            static T current_t = h;
            if (min_id < 0) {
                for (int i = range[0]; i < range[1]; ++i) {
                    VECTOR<T, dim + 1> &dbcI = std::get<0>(DBC.Get_Unchecked(i));
                    VECTOR<T, dim + 1> &rest_dbcI = std::get<0>(rest_space.Get_Unchecked(i));
                    T rx = rest_dbcI[1];
                    T rz = rest_dbcI[3];
                    if (rx > v * current_t && rx < min_rx) {
                        min_rx = rx;
                        min_id = i;
                    }
                }
            }
            if (min_id < 0) return;
            VECTOR<T, dim + 1> &center = std::get<0>(DBC.Get_Unchecked(min_id));
            VECTOR<T, dim + 1> &rest_center = std::get<0>(rest_space.Get_Unchecked(min_id));
            T delta = std::atan2(center[3] - r, center[1]) + 0.5 * M_PI;

            printf("Found %.10f %.10f\n", min_rx, rest_center[1]);

            for (int i = range[0]; i < range[1]; ++i) {
                VECTOR<T, dim + 1> &dbcI = std::get<0>(DBC.Get_Unchecked(i));
                VECTOR<T, dim + 1> &rest_dbcI = std::get<0>(rest_space.Get_Unchecked(i));
                if (rest_dbcI[1] <= rest_center[1]) continue;
                T dx = rest_dbcI[1] - rest_center[1];
                T dz = rest_dbcI[3] - rest_center[3];
                T rotated_dx = std::cos(delta) * dx - std::sin(delta) * dz;
                T rotated_dz = std::sin(delta) * dx + std::cos(delta) * dz;
                dbcI[1] = center[1] + rotated_dx;
                dbcI[3] = center[3] + rotated_dz;
            }
            current_t += h;
            return;
        }

        //TODO: parallel the following loop
        bool first = true;
        VECTOR<T, dim> bboxMin;
        VECTOR<T, dim> bboxMax;
        for (int i = range[0]; i < range[1]; ++i) {
            VECTOR<T, dim + 1>& dbcI = std::get<0>(DBC.Get_Unchecked(i));
            VECTOR<T, dim> x;
            for (int dimI = 0; dimI < dim; ++dimI)
                x[dimI] = dbcI[dimI + 1];
            if (first) {
                first = false;
                bboxMin = x;
                bboxMax = x;
            }
            else {
                for (int dimI = 0; dimI < dim; ++dimI) {
                    if (bboxMax(dimI) < x(dimI)) {
                        bboxMax(dimI) = x(dimI);
                    }
                    if (bboxMin(dimI) > x(dimI)) {
                        bboxMin(dimI) = x(dimI);
                    }
                }
            }
        }
        VECTOR<T, dim> bboxMid = (bboxMin + bboxMax) * 0.5;
        for (int i = range[0]; i < range[1]; ++i) {
            VECTOR<T, dim + 1>& dbcI = std::get<0>(DBC.Get_Unchecked(i));
            if (scaleVel) {
                for (int dimI = 0; dimI <= 0; ++dimI)
                    dbcI[dimI + 1] = bboxMid[dimI] + (dbcI[dimI + 1] - bboxMid[dimI]) * (1. + scaleVel);
                for (int dimI = 2; dimI <= 2; ++dimI)
                    dbcI[dimI + 1] = bboxMid[dimI] + (dbcI[dimI + 1] - bboxMid[dimI]) * (1. + scaleVel);
            }

            if (angVelDeg) {
                if constexpr (dim == 2) {
                    T rotAngRad = angVelDeg / 180 * M_PI * h;
                    MATRIX<T, dim> rotMtr;
                    rotMtr(0, 0) = std::cos(rotAngRad);
                    rotMtr(0, 1) = -std::sin(rotAngRad);
                    rotMtr(1, 0) = -rotMtr(0, 1);
                    rotMtr(1, 1) = rotMtr(0, 0);

                    VECTOR<T, dim> x(dbcI[1] - rotCenter[0], dbcI[2] - rotCenter[1]);
                    VECTOR<T, dim> rotx = rotMtr * x;
                    dbcI[1] = rotx[0] + rotCenter[0];
                    dbcI[2] = rotx[1] + rotCenter[1];
                }
                else {
                    T rotAngRad = angVelDeg / 180 * M_PI * h;
                    const Eigen::Matrix3d rotMtr = Eigen::AngleAxis<double>(rotAngRad,
                        Eigen::Vector3d(rotAxis[0], rotAxis[1], rotAxis[2])).toRotationMatrix();
                    
                    const Eigen::Vector3d x(dbcI[1] - rotCenter[0], dbcI[2] - rotCenter[1], dbcI[3] - rotCenter[2]);
                    const Eigen::Vector3d rotx = rotMtr * x;
                    dbcI[1] = rotx[0] + rotCenter[0];
                    dbcI[2] = rotx[1] + rotCenter[1];
                    dbcI[3] = rotx[2] + rotCenter[2];
                }
            }

            dbcI[1] += v[0] * h;
            dbcI[2] += v[1] * h;
            if constexpr (dim == 3) {
                dbcI[3] += v[2] * h;
            }
        }
    });
}

template <class T, int dim>
void Reset_Dirichlet(
    MESH_NODE<T, dim>& X,
    VECTOR_STORAGE<T, dim + 1>& DBC)
{
    DBC.Par_Each([&](int id, auto data) {
        auto &[dbcI] = data;
        int vI = dbcI[0];
        const VECTOR<T, dim>& coord = std::get<0>(X.Get_Unchecked(vI));
        dbcI[1] = coord[0];
        dbcI[2] = coord[1];
        if constexpr (dim == 3) {
            dbcI[3] = coord[2];
        }
    });
}

template <class T, int dim = 3>
void Load_Dirichlet(
    const std::string& filePath,
    int vIndOffset,
    const VECTOR<T, dim>& translate,
    VECTOR_STORAGE<T, dim + 1>& DBC)
{
    MESH_NODE<T, dim> X;
    MESH_ELEM<2> triangles;
    VECTOR<int, 4> counter = Read_TriMesh_Obj(filePath, X, triangles);
    if (counter[0] < 0) {
        return;
    }
    DBC.Par_Each([&](int id, auto data) {
        auto &[dbcI] = data;
        int vI = dbcI[0] - vIndOffset;
        const VECTOR<T, dim>& coord = std::get<0>(X.Get_Unchecked(vI));
        dbcI[1] = coord[0] + translate[0];
        dbcI[2] = coord[1] + translate[1];
        if constexpr (dim == 3) {
            dbcI[3] = coord[2] + translate[2];
        }
    });
}

template <class T, int dim = 3>
void Boundary_Dirichlet(
    MESH_NODE<T, dim>& X,
    MESH_ELEM<dim - 1>& Tri,
    VECTOR_STORAGE<T, dim + 1>& DBC)
{
    std::vector<int> boundaryNode;
    std::vector<VECTOR<int, 2>> boundaryEdge;
    Find_Boundary_Edge_And_Node(X.size, Tri, boundaryNode, boundaryEdge);
    
    for (const auto& vI : boundaryNode) {
        const VECTOR<T, dim>& x = std::get<0>(X.Get_Unchecked(vI));
        DBC.Append(VECTOR<T, dim + 1>(vI, x[0], x[1], x[2]));
    }
}

template <class T, int dim>
void Turn_Dirichlet(
    DBC_MOTION<T, dim>& DBCMotion)
{
    DBCMotion.Each([&](int id, auto data) {
        auto &[range, v, rotCenter, rotAxis, angVelDeg, scaleVel] = data;
        v = -v;
    });
}

template <class T, int dim>
void Magnify_Body_Force(
    MESH_NODE<T, dim>& X, // mid-surface node coordinates
    const VECTOR<T, dim>& relBoxMin,
    const VECTOR<T, dim>& relBoxMax,
    T magnifyFactor,
    std::vector<T>& b)
{
    DBC_MOTION<T, dim> DBCMotion;
    VECTOR_STORAGE<T, dim + 1> DBC;
    Init_Dirichlet<T, dim>(X, relBoxMin, relBoxMax, VECTOR<T, dim>(0),
        VECTOR<T, dim>(0), VECTOR<T, dim>(0), 0, 0, DBC, DBCMotion);
    DBC.Each([&](int id, auto data) {
        auto &[dbcI] = data;
        int vI = dbcI(0);

        b[vI * dim] *= magnifyFactor;
        b[vI * dim + 1] *= magnifyFactor;
        if constexpr (dim == 3) {
            b[vI * dim + 2] *= magnifyFactor;
        }
    });
}

template <class T, int dim>
void Pop_Back_Dirichlet(
    DBC_MOTION<T, dim>& DBCMotion,
    VECTOR_STORAGE<T, dim + 1>& DBC)
{
    int counter = 0;
    DBCMotion.Each([&](int id, auto data) {
        auto &[range, v, rotCenter, rotAxis, angVelDeg, scaleVel] = data;
        if (counter == DBCMotion.size - 1) {
            for (int i = range[1] - 1; i >= range[0]; --i) {
                DBC.Remove(i);
            }
        }
        ++counter;
    });
    DBCMotion.Remove(DBCMotion.size - 1);
}

std::vector<VECTOR<double, 3>> done;
std::vector<VECTOR<double, 3>> doing;
double start_x, start_y;
void Interact_Add(MESH_NODE<double, 2>& X, double x, double y) {
    start_x = x;
    start_y = y;
    int cnt = 0;
    X.Each([&](int id, auto data) {
        auto& [v] = data;
        if (VECTOR<double, 2>(v(0) - x, v(1) - y).length() < 0.02) {
            ++cnt;
            doing.emplace_back(id, v(0), v(1));
        }
    });
    printf(">>> %d points added.\n", cnt);
}

void Interact_Move(double x, double y) {
    double delta_x = x - start_x;
    double delta_y = y - start_y;
    for (auto& i : doing) {
        i(1) += delta_x;
        i(2) += delta_y;
    }
    start_x = x;
    start_y = y;
    printf(">>> %d points doing. %d points done.\n", (int)doing.size(), (int)done.size());
}

void Interact_Done() {
    for (auto& i : doing)
        done.push_back(i);
    doing.clear();
}

void Retrive_DBC(VECTOR_STORAGE<double, 3>& DBC) {
    DBC = VECTOR_STORAGE<double, 3>();
    DBC.Reserve(doing.size() + done.size());
    int DBCCount = 0;
    for (auto& i : doing) DBC.Insert(DBCCount++, i);
    for (auto& i : done) DBC.Insert(DBCCount++, i);
}

std::vector<VECTOR<double, 4>> done_3d;
template <class T>
void Interact_Add_3D(BASE_STORAGE<VECTOR<T, 3>>& nodes, BASE_STORAGE<int>& TriVI2TetVI, BASE_STORAGE<VECTOR<int, 3>>& faces, int picked) {
    if (picked >= 0) {
        for (int i = 0; i < 3; ++i) {
            int TriVI = std::get<0>(faces.Get_Unchecked(picked))(i);
            int TetVI = std::get<0>(TriVI2TetVI.Get_Unchecked(TriVI));
            VECTOR<T, 3> pos = std::get<0>(nodes.Get_Unchecked(TetVI));
            int exist = -1;
            for (int i = 0; i < done_3d.size(); ++i)
                if ((int)done_3d[i](0) == (int)TetVI)
                    exist = i;
            if (exist == -1) {
                done_3d.emplace_back(TetVI, pos(0) + 0.01, pos(1), pos(2));
            } else {
                done_3d[exist](1) += 0.01;
            }
        }
    }
}
void Retrive_DBC_3D(VECTOR_STORAGE<double, 4>& DBC) {
    DBC = VECTOR_STORAGE<double, 4>();
    DBC.Reserve(done_3d.size());
    int DBCCount = 0;
    for (auto& i : done_3d) DBC.Insert(DBCCount++, i);
}


//#####################################################################
void Export_Boundary_Condition(py::module& m) {
    m.def("Init_Dirichlet", &Init_Dirichlet<double, 2>);
    m.def("Init_Dirichlet", &Init_Dirichlet<double, 3>);

    m.def("Step_Dirichlet", &Step_Dirichlet<double, 2>);
    m.def("Step_Dirichlet", &Step_Dirichlet<double, 3>);

    m.def("Reset_Dirichlet", &Reset_Dirichlet<double, 2>);
    m.def("Reset_Dirichlet", &Reset_Dirichlet<double, 3>);

    m.def("Turn_Dirichlet", &Turn_Dirichlet<double, 2>);
    m.def("Turn_Dirichlet", &Turn_Dirichlet<double, 3>);

    m.def("Load_Dirichlet", &Load_Dirichlet<double, 2>);
    m.def("Load_Dirichlet", &Load_Dirichlet<double, 3>);

    m.def("Boundary_Dirichlet", &Boundary_Dirichlet<double, 3>);

    m.def("Magnify_Body_Force", &Magnify_Body_Force<double, 2>);
    m.def("Magnify_Body_Force", &Magnify_Body_Force<double, 3>);

    m.def("Pop_Back_Dirichlet", &Pop_Back_Dirichlet<double, 2>);
    m.def("Pop_Back_Dirichlet", &Pop_Back_Dirichlet<double, 3>);

    m.def("Interact_Add", &Interact_Add);
    m.def("Interact_Move", &Interact_Move);
    m.def("Interact_Done", &Interact_Done);
    m.def("Retrive_DBC", &Retrive_DBC);

    m.def("Interact_Add_3D", &Interact_Add_3D<double>);
    m.def("Retrive_DBC_3D", &Retrive_DBC_3D);
}

}
