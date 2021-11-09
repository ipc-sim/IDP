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
                        if (PARAMETER::Get("Dirichlet_circle", (T)-1) < 0 || r <= PARAMETER::Get("Dirichlet_circle", (T)-1))
                            DBC.Insert(DBCCount++, VECTOR<T, dim + 1>(id, x(0), x(1), x(2)));
                        std::cout << " " << id;
                    }
                }
                else {
                    if (x(0) >= rangeMin(0) && x(0) <= rangeMax(0) &&
                        x(1) >= rangeMin(1) && x(1) <= rangeMax(1))
                    {
                        T r = (x - rangeMid).length();
                        if (PARAMETER::Get("Dirichlet_circle", (T)-1) < 0 || r <= PARAMETER::Get("Dirichlet_circle", (T)-1))
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
    DBCMotion.Each([&](int id, auto data) {
        auto &[range, v, rotCenter, rotAxis, angVelDeg, scaleVel] = data;
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
                for (int dimI = 0; dimI < dim; ++dimI)
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

    m.def("Magnify_Body_Force", &Magnify_Body_Force<double, 2>);
    m.def("Magnify_Body_Force", &Magnify_Body_Force<double, 3>);

    m.def("Pop_Back_Dirichlet", &Pop_Back_Dirichlet<double, 2>);
    m.def("Pop_Back_Dirichlet", &Pop_Back_Dirichlet<double, 3>);
}

}
