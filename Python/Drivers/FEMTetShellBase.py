import sys
sys.path.insert(0, "../../build")
import os
try:
    os.mkdir("output")
except OSError:
    pass

from JGSL import *
from .SimulationBase import SimulationBase


class FEMTetShellBase(SimulationBase):
    def __init__(self, precision, dim):
        super().__init__(precision, dim)
        self.X0 = Storage.V2dStorage() if self.dim == 2 else Storage.V3dStorage()
        self.X = Storage.V2dStorage() if self.dim == 2 else Storage.V3dStorage()
        self.Elem = Storage.V3iStorage() if self.dim == 2 else Storage.V4iStorage()
        self.nodeAttr = Storage.V2dV2dV2dSdStorage() if self.dim == 2 else Storage.V3dV3dV3dSdStorage()
        self.elemAttr = Storage.M2dM2dSdStorage() if self.dim == 2 else Storage.M3dM3dSdStorage()
        self.elasticity = FIXED_COROTATED_2.Create() if self.dim == 2 else FIXED_COROTATED_3.Create() #TODO: different material switch
        self.DBC = Storage.V3dStorage() if self.dim == 2 else Storage.V4dStorage()
        self.DBCMotion = Storage.V2iV2dV2dV2dSdStorage() if self.dim == 2 else Storage.V2iV3dV3dV3dSdStorage()
        self.gravity = self.Vec(0, -9.81) if self.dim == 2 else self.Vec(0, -9.81, 0)
        if self.dim == 3:
            self.TriVI2TetVI = Storage.SiStorage() #TODO: together?
            self.Tri = Storage.V3iStorage()
        self.PNIterCount = 0
        self.PNTol = 1e-3
        self.withCollision = True

    def set_object(self, filePath, velocity, p_density, E, nu):
        if self.dim == 2:
            meshCounter = MeshIO.Read_TriMesh_Obj(filePath, self.X0, self.Elem)
        else:
            meshCounter = MeshIO.Read_TetMesh_Vtk(filePath, self.X0, self.Elem)
            MeshIO.Find_Surface_TriMesh(self.X0, self.Elem, self.TriVI2TetVI, self.Tri)
        MeshIO.Append_Attribute(self.X0, self.X)
        vol = Storage.SdStorage()
        FEM.Compute_Vol_And_Inv_Basis(self.X0, self.Elem, vol, self.elemAttr)
        if self.dim == 2:
            FIXED_COROTATED_2.Append_FEM(self.elasticity, meshCounter, vol, E, nu)
        else:
            FIXED_COROTATED_3.Append_FEM(self.elasticity, meshCounter, vol, E, nu)
        FEM.Compute_Mass_And_Init_Velocity(self.X0, self.Elem, vol, p_density, velocity, self.nodeAttr)

    def set_DBC(self, DBCRangeMin, DBCRangeMax, v, rotCenter, rotAxis, angVelDeg):
        FEM.Init_Dirichlet(self.X0, DBCRangeMin, DBCRangeMax,  v, rotCenter, rotAxis, angVelDeg, self.DBC, self.DBCMotion)

    #TODO: def add_object for multi-object system

    def advance_one_time_step(self, dt):
        #TODO: self.tol
        FEM.Step_Dirichlet(self.DBCMotion, dt, self.DBC)
        # FEM.TimeStepper.ImplicitEuler.Check_Gradient_FEM(self.Elem, self.DBC, \
        #     self.gravity, self.dt, self.X, self.X0, self.nodeAttr, self.elemAttr, self.elasticity)
        self.PNIterCount += FEM.TimeStepper.ImplicitEuler.Advance_One_Step_Shell(self.Elem, self.DBC, \
            self.gravity, self.dt, self.PNTol, self.withCollision, self.X, self.X0, self.nodeAttr, self.elemAttr, self.elasticity)
        print("Total PN iteration count: ", self.PNIterCount, "\n")

    def write(self, frame_idx):
        MeshIO.Write_TriMesh_Obj(self.X, self.Elem, self.output_folder + str(frame_idx) + ".obj") if self.dim == 2 \
            else MeshIO.Write_Surface_TriMesh_Obj(self.X, self.TriVI2TetVI, self.Tri, \
                self.output_folder + str(frame_idx) + ".obj")