import sys
sys.path.insert(0, "../../build")
import os
try:
    os.mkdir("output")
except OSError:
    pass

from JGSL import *
from .SimulationBase import SimulationBase


class FEMShellBase(SimulationBase):
    def __init__(self, precision, dim):
        super().__init__(precision, dim)
        self.X = Storage.V2dStorage() if self.dim == 2 else Storage.V3dStorage()
        self.Elem = Storage.V2iStorage() if self.dim == 2 else Storage.V3iStorage()
        self.X_mesh = Storage.V2dStorage() if self.dim == 2 else Storage.V3dStorage()
        self.Elem_mesh = Storage.V3iStorage()
        self.nodeAttr = Storage.V2dV2dV2dSdStorage() if self.dim == 2 else Storage.V3dV3dV3dSdStorage()
        self.nodeAttr_mesh = Storage.V2dV2dV2dSdStorage() if self.dim == 2 else Storage.V3dV3dV3dSdStorage()
        self.massMatrix = CSR_MATRIX_D()
        self.elemAttr = Storage.M2dM2dSdStorage() if self.dim == 2 else Storage.M3dM3dSdStorage()
        self.elasticity = FIXED_COROTATED_2.Create() if self.dim == 2 else FIXED_COROTATED_3.Create() #TODO: different material switch
        self.DBC = Storage.V3dStorage() if self.dim == 2 else Storage.V4dStorage()
        self.DBCMotion = Storage.V2iV2dV2dV2dSdStorage() if self.dim == 2 else Storage.V2iV3dV3dV3dSdStorage()
        self.gravity = self.Vec(0, -9.81) if self.dim == 2 else self.Vec(0, -9.81, 0)
        self.bodyForce = StdVectorXd()
        self.gamma = StdVectorXd()
        self.lambdaq = StdVectorXd()
        self.withCollision = False
        self.PNIterCount = 0
        self.PNTol = 1e-3
        self.dHat2 = 1e-6
        self.kappa = Vector2d(1e5, 0)
        self.staticSolve = False
        self.t = 0.0
        self.MDBC_tmax = 1e10

    def add_shell(self, shellType, numElem, length, thickness0, thickness1, translate, rotDeg): # 2D
        FEM.Shell.Add_Shell(shellType, numElem, length, thickness0, thickness1, translate, rotDeg, \
            self.X, self.Elem, self.X_mesh, self.Elem_mesh, self.nodeAttr_mesh)

    def add_shell_3D(self, filePath, thickness, translate): # 3D
        FEM.Shell.Add_Shell(filePath, thickness, translate, \
            self.X, self.Elem, self.X_mesh, self.Elem_mesh, self.nodeAttr_mesh)

    def initialize(self, p_density, E, nu, gammaAmt, lambdaAmt, caseI):
        if caseI != 0:
            self.gravity = self.Vec(0, 0) if self.dim == 2 else self.Vec(0, 0, 0)
        FEM.Shell.Initialize_Shell(p_density, E, nu, gammaAmt, lambdaAmt, \
            self.gamma, self.lambdaq, self.X, self.Elem, \
            self.nodeAttr, self.massMatrix, self.gravity, self.bodyForce, self.elemAttr, self.elasticity)
        FEM.Shell.Initialize_Displacement(self.X, self.Elem, caseI)

    def set_DBC(self, DBCRangeMin, DBCRangeMax, v, rotCenter, rotAxis, angVelDeg):
        FEM.Shell.Init_Dirichlet(self.X, self.X_mesh, DBCRangeMin, DBCRangeMax,  v, rotCenter, rotAxis, angVelDeg, self.DBC, self.DBCMotion)

    def advance_one_time_step(self, dt):
        #TODO: self.tol
        if self.t < self.MDBC_tmax:
            FEM.Step_Dirichlet(self.DBCMotion, dt, self.DBC)
        self.PNIterCount = self.PNIterCount + FEM.Shell.Advance_One_Step_IE(self.gamma, self.lambdaq, self.Elem, self.DBC, \
            self.bodyForce, self.dt, self.PNTol, self.withCollision, self.dHat2, self.kappa, self.staticSolve, \
            self.X, self.nodeAttr, self.massMatrix, self.elemAttr, self.elasticity, \
            self.X_mesh, self.Elem_mesh, self.nodeAttr_mesh)
        # symplectic Euler:
        # FEM.Shell.Advance_One_Step_SE(self.gamma, self.lambdaq, self.Elem, self.DBC, \
        #     self.bodyForce, self.dt, \
        #     self.X, self.nodeAttr, self.massMatrix, self.elemAttr, self.elasticity)
        self.t += dt
        print("Total PN iteration count: ", self.PNIterCount, "\n")

    def write(self, frame_idx):
        FEM.Shell.Update_Render_Mesh_Node(self.X, self.X_mesh)
        MeshIO.Write_TriMesh_Obj(self.X_mesh, self.Elem_mesh, self.output_folder + str(frame_idx) + ".obj")

    def write_node_coordinate(self, filePath):
        FEM.Shell.Write_Node_Coordinate(self.X_mesh, filePath)