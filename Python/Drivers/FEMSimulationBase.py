import sys
sys.path.insert(0, "../../build")
import os
try:
    os.mkdir("output")
except OSError:
    pass

from JGSL import *
from .SimulationBase import SimulationBase


class FEMSimulationBase(SimulationBase):
    def __init__(self, precision, dim, model="FCR"):
        super().__init__(precision, dim)
        self.X0 = Storage.V2dStorage() if self.dim == 2 else Storage.V3dStorage()
        self.X = Storage.V2dStorage() if self.dim == 2 else Storage.V3dStorage()
        self.Elem = Storage.V3iStorage() if self.dim == 2 else Storage.V4iStorage()
        self.Extra_Mesh = Storage.V3iStorage()
        self.nodeAttr = Storage.V2dV2dV2dSdStorage() if self.dim == 2 else Storage.V3dV3dV3dSdStorage()
        self.elemAttr = Storage.M2dM2dSdStorage() if self.dim == 2 else Storage.M3dM3dSdStorage()
        self.model = model
        Set_Parameter("Elasticity_model", self.model)
        self.elasticity = FIXED_COROTATED_2.Create() if self.dim == 2 else FIXED_COROTATED_3.Create() #TODO: different material switch
        self.withPlasticity = False
        self.DBC = Storage.V3dStorage() if self.dim == 2 else Storage.V4dStorage()
        self.DBCMotionList = []
        self.gravity = self.Vec(0, -9.81) if self.dim == 2 else self.Vec(0, -9.81, 0)
        if self.dim == 3:
            self.TriVI2TetVI = Storage.SiStorage() #TODO: together?
            self.Tri = Storage.V3iStorage()
        self.PNIterCountSU = [0] * 5
        self.PNIterCount = 0
        self.PNTol = 1e-2
        self.meshCounter = Vector4i()
        self.withCollision = True
        self.useNewton = True
        self.dHat2 = 1e-6
        self.EIPC = False
        self.kappa = Vector3d(1e4, 0, 0) #1e9 for 2d fracture
        self.mu = 0
        self.epsv2 = 1e-6
        self.staticSolve = False
        self.withShapeMatching = False
        self.enableFracture = False
        self.debrisAmt = -1
        self.strengthenFactor = 2
        self.edge_dupV = StdVectorArray4i() if self.dim == 2 else StdVectorArray6i()
        self.isFracture_edge = StdVectorXi()
        self.incTriV_edge = StdVectorVector4i()
        self.incTriRestDist2_edge = StdVectorXd()
        self.fractureRatio2 = 2
        self.finalV2old = StdVectorXi()
        self.rho0 = 0.0
        self.callback_func = None
        self.callback_frame = -1
        self.hero_counter = None
        Set_Parameter("Zero_velocity", True)
        Set_Parameter("Exit_When_Done", False)
        Set_Parameter("Target_Psi", "NO")
        Set_Parameter("Output_Psi_AtLeast", "1")
        self.bdK = 1.1
        self.bdstiff = 0

    def set_object(self, filePath, velocity, p_density, E, nu):
        self.add_object(filePath, self.Vec(0), self.Vec(0), self.Vec(0), 0, self.Vec(1))
        self.initialize_added_objects(velocity, p_density, E, nu)

    def add_object(self, filePath, translate, rotCenter, rotAxis, rotDeg, scale, hero=False):
        if self.dim == 2:
            self.meshCounter = MeshIO.Read_TriMesh_Obj(filePath, self.X0, self.Elem)
        else:
            if filePath[-4:] == '.obj':
                self.meshCounter = MeshIO.Read_TriMesh_Obj(filePath, self.X0, self.Extra_Mesh)
            if filePath[-4:] == '.vtk':
                self.meshCounter = MeshIO.Read_TetMesh_Vtk(filePath, self.X0, self.Elem)
            if filePath[-4:] == '.tet':
                self.meshCounter = MeshIO.Read_TetMesh_Tet(filePath, self.X0, self.Elem)
        if hero:
            if self.hero_counter is None:
                self.hero_counter = self.meshCounter
            else:
                self.hero_counter = MeshIO.Merge_Heros(self.hero_counter, self.meshCounter)
        MeshIO.Transform_Points(translate, rotCenter, rotAxis, rotDeg, scale, self.meshCounter, self.X0)

    def add_object_tex_2D(self, filePath, translate, rotCenter, rotAxis, rotDeg, scale, hero=False):
        X_world = Storage.V2dStorage()
        Elem_world = Storage.V3iStorage()
        if self.dim == 2:
            self.meshCounter = MeshIO.Read_TriMesh_With_Tex_Obj(filePath, X_world, self.X0, Elem_world, self.Elem)
        else:
            print('add_object_tex_2D only support 2d')
            exit()
        MeshIO.Transform_Points(translate, rotCenter, rotAxis, rotDeg, scale, self.meshCounter, self.X0)
    
    #TODO: objects with different density, E, and nu
    def initialize_added_objects(self, velocity, p_density, E, nu):
        if self.dim == 3:
            MeshIO.Find_Surface_TriMesh(self.X0, self.Elem, self.TriVI2TetVI, self.Tri)
        MeshIO.Append_Attribute(self.X0, self.X)
        vol = Storage.SdStorage()
        FEM.Compute_Vol_And_Inv_Basis(self.X0, self.Elem, vol, self.elemAttr)
        if self.dim == 2:
            FIXED_COROTATED_2.Append_All_FEM(self.elasticity, self.meshCounter, vol, E, nu)
        else:
            FIXED_COROTATED_3.Append_All_FEM(self.elasticity, self.meshCounter, vol, E, nu)
        FEM.Compute_Mass_And_Init_Velocity(self.X0, self.Elem, vol, p_density, velocity, self.nodeAttr)
        self.rho0 = p_density
        if self.enableFracture:
            FEM.Fracture.Initialize_Fracture(self.X0, self.Elem, self.debrisAmt, self.fractureRatio2, self.strengthenFactor, self.edge_dupV, \
                self.isFracture_edge, self.incTriV_edge, self.incTriRestDist2_edge)
        if self.EIPC:
            FEM.TimeStepper.ImplicitEuler.Initialize_Elastic_IPC(self.X0, self.Elem, self.dt, E, nu, self.dHat2, self.kappa)

    def initialize_added_objects_tex_2D(self, filePath, velocity, p_density, E, nu):
        if self.dim == 3:
            print('initialize_added_objects_tex_2D only support 2d')
            exit()
        MeshIO.Append_Attribute(self.X0, self.X)
        
        X_world = Storage.V3dStorage()
        Elem_world = Storage.V3iStorage()
        MeshIO.Read_TriMesh_Obj(filePath, X_world, Elem_world)
        vol = Storage.SdStorage()
        FEM.Compute_Vol_And_Inv_Basis_Tex(X_world, Elem_world, vol, self.elemAttr)
        
        FIXED_COROTATED_2.Append_All_FEM(self.elasticity, self.meshCounter, vol, E, nu)
        FEM.Compute_Mass_And_Init_Velocity(self.X0, self.Elem, vol, p_density, velocity, self.nodeAttr)
        self.rho0 = p_density

    def set_plasticity(self, E, nu, yield_stress, fail_stress, xi):
        self.withPlasticity = True
        Set_Parameter("Von_Mises_E", E)
        Set_Parameter("Von_Mises_nu", nu)
        Set_Parameter("Von_Mises_yield_stress", yield_stress)
        Set_Parameter("Von_Mises_fail_stress", fail_stress)
        Set_Parameter("Von_Mises_xi", xi)

    def initialize_OIPC(self, dHat2, stiffMult = 1):
        self.EIPC = False
        self.dHat2 = FEM.DiscreteShell.Initialize_OIPC_VM(dHat2, self.nodeAttr, self.kappa, stiffMult)

    def set_DBC(self, DBCRangeMin, DBCRangeMax, v, rotCenter, rotAxis, angVelDeg, scaleVel=0., durationStart=0, durationEnd=1e30):
        DBCMotion = Storage.V2iV2dV2dV2dSdSdStorage() if self.dim == 2 else Storage.V2iV3dV3dV3dSdSdStorage()
        FEM.Init_Dirichlet(self.X0, DBCRangeMin, DBCRangeMax,  v, rotCenter, rotAxis, angVelDeg, scaleVel, self.DBC, DBCMotion, Vector4i(0, 0, 1000000000, -1))
        self.DBCMotionList.append((durationStart, durationEnd, DBCMotion))

    def set_DBC_wrt_X(self, DBCRangeMin, DBCRangeMax, v, rotCenter, rotAxis, angVelDeg, scaleVel=0., durationStart=0, durationEnd=1e30):
        DBCMotion = Storage.V2iV2dV2dV2dSdSdStorage() if self.dim == 2 else Storage.V2iV3dV3dV3dSdSdStorage()
        FEM.Init_Dirichlet(self.X, DBCRangeMin, DBCRangeMax,  v, rotCenter, rotAxis, angVelDeg, scaleVel, self.DBC, DBCMotion, Vector4i(0, 0, 1000000000, -1))
        self.DBCMotionList.append((durationStart, durationEnd, DBCMotion))

    #TODO: def add_object for multi-object system

    def advance_one_time_step(self, dt):
        #TODO: self.tol

        if self.current_frame == self.callback_frame:
            if self.callback_func:
                self.callback_func()
                self.callback_func = None

        for durationStart, durationEnd, DBCMotion in self.DBCMotionList:
            if durationStart <= self.current_frame < durationEnd:
                FEM.Step_Dirichlet(DBCMotion, dt, self.DBC)

        # if self.current_frame < Get_Parameter("DBC_stop_frame", self.frame_num):
        #     FEM.Step_Dirichlet(self.DBCMotion, dt, self.DBC)
        # else:
        #     FEM.Reset_Dirichlet(self.X, self.DBC)
        #     self.dt = 1
        #     Set_Parameter("Terminate", True)
        # FEM.TimeStepper.ImplicitEuler.Check_Gradient_FEM(self.Elem, self.DBC, \
        #     self.gravity, self.dt, self.X, self.X0, self.nodeAttr, self.elemAttr, self.elasticity)

        lastPNIter = self.PNIterCount
        if self.useNewton:
            if self.EIPC:
                self.PNIterCount += FEM.TimeStepper.ImplicitEuler.Advance_One_Step_EIPC(self.Elem, self.DBC, \
                    self.gravity, self.dt, self.PNTol, self.withCollision, self.dHat2, self.kappa, self.mu, self.epsv2, self.bdstiff, self.bdK, \
                    self.staticSolve, self.withShapeMatching, self.output_folder, self.current_frame, self.X, self.X0, self.nodeAttr, self.elemAttr, self.elasticity)
            else:
                self.PNIterCount += FEM.TimeStepper.ImplicitEuler.Advance_One_Step(self.Elem, self.DBC, \
                    self.gravity, self.dt, self.PNTol, self.withCollision, self.dHat2, self.kappa, self.mu, self.epsv2, self.bdstiff, self.bdK, \
                    self.staticSolve, self.withShapeMatching, self.output_folder, self.current_frame, self.X, self.X0, self.nodeAttr, self.elemAttr, self.elasticity, self.Extra_Mesh)
        else:
            self.PNIterCount += FEM.TimeStepper.ImplicitEuler.Advance_One_Step_SU(self.Elem, self.DBC, \
                self.gravity, self.dt, self.PNTol, self.withCollision, self.dHat2, self.kappa, self.mu, self.epsv2, self.bdstiff, self.bdK, \
                self.output_folder, self.current_frame, self.X, self.X0, self.nodeAttr, self.elemAttr, self.elasticity)

        FEM.Reset_Dirichlet(self.X, self.DBC)
        if self.withPlasticity:
            Von_Mises_Project_Strain(self.Elem, self.X, self.elemAttr, self.elasticity)

        print("Total PN iteration count: ", self.PNIterCount, "\n")
        # if self.enableFracture:
        #     if FEM.Fracture.Edge_Fracture(self.X, self.Elem, self.incTriV_edge, self.incTriRestDist2_edge, \
        #         self.fractureRatio2, self.isFracture_edge):
        #         print("edge fractured")
        #         FEM.Fracture.Node_Fracture(self.edge_dupV, self.isFracture_edge, \
        #             self.X, self.Elem, self.finalV2old)
        #         print("node fractured")
        #         FEM.Fracture.Update_Fracture(self.Elem, self.rho0, self.finalV2old, \
        #             self.X0, self.nodeAttr, self.DBC, self.DBCMotion, self.withCollision, self.dHat2, self.X)
        #         if self.dim == 3:
        #             MeshIO.Find_Surface_TriMesh(self.X0, self.Elem, self.TriVI2TetVI, self.Tri)
        #         print("fracture updated")

        if lastPNIter == self.PNIterCount:
            print('no progress to make')
            exit()

    def write(self, frame_idx):
        fn = self.output_folder + str(frame_idx) + ".obj"
        if self.dim == 2:
            MeshIO.Write_TriMesh_Obj(self.X, self.Elem, fn)
        else:
            MeshIO.Write_Surface_TriMesh_Obj(self.X, self.TriVI2TetVI, self.Tri, fn)
        if self.hero_counter:
            MeshIO.Write_Surface_TriMesh_Obj_Regional(self.X, self.TriVI2TetVI, self.Tri, self.output_folder + "hero_" + str(frame_idx) + ".obj", self.hero_counter)
    
    def write_com(self):
        FEM.TimeStepper.ImplicitEuler.Write_COM(self.X, self.nodeAttr, self.output_folder + "com.txt")