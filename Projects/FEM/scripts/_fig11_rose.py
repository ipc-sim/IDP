import sys
sys.path.insert(0, "../../../Python")
sys.path.insert(0, "../../../build")
import Drivers
from JGSL import *
import math

if __name__ == "__main__":
    sim = Drivers.FEMSimulationBase("double", 3, "NH")

    E = 10000

    # for i in range(60):
    #     sim.add_object("../input/long_petal.vtk", Vector3d(i * 0.007, 0, 0),
    #                    Vector3d(0, 0, 0), Vector3d(1, 0, 0), 90, Vector3d(1, 0.5, 1))
    sim.add_object("../input/long_petals.vtk", Vector3d(0, 0, 0),
                   Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(1.5, 1, 1))
    sim.set_DBC(Vector3d(-0.1, -0.1, -0.1), Vector3d(1.1, 0.05, 1.1),
                Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0)
    sim.set_DBC(Vector3d(-0.1, 0.95, -0.1), Vector3d(1.1, 1.1, 1.1),
                Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0)


    def func():
        Set_Parameter("rose_base_v", 0)
        sim.DBC = Storage.V3dStorage() if sim.dim == 2 else Storage.V4dStorage()
        sim.DBCMotionList = []
        sim.set_DBC_wrt_X(Vector3d(-0.1, -0.1, -0.1), Vector3d(1.1, 0.05, 1.1),
                          Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, -0.1)
    sim.callback_frame = 100
    sim.callback_func = func

    Set_Parameter("rose_base_v", 0.1)

    sim.initialize_added_objects(Vector3d(0, 0, 0), 1000, E, 0.4)

    sim.dt = 0.04
    sim.frame_dt = 0.04
    sim.frame_num = 120
    sim.gravity = Vector3d(0, 0, 0)
    sim.initialize_OIPC(1e-6)
    sim.adjust_camera(2.5, [0, 0, 0])
    sim.run()
