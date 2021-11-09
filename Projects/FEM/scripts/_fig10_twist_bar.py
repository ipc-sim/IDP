import sys
sys.path.insert(0, "../../../Python")
sys.path.insert(0, "../../../build")
import Drivers
from JGSL import *

if __name__ == "__main__":
    sim = Drivers.FEMSimulationBase("double", 3, "NH")

    sim.add_object("../input/bar.vtk", Vector3d(0, 0, 0),
                   Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(0.1, 0.1, 0.1))

    sim.set_DBC(Vector3d(0, -0.1, -0.1), Vector3d(0.06, 1.1, 1.1),
                Vector3d(0, 0, 0), Vector3d(0, 0.06, 0), Vector3d(0, 0, 1), -90, 0, 0, 26)
    sim.set_DBC(Vector3d(0.94, -0.1, -0.1), Vector3d(1, 1.1, 1.1),
                Vector3d(0, 0, 0), Vector3d(0, 0.06, 0), Vector3d(0, 0, 1), 90, 0, 0, 26)

    def func():
        Set_Parameter("Dirichlet_circle", 0.04)
        sim.set_DBC_wrt_X(Vector3d(-0.1, 0.00, -0.1), Vector3d(1.1, 0.15, 1.1),
                          Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(0, 1, 0), -180)
    sim.callback_frame = 26
    sim.callback_func = func

    sim.initialize_added_objects(Vector3d(0, 0, 0), 1000, 1e5, 0.4)

    sim.dt = 0.04
    sim.frame_dt = 0.04
    sim.frame_num = 150
    sim.gravity = Vector3d(0, 0, 0)
    sim.initialize_OIPC(1e-6)
    sim.adjust_camera(2, [0, -0.4, 0])
    sim.run()