import sys
sys.path.insert(0, "../../../Python")
sys.path.insert(0, "../../../build")
import Drivers
from JGSL import *

if __name__ == "__main__":
    sim = Drivers.FEMSimulationBase("double", 3, "NH")

    case = int(sys.argv[1])
    fn = ["../input/Armadillo2.5K.vtk", "../input/Armadillo13K.vtk", "../input/Armadillo28K.vtk", "../input/armadillo_rest273K.tet", "../input/armadillo.tet"][case]
    if case == 3:
        sim.add_object(fn, Vector3d(-0.05, 0.65, 0.01),
                       Vector3d(0, 0, 0), Vector3d(0, 1, 0), 180, Vector3d(3, 3, 3))
    else:
        sim.add_object(fn, Vector3d(0, 0, 0),
                       Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(0.02, 0.02, 0.02))
    # left hand, right hand, left foot, right foot
    sim.set_DBC(Vector3d(-0.1, 0.5, -0.1), Vector3d(0.05, 1.1, 0.22),
                Vector3d(0, 0, 0), Vector3d(-0.4, 1.15, -0.6), Vector3d(1, 1.5, 0), -90, 0, 0, 25)
    sim.set_DBC(Vector3d(0.95, 0.5, -0.1), Vector3d(1.1, 1.1, 0.22),
                Vector3d(0, 0, 0), Vector3d(0.4, 1.25, -0.7), Vector3d(0.7, -1.5, 0), -90, 0, 25, 54)
    sim.set_DBC(Vector3d(-0.1, -0.1, -0.1), Vector3d(1.1, 0.2, 1.1),
                Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(0, 0, 0), 0)

    def func():
        sim.DBC = Storage.V3dStorage() if sim.dim == 2 else Storage.V4dStorage()
        sim.DBCMotionList = []
        sim.set_DBC_wrt_X(Vector3d(-0.1, -0.1, -0.1), Vector3d(1.1, 0.2, 1.1),
                          Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(0, 0, 0), 0)
        sim.set_DBC_wrt_X(Vector3d(-0.1, 0.5, -0.1), Vector3d(1.1, 1.1, 0.5),
                          Vector3d(0, 0, 0), Vector3d(0, 0.2, 0.3), Vector3d(1, 0, 0), -60)
    sim.callback_frame = 54
    sim.callback_func = func
    sim.set_DBC(Vector3d(-0.1, 0.5, -0.1), Vector3d(0.05, 1.1, 0.22),
                Vector3d(0, 0, 0), Vector3d(0, 0.2, 0.3), Vector3d(1, 0, 0), -180, 0, 54, 80)
    sim.set_DBC(Vector3d(0.95, 0.5, -0.1), Vector3d(1.1, 1.1, 0.22),
                Vector3d(0, 0, 0), Vector3d(0, 0.2, 0.3), Vector3d(1, 0, 0), -180, 0, 54, 80)

    sim.initialize_added_objects(Vector3d(0, 0, 0), 1000, 1e6, 0.4)

    sim.dt = 0.04
    sim.frame_dt = 0.04
    sim.frame_num = 120
    sim.gravity = Vector3d(0, 0, 0)
    sim.initialize_OIPC(1e-6)
    sim.adjust_camera(2.4, [0, 0, 0])
    sim.run()