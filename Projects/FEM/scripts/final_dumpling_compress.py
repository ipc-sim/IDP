import sys
sys.path.insert(0, "../../../Python")
sys.path.insert(0, "../../../build")
import Drivers
from JGSL import *

if __name__ == "__main__":
    sim = Drivers.FEMSimulationBase("double", 3, "NH")

    E = 100000
    yield_stress = 10000

    sim.add_object("../input/cube_center.vtk", Vector3d(0, 1.6, 0),
                   Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(2.5, 0.01, 2.5))
    sim.add_object("../input/cube_center.vtk", Vector3d(0, -1.6, 0),
                   Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(4, 0.01, 4))
    sim.set_DBC(Vector3d(-0.1, 0.8, -0.1), Vector3d(1.1, 1.1, 1.1),
                Vector3d(0, -4.3, 0), Vector3d(0, 0, 0), Vector3d(0, 0, 0), 0, 0, 0, 10)
    sim.set_DBC(Vector3d(-0.1, 0.8, -0.1), Vector3d(1.1, 1.1, 1.1),
                Vector3d(0, 70, 0), Vector3d(0, 0, 0), Vector3d(0, 0, 0), 0, 0, 10, 11)
    sim.set_DBC(Vector3d(-0.1, -0.1, -0.1), Vector3d(1.1, 0.2, 1.1),
                Vector3d(0, 40, 0), Vector3d(0, 0, 0), Vector3d(0, 0, 0), 0, 0, 0, 2)
    sim.set_DBC(Vector3d(-0.1, -0.1, -0.1), Vector3d(1.1, 0.2, 1.1),
                Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(0, 0, 0), 0, 0, 2, 10)
    sim.set_DBC(Vector3d(-0.1, -0.1, -0.1), Vector3d(1.1, 0.2, 1.1),
                Vector3d(0, -70, 0), Vector3d(0, 0, 0), Vector3d(0, 0, 0), 0, 0, 10, 11)

    sim.add_object("../input/sphere51K.vtk", Vector3d(0, 0.55, 0),
                   Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(1, 1, 1), True)

    sim.initialize_added_objects(Vector3d(0, 0, 0), 1000, E, 0.4)
    sim.set_plasticity(E, 0.4, yield_stress, 1.e30, 0.)

    sim.dt = 0.04
    sim.frame_dt = 0.04
    sim.frame_num = 30
    sim.gravity = Vector3d(0, 0, 0)
    sim.initialize_OIPC(1e-6)
    sim.adjust_camera(2.5, [0, 0, 0])
    sim.run()
