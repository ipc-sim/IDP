import sys
sys.path.insert(0, "../../../Python")
sys.path.insert(0, "../../../build")
import Drivers
from JGSL import *

if __name__ == "__main__":
    sim = Drivers.FEMSimulationBase("double", 3, "NH")

    E = 100000
    yield_stress = 100000

    sim.add_object("tmp/dumpling_compressed.vtk", Vector3d(0, 0, 0),
                   Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(1, 1, 1), True)
    Set_Parameter("Dirichlet_zero_ring", 1.1)
    sim.set_DBC(Vector3d(-0.1, -0.1, -0.1), Vector3d(0.44, 1.1, 1.1),
                Vector3d(0, 0, 0), Vector3d(0, 0.1, 0), Vector3d(0, 0, 1), -190, 0, 0, 13)
    sim.set_DBC(Vector3d(0.56, -0.1, -0.1), Vector3d(1.1, 1.1, 1.1),
                Vector3d(0, 0, 0), Vector3d(0, 0.1, 0), Vector3d(0, 0, 1), 190, 0, 0, 13)

    sim.add_object("../input/sphere1K.vtk", Vector3d(0, .6, 0),
                   Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(1, 1, 2), True)
    sim.set_DBC(Vector3d(0.4, 0.4, 0.4), Vector3d(0.6, 0.6, 0.6),
                Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0)

    sim.initialize_added_objects(Vector3d(0, 0, 0), 1000, E, 0.4)
    sim.set_plasticity(E, 0.4, yield_stress, 1.e30, 0.)

    sim.dt = 0.04
    sim.frame_dt = 0.04
    sim.frame_num = 20
    sim.gravity = Vector3d(0, 0, 0)
    sim.initialize_OIPC(1e-6)
    sim.adjust_camera(2.5, [0, 0, 0])
    sim.run()
