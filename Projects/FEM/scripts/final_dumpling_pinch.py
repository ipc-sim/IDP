import sys
sys.path.insert(0, "../../../Python")
sys.path.insert(0, "../../../build")
import Drivers
from JGSL import *

if __name__ == "__main__":
    sim = Drivers.FEMSimulationBase("double", 3, "NH")

    E = 10000
    yield_stress = 1000

    sim.add_object("tmp/dumpling_compressed.vtk", Vector3d(0, 0, 0),
                   Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(1.1, 1.1, 1.1))
    Set_Parameter("Dirichlet_zero_ring", 1.2)
    sim.set_DBC(Vector3d(-0.1, -0.1, -0.1), Vector3d(0.4, 1.1, 1.1),
                Vector3d(0, 0, 0), Vector3d(0, 0.1, 0), Vector3d(0, 0, 1), -185, 0, 0, 20)
    sim.set_DBC(Vector3d(0.6, -0.1, -0.1), Vector3d(1.1, 1.1, 1.1),
                Vector3d(0, 0, 0), Vector3d(0, 0.1, 0), Vector3d(0, 0, 1), 185, 0, 0, 20)


    # sim.add_object("../input/cube_center.vtk", Vector3d(-0.3, 1.45, 0 - 0.2),
    #                Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(0.02, 0.5, 0.4))
    # sim.add_object("../input/cube_center.vtk", Vector3d(0.3, 1.45, 0 + 0.2),
    #                Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(0.02, 0.5, 0.4))
    # sim.add_object("../input/cube_center.vtk", Vector3d(-0.3, 1.25574, 0.725 - 0.2),
    #                Vector3d(0, 0, 0), Vector3d(1, 0, 0), 30, Vector3d(0.02, 0.5, 0.4))
    # sim.add_object("../input/cube_center.vtk", Vector3d(0.3, 1.25574, 0.725 + 0.2),
    #                Vector3d(0, 0, 0), Vector3d(1, 0, 0), 30, Vector3d(0.02, 0.5, 0.4))
    # sim.add_object("../input/cube_center.vtk", Vector3d(-0.3, 0.725, 1.25574 - 0.2),
    #                Vector3d(0, 0, 0), Vector3d(1, 0, 0), 60, Vector3d(0.02, 0.5, 0.4))
    # sim.add_object("../input/cube_center.vtk", Vector3d(0.3, 0.725, 1.25574 + 0.2),
    #                Vector3d(0, 0, 0), Vector3d(1, 0, 0), 60, Vector3d(0.02, 0.5, 0.4))
    # sim.add_object("../input/cube_center.vtk", Vector3d(-0.3, 1.25574, -0.725 - 0.2),
    #                Vector3d(0, 0, 0), Vector3d(1, 0, 0), -30, Vector3d(0.02, 0.5, 0.4))
    # sim.add_object("../input/cube_center.vtk", Vector3d(0.3, 1.25574, -0.725 + 0.2),
    #                Vector3d(0, 0, 0), Vector3d(1, 0, 0), -30, Vector3d(0.02, 0.5, 0.4))
    # sim.add_object("../input/cube_center.vtk", Vector3d(-0.3, 0.725, -1.25574 - 0.2),
    #                Vector3d(0, 0, 0), Vector3d(1, 0, 0), -60, Vector3d(0.02, 0.5, 0.4))
    # sim.add_object("../input/cube_center.vtk", Vector3d(0.3, 0.725, -1.25574 + 0.2),
    #                Vector3d(0, 0, 0), Vector3d(1, 0, 0), -60, Vector3d(0.02, 0.5, 0.4))


    # sim.set_DBC(Vector3d(-0.1, -0.1, -0.1), Vector3d(0.5, 1.1, 1.1),
    #             Vector3d(1, 0, 1), Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0)
    # sim.set_DBC(Vector3d(0.5, -0.1, -0.1), Vector3d(1.1, 1.1, 1.1),
    #             Vector3d(-1, 0, -1), Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0)

    sim.add_object("tmp/dumpling_pushed.vtk", Vector3d(0, 0, 0),
                   Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(1, 1, 1), True)

    sim.initialize_added_objects(Vector3d(0, 0, 0), 1000, E, 0.4)
    sim.set_plasticity(E, 0.4, yield_stress, 1.e30, 0.)

    sim.dt = 0.04
    sim.frame_dt = 0.04
    sim.frame_num = 60
    sim.gravity = Vector3d(0, 0, 0)
    sim.initialize_OIPC(1e-6)
    sim.adjust_camera(1.0, [0, 0, 0])
    sim.run()
