import sys
sys.path.insert(0, "../../../Python")
sys.path.insert(0, "../../../build")
import Drivers
from JGSL import *

if __name__ == "__main__":
    sim = Drivers.FEMSimulationBase("double", 3, "NH")

    E = 10000
    yield_stress = 10000

    a = 1.1
    b = 2
    sim.add_object("../input/sphere1K.vtk", Vector3d(-0.15, 1.45 * a, 0 * a),
                   Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(0.15, 0.3 * b, 0.15))
    sim.add_object("../input/sphere1K.vtk", Vector3d(0.15, 1.45 * a, 0 * a),
                   Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(0.15, 0.3 * b, 0.15))
    sim.add_object("../input/sphere1K.vtk", Vector3d(-0.15, 1.25574 * a, 0.725 * a),
                   Vector3d(0, 0, 0), Vector3d(1, 0, 0), 30, Vector3d(0.15, 0.3 * b, 0.15))
    sim.add_object("../input/sphere1K.vtk", Vector3d(0.15, 1.25574 * a, 0.725 * a),
                   Vector3d(0, 0, 0), Vector3d(1, 0, 0), 30, Vector3d(0.15, 0.3 * b, 0.15))
    sim.add_object("../input/sphere1K.vtk", Vector3d(-0.15, 0.725 * a, 1.25574 * a),
                   Vector3d(0, 0, 0), Vector3d(1, 0, 0), 60, Vector3d(0.15, 0.3 * b, 0.15))
    sim.add_object("../input/sphere1K.vtk", Vector3d(0.15, 0.725 * a, 1.25574 * a),
                   Vector3d(0, 0, 0), Vector3d(1, 0, 0), 60, Vector3d(0.15, 0.3 * b, 0.15))
    sim.add_object("../input/sphere1K.vtk", Vector3d(-0.15, 1.25574 * a, -0.725 * a),
                   Vector3d(0, 0, 0), Vector3d(1, 0, 0), -30, Vector3d(0.15, 0.3 * b, 0.15))
    sim.add_object("../input/sphere1K.vtk", Vector3d(0.15, 1.25574 * a, -0.725 * a),
                   Vector3d(0, 0, 0), Vector3d(1, 0, 0), -30, Vector3d(0.15, 0.3 * b, 0.15))
    sim.add_object("../input/sphere1K.vtk", Vector3d(-0.15, 0.725 * a, -1.25574 * a),
                   Vector3d(0, 0, 0), Vector3d(1, 0, 0), -60, Vector3d(0.15, 0.3 * b, 0.15))
    sim.add_object("../input/sphere1K.vtk", Vector3d(0.15, 0.725 * a, -1.25574 * a),
                   Vector3d(0, 0, 0), Vector3d(1, 0, 0), -60, Vector3d(0.15, 0.3 * b, 0.15))

    sim.set_DBC(Vector3d(-0.1, -0.1, -0.1), Vector3d(1.1, 1.1, 0.2),
                Vector3d(0, 0, 0), Vector3d(0, 0.725, -1.25574), Vector3d(0, 0.5, -0.866), 180)
    sim.set_DBC(Vector3d(-0.1, -0.1, 0.2), Vector3d(1.1, 1.1, 0.4),
                Vector3d(0, 0, 0), Vector3d(0, 1.25574, -0.725), Vector3d(0, 0.866, -0.5), 180)
    sim.set_DBC(Vector3d(-0.1, -0.1, 0.4), Vector3d(1.1, 1.1, 0.6),
                Vector3d(0, 0, 0), Vector3d(0, 1.45, 0), Vector3d(0, 1, 0), 180)
    sim.set_DBC(Vector3d(-0.1, -0.1, 0.6), Vector3d(1.1, 1.1, 0.8),
                Vector3d(0, 0, 0), Vector3d(0, 1.25574, 0.725), Vector3d(0, 0.866, 0.5), 180)
    sim.set_DBC(Vector3d(-0.1, -0.1, 0.8), Vector3d(1.1, 1.1, 1.1),
                Vector3d(0, 0, 0), Vector3d(0, 0.725, 1.25574), Vector3d(0, 0.5, 0.866), 180)

    sim.add_object("tmp/dumpling_folded.vtk", Vector3d(0, 0, 0),
                   Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(1, 1, 1), True)

    sim.initialize_added_objects(Vector3d(0, 0, 0), 1000, E, 0.4)
    sim.set_plasticity(E, 0.4, yield_stress, 1.e30, 0.)

    sim.dt = 0.04
    sim.frame_dt = 0.04
    sim.frame_num = 20
    sim.gravity = Vector3d(0, 0, 0)
    sim.initialize_OIPC(1e-6)
    sim.adjust_camera(1.0, [0, 0, 0])
    sim.run()
