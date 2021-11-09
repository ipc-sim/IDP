import sys
sys.path.insert(0, "../../../Python")
sys.path.insert(0, "../../../build")
import Drivers
from JGSL import *
import math

if __name__ == "__main__":
    sim = Drivers.FEMSimulationBase("double", 3, "NH")

    E = 10000

    for i in range(3):
        slice = 360 / 3
        x = 0.024 * math.cos(i * slice / 180 * math.pi)
        z = 0.024 * math.sin(i * slice / 180 * math.pi)
        sim.add_object("../input/single_petal.vtk", Vector3d(x, 0, z),
                       Vector3d(0, 0, 0), Vector3d(0, 1, 0), -90 - i * slice, Vector3d(1, 1, 1))
    for i in range(5):
        slice = 360 / 5
        x = 0.045 * math.cos((i + 0.5) * slice / 180 * math.pi)
        z = 0.045 * math.sin((i + 0.5) * slice / 180 * math.pi)
        sim.add_object("../input/single_petal.vtk", Vector3d(x, 0, z),
                       Vector3d(0, 0, 0), Vector3d(0, 1, 0), -90 - (i + 0.5) * slice, Vector3d(1, 1, 1))
    for i in range(7):
        slice = 360 / 7
        x = 0.07 * math.cos(i * slice / 180 * math.pi)
        z = 0.07 * math.sin(i * slice / 180 * math.pi)
        sim.add_object("../input/single_petal.vtk", Vector3d(x, 0, z),
                       Vector3d(0, 0, 0), Vector3d(0, 1, 0), -90 - i * slice, Vector3d(1, 1, 1))
    sim.set_DBC(Vector3d(-0.1, -0.1, -0.1), Vector3d(1.1, 0.1, 1.1),
                Vector3d(0, 0, 0), Vector3d(1, 0, 0), Vector3d(0, 0, 0), 0, -0.1)

    sim.initialize_added_objects(Vector3d(0, 0, 0), 1000, E, 0.4)

    sim.dt = 0.04
    sim.frame_dt = 0.04
    sim.frame_num = 100
    sim.gravity = Vector3d(0, 0, 0)
    sim.initialize_OIPC(1e-6)
    sim.adjust_camera(2.5, [0, 0, 0])
    sim.run()
