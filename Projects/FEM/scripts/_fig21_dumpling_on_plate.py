import sys
import random
sys.path.insert(0, "../../../Python")
sys.path.insert(0, "../../../build")
import Drivers
from JGSL import *

if __name__ == "__main__":
    sim = Drivers.FEMSimulationBase("double", 3, "NH")

    sim.add_object("../input/plate.obj", Vector3d(0, 0, 0),
                   Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(1, 1, 1))
    sim.set_DBC(Vector3d(-0.1, -0.1, -0.1), Vector3d(1.1, 1.1, 1.1),
                Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(0, 0, 0), 0)

    random.seed(10)
    for i in range(2):
        for j in range(2):
            h = random.random() * 0.2 + 0.03
            angle = random.random() * 360
            sim.add_object("../input/dumpling16K_inside.vtk", Vector3d(i * 0.1 - 0.05, h, j * 0.1 - 0.05),
                           Vector3d(0, 0, 0), Vector3d(0, 1, 0), angle, Vector3d(0.04, 0.04, 0.04))
    for i in range(2):
        for j in range(2):
            h = random.random() * 0.2 + 0.03
            angle = random.random() * 360
            sim.add_object("../input/dumpling16K_inside.vtk", Vector3d(i * 0.1 - 0.05, h + 1.2, j * 0.1 - 0.05),
                           Vector3d(0, 0, 0), Vector3d(0, 1, 0), angle, Vector3d(0.04, 0.04, 0.04))
    print("===============")

    random.seed(10)
    for i in range(2):
        for j in range(2):
            h = random.random() * 0.2 + 0.03
            angle = random.random() * 360
            sim.add_object("../input/dumpling16K_outside.vtk", Vector3d(i * 0.1 - 0.05, h, j * 0.1 - 0.05),
                           Vector3d(0, 0, 0), Vector3d(0, 1, 0), angle, Vector3d(0.04, 0.04, 0.04))
    for i in range(2):
        for j in range(2):
            h = random.random() * 0.2 + 0.03
            angle = random.random() * 360
            sim.add_object("../input/dumpling16K_outside.vtk", Vector3d(i * 0.1 - 0.05, h + 1.2, j * 0.1 - 0.05),
                           Vector3d(0, 0, 0), Vector3d(0, 1, 0), angle, Vector3d(0.04, 0.04, 0.04))

    sim.add_object("../input/sphere1K.vtk", Vector3d(0.5, 0.03, 0),
                   Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(0.2, 0.2, 0.2))
    sim.set_DBC(Vector3d(0.5, -0.1, -0.1), Vector3d(1.1, 1.1, 1.1),
                Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(0, 0, 0), 0, 0, 0, 20)
    sim.set_DBC(Vector3d(0.5, -0.1, -0.1), Vector3d(1.1, 1.1, 1.1),
                Vector3d(-0.4, 0, 0), Vector3d(0, 0, 0), Vector3d(0, 0, 0), 0, 0, 20, 45)
    sim.set_DBC(Vector3d(0.5, -0.1, -0.1), Vector3d(1.1, 1.1, 1.1),
                Vector3d(0.4, 0, 0), Vector3d(0, 0, 0), Vector3d(0, 0, 0), 0, 0, 45, 70)

    Set_Parameter("dumpling_on_plate", 1)
    Set_Parameter("dumpling_on_plate_threshold", 19288)

    sim.initialize_added_objects(Vector3d(0, 0, 0), 1000, 10000, 0.4)

    sim.dt = 0.04
    sim.frame_dt = 0.04
    sim.frame_num = 120
    sim.gravity = Vector3d(0, -9.8, 0)
    sim.initialize_OIPC(1e-6)
    sim.adjust_camera(1, [0, 1.0, 0])
    sim.run()