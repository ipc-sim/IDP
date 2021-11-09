import sys
sys.path.insert(0, "../../../Python")
sys.path.insert(0, "../../../build")
import Drivers
from JGSL import *
import math
import random

if __name__ == "__main__":
    sim = Drivers.FEMSimulationBase("double", 3, "NH")

    E = 100000
    sim.add_object("../input/vase.vtk", Vector3d(0, 0, 0),
                   Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(1, 1, 1))
    sim.set_DBC(Vector3d(-0.1, -0.1, -0.1), Vector3d(1.1, 1.1, 1.1),
                Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(0, 0, 0), 0, -0.1)

    random.seed(12)
    for i in range(4):
        for j in range(4):
            h = int(random.random() * 8)
            if i * j == 0 and i + j == 2:
                h = 0
            if h > 3:
                sim.add_object("../input/leaf.vtk", Vector3d(i * 0.2, 0, j * 0.2),
                               Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(1, 1, 1))
            else:
                sim.add_object("../input/wood.vtk", Vector3d(i * 0.2, 0, j * 0.2),
                               Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(1, 1, 1))
    print("===========")
    Set_Parameter("final_flower_fruit_id", 11889)
    Set_Parameter("final_flower_fruit_rho", 20.0)
    random.seed(12)
    for i in range(4):
        for j in range(4):
            h = int(random.random() * 8)
            if i * j == 0 and i + j == 2:
                h = 0
            if h == 0:
                sim.add_object("../input/rose.vtk", Vector3d(i * 0.2, 0, j * 0.2),
                               Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(1, 1, 1))
            elif h == 1:
                sim.add_object("../input/rhododendron.vtk", Vector3d(i * 0.2, 0, j * 0.2),
                               Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(1, 1, 1))
            elif h == 2:
                sim.add_object("../input/justicia.vtk", Vector3d(i * 0.2, 0, j * 0.2),
                               Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(1, 1, 1))
            elif h == 3:
                sim.add_object("../input/fruit.vtk", Vector3d(i * 0.2, 0, j * 0.2),
                               Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(1, 1, 1))

    sim.initialize_added_objects(Vector3d(0, 0, 0), 1000, E, 0.4)

    sim.dt = 0.04
    sim.frame_dt = 0.04
    sim.frame_num = 1000
    sim.gravity = Vector3d(0, -5, 0)
    sim.initialize_OIPC(1e-6)
    sim.adjust_camera(2.5, [0, 0, 0])
    sim.run()
