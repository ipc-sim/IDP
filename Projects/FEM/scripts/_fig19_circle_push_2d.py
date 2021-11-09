import sys
sys.path.insert(0, "../../../Python")
sys.path.insert(0, "../../../build")
import Drivers
from JGSL import *

if __name__ == "__main__":
    sim = Drivers.FEMSimulationBase("double", 2, "NH")

    sim.add_object("../input/circle.obj", Vector2d(0, 0),
                   Vector2d(0, 0), Vector2d(1, 0), 0, Vector2d(1, 1))
    sim.initialize_added_objects(Vector2d(0, 0), 1000, 1e6, 0.4)

    sim.set_DBC(Vector2d(0, 0.475), Vector2d(0.05, 0.525), Vector2d(1, 0), Vector2d(0, 0), Vector2d(0, 0), 0)
    sim.set_DBC(Vector2d(0.95, 0.475), Vector2d(1.00, 0.525), Vector2d(-1, 0), Vector2d(0, 0), Vector2d(0, 0), 0)
    sim.set_DBC(Vector2d(0.475, 0), Vector2d(0.525, 0.05), Vector2d(0, 1), Vector2d(0, 0), Vector2d(0, 0), 0)
    sim.set_DBC(Vector2d(0.475, 0.95), Vector2d(0.525, 1.00), Vector2d(0, -1), Vector2d(0, 0), Vector2d(0, 0), 0)

    sim.dt = 0.04
    sim.frame_dt = 0.04
    sim.frame_num = 20
    sim.gravity = Vector2d(0, 0)
    sim.initialize_OIPC(1e-6)
    sim.adjust_camera(2. / 3, [0, 0])

    sim.run()
