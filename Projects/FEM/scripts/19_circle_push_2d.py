import sys
sys.path.insert(0, "../../../Python")
sys.path.insert(0, "../../../build")
import Drivers
from JGSL import *

if __name__ == "__main__":
    model = "NH"
    if len(sys.argv) > 1:
        model = sys.argv[1]

    size = ''
    if len(sys.argv) > 2:
        size = sys.argv[2]

    handleSize = 0.05
    if len(sys.argv) > 3:
        handleSize = float(sys.argv[3])
    
    sim = Drivers.FEMSimulationBase("double", 2, model)
    Set_Parameter("Output_Psi", "YES")

    sim.add_object("../input/circle" + size + ".obj", Vector2d(0, 0),
                   Vector2d(0, 0), Vector2d(1, 0), 0, Vector2d(1, 1))
    sim.initialize_added_objects(Vector2d(0, 0), 1000, 2e4, 0.4)

    sim.set_DBC(Vector2d(0, 0.5 - handleSize / 2), Vector2d(handleSize, 0.5 + handleSize / 2), Vector2d(1, 0), Vector2d(0, 0), Vector2d(0, 0), 0)
    sim.set_DBC(Vector2d(1 - handleSize, 0.5 - handleSize / 2), Vector2d(1.00, 0.5 + handleSize / 2), Vector2d(-1, 0), Vector2d(0, 0), Vector2d(0, 0), 0)
    sim.set_DBC(Vector2d(0.5 - handleSize / 2, 0), Vector2d(0.5 + handleSize / 2, handleSize), Vector2d(0, 1), Vector2d(0, 0), Vector2d(0, 0), 0)
    sim.set_DBC(Vector2d(0.5 - handleSize / 2, 1 - handleSize), Vector2d(0.5 + handleSize / 2, 1.00), Vector2d(0, -1), Vector2d(0, 0), Vector2d(0, 0), 0)

    sim.dt = 0.04
    sim.frame_dt = 0.04
    sim.frame_num = 15
    sim.gravity = Vector2d(0, 0)
    sim.initialize_OIPC(1e-6)
    sim.adjust_camera(2. / 3, [0, 0])

    sim.run()
