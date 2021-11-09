import sys
sys.path.insert(0, "../../../Python")
sys.path.insert(0, "../../../build")
import Drivers
from JGSL import *

if __name__ == "__main__":

    model = "SD"
    if len(sys.argv) > 1:
        model = sys.argv[1]
    
    sim = Drivers.FEMSimulationBase("double", 2, model)
    Set_Parameter("Output_Psi", "YES")
    Set_Parameter("Exit_When_Done", True)
    sim.gravity = Vector2d(0, 0)

    meshName = "cone"
    if len(sys.argv) > 2:
        meshName = sys.argv[2]

    targetPsi = -1
    if len(sys.argv) > 3:
        targetPsi = float(sys.argv[3])

    sim.withCollision = True
    if len(sys.argv) > 4:
        sim.withCollision = bool(int(sys.argv[4]))


    sim.add_object_tex_2D("../input/withUV/" + meshName + ".obj", Vector2d(0, 0), 
        Vector2d(0, 0), Vector2d(1, 0), 0, Vector2d(1, 1))
    sim.initialize_added_objects_tex_2D("../input/withUV/" + meshName + ".obj",
        Vector2d(0, 0), 1000, 2e4, 0.4)

    if targetPsi < 0:
        # static solve
        # DBCRangeMin, DBCRangeMax,
        # v, rotCenter, rotAxis, angVelDeg
        sim.set_DBC(Vector2d(-0.1, -0.1), Vector2d(1e-4, 1.1),
                    Vector2d(0, 0), Vector2d(0, 0), Vector2d(1, 0), 0)
        
        sim.staticSolve = True
        sim.dt = 1
        sim.frame_dt = 1
        sim.frame_num = 1
        Set_Parameter("Target_Psi", "-1")
    else:
        # dynamic solve with target psi
        sim.dt = 0.04
        sim.frame_dt = 0.04
        sim.frame_num = 10000
        Set_Parameter("Target_Psi", str(targetPsi))
        # Set_Parameter("Target_Psi", "5.850865e+03")
        # Set_Parameter("Target_Psi", "5.814576e+03")

    if sim.withCollision:
        sim.initialize_OIPC(1e-6)

    sim.run()
