import sys
sys.path.insert(0, "../../Python")
import Drivers
from JGSL import *

if __name__ == "__main__":
    sim = Drivers.FEMDiscreteShellBase("double", 3)

    meshName = 'cat'
    if len(sys.argv) > 1:
        meshName = sys.argv[1]
    
    smoothIntensity = 0.5
    if len(sys.argv) > 2:
        smoothIntensity = float(sys.argv[2])

    sim.normalFlowMag = 0.005
    if len(sys.argv) > 3:
        sim.normalFlowMag = float(sys.argv[3])

    sim.frame_num = 10
    if len(sys.argv) > 4:
        sim.frame_num = int(sys.argv[4])

    sim.add_shell_3D("input/" + meshName + ".obj", Vector3d(0, 0, 0), \
        Vector3d(0, 0, 0), Vector3d(0, 0, 0), 0)
    

    sim.dt = 1
    sim.frame_dt = 1
    sim.gravity = Vector3d(0, 0, 0)
    sim.flow = (smoothIntensity > 0)
    if sim.flow:
        sim.dt *= smoothIntensity
        sim.frame_dt *= smoothIntensity
        sim.normalFlowMag /= smoothIntensity * smoothIntensity

    sim.initialize(1, 0, 0, 1, 0)
    sim.bendingStiffMult = 0

    sim.withCollision = True
    sim.initialize_OIPC(1e-3, 0)

    sim.run()
