import sys
sys.path.insert(0, "../../Python")
import Drivers
from JGSL import *

if __name__ == "__main__":
    sim = Drivers.FEMDiscreteShellBase("double", 3)

    seqName = 'Rumba_Dancing_unfixed'
    if len(sys.argv) > 1:
        seqName = sys.argv[1]

    dhat = 1e-2
    if len(sys.argv) > 2:
        dhat = float(sys.argv[2])
    
    kappaMult = 1
    if len(sys.argv) > 3:
        kappaMult = float(sys.argv[3])
    
    # load mannequin rest shape
    sim.add_shell_with_scale_3D("input/wm2_15k.obj", Vector3d(0, 0.75, 0), Vector3d(1, 1, 1),\
        Vector3d(0, 0, 0), Vector3d(1, 0, 0), -90)
    sim.set_DBC(Vector3d(-0.1, 0.867, -0.1), Vector3d(1.1, 1.1, 1.1), Vector3d(0, 0, 0), 
        Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0)

    sim.dt = 0.04
    sim.frame_dt = 0.04
    sim.frame_num = 180
    sim.withCollision = True
    sim.gravity = Vector3d(0, 0, 0)


    sim.initialize(1000, 100, 0.4, 0.01, 0)
    sim.bendingStiffMult = 1

    sim.initialize_OIPC(dhat, 0, kappaMult)

    sim.lv_fn = 1
    sim.seqDBCPath = 'input/' + seqName

    sim.run()
