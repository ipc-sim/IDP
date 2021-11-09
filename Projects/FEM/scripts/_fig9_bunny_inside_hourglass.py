import sys
sys.path.insert(0, "../../../Python")
sys.path.insert(0, "../../../build")
import Drivers
from JGSL import *

if __name__ == "__main__":
    sim = Drivers.FEMSimulationBase("double", 3, "NH")

    sim.add_object("../input/hourglass.obj", Vector3d(0, 0, 0),
                   Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(1, 1, 1))
    sim.set_DBC(Vector3d(-0.1, -0.1, -0.1), Vector3d(1.1, 1.1, 1.1),
                Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(0, 0, 0), 0)

    case = int(sys.argv[1])
    files = ["../input/bunnyx30_13K.vtk",
             "../input/bunnyx30_54K.vtk",
             "../input/octocat6K.vtk",
             "../input/octocat23K.vtk"]
    if case < 2:
        sim.add_object(files[case], Vector3d(0, 0, 0),
                       Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(1./30*1.5, 1./30*1.5, 1./30*1.5))
    else:
        sim.add_object(files[case], Vector3d(0, 0, 0),
                       Vector3d(0, 0, 0), Vector3d(1, 0, 0),-90, Vector3d(1., 1., 1.))

    # Set_Parameter("Init_F_script", "bunny_inside_box")
    # Set_Parameter("bunny_inside_box_scale", 3.)
    Set_Parameter("Progressively_Update_F_scale", 0.1)
    Set_Parameter("Progressively_Update_F_frame", int(sys.argv[2]))

    sim.initialize_added_objects(Vector3d(0, 0, 0), 1000, 1e4, 0.4)

    sim.dt = 0.04
    sim.frame_dt = 0.04
    sim.frame_num = int(sys.argv[3])
    sim.gravity = Vector3d(0, 0, 0)
    sim.initialize_OIPC(1e-6)
    sim.adjust_camera(1, [0, 0, 0])
    sim.run()