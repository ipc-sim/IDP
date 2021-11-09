import sys
import os
import re
import numpy as np
import meshio

sys.path.insert(0, "../../build")
from JGSL import *

def make_directory(folder):
    try:
        os.mkdir(folder)
    except OSError:
        pass

class SimulationBase:
    def __init__(self, precision, dim):
        Kokkos_Initialize()
        self.precision = precision
        self.dim = dim
        assert precision == "float" or precision == "double"
        assert dim == 2 or dim == 3
        self.set_type()
        self.dt = 0.01
        self.dx = 0.01
        self.gravity = self.Vec()
        self.frame_dt = 1.0 / 24
        self.frame_num = 240
        self.current_frame = 0
        self.symplectic = True
        self.output_folder = "output/" + os.path.splitext(os.path.basename(sys.argv[0]))[0] + "/"
        make_directory(self.output_folder)
        if len(sys.argv) > 1:
            self.output_folder += sys.argv[1]
            for i in range(2, len(sys.argv)):
                self.output_folder += "_" + sys.argv[i]
            self.output_folder += "/"
        make_directory(self.output_folder)
        self.register_logger()
        self.update_scale = None
        self.update_offset = None

    def register_logger(self):
        class Logger(object):
            def __init__(self, output_folder):
                log_folder = output_folder + "log/"
                make_directory(log_folder)
                Set_Parameter("Basic.log_folder", log_folder)
                self.terminal = sys.stdout
                self.log = open(log_folder + "log.txt", "w")

            def write(self, decorated_message):
                raw_message = re.sub(r'\x1b\[[\d;]+m', '', decorated_message);
                self.terminal.write(decorated_message)
                self.log.write(raw_message)

            def flush(self):
                pass
        sys.stdout = Logger(self.output_folder)

    def set_type(self):
        if self.precision == "float":
            self.Scalar = Scalarf
            # Vec2, Vec3, Vec4 = Vector2f, Vector3f, Vector4f
            # Mat2, Mat3, Mat4 = Matrix2f, Matrix3f, Matrix4f
            self.Vec = Vector2f if self.dim == 2 else Vector3f
            self.Mat = Matrix2f if self.dim == 2 else Matrix3f
        else:
            self.Scalar = Scalard
            # Vec2, Vec3, Vec4 = Vector2d, Vector3d, Vector4d
            # Mat2, Mat3, Mat4 = Matrix2d, Matrix3d, Matrix4d
            self.Vec = Vector2d if self.dim == 2 else Vector3d
            self.Mat = Matrix2d if self.dim == 2 else Matrix3d

    def advance_one_time_step(self, dt):
        pass

    def advance_one_frame(self, f):
        remain_dt = self.frame_dt
        while remain_dt > 0:
            if remain_dt > self.dt * 2:
                current_dt = self.dt
            elif remain_dt > self.dt:
                current_dt = remain_dt / 2
            else:
                current_dt = remain_dt
            self.advance_one_time_step(current_dt)
            print("Advance dt with %.2e" % current_dt)
            remain_dt -= current_dt
            TIMER_FLUSH(f, self.frame_num, self.frame_dt - remain_dt, self.frame_dt)
            if Get_Parameter("Terminate", False):
                break

    def write(self, frame_idx):
        pass

    def run(self):
        self.write(0)
        # do it twice to make sure the image is shown
        for f in range(self.frame_num):
            self.current_frame = f + 1
            self.advance_one_frame(f + 1)
            self.write(f + 1)
            if Get_Parameter("Terminate", False):
                break
        # Kokkos_Finalize() # will cause the following error:
        # libc++abi.dylib: terminating with uncaught exception of type std::runtime_error: Kokkos allocation "storage" is being deallocated after Kokkos::finalize was called

    def adjust_camera(self, scale, offset):
        self.update_scale = scale
        self.update_offset = offset

    def write_image(self, index):
        input_fn = self.output_folder + str(index) + '.obj'
        output_fn = self.output_folder + f'images/{index:06d}.png'
        mesh = meshio.read(input_fn)
        particles = mesh.points[:, :self.dim] * self.camera_scale + self.camera_offset
        vertices = mesh.cells[0].data
        if self.dim == 2:
            for i in range(len(vertices)):
                for j in range(3):
                    a, b = vertices[i, j], vertices[i, (j + 1) % 3]
                    self.gui.line((particles[a, 0], particles[a, 1]),
                             (particles[b, 0], particles[b, 1]),
                             radius=1,
                             color=0x4FB99F)
            self.gui.show(output_fn)
        else:
            self.t3_model.vi.from_numpy(particles.astype(np.float32))
            self.t3_model.faces.from_numpy(vertices.astype(np.int32))
            self.t3_camera.from_mouse(self.gui)
            self.t3_scene.render()
            self.gui.set_image(self.t3_camera.img)
            self.gui.show(output_fn)

    def generate_gif(self):
        framerate = int(1. / self.frame_dt)
        # generate mp4
        try:
            os.remove(self.output_folder + 'images/video.mp4')
        except OSError:
            pass
        os.system('ffmpeg -framerate ' + str(framerate) + ' -i "' + self.output_folder + 'images/%06d.png" -c:v libx264 -profile:v high -crf 10 -pix_fmt yuv420p -threads 20 ' + self.output_folder + 'images/video.mp4')
        # convert mp4 to gif
        os.system('ffmpeg -i ' + self.output_folder + 'images/video.mp4 -vf scale=320:-1 -r 24 -f image2pipe -vcodec ppm - | convert -delay 5 -loop 0 - ' + self.output_folder[:-1] + '.gif')
