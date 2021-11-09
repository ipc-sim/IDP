import sys
sys.path.insert(0, "../../build")
import os
try:
    os.mkdir("output")
except OSError:
    pass

from JGSL import *
from .SimulationBase import SimulationBase


class FLIPSimulationBase(SimulationBase):
    def __init__(self, precision, dim, pic_r):
        super().__init__(precision, dim)
        self.gravity = self.Vec(0, -9.81) if self.dim == 2 else self.Vec(0, -9.81, 0)
        # simulation properties
        self.pic_ratio = pic_r
        self.rho = 0.01
        self.particles = Create_FLIP_Particles(self.Vec())  # create flip particles!

        self.low = self.Vec(0.0, 0.0) if self.dim == 2 else self.Vec(0, 0, 0)
        self.up = self.Vec(1.0, 1.0) if self.dim == 2 else self.Vec(1, 1, 1)

    def add_particles(self, lower, upper):
        self.ppc = 4.0  # how many particles in one cell
        vel = self.Vec(0, 0) if self.dim == 2 else self.Vec(0, 0, 0)
        # print("vel:", vel)
        handle = Sampler.Uniform_Sample_In_Box_flip(self.particles, lower, upper, vel, self.dx, self.ppc)
        return handle

    def advance_one_time_step(self, dt):
        # print("start!")
        _Type_Grid = ClassifyGridType(self.particles, self.dx, self.low, self.up)  # get the type grid
        # print("ClassifyGridType")

        _VX = Create_Vel_Grid(self.Vec())
        _VY = Create_Vel_Grid(self.Vec())
        _VZ = Create_Vel_Grid(self.Vec())

        # AddExternalA2P(self.particles, self.gravity, dt);

        if self.dim == 2:
            Particles_To_Grid_Flip(self.particles, self.dx, dt, _VX, _VY)  # particle to grid bilinear transfer
        elif self.dim == 3:
            Particles_To_Grid_Flip_S(self.particles, self.dx, dt, _VX, 0)
            Particles_To_Grid_Flip_S(self.particles, self.dx, dt, _VY, 1)
            Particles_To_Grid_Flip_S(self.particles, self.dx, dt, _VZ, 2)

        # print("Particles_To_Grid_Flip")

        _Prev_VX = Create_Vel_Grid(self.Vec())
        _Prev_VY = Create_Vel_Grid(self.Vec())

        if self.dim == 3:
            _Prev_VZ = Create_Vel_Grid(self.Vec())
            SavePrevVelGrid(_Prev_VZ, _VZ)

        SavePrevVelGrid(_Prev_VX,  _VX)
        SavePrevVelGrid(_Prev_VY,  _VY)
        # print("SavePrevVelGrid")

        AddExternalA(_Type_Grid, _VX, _VY, _VZ, self.gravity, dt)
        # print("AddExternalA")

        EnforceDirichlet(_VX, _VY, _VZ, _Type_Grid)
        # print("EnforceDirichlet")

        Extrapolating_Velocity(_VX, _Type_Grid, 2)
        Extrapolating_Velocity(_VY, _Type_Grid, 2)

        if self.dim == 3:
            Extrapolating_Velocity(_VZ, _Type_Grid, 2)
        # print("ExtrapolatingVelocityIndividual")

        EnforceDirichlet(_VX, _VY, _VZ, _Type_Grid)
        # print("EnforceDirichlet")

        SolvePressure(self.particles, _VX, _VY,  _VZ, _Type_Grid, self.rho, self.dx, dt, self.low, self.up)
        # print("SolvePressure")

        Extrapolating_Velocity(_VX, _Type_Grid, 2)
        Extrapolating_Velocity(_VY, _Type_Grid, 2)

        if self.dim == 3:
            Extrapolating_Velocity(_VZ, _Type_Grid, 2)
        # print("ExtrapolatingVelocityIndividual")

        EnforceDirichlet(_VX, _VY, _VZ, _Type_Grid)
        # print("EnforceDirichlet")

        _diffVX = Create_Vel_Grid(self.Vec())
        _diffVY = Create_Vel_Grid(self.Vec())

        if self.dim == 3:
            _diffVZ = Create_Vel_Grid(self.Vec())

        CreateDiffVelGrid(_Prev_VX, _VX, _diffVX)
        CreateDiffVelGrid(_Prev_VY, _VY, _diffVY)

        if self.dim == 3:
            CreateDiffVelGrid(_Prev_VZ, _VZ, _diffVZ)
        # print("CreateDiffVelGrid")

        if self.dim == 2:
            Grid_To_Particles_Flip(_VX, _VY, _diffVX, _diffVY, self.particles, self.dx, dt, self.pic_ratio)
        elif self.dim == 3:
            Grid_To_Particles_Flip_S(_VX, _diffVX, self.particles, self.dx, dt, self.pic_ratio, 0)
            Grid_To_Particles_Flip_S(_VY, _diffVY, self.particles, self.dx, dt, self.pic_ratio, 1)
            Grid_To_Particles_Flip_S(_VZ, _diffVZ, self.particles, self.dx, dt, self.pic_ratio, 2)

        # print("Grid_To_Particles_Flip")
        AdvectParticles(_Type_Grid, self.particles, self.dt, self.dx)
        # print("AdvectParticles")


    def run(self):
        super().run()

    def write(self, frame_idx):
        Visualizer.Dump_FLIP_Particles(self.particles, self.output_folder + "partio" + str(frame_idx) + ".bgeo")
        # Visualizer.Dump_Particles_WithLogJp(self.X, self.logJp, self.output_folder + "logJp" + str(frame_idx) + ".bgeo")