import sys
sys.path.insert(0, "../../build")
import os
try:
    os.mkdir("output")
except OSError:
    pass

from JGSL import *
from JGSL_MPM import *
from .SimulationBase import SimulationBase

class MPMSimulationModel:
    def __init__(self, sim, model):
        self.sim = sim
        self.model = model
        self.storage = model.Create()
        self.updated_storage = None

    def append(self):
        # Please overwrite this function when you have a new model
        pass
    
    def compute_stress(self, stress_storage):
        self.model.Compute_Kirchoff_Stress(self.storage, stress_storage)

    def evolve_strain(self, particles, dt):
        self.model.Evolve_Strain(self.storage, particles, dt)

class FixedCorotatedModel(MPMSimulationModel):
    def __init__(self, sim):
        model = FIXED_COROTATED_2 if sim.dim == 2 else FIXED_COROTATED_3
        super().__init__(sim, model)

    def append(self, handle, vol0, E = 4000000, nu = 0.3):
        self.model.Append(self.storage, handle, vol0, E, nu)

class EquationOfStateModel(MPMSimulationModel):
    def __init__(self, sim):
        model = EQUATION_OF_STATE_2 if sim.dim == 2 else EQUATION_OF_STATE_3
        super().__init__(sim, model)

    def append(self, handle, vol0, bulk = 400000, gamma = 7):
        self.model.Append(self.storage, handle, vol0, bulk, gamma)

class LinearCorotatedModel(MPMSimulationModel):
    def __init__(self, sim):
        model = LINEAR_COROTATED_2 if sim.dim == 2 else LINEAR_COROTATED_3
        super().__init__(sim, model)

    def append(self, handle, vol0, E = 50, nu = 0.3):
        self.model.Append(self.storage, handle, vol0, E, nu)

class NeohookeanBordenModel(MPMSimulationModel):
    def __init__(self, sim):
        model = NEOHOOKEAN_BORDEN_2 if sim.dim == 2 else NEOHOOKEAN_BORDEN_3
        super().__init__(sim, model)

    def append(self, handle, vol0, E = 300, nu = 0.2):
        self.model.Append(self.storage, handle, vol0, E, nu)

class ParticleHandle:
    def __init__(self, sim, handle, vol0):
        self.sim = sim
        self.handle = handle
        self.vol0 = vol0

    def of_model(self, model_name, **kwargs):
        self.sim.models[model_name].append(self.handle, self.vol0, **kwargs)
        return self

    def of_combined_model(self, model_name1, model_name2, **kwargs):
        self.sim.models[model_name1].append(self.handle, self.vol0, **kwargs)
        self.sim.models[model_name2].append(self.handle, self.vol0, **kwargs)
        return self

class MPMSimulationBase(SimulationBase):
    def __init__(self, precision, dim):
        super().__init__(precision, dim)
        self.particles = Create_MPM_Particles(self.Vec())
        if dim == 2:
            self.collider = COLLIDER_2D()
        else:
            self.collider = COLLIDER_3D()
        self.ppc = 4.0

        self.models = {
            "fixed_corotated": FixedCorotatedModel(self),
            "equation_of_state": EquationOfStateModel(self),
            "linear_corotated": LinearCorotatedModel(self),
            "neohookean_borden": NeohookeanBordenModel(self)
        }

        self.dof = None
        self.newton_tol = 1e-1
        self.newton_maxiter = 10
        self.linesearch = False

    def add_particles(self, lower, upper, rho):
        vol0 = self.dx * self.dx / self.ppc
        m0 = rho * vol0
        if self.dim == 2:
            handle = Sampler.Uniform_Sample_In_Box(self.particles, lower, upper, self.Vec(0, 0), m0, self.dx, self.ppc)
        else:
            handle = Sampler.Uniform_Sample_In_Box(self.particles, lower, upper, self.Vec(0, 0, 0), m0, self.dx, self.ppc)
        return ParticleHandle(self, handle, vol0)

    def add_collision_object(self, shape):
        self.collider.Add(shape)

    def run(self):
        self.stress = Create_MPM_Stress(self.particles)
        super().run()

    def update_state(self, dv, grid, dt):
        grad_v = Compute_GradV(self.dof, dv, grid, self.particles, self.dx)
        for model in self.models.values():
            model.update_strain(grad_v, self.particles, dt)
    
    def total_energy(self, dv, grid, dt):
        E = Inertia_Energy(self.dof, dv, grid, self.gravity, dt)
        for model in self.models.values():
            E += model.elastic_energy()
        return E
    
    def implicit_euler(self, grid, dt):
        fcr = self.models['fixed_corotated'].storage
        eos = self.models['equation_of_state'].storage
        lcr = self.models['linear_corotated'].storage
        nkb = self.models['neohookean_borden'].storage
        fcr_moved = Copy_Elasticity_Attr(fcr)
        eos_moved = Copy_Elasticity_Attr(eos)
        lcr_moved = Copy_Elasticity_Attr(lcr)
        nkb_moved = Copy_Elasticity_Attr(nkb)
        Implicit_Update(self.particles, grid, fcr, eos, lcr, nkb, fcr_moved, eos_moved, lcr_moved, nkb_moved, self.collider, self.gravity, self.dx, dt, 1)

    def advance_one_time_step(self, dt):
        Clear_MPM_Stress(self.stress)
        
        for model in self.models.values():
            model.compute_stress(self.stress)

        _V_and_m = Particles_To_Grid(self.particles, self.stress, self.gravity, self.dx, dt, self.symplectic)
        Multi_Object_Collision(_V_and_m, self.collider, self.dx)
        if not self.symplectic:
            self.implicit_euler(_V_and_m, dt)

        Grid_To_Particles(_V_and_m, self.particles, self.dx, dt)

        for model in self.models.values():
            model.evolve_strain(self.particles, dt)

    def write(self, frame_idx):
        Visualizer.Dump_Particles(self.particles, self.output_folder + "partio" + str(frame_idx) + ".bgeo")
        #Visualizer.Dump_Particles_WithLogJp(self.X, self.logJp, self.output_folder + "logJp" + str(frame_idx) + ".bgeo")