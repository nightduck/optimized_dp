import heterocl as hcl
from odp.computeGraphs.CustomGraphFunctions import my_abs, power, my_sqrt

""" Single DOF Bouc-Wen Model, augmented to couple nonlinear and linear dynamics

u_dot = u_dot
u_ddot = (input - c * u_dot - alpha * k * u - (1 - alpha) * k * z) / m
z_dot = A * u_dot - beta * |u_dot| * |z|^(n-1) - gamma * u_dot * |z|^n
u_dot_linear = u_dot_linear
u_ddot_linear = (input - c * u_dot_linear - alpha * k * u_linear - (1 - alpha) * k * z_linear) / m
z_dot_linear = A * u_dot_linear

"""

class SphereGrower:
    def __init__(self, x=[0,0,0], uMin = [-1], uMax = [1], dMin = [-0.25], dMax = [0.25], uMode="min", dMode="max"):
        """
        Creates a Bouc-Wen model with the following states:
        u: displacement
        u_dot: velocity
        z: hysteric displacement
        u_linear: displacement, modelled linearly
        u_dot_linear: velocity, modelled linearly
        z_linear: hysteric displacement, modelled linearly
        """
        self.x = x
        self.uMax = uMax
        self.uMin = uMin
        self.dMax = dMax
        self.dMin = dMin
        assert(uMode in ["min", "max"])
        self.uMode = uMode
        if uMode == "min":
            assert(dMode == "max")
        else:
            assert(dMode == "min")
        self.dMode = dMode

    def opt_ctrl(self, t, state, spat_deriv):
        """
        """
        opt_f = hcl.scalar(self.uMax[0], "opt_f")
        in2 = hcl.scalar(0, "in2")
        in3 = hcl.scalar(0, "in3")
        in4 = hcl.scalar(0, "in4")

        with hcl.if_(self.uMode == "min"):
            opt_f[0] = self.uMin[0]
        with hcl.if_(self.uMode == "max"):
            opt_f[0] = self.uMax[0]
        return (opt_f[0], in2[0], in3[0])

    def opt_dstb(self, t, state, spat_deriv):
        """
        """
        opt_d = hcl.scalar(self.dMax[0], "opt_d")
        in2 = hcl.scalar(0, "in2")
        in3 = hcl.scalar(0, "in3")
        in4 = hcl.scalar(0, "in4")

        with hcl.if_(self.dMode == "min"):
            opt_d[0] = self.dMin[0]
        with hcl.elif_(self.dMode == "max"):
            opt_d[0] = self.dMax[0]
        return (opt_d[0], in2[0], in3[0])

    def dynamics(self, t, state, uOpt, dOpt):
        """
        """
        x_dot = hcl.scalar(0, "x_dot")
        y_dot = hcl.scalar(0, "y_dot")
        z_dot = hcl.scalar(0, "z_dot")

        normalized_direction = hcl.scalar(0, "normalized_direction")
        normalized_direction[0] = my_sqrt(state[0]*state[0] + state[1]*state[1] + state[2]*state[2])

        x_dot[0] = (uOpt[0] + dOpt[0]) * state[0] / normalized_direction[0]
        y_dot[0] = (uOpt[0] + dOpt[0]) * state[1] / normalized_direction[0]
        z_dot[0] = (uOpt[0] + dOpt[0]) * state[2] / normalized_direction[0]

        return (x_dot[0], y_dot[0], z_dot[0])
        
        
