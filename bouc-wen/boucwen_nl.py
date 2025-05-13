import heterocl as hcl
from odp.computeGraphs.CustomGraphFunctions import my_abs, power

""" Single DOF Bouc-Wen Model, augmented to couple nonlinear and linear dynamics

u_dot = u_dot
u_ddot = (input - c * u_dot - alpha * k * u - (1 - alpha) * k * z) / m
z_dot = A * u_dot - beta * |u_dot| * |z|^(n-1) - gamma * u_dot * |z|^n
u_dot_linear = u_dot_linear
u_ddot_linear = (input - c * u_dot_linear - alpha * k * u_linear - (1 - alpha) * k * z_linear) / m
z_dot_linear = A * u_dot_linear

"""

class BoucWenNL:
    def __init__(self, A, alpha, beta, gamma, k, c, n, m, x=[0,0,0], uMin = [-1], uMax = [1], dMin = [0], dMax = [0], uMode="min", dMode="max"):
        """
        Creates a Bouc-Wen model with the following states:
        u: displacement
        u_dot: velocity
        z: hysteric displacement
        u_linear: displacement, modelled linearly
        u_dot_linear: velocity, modelled linearly
        z_linear: hysteric displacement, modelled linearly
        """
        self.A = A
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.k = k
        self.c = c
        self.n = n
        self.m = m
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

        # Declare a variable
        a_term = hcl.scalar(0, "a_term")
        # use the scalar by indexing 0 everytime
        a_term[0] = spat_deriv[1] * (-self.c * state[1] - self.alpha * self.k * state[0] - (1 - self.alpha) * self.k * state[2]) / self.m

        if self.uMode == "max":
            with hcl.if_(spat_deriv[1] > 0): # If spat_deriv[1] > 0
                opt_f[0] = self.uMax[0]
            with hcl.elif_(spat_deriv[1] < 0): # If spat_deriv[1] < 0
                opt_f[0] = self.uMin[0]
        else:
            with hcl.if_(spat_deriv[1] > 0): # If spat_deriv[1] > 0
                opt_f[0] = self.uMin[0]
            with hcl.elif_(spat_deriv[1] < 0): # If spat_deriv[1] < 0
                opt_f[0] = self.uMax[0]

        # # with hcl.if_(a_term >= 0):
        # with hcl.if_(self.uMode == "min"):
        #     opt_f[0] = self.uMin[0]
        # # with hcl.elif_(a_term < 0):
        # with hcl.if_(self.uMode == "max"):
        #     opt_f[0] = self.uMax[0]
        return (opt_f[0], in2[0], in3[0])

    def opt_dstb(self, t, state, spat_deriv):
        """
        """
        opt_d = hcl.scalar(self.dMax[0], "opt_d")
        in2 = hcl.scalar(0, "in2")
        in3 = hcl.scalar(0, "in3")
        in4 = hcl.scalar(0, "in4")

        # Declare a variable
        b_term = hcl.scalar(0, "b_term")
        # use the scalar by indexing 0 everytime
        b_term[0] = spat_deriv[1] * (-self.c * state[1] - self.alpha * self.k * state[0] - (1 - self.alpha) * self.k * state[2]) / self.m

        if self.dMode == "max":
            with hcl.if_(spat_deriv[1] > 0): # If spat_deriv[1] > 0
                opt_d[0] = self.dMax[0]
            with hcl.elif_(spat_deriv[1] < 0): # If spat_deriv[1] < 0
                opt_d[0] = self.dMin[0]
        else:
            with hcl.if_(spat_deriv[1] > 0): # If spat_deriv[1] > 0
                opt_d[0] = self.dMin[0]
            with hcl.elif_(spat_deriv[1] < 0): # If spat_deriv[1] < 0
                opt_d[0] = self.dMax[0]
                
        # # with hcl.if_(b_term[0] >= 0):
        # with hcl.if_(self.dMode == "min"):
        #     opt_d[0] = self.dMin[0]
        # # with hcl.elif_(b_term[0] < 0):
        # with hcl.if_(self.dMode == "max"):
        #     opt_d[0] = self.dMax[0]
        
        return (opt_d[0], in2[0], in3[0])

    def dynamics(self, t, state, uOpt, dOpt):
        """
        """
        u_dot = hcl.scalar(0, "u_dot")
        u_ddot = hcl.scalar(0, "u_ddot")
        z_dot = hcl.scalar(0, "z_dot")

        u_dot[0] = state[1]
        u_ddot[0] = (uOpt[0] - self.c * u_dot[0] - self.alpha * self.k * state[0] - (1 - self.alpha) * self.k * state[2]) / self.m
        z_dot[0] = self.A * u_dot[0] - self.beta * my_abs(u_dot[0]) * power(state[2], self.n - 1) - self.gamma * u_dot[0] * power(state[2], self.n)

        return (u_dot[0], u_ddot[0], z_dot[0])
        
        
