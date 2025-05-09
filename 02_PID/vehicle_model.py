import numpy as np

class VehicleModel(object):
    def __init__(self, step_time, R, force_ratio, force_bias, m=1.0):
        self.Y  = np.array([[1.0], [0.0]])
        self.A = np.array([[1.0, step_time], [0.0, 1.0]])
        self.B = np.array([[0],[step_time/m]])
        self.C = np.array([[1.0, 0.0]])
        self.r_f = force_ratio
        self.R = R
        self.bias = np.array([[0], [force_bias/m*step_time]])
        self.y_measure = self.C @ self.Y
    
    def ControlInput(self, u):
        self.Y = self.A @ self.Y  +  self.B * u * self.r_f + self.bias
        self.y_measure = self.C @ self.Y + np.random.normal(0.0, self.R)