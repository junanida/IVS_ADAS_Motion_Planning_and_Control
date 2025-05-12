import numpy as np
import matplotlib.pyplot as plt

from VehicleModel_Long import VehicleModel_Long

class PID_Controller_Speed(object):
    def __init__(self, reference, measure, step_time, P_Gain=0.25, D_Gain=4.0, I_Gain=0.015):
        # Code
        self.reference = reference
        self.measure = measure
        self.step_time = step_time
        self.Kp = P_Gain
        self.Kd = D_Gain
        self.Ki = I_Gain
        
        self.prev_error = 0.0
        self.integral = 0.0
        self.u = 0.0
    def ControllerInput(self, reference, measure):
        # Code
        error = reference - measure
        self.integral += error * self.step_time
        derivative = (error - self.prev_error) / self.step_time
        
        self.u = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        
        self.prev_error = error

        

if __name__ == "__main__":
    
    step_time = 0.1
    simulation_time = 50.0
    m = 500.0
    
    reference_speed = 30.0
    Vx = []
    ax = []
    time = []
    plant = VehicleModel_Long(step_time, m, 0.5, 0.0, 0.0)
    controller = PID_Controller_Speed(reference_speed, plant.vx, step_time)
    for i in range(int(simulation_time/step_time)):
        time.append(step_time*i)
        Vx.append(plant.vx)
        ax.append(plant.ax)
        controller.ControllerInput(reference_speed,plant.vx)
        plant.update(controller.u)
        
    plt.figure(1)
    plt.plot(time, Vx,'r-',label = "Vx")
    plt.xlabel('time [s]')
    plt.ylabel('Vx [m/s]')
    plt.legend(loc="best")
    plt.axis("equal")
    plt.grid(True)
    plt.show()

    