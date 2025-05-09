from vehicle_model import VehicleModel
import numpy as np
import matplotlib.pyplot as plt

class PID_Controller(object):
    def __init__(self, target, y_init, step_time, Kp=0.5, Ki=0.0, Kd=1.0):
        # Code
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.target = target
        self.e_prev = target - y_init
        self.e_sum = 0.0
        self.step_time = step_time
        self.u = 0.0
    
    def ControllerInput(self, target, y_estimate):
        # Code
        e = target - y_estimate
        self.e_sum += e * self.step_time
        de = (e - self.e_prev) / self.step_time
        self.u = self.Kp * e + self.Ki * self.e_sum + self.Kd * de
        self.e_prev = e

class LowPassFilter:
    def __init__(self, y_init, alpha=0.2):  # alpha = filter gain
        # Code
        self.y_estimate = y_init
        self.alpha = alpha
    
    def estimate(self, y_measure):
        # Code
        self.y_estimate = self.alpha * y_measure + (1 - self.alpha) * self.y_estimate


if __name__ == "__main__":
    target_y = 0.0
    measure_y =[]
    estimated_y = []
    time = []
    step_time = 0.1
    simulation_time = 30   
    plant = VehicleModel(step_time, 0.25, 0.99, 0.05)
    estimator = LowPassFilter(plant.y_measure[0][0])
    controller = PID_Controller(target_y, plant.y_measure[0][0], step_time)
    
    for i in range(int(simulation_time/step_time)):
        time.append(step_time*i)
        measure_y.append(plant.y_measure[0][0])
        estimated_y.append(estimator.y_estimate)
        estimator.estimate(plant.y_measure[0][0])
        controller.ControllerInput(target_y, estimator.y_estimate)
        plant.ControlInput(controller.u)
    
    plt.figure()
    plt.plot([0, time[-1]], [target_y, target_y], 'k-', label="reference")
    plt.plot(time, measure_y,'r-',label = "Vehicle Position(Measure)")
    plt.plot(time, estimated_y,'c-',label = "Vehicle Position(Estimator)")
    plt.xlabel('time (s)')
    plt.ylabel('signal')
    plt.legend(loc="best")
    plt.axis("equal")
    plt.grid(True)
    plt.show()
