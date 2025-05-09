from vehicle_model import VehicleModel
import numpy as np
import matplotlib.pyplot as plt

class P_Controller(object):
    def __init__(self, P_Gain=0.5):
        # Code
        self.Kp = P_Gain  # P 이득 저장
        self.u = 0.0      # 제어 입력 초기값
        
    def ControllerInput(self, reference, measure):
        # Code
        error = reference - measure  # 오차 계산
        self.u = self.Kp * error     # P 제어기 법칙


if __name__ == "__main__":
    target_y = 0.0
    measure_y =[]
    time = []
    step_time = 0.1
    simulation_time = 30   
    plant = VehicleModel(step_time, 0.0, 0.99, 0.1)
    controller = P_Controller()
    
    for i in range(int(simulation_time/step_time)):
        time.append(step_time*i)
        measure_y.append(plant.y_measure[0][0])
        controller.ControllerInput(target_y, plant.y_measure[0][0])
        plant.ControlInput(controller.u)
    
    plt.figure()
    plt.plot([0, time[-1]], [target_y, target_y], 'k-', label="reference")
    plt.plot(time, measure_y,'r-',label = "Vehicle Position")
    plt.xlabel('time (s)')
    plt.ylabel('signal')
    plt.legend(loc="best")
    plt.axis("equal")
    plt.grid(True)
    plt.show()
