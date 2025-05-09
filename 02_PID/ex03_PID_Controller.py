from vehicle_model import VehicleModel
import numpy as np
import matplotlib.pyplot as plt

class PID_Controller(object):
    def __init__(self, reference, measure, step_time, P_Gain=1.0, D_Gain=1.2, I_Gain=0.05):
        # Code
        self.Kp = P_Gain # 비례 이득
        self.Kd = D_Gain # 미분 이득
        self.Ki = I_Gain # 적분 이득
        self.step_time = step_time # 시간 간격
        self.prev_error = reference - measure # 이전 오차
        self.integral = 0.0 # 적분값 초기화
        self.u = 0.0 # 제어 입력 초기화
    
    def ControllerInput(self, reference, measure):
        # Code
        error = reference - measure # 현재 오차
        self.integral += error * self.step_time # 적분 계산
        derivative = (error - self.prev_error) / self.step_time # 미분 계산

        # PID 제어 입력
        self.u = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        self.prev_error = error # 이전 오차
        
        
if __name__ == "__main__":
    target_y = 0.0
    measure_y =[]
    time = []
    step_time = 0.1
    simulation_time = 30   
    plant = VehicleModel(step_time, 0.0, 0.99, 0.05)
    controller = PID_Controller(target_y, plant.y_measure[0][0], step_time)
    
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
