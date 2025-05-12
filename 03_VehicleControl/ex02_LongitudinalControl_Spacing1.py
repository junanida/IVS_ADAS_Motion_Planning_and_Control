import numpy as np
import matplotlib.pyplot as plt

from VehicleModel_Long import VehicleModel_Long

class PID_Controller_ConstantSpace(object):
    def __init__(self, step_time, target_x, ego_x, constantSpace=30.0, P_Gain=0.4, D_Gain=0.25, I_Gain=0.07):
        self.space = constantSpace
        # Code
        self.P_Gain = P_Gain       # 비례 이득
        self.D_Gain = D_Gain       # 미분 이득
        self.I_Gain = I_Gain       # 적분 이득
        self.step_time = step_time
        self.prev_error = 0.0      # 이전 오차
        self.integral = 0.0        # 누적 적분 오차
        self.u = 0.0               # 제어 입력 (가속도/감속도)
    
    def ControllerInput(self, target_x, ego_x):
        # Code
        # 오차 계산
        error = (target_x - ego_x) - self.space

        # 비례 항 계산
        P_term = self.P_Gain * error

        # 적분 항 계산
        self.integral += error * self.step_time
        I_term = self.I_Gain * self.integral

        # 미분 항 계산
        derivative = (error - self.prev_error) / self.step_time
        D_term = self.D_Gain * derivative

        # 제어 입력 계산
        self.u = P_term + I_term + D_term

        # 이전 오차 업데이트
        self.prev_error = error
        

if __name__ == "__main__":
    
    step_time = 0.1
    simulation_time = 50.0
    m = 500.0
    
    vx_ego = []
    vx_target = []
    x_space = []
    time = []
    target_vehicle = VehicleModel_Long(step_time, m, 0.0, 30.0, 10.0)
    ego_vehicle = VehicleModel_Long(step_time, m, 0.5, 0.0, 10.0)
    controller = PID_Controller_ConstantSpace(step_time, target_vehicle.x, ego_vehicle.x)
    for i in range(int(simulation_time/step_time)):
        time.append(step_time*i)
        vx_ego.append(ego_vehicle.vx)
        vx_target.append(target_vehicle.vx)
        x_space.append(target_vehicle.x-ego_vehicle.x)
        controller.ControllerInput(target_vehicle.x, ego_vehicle.x)
        ego_vehicle.update(controller.u)
        target_vehicle.update(0.0)
        
        
    plt.figure(1)
    plt.plot(time, vx_ego,'r-',label = "ego_vx [m/s]")
    plt.plot(time, vx_target,'b-',label = "target_vx [m/s]")
    plt.xlabel('time [s]')
    plt.ylabel('Vx')
    plt.legend(loc="best")
    plt.axis("equal")
    plt.grid(True)
    
    plt.figure(2)
    plt.plot([0, time[-1]], [controller.space, controller.space], 'k-', label="reference")
    plt.plot(time, x_space,'b-',label = "space [m]")
    plt.xlabel('time [s]')
    plt.ylabel('x')
    plt.legend(loc="best")
    plt.axis("equal")
    plt.grid(True)    
    
    plt.show()

    