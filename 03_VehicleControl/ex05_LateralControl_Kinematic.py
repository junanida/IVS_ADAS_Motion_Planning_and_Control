import numpy as np
import matplotlib.pyplot as plt

from VehicleModel_Lat import VehicleModel_Lat

class PID_Controller_Kinematic(object):
    def __init__(self, step_time, target_y, current_y, P_Gain=0.8, D_Gain=1.2, I_Gain=0.001):
        # Code
        self.step_time = step_time
        self.target_y = target_y
        self.current_y = current_y

        # PID 파라미터
        self.P_Gain = P_Gain
        self.D_Gain = D_Gain
        self.I_Gain = I_Gain

        # PID 제어 변수
        self.prev_error = 0.0
        self.integral = 0.0
        self.u = 0.0  # 스티어링 각도 입력

    def ControllerInput(self, target_y, current_y):
        # Code
        # 오차 계산
        error = target_y - current_y

        # 비례 항
        P_term = self.P_Gain * error

        # 적분 항
        self.integral += error * self.step_time
        I_term = self.I_Gain * self.integral

        # 미분 항
        derivative = (error - self.prev_error) / self.step_time
        D_term = self.D_Gain * derivative

        # 제어 입력 계산 (스티어링 각도 u)
        self.u = P_term + I_term + D_term

        # 이전 오차 업데이트
        self.prev_error = error

    
if __name__ == "__main__":
    step_time = 0.1
    simulation_time = 30.0
    Vx = 3.0
    Y_ref = 4.0
    
    time = []
    X_ego = []
    Y_ego = []
    ego_vehicle = VehicleModel_Lat(step_time, Vx)
    controller = PID_Controller_Kinematic(step_time, Y_ref, ego_vehicle.Y)
    for i in range(int(simulation_time/step_time)):
        time.append(step_time*i)
        X_ego.append(ego_vehicle.X)
        Y_ego.append(ego_vehicle.Y)
        controller.ControllerInput(Y_ref, ego_vehicle.Y)
        ego_vehicle.update(controller.u, Vx)

        
    plt.figure(1)
    plt.plot(X_ego, Y_ego,'b-',label = "Position")
    plt.plot([0, X_ego[-1]], [Y_ref, Y_ref], 'k:',label = "Reference")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc="best")
#    plt.axis("best")
    plt.grid(True)    
    plt.show()


