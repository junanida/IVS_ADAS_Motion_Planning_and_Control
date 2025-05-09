from vehicle_model import VehicleModel
import numpy as np
import matplotlib.pyplot as plt

class PID_Controller(object):
    def __init__(self, target, y_init, step_time, Kp=1.0, Ki=0.05, Kd=1.2):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.target = target
        self.e_prev = target - y_init
        self.e_sum = 0.0
        self.step_time = step_time
        self.u = 0.0
    
    def ControllerInput(self, target, y_estimate):
        e = target - y_estimate
        self.e_sum += e * self.step_time
        de = (e - self.e_prev) / self.step_time
        self.u = self.Kp * e + self.Ki * self.e_sum + self.Kd * de
        self.e_prev = e
        
class KalmanFilter:
    def __init__(self, y_init, dt=0.1):
        # 상태: x = [y; y_dot]
        self.dt = dt
        self.x_estimate = np.array([[y_init], [0.0]])  # 초기 위치, 속도
        self.P = np.eye(2)  # 오차 공분산
        
        # 시스템 모델
        self.A = np.array([[1, dt],
                           [0, 1]])
        self.B = np.array([[0],
                           [dt]])
        self.H = np.array([[1, 0]])  # 측정 모델
        
        # 공분산
        self.Q = np.array([[5e-4, 0],
                           [0, 5e-4]])  # process noise
        self.R = np.array([[1e-6]])      # measurement noise

    def estimate(self, z, u):
        # 예측
        self.x_estimate = self.A @ self.x_estimate + self.B * u
        self.P = self.A @ self.P @ self.A.T + self.Q
        
        # 칼만 게인
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        
        # 업데이트
        y_tilde = z - (self.H @ self.x_estimate)
        self.x_estimate = self.x_estimate + K @ y_tilde
        self.P = (np.eye(2) - K @ self.H) @ self.P
        
        
if __name__ == "__main__":
    target_y = 0.0
    measure_y =[]
    estimated_y = []
    time = []
    step_time = 0.1
    simulation_time = 30   
    plant = VehicleModel(step_time, 0.25, 0.99, 0.05)
    estimator = KalmanFilter(plant.y_measure[0][0])
    controller = PID_Controller(target_y, plant.y_measure[0][0], step_time)
    
    for i in range(int(simulation_time/step_time)):
        time.append(step_time*i)
        measure_y.append(plant.y_measure[0][0])
        estimated_y.append(estimator.x_estimate[0][0])
        estimator.estimate(plant.y_measure[0][0],controller.u)
        controller.ControllerInput(target_y, estimator.x_estimate[0][0])
        plant.ControlInput(controller.u)
    
    plt.figure()
    plt.plot([0, time[-1]], [target_y, target_y], 'k-', label="reference")
    plt.plot(time, measure_y,'r:',label = "Vehicle Position(Measure)")
    plt.plot(time, estimated_y,'c-',label = "Vehicle Position(Estimator)")
    plt.xlabel('time (s)')
    plt.ylabel('signal')
    plt.legend(loc="best")
    plt.axis("equal")
    plt.grid(True)
    plt.show()
