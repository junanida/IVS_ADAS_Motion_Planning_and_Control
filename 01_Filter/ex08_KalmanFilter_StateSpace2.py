import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, y_measure_init, step_time=0.1, m=10.0, k=100.0, b=20.0, Q_x=0.02, Q_v=0.05, R=5.0, errorCovariance_init=10.0):
        # 시스템 파라미터 초기화
        self.A = np.array([[1.0, step_time],
                           [-step_time * k / m, 1.0 - (step_time * b / m)]])  # 상태 천이 행렬
        self.B = np.array([[0.0], [step_time / m]])  # 제어 입력 행렬
        self.C = np.array([[1.0, 0.0]])  # 측정 행렬
        self.D = 0.0  # 관찰 모델의 바이어스
        self.Q = np.array([[Q_x, 0.0], [0.0, Q_v]])  # 프로세스 잡음 공분산
        self.R = R  # 측정 잡음 공분산
        self.x_estimate = np.array([[y_measure_init], [0.0]])  # 초기 상태 추정값 (위치, 속도)
        self.P_estimate = np.array([[errorCovariance_init, 0.0],
                                     [0.0, errorCovariance_init]])  # 초기 오차 공분산

    def estimate(self, y_measure, input_u):
        # 예측 단계
        x_pred = self.A @ self.x_estimate + self.B * input_u  # 상태 예측
        P_pred = self.A @ self.P_estimate @ self.A.T + self.Q  # 오차 공분산 예측

        # 업데이트 단계 (칼만 이득 계산)
        S = self.C @ P_pred @ self.C.T + self.R  # 측정 오차
        K = P_pred @ self.C.T @ np.linalg.inv(S)  # 칼만 이득

        # 측정 업데이트
        y_pred = self.C @ x_pred  # 예측된 측정값
        self.x_estimate = x_pred + K @ (y_measure - y_pred)  # 상태 업데이트
        self.P_estimate = (np.eye(2) - K @ self.C) @ P_pred  # 오차 공분산 업데이트
        
if __name__ == "__main__":
    signal = pd.read_csv("01_filter/Data/example08.csv")

    y_estimate = KalmanFilter(signal.y_measure[0])
    for i, row in signal.iterrows():
        y_estimate.estimate(signal.y_measure[i],signal.u[i])
        signal.y_estimate[i] = y_estimate.x_estimate[0][0]

    plt.figure()
    plt.plot(signal.time, signal.y_measure,'k.',label = "Measure")
    plt.plot(signal.time, signal.y_estimate,'r-',label = "Estimate")
    plt.xlabel('time (s)')
    plt.ylabel('signal')
    plt.legend(loc="best")
    plt.axis("equal")
    plt.grid(True)
    plt.show()



