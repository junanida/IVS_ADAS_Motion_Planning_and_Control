import numpy as np
import matplotlib.pyplot as plt

from VehicleModel_Lat import VehicleModel_Lat
from ex06_GlobalFrame2LocalFrame import Global2Local
from ex06_GlobalFrame2LocalFrame import PolynomialFitting
from ex06_GlobalFrame2LocalFrame import PolynomialValue


if __name__ == "__main__":
    step_time = 0.1
    simulation_time = 30.0
    Vx = 3.0  # 차량 속도
    X_ref = np.arange(0.0, 100.0, 0.1)
    Y_ref = 2.0 - 2 * np.cos(X_ref / 10)
    num_degree = 3
    num_point = 5
    x_local = np.arange(0.0, 10.0, 0.5)

    class StanleyMethod(object):
        def __init__(self, step_time, k=3.5, wheelbase=2.5):
            self.k = k  # Stanley Method gain (측면 오차에 대한 보정 정도)
            self.L = wheelbase  # 차량 축간 거리
            self.u = 0.0  # 조향 각 (스티어링 입력)

        def ControllerInput(self, local_points, yaw_angle, vx):
            # Lookahead 거리 내 가장 가까운 점 탐색
            target = None
            lookahead_dist = 2.0  # 조정 가능한 Lookahead 거리
            for pt in local_points:
                if pt[0] > 0 and np.hypot(pt[0], pt[1]) >= lookahead_dist:
                    target = pt
                    break

            if target is None:
                self.u = 0.0
                return

            lx, ly = target[0], target[1]

            # 경로 각도 계산
            path_yaw = np.arctan2(ly, lx + 1e-6)

            # Stanley Controller 계산
            delta = path_yaw + np.arctan2(self.k * ly, vx + 1e-5)
            self.u = delta

    # 시뮬레이션 변수 초기화
    time = []
    X_ego = []
    Y_ego = []
    ego_vehicle = VehicleModel_Lat(step_time, Vx)

    frameconverter = Global2Local(num_point)
    polynomialfit = PolynomialFitting(num_degree, num_point)
    polynomialvalue = PolynomialValue(num_degree, np.size(x_local))

    # 컨트롤러 초기화
    controller = StanleyMethod(step_time)  # k 값 조정

    # 시뮬레이션 루프
    for i in range(int(simulation_time / step_time)):
        time.append(step_time * i)
        X_ego.append(ego_vehicle.X)
        Y_ego.append(ego_vehicle.Y)

        # Reference 경로를 Local Frame으로 변환
        X_ref_convert = np.arange(ego_vehicle.X, ego_vehicle.X + 5.0, 1.0)
        Y_ref_convert = 2.0 - 2 * np.cos(X_ref_convert / 10)
        Points_ref = np.transpose(np.array([X_ref_convert, Y_ref_convert]))
        frameconverter.convert(Points_ref, ego_vehicle.Yaw, ego_vehicle.X, ego_vehicle.Y)

        # Stanley Controller 입력 생성
        controller.ControllerInput(frameconverter.LocalPoints, ego_vehicle.Yaw, Vx)

        # 차량 상태 업데이트
        ego_vehicle.update(controller.u, Vx)

    # 결과 그래프 출력
    plt.figure(1)
    plt.plot(X_ref, Y_ref, 'k-', label="Reference")
    plt.plot(X_ego, Y_ego, 'b-', label="Position")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()
