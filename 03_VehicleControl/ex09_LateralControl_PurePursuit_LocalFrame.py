import numpy as np
import matplotlib.pyplot as plt

from VehicleModel_Lat import VehicleModel_Lat
from ex06_GlobalFrame2LocalFrame import Global2Local
from ex06_GlobalFrame2LocalFrame import PolynomialFitting
from ex06_GlobalFrame2LocalFrame import PolynomialValue

    
if __name__ == "__main__":
    step_time = 0.1
    simulation_time = 30.0
    Vx = 3.0
    X_ref = np.arange(0.0, 100.0, 0.1)
    Y_ref = 2.0-2*np.cos(X_ref/10)
    num_degree = 3
    num_point = 5
    x_local = np.arange(0.0, 10.0, 0.5)

    class PurePursuit:
        def __init__(self, step_time, Ld=2.0, wheelbase=2.5):
            self.Ld = Ld  # Look-ahead 거리
            self.L = wheelbase
            self.u = 0.0

        def ControllerInput(self, local_points):
            # 가장 가까운 look-ahead point 찾기
            target = None
            for pt in local_points:
                if pt[0] > 0 and np.hypot(pt[0], pt[1]) >= self.Ld:
                    target = pt
                    break
            if target is None:
                self.u = 0.0
                return

            lx, ly = target[0], target[1]
            # 조향각 계산 (Pure Pursuit 공식)
            self.u = np.arctan2(2 * self.L * ly, self.Ld ** 2)

    
    time = []
    X_ego = []
    Y_ego = []
    ego_vehicle = VehicleModel_Lat(step_time, Vx)

    frameconverter = Global2Local(num_point)
    polynomialfit = PolynomialFitting(num_degree,num_point)
    polynomialvalue = PolynomialValue(num_degree,np.size(x_local))
    #controller = PurePursuit(step_time, polynomialfit.coeff, Vx)
    controller = PurePursuit(step_time)
    for i in range(int(simulation_time/step_time)):
        time.append(step_time*i)
        X_ego.append(ego_vehicle.X)
        Y_ego.append(ego_vehicle.Y)
        X_ref_convert = np.arange(ego_vehicle.X, ego_vehicle.X+5.0, 1.0)
        Y_ref_convert = 2.0-2*np.cos(X_ref_convert/10)
        Points_ref = np.transpose(np.array([X_ref_convert, Y_ref_convert]))
        frameconverter.convert(Points_ref, ego_vehicle.Yaw, ego_vehicle.X, ego_vehicle.Y)
        polynomialfit.fit(frameconverter.LocalPoints)
        polynomialvalue.calculate(polynomialfit.coeff, x_local)
        # controller.ControllerInput(polynomialfit.coeff, Vx)
        controller.ControllerInput(frameconverter.LocalPoints)
        ego_vehicle.update(controller.u, Vx)

        
    plt.figure(1)
    plt.plot(X_ref, Y_ref,'k-',label = "Reference")
    plt.plot(X_ego, Y_ego,'b-',label = "Position")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc="best")
#    plt.axis("best")
    plt.grid(True)    
    plt.show()


