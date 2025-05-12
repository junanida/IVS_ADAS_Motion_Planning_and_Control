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

    class PD_Controller(object):
        def __init__(self, step_time, Kp=1.0, Kd=0.1):
            self.dt = step_time
            self.Kp = Kp
            self.Kd = Kd
            self.prev_error = 0.0
            self.u = 0.0

        def ControllerInput(self, reference_coeff, vx):
            # 1. Lookahead 위치의 reference y값 = f(x_look)
            x_look = 2.0
            y_look = reference_coeff[0]*x_look**3 + reference_coeff[1]*x_look**2 + reference_coeff[2]*x_look + reference_coeff[3]

            # 2. PD 제어기 (y_look: 목표 lateral 위치, ego Y=0)
            error = y_look - 0.0
            derivative = (error - self.prev_error) / self.dt
            delta_fb = self.Kp * error + self.Kd * derivative
            self.prev_error = error

            # 3. Feedforward 조향각 (곡률 기반)
            curvature = 6 * reference_coeff[0]*x_look + 2 * reference_coeff[1]  # y''(x) / (1 + y'(x)^2)^1.5 생략 버전
            L = 2.9  # wheelbase (m)
            delta_ff = np.arctan(L * curvature)

            self.u = delta_ff + delta_fb

    
    time = []
    X_ego = []
    Y_ego = []
    ego_vehicle = VehicleModel_Lat(step_time, Vx)

    frameconverter = Global2Local(num_point)
    polynomialfit = PolynomialFitting(num_degree,num_point)
    polynomialvalue = PolynomialValue(num_degree,np.size(x_local))
    # controller = PID_Controller_Kinematic(step_time, polynomialfit.coeff, Vx)
    controller = PD_Controller(step_time)
    
    for i in range(int(simulation_time/step_time)):
        time.append(step_time*i)
        # X_ego.append(ego_vehicle.X)
        # Y_ego.append(ego_vehicle.Y)
        X_ego.append(float(ego_vehicle.X))
        Y_ego.append(float(ego_vehicle.Y))
        X_ref_convert = np.arange(ego_vehicle.X, ego_vehicle.X+5.0, 1.0)
        Y_ref_convert = 2.0-2*np.cos(X_ref_convert/10)
        Points_ref = np.transpose(np.array([X_ref_convert, Y_ref_convert]))
        frameconverter.convert(Points_ref, ego_vehicle.Yaw, ego_vehicle.X, ego_vehicle.Y)
        polynomialfit.fit(frameconverter.LocalPoints)
        polynomialvalue.calculate(polynomialfit.coeff, x_local)
        controller.ControllerInput(polynomialfit.coeff, Vx)
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


