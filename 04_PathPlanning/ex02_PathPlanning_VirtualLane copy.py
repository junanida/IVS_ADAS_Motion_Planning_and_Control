import numpy as np
import matplotlib.pyplot as plt
from lane_2 import lane
from ex01_PathPlanning_BothLane import Global2Local, Polyfit, VehicleModel_Lat, PurePursuit
prev_coeff_path = [0.0, 0.0, 0.0, 0.0]

class LaneWidthEstimator(object):
    def __init__(self, Lw_init=3.0):
        self.Lw = Lw_init
        self.valid = False  # 이번 frame에서 update 성공 여부

    def update(self, coeff_L, coeff_R, isLaneValid_L, isLaneValid_R):
        self.valid = False

        # 유효한 포인트 수가 충분한지 검사
        min_valid_points = 5
        if (np.sum(isLaneValid_L) < min_valid_points) or (np.sum(isLaneValid_R) < min_valid_points):
            return

        mid_x = np.linspace(0.0, 5.0, num=10)
        y_L = np.polyval(coeff_L, mid_x)
        y_R = np.polyval(coeff_R, mid_x)

        # NaN/Inf 제거
        valid_mask = (
            ~np.isnan(y_L) & ~np.isnan(y_R) &
            ~np.isinf(y_L) & ~np.isinf(y_R)
        )
        y_L = y_L[valid_mask]
        y_R = y_R[valid_mask]

        if len(y_L) < 3:
            return

        lane_widths = np.abs(y_R - y_L)
        mean_width = np.mean(lane_widths)

        if 2.0 <= mean_width <= 5.0:
            self.Lw = 0.9 * self.Lw + 0.1 * mean_width
            self.valid = True

def EitherLane2Path(coeff_L, coeff_R, isLaneValid_L, isLaneValid_R, Lw):
    MIN_LANE_WIDTH = 2.5 
    valid_L = np.sum(isLaneValid_L) >= 5
    valid_R = np.sum(isLaneValid_R) >= 5

    if valid_L and valid_R:
        coeff = (np.array(coeff_L) + np.array(coeff_R)) / 2.0
        return coeff
    elif valid_L:
        shifted_coeff = np.array(coeff_L, dtype=float)
        shifted_coeff[-1] -= max(Lw, MIN_LANE_WIDTH) / 2.0
        return shifted_coeff
    elif valid_R:
        shifted_coeff = np.array(coeff_R, dtype=float)
        shifted_coeff[-1] += max(Lw, MIN_LANE_WIDTH) / 2.0
        return shifted_coeff
    else:
        return None

        
if __name__ == "__main__":
    step_time = 0.1
    simulation_time = 30.0
    Vx = 3.0
    X_lane = np.arange(0.0, 100.0, 0.1)
    Y_lane_L, Y_lane_R, Valid_L, Valid_R = lane(X_lane)
    
    LaneWidth = LaneWidthEstimator()
    ego_vehicle = VehicleModel_Lat(step_time, Vx)
    controller = PurePursuit()
    
    time = []
    X_ego = []
    Y_ego = []
    
    for i in range(int(simulation_time/step_time)):
        time.append(step_time*i)
        X_ego.append(ego_vehicle.X)
        Y_ego.append(ego_vehicle.Y)
        # Lane Info
        X_ref = np.arange(ego_vehicle.X, ego_vehicle.X+5.0, 1.0)
        Y_ref_L, Y_ref_R, isLaneValid_L, isLaneValid_R = lane(X_ref)
        # Global points (front 5 meters from the ego vehicle)
        global_points_L = np.transpose(np.array([X_ref, Y_ref_L])).tolist()
        global_points_R = np.transpose(np.array([X_ref, Y_ref_R])).tolist()
        # Converted to local frame
        local_points_L = Global2Local(global_points_L, ego_vehicle.Yaw, ego_vehicle.X, ego_vehicle.Y)
        local_points_R = Global2Local(global_points_R, ego_vehicle.Yaw, ego_vehicle.X, ego_vehicle.Y)
        # 3th order fitting
        coeff_L = Polyfit(local_points_L, num_order=3)
        coeff_R = Polyfit(local_points_R, num_order=3)
        # Lane to path
        LaneWidth.update(coeff_L, coeff_R, isLaneValid_L, isLaneValid_R)
        coeff_path = EitherLane2Path(coeff_L, coeff_R, isLaneValid_L, isLaneValid_R, LaneWidth.Lw)
        
        # Controller input
        controller.ControllerInput(coeff_path, Vx)
        ego_vehicle.update(controller.u, Vx)
        
    plt.figure(1, figsize=(13,2))
    plt.plot(X_lane, Y_lane_L,'k--')
    plt.plot(X_lane, Y_lane_R,'k--',label = "Reference")
    plt.plot(X_ego, Y_ego,'b.',label = "Vehicle Position")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc="best")
    plt.grid(True)    
    plt.show()
