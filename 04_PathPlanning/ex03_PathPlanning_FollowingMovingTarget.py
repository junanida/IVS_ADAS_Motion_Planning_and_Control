import numpy as np
import matplotlib.pyplot as plt
from lane_1 import lane
from ex01_PathPlanning_BothLane import Global2Local, Polyfit, Polyval, BothLane2Path, VehicleModel_Lat, PurePursuit

class LeadingVehiclePos(object):
    def __init__(self, num_data_store=5):
        self.max_num_array = num_data_store
        self.PosArray = [] 

    def update(self, pos_lead, Vx, yawrate, step_time):
        if len(self.PosArray) >= self.max_num_array:
            self.PosArray.pop(0)
        self.PosArray.append([pos_lead[0][0], pos_lead[0][1], Vx, yawrate])

def HeadingAngleEstimation(coeff_path, PosArray):
    # Using the position array to estimate heading angle
    if len(PosArray) < 2:
        return 0.0  # Return a default angle if not enough data
    delta_x = PosArray[-1][0] - PosArray[-2][0]
    delta_y = PosArray[-1][1] - PosArray[-2][1]
    angle = np.arctan2(delta_y, delta_x)
    return angle

def TargetFollowingPath(PosArray):
    if len(PosArray) < 1:
        return None  # No valid path to follow
    
    # Using the position data to create a following path
    lead_pos = PosArray[-1]
    target_x = lead_pos[0] + 5.0  # Follow the leading vehicle with 5m offset ahead
    target_y = lead_pos[1]
    coeff_path = np.polyfit([lead_pos[0], target_x], [lead_pos[1], target_y], 1)  # Linear path
    return coeff_path

if __name__ == "__main__":
    step_time = 0.1
    simulation_time = 30.0
    Vx = 3.0
    X_lane = np.arange(0.0, 300.0, 0.1)
    Y_lane_L, Y_lane_R = lane(X_lane)
    
    leading_vehicle = VehicleModel_Lat(step_time, Vx)
    ego_vehicle = VehicleModel_Lat(step_time, Vx, Pos=[-10.0, 0.0, 0.0])
    controller_lead = PurePursuit()
    controller_ego = PurePursuit()
    leading_vehicle_pos = LeadingVehiclePos()
    
    time = []
    X_lead = []
    Y_lead = []
    X_ego = []
    Y_ego = []
    plt.figure(figsize=(13, 2))
    
    for i in range(int(simulation_time / step_time)):
        time.append(step_time * i)
        X_lead.append(leading_vehicle.X)
        Y_lead.append(leading_vehicle.Y)
        X_ego.append(ego_vehicle.X)
        Y_ego.append(ego_vehicle.Y)

        # Lane for leading vehicle (not used for path planning but just for reference)
        X_ref = np.arange(leading_vehicle.X, leading_vehicle.X + 300.0, 1.0)
        Y_ref_L, Y_ref_R = lane(X_ref)
        global_points_L = np.transpose(np.array([X_ref, Y_ref_L])).tolist()
        global_points_R = np.transpose(np.array([X_ref, Y_ref_R])).tolist()
        local_points_L = Global2Local(global_points_L, leading_vehicle.Yaw, leading_vehicle.X, leading_vehicle.Y)
        local_points_R = Global2Local(global_points_R, leading_vehicle.Yaw, leading_vehicle.X, leading_vehicle.Y)
        coeff_L = Polyfit(local_points_L, num_order=3)
        coeff_R = Polyfit(local_points_R, num_order=3)
        coeff_path_lead = BothLane2Path(coeff_L, coeff_R)

        # Path for ego vehicle following the leading vehicle
        pos_lead = Global2Local([[leading_vehicle.X, leading_vehicle.Y]], ego_vehicle.Yaw, ego_vehicle.X, ego_vehicle.Y)
        leading_vehicle_pos.update(pos_lead, Vx, ego_vehicle.yawrate, step_time)
        coeff_path_ego = TargetFollowingPath(leading_vehicle_pos.PosArray)

        # Controller input for the leading vehicle
        controller_lead.ControllerInput(coeff_path_lead, Vx)
        controller_ego.ControllerInput(coeff_path_ego, Vx)
        
        # Update both leading and ego vehicle
        leading_vehicle.update(controller_lead.u, Vx)
        ego_vehicle.update(controller_ego.u, Vx)

        # Plot the vehicle positions
        plt.plot(ego_vehicle.X, ego_vehicle.Y, 'bo')
        plt.plot(leading_vehicle.X, leading_vehicle.Y, 'ro')
        plt.axis("equal")
        plt.pause(0.01)

    plt.show()
