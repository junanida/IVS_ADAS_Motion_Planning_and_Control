import numpy as np
import math
import matplotlib.pyplot as plt
import random
from map_3 import map

show_animation  = True

class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.heading = 0.0
        self.f = 0
        self.g = 0
        self.h = 0

# Check if position of node is same( if distance < threshold, regard as same node)
def isSamePosition(node_1, node_2, epsilon_position=0.1):
    dx = node_1.position[0] - node_2.position[0]
    dy = node_1.position[1] - node_2.position[1]
    return math.hypot(dx, dy) < epsilon_position # True or False

def isSameYaw(node_1, node_2, epsilon_yaw=0.05):
    dyaw = abs(node_1.heading - node_2.heading)
    dyaw = dyaw if dyaw < math.pi else 2*math.pi - dyaw
    return dyaw < epsilon_yaw # True or False

# Action set, Moving only forward direction              
def get_action(R,Vx,delta_time_step):
    yaw_rate = Vx/R
    distance_travel = Vx*delta_time_step
    # yaw_rate, delta_time_step, cost
    action_set = [[yaw_rate, delta_time_step, distance_travel], 
                  [-yaw_rate, delta_time_step, distance_travel],
                  [yaw_rate/2, delta_time_step, distance_travel],
                  [-yaw_rate/2, delta_time_step, distance_travel],
                  [0.0, delta_time_step, distance_travel]]
    return action_set

# Vehicle movement
def vehicle_move(position_parent, yaw_rate, delta_time, Vx):
    x_parent = position_parent[0]
    y_parent  = position_parent[1]
    yaw_parent = position_parent[2]
    # if yaw_rate != 0 (left or right turn)
    if abs(yaw_rate) > 1e-5:  # 회전
        R = Vx / yaw_rate
        cx = x_parent - R * math.sin(yaw_parent)
        cy = y_parent + R * math.cos(yaw_parent)
        delta_yaw = yaw_rate * delta_time
        yaw_child = yaw_parent + delta_yaw
        x_child = cx + R * math.sin(yaw_child)
        y_child = cy - R * math.cos(yaw_child)

    # move straight
    else:  # 직진
        x_child = x_parent + Vx * delta_time * math.cos(yaw_parent)
        y_child = y_parent + Vx * delta_time * math.sin(yaw_parent)
        yaw_child = yaw_parent
    # yaw processing
    if yaw_child > 2*np.pi:
        yaw_child = yaw_child - 2*np.pi
    if yaw_child < 0:
        yaw_child = yaw_child + 2*np.pi
    # return position : [x, y, yaw]
    return [x_child, y_child, yaw_child]

# Collision check : path overlaps with any of obstacle
def collision_check(position_parent, yaw_rate, delta_time_step, obstacle_list, Vx):
    num_checkpoints = 10
    for i in range(1, num_checkpoints + 1):
        dt = delta_time_step * (i / num_checkpoints)
        pos = vehicle_move(position_parent, yaw_rate, dt, Vx)
        x, y = pos[0], pos[1]
        for obs in obstacle_list:
            dx = x - obs[0]
            dy = y - obs[1]
            if math.hypot(dx, dy) <= obs[2]:
                return True
    return False


# Check if the node is in the searching space
def isNotInSearchingSpace(position_child, space):
    x, y = position_child[0], position_child[1]
    x_min, x_max, y_min, y_max = space
    return not (x_min <= x <= x_max and y_min <= y <= y_max)

def heuristic(cur_node, goal_node):
    dist = np.sqrt((cur_node.position[0] - goal_node.position[0])**2 + (cur_node.position[1]  - goal_node.position[1])**2)
    return dist

def a_star(start, goal, space, obstacle_list, R, Vx, delta_time_step, weight):
    start_node = Node(None, start)
    start_node.heading = start[2]  # heading 정보 포함
    start_node.g = 0
    start_node.h = heuristic(start_node, Node(None, goal))
    start_node.f = start_node.g + weight * start_node.h

    open_list = [start_node]
    closed_list = []

    while open_list:
        cur_node = min(open_list, key=lambda n: n.f)
        open_list.remove(cur_node)

        if isSamePosition(cur_node, Node(None, goal), 0.6):
            path = []
            while cur_node is not None:
                path.append(cur_node.position)
                cur_node = cur_node.parent
            return path[::-1]

        closed_list.append(cur_node)

        for action in get_action(R, Vx, delta_time_step):
            yaw_rate, dt, cost = action
            pos_child = vehicle_move(cur_node.position, yaw_rate, dt, Vx)
            if isNotInSearchingSpace(pos_child, space):
                continue
            if collision_check(cur_node.position, yaw_rate, dt, obstacle_list, Vx):
                continue

            child = Node(cur_node, pos_child)
            child.heading = pos_child[2]
            child.g = cur_node.g + cost
            child.h = heuristic(child, Node(None, goal))
            child.f = child.g + weight * child.h

            if any(isSamePosition(child, n) and isSameYaw(child, n) for n in closed_list):
                continue

            # Open 리스트에 더 나은 게 있으면 skip
            skip = False
            for n in open_list:
                if isSamePosition(child, n) and isSameYaw(child, n) and child.g >= n.g:
                    skip = True
                    break
            if not skip:
                open_list.append(child)

        # 시각화
        if show_animation:
            plt.plot(cur_node.position[0], cur_node.position[1], 'yo', alpha=0.3)
            if len(closed_list) % 100 == 0:
                plt.pause(0.01)

    return []
                
                

def main():

    start, goal, obstacle_list, space = map()

    if show_animation == True:
        theta_plot = np.linspace(0,1,101) * np.pi * 2
        plt.figure(figsize=(8,8))
        plt.plot(start[0], start[1], 'bs',  markersize=7)
        plt.text(start[0], start[1]+0.5, 'start', fontsize=12)
        plt.plot(goal[0], goal[1], 'rs',  markersize=7)
        plt.text(goal[0], goal[1]+0.5, 'goal', fontsize=12)
        for i in range(len(obstacle_list)):
            x_obstacle = obstacle_list[i][0] + obstacle_list[i][2] * np.cos(theta_plot)
            y_obstacle = obstacle_list[i][1] + obstacle_list[i][2] * np.sin(theta_plot)
            plt.plot(x_obstacle, y_obstacle,'k-')
        plt.axis(space)
        plt.grid(True)
        plt.xlabel("X [m]"), plt.ylabel("Y [m]")
        plt.title("Hybrid a star algorithm", fontsize=20)

    opt_path = a_star(start, goal, space, obstacle_list, R=5.0, Vx=2.0, delta_time_step=0.5, weight=1.1)
    print("Optimal path found!")
    opt_path = np.array(opt_path)
    if show_animation == True:
        plt.plot(opt_path[:,0], opt_path[:,1], "m.-")
        plt.show()


if __name__ == "__main__":
    main()

    
