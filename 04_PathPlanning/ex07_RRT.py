import numpy as np
import matplotlib.pyplot as plt
from map_4 import map


class Node(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

    def set_parent(self, parent):
        self.parent = parent


class RRT(object):
    def __init__(self, start, goal, space, obstacle_list, success_dist_thres=1.0):
        self.start_node = Node(start[0], start[1])  # node (x, y)
        self.goal_node = Node(goal[0], goal[1])  # node (x, y)
        self.space = space  # (min_x, max_x, min_y, max_y)
        self.obstalce_list = obstacle_list  # list of (x, y ,r)
        self.node_list = []

        # options
        self.max_iter = 5000
        self.goal_sample_rate = 0.1
        self.min_u = 1.0
        self.max_u = 3.0
        self.success_dist_thres = success_dist_thres
        self.collision_check_step = 0.2
        self.stepsize = 0.5

    def plan(self):
        self.node_list = [self.start_node]
        for i in range(self.max_iter):
            # Get random node
            rand_node = self.get_random_node()
            # Find nearest node
            nearest_node = self.find_nearest_node(self.node_list, rand_node)
            # Create new node
            u = self.stepsize * self.get_random_input(self.min_u, self.max_u)
            new_node = self.create_child_node(nearest_node, rand_node, u)
            # Collision check(Node, Path)
            node_collide = self.is_collide(new_node, self.obstalce_list)
            if node_collide:
                continue
            collide = self.is_path_collide(nearest_node, new_node, self.obstalce_list, self.collision_check_step)
            if collide:
                continue
            # Add to tree
            new_node.set_parent(nearest_node)
            self.node_list.append(new_node)
            # Goal check
            goal_reached = self.check_goal(new_node, self.success_dist_thres)
            if goal_reached:
                print(" [-] GOAL REACHED")
                return self.backtrace_path(new_node)
        return None

    @staticmethod
    def is_same_node(node1, node2):
        # Return True if both nodes are at the same position
        return node1.x == node2.x and node1.y == node2.y

    def backtrace_path(self, node):
        current_node = node
        path = [current_node]
        reached_start_node = self.is_same_node(current_node, self.start_node)
        while not reached_start_node:
            current_node = current_node.parent
            path.append(current_node)
            reached_start_node = self.is_same_node(current_node, self.start_node)
        return path[::-1]

    def get_random_node(self):
        # Occasionally set random node as goal node
        if np.random.rand() > self.goal_sample_rate:
            rand_x = np.random.uniform(self.space[0], self.space[1])
            rand_y = np.random.uniform(self.space[2], self.space[3])
            return Node(rand_x, rand_y)
        else:
            return self.goal_node

    def check_goal(self, node, success_dist_thres):
        # Check if the node is within the success distance threshold from goal
        dist = np.sqrt((node.x - self.goal_node.x)**2 + (node.y - self.goal_node.y)**2)
        return dist <= success_dist_thres

    @staticmethod
    def create_child_node(nearest_node, rand_node, u):
        # Create new node based on the nearest node, random node, and control input u
        angle = np.arctan2(rand_node.y - nearest_node.y, rand_node.x - nearest_node.x)
        new_x = nearest_node.x + u * np.cos(angle)
        new_y = nearest_node.y + u * np.sin(angle)
        return Node(new_x, new_y)

    @staticmethod
    def get_random_input(min_u, max_u):
        # Random input control in the range [min_u, max_u]
        return np.random.uniform(min_u, max_u)

    @staticmethod
    def find_nearest_node(node_list, rand_node):
        # Find the nearest node from the node list to the random node
        min_dist = float("inf")
        nearest_node = None
        for node in node_list:
            dist = np.sqrt((node.x - rand_node.x)**2 + (node.y - rand_node.y)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
        return nearest_node

    @staticmethod
    def is_collide(node, obstacle_list):
        # Check if the node collides with any obstacle
        for (ox, oy, r) in obstacle_list:
            dist = np.sqrt((node.x - ox)**2 + (node.y - oy)**2)
            if dist <= r:
                return True
        return False

    @staticmethod
    def is_path_collide(node_from, node_to, obstacle_list, check_step=0.2):
        # Check if the path between node_from and node_to collides with any obstacle
        steps = int(np.linalg.norm([node_from.x - node_to.x, node_from.y - node_to.y]) / check_step)
        for i in range(steps):
            x = node_from.x + (node_to.x - node_from.x) * (i / steps)
            y = node_from.y + (node_to.y - node_from.y) * (i / steps)
            for (ox, oy, r) in obstacle_list:
                dist = np.sqrt((x - ox)**2 + (y - oy)**2)
                if dist <= r:
                    return True
        return False


if __name__ == "__main__":
    start, goal, space, obstacle_list = map()

    success_dist_thres = 1.0
    rrt = RRT(start, goal, space, obstacle_list, success_dist_thres)
    path = rrt.plan()
    for node in path:
        print(" [-] x = %.2f, y = %.2f " % (node.x, node.y))

    # draw result
    _t = np.linspace(0, 2*np.pi, 30)
    for obs in obstacle_list:
        x, y, r = obs
        _x = x + r * np.cos(_t)
        _y = y + r * np.sin(_t)

        plt.plot(_x, _y, 'k-')

    goal_x = goal[0] + success_dist_thres * np.cos(_t)
    goal_y = goal[1] + success_dist_thres * np.sin(_t)
    plt.plot(goal_x, goal_y, 'g--')

    for i in range(len(path)-1):
        node_i = path[i]
        node_ip1 = path[i+1]
        plt.plot([node_i.x, node_ip1.x], [node_i.y, node_ip1.y], 'r.-')

    plt.axis("equal")
    plt.show()
