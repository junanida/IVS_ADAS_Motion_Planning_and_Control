import numpy as np
import math
import matplotlib.pyplot as plt
import random
from map_2 import map
import heapq

class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.f = 0
        self.g = 0
        self.h = 0

    def __eq__(self, other):
        if self.position == other.position:
            return True
    def __lt__(self, other):  # 이걸 추가!
        return self.f < other.f
    def update_f(self, weight=1.0):
        self.f = self.g + weight * self.h

def heuristic(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.hypot(dx, dy)

show_animation  = True
def get_action():
    return [
        [0, 1, 1],    # 상
        [0, -1, 1],   # 하
        [-1, 0, 1],   # 좌
        [1, 0, 1],    # 우
        [-1, 1, math.sqrt(2)],   # 좌상
        [-1, -1, math.sqrt(2)],  # 좌하
        [1, 1, math.sqrt(2)],    # 우상
        [1, -1, math.sqrt(2)]    # 우하
    ]


def collision_check(omap, node):
    x, y = node.position
    for ox, oy in zip(omap[0], omap[1]):
        if x == ox and y == oy:
            return True
    return False


def a_star(start, goal, omap, weight=10.0):
    start_node = Node(None, start)
    goal_node = Node(None, goal)
    start_node.g = 0
    start_node.h = heuristic(start_node.position, goal_node.position)
    start_node.update_f(weight)

    open_list = []
    heapq.heappush(open_list, (start_node.f, start_node))
    closed_list = []

    while open_list:
        _, current = heapq.heappop(open_list)
        closed_list.append(current)

        if current == goal_node:
            path = []
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]

        for dx, dy, cost in get_action():
            new_pos = (current.position[0] + dx, current.position[1] + dy)
            child = Node(current, new_pos)

            if collision_check(omap, child):
                continue
            if child in closed_list:
                continue

            child.g = current.g + cost
            child.h = heuristic(child.position, goal_node.position)
            child.update_f(weight)

            # 더 나은 경로라면 open_list에 넣기
            skip = False
            for _, open_node in open_list:
                if child == open_node and child.g >= open_node.g:
                    skip = True
                    break
            if not skip:
                heapq.heappush(open_list, (child.f, child))

        if show_animation:
            plt.plot(current.position[0], current.position[1], 'yo', alpha=0.3)
            if len(closed_list) % 100 == 0:
                plt.pause(0.01)

    return []


def main():
    start, goal, omap = map()

    if show_animation:
        plt.figure(figsize=(8, 8))
        plt.plot(start[0], start[1], 'bs', markersize=7)
        plt.text(start[0], start[1] + 0.5, 'start', fontsize=12)
        plt.plot(goal[0], goal[1], 'rs', markersize=7)
        plt.text(goal[0], goal[1] + 0.5, 'goal', fontsize=12)
        plt.plot(omap[0], omap[1], '.k', markersize=10)
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("X [m]"), plt.ylabel("Y [m]")
        plt.title("A-Star algorithm", fontsize=20)

    opt_path = a_star(start, goal, omap)
    print("Optimal path found!")
    opt_path = np.array(opt_path)

    if show_animation:
        plt.plot(opt_path[:, 0], opt_path[:, 1], "m.-")
        plt.show()


if __name__ == "__main__":
    main()
