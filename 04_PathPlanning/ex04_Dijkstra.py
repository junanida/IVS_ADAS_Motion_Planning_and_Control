import numpy as np
import math
import matplotlib.pyplot as plt
import random
from map_1 import map  # 반드시 map_1.py 파일이 있어야 합니다.

show_animation  = True

class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.f = 0  # 현재까지 누적 비용

    def __eq__(self, other):
        return self.position == other.position

def get_action():
    # dx, dy, cost (cost는 대각선인 경우 sqrt(2), 직선은 1)
    action_set = [
        [1, 0, 1],
        [0, 1, 1],
        [-1, 0, 1],
        [0, -1, 1],
        [1, 1, math.sqrt(2)],
        [-1, -1, math.sqrt(2)],
        [1, -1, math.sqrt(2)],
        [-1, 1, math.sqrt(2)],
    ]
    return action_set

def collision_check(omap, node_pos):
    x, y = node_pos
    for ox, oy in zip(omap[0], omap[1]):
        if x == ox and y == oy:
            return True  # 충돌 발생
    return False  # 충돌 없음

def dijkstra(start, goal, map_obstacle):
    start_node = Node(None, tuple(start))
    goal_node = Node(None, tuple(goal))
    
    open_list = []
    closed_list = []

    open_list.append(start_node)
    
    while open_list:
        # 가장 비용이 적은 노드를 선택
        cur_node = min(open_list, key=lambda n: n.f)
        open_list.remove(cur_node)
        closed_list.append(cur_node)

        # 목적지에 도달했으면 경로 추적
        if cur_node == goal_node:
            path = []
            current = cur_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]  # 역순 반환

        action_set = get_action()
        for action in action_set:
            new_pos = (cur_node.position[0] + action[0],
                       cur_node.position[1] + action[1])

            # 충돌 검사
            if collision_check(map_obstacle, new_pos):
                continue

            # 새로운 노드 생성
            child = Node(cur_node, new_pos)
            child.f = cur_node.f + action[2]

            # 이미 닫힌 리스트에 있다면 패스
            if any(closed_node == child for closed_node in closed_list):
                continue

            # 같은 위치의 노드가 오픈 리스트에 있다면 더 짧은 경로면 교체
            for open_node in open_list:
                if child == open_node and child.f >= open_node.f:
                    break
            else:
                open_list.append(child)

        if show_animation:
            plt.plot(cur_node.position[0], cur_node.position[1], 'yo', alpha=0.5)
            if len(closed_list) % 100 == 0:
                plt.pause(0.01)

    return None  # 경로를 찾지 못한 경우

def main():
    start, goal, omap = map()

    if show_animation:
        plt.figure(figsize=(8,8))
        plt.plot(start[0], start[1], 'bs',  markersize=7)
        plt.text(start[0], start[1]+0.5, 'start', fontsize=12)
        plt.plot(goal[0], goal[1], 'rs',  markersize=7)
        plt.text(goal[0], goal[1]+0.5, 'goal', fontsize=12)
        plt.plot(omap[0], omap[1], '.k',  markersize=10)
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("X [m]"), plt.ylabel("Y [m]")
        plt.title("Dijkstra algorithm", fontsize=20)

    opt_path = dijkstra(start, goal, omap)
    
    if opt_path is not None:
        print("Optimal path found!")
        opt_path = np.array(opt_path)
        if show_animation:
            plt.plot(opt_path[:,0], opt_path[:,1], "m.-")
            plt.show()
    else:
        print("Path not found.")

if __name__ == "__main__":
    main()
