import numpy as np
from time import time
from itertools import product
from collections import deque
from datetime import datetime
import math
import json
import solveSections


def open_coordinates(TSP_file):
    XY = []
    with open(TSP_file, 'rt') as f:
        for row in f:
            XY.append(list(map(float, row.strip().split(','))))
    return XY


def open_course(course_file, is_json):
    course = []
    with open(course_file, 'rt') as f:
        if not is_json:
            for row in f:
                course.append(list(map(int, row.replace(',', ' ').strip().split(' '))))
        else:
            course = json.loads(f.readline().strip())
    return course


def init(TSP_file='TSP.csv', course_file='TSP_03.csv', is_json=False):
    XY = open_coordinates(TSP_file)
    courses = open_course(course_file, is_json)

    return XY, courses


def make_edge(course):
    if len(course) < 1:
        return []

    edges = []
    before_node = course[-1]

    for node in course:
        edges.append((before_node, node))
        before_node = node

    return edges


def make_course(edges):
    next_course = {}
    course = [edges[0][1]]

    for start, end in edges:
        next_course[start] = end

    for _ in range(len(next_course) - 1):
        course.append(next_course[course[-1]])

    return course


DP={}
def euclidean_distance(x, y):
    key = (*x, *y)
    if key not in DP:
        DP[key] = np.linalg.norm(np.array(x) - np.array(y))

    return DP[key]

def calculate_distance(object, XY):
    if not object:
        return 0

    edges = []
    if type(object[0]) == int:
        edges = make_edge(object)
    else:
        edges = object

    distance = 0.0
    for start, end in edges:
        distance += euclidean_distance(XY[start], XY[end])
    return distance


def merge_two_sector(edges1, edges2, XY):
    edges1 = edges1[:]
    edges2 = edges2[:]
    new_edges = []

    # 한쪽 구역이 비어 있는 경우
    if len(edges1) == 0 or len(edges2) == 0:
        new_edges = edges1 + edges2
    else:
        base_distance = calculate_distance(edges1, XY) + calculate_distance(edges2, XY)
        best_distance = float('inf')
        best_combination = None

        for edge1, edge2 in product(edges1, edges2):
            distance = base_distance \
                       - euclidean_distance(XY[edge1[0]], XY[edge1[1]]) \
                       - euclidean_distance(XY[edge2[0]], XY[edge2[1]])

            not_cross_distance = distance \
                                 + euclidean_distance(XY[edge1[0]], XY[edge2[0]]) \
                                 + euclidean_distance(XY[edge1[1]], XY[edge2[1]])
            cross_distance = distance \
                             + euclidean_distance(XY[edge1[0]], XY[edge2[1]]) \
                             + euclidean_distance(XY[edge1[1]], XY[edge2[0]])

            if not_cross_distance < best_distance:
                best_distance = not_cross_distance
                best_combination = (edge1, edge2, False)
            if cross_distance < best_distance:
                best_distance = cross_distance
                best_combination = (edge1, edge2, True)

        edge1, edge2, cross = best_combination

        edges1.remove(edge1)
        edges2.remove(edge2)

        new_edges = edges1
        if cross:
            new_edges += edges2
            new_edges.append((edge1[0], edge2[1]))
            new_edges.append((edge2[0], edge1[1]))
        else:
            new_edges += list(map(lambda x: tuple(reversed(x)), edges2))
            new_edges.append((edge1[0], edge2[0]))
            new_edges.append((edge2[1], edge1[1]))

    return new_edges


def merge(edges, XY, verbose=1):
    if verbose:
        start = time()
        print(datetime.now())

    while len(edges) != 1:
        new_edges = []
        for i in range(len(edges) // 2):
            new_edges.append(merge_two_sector(edges[i * 2], edges[i * 2 + 1], XY))
        if len(edges) % 2 == 1:
            new_edges.append(edges[-1])
        edges = new_edges

        if verbose:
            print(f'\r#{len(edges)}', end='')

    if verbose:
        print('\rFinish %ds' % (time() - start))
        print(datetime.now())
    return edges[0]


def find_neighbors(idx, N):
    neighbors = []

    row = idx // N
    col = idx % N
    directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

    for drow, dcol in directions:
        nrow = row + drow
        ncol = col + dcol
        if 0 <= nrow < N and 0 <= ncol < N:
            neighbors.append(nrow * N + ncol)
    return neighbors


def merge_greedy(edges, start, XY):
    N = int(math.sqrt(len(edges)))

    idx = start
    edge = edges[start]
    visited = {start: True}
    neighbors = {}

    for _ in range(N ** 2 - 1):
        for neighbor in find_neighbors(idx, N):
            if neighbor not in visited:
                neighbors[neighbor] = True

        best_distance = float('inf')
        best_idx = None
        best_edge = None

        for neighbor in neighbors.keys():
            new_edge = merge_two_sector(edge, edges[neighbor], XY)
            distance = calculate_distance(new_edge, XY)

            if distance < best_distance:
                best_distance = distance
                best_idx = neighbor
                best_edge = new_edge

        idx = best_idx
        edge = best_edge
        visited[idx] = True
        del neighbors[idx]

    return edge


def merge_with_tree(edges, XY, verbose=1):
    if verbose:
        start = time()
        print(datetime.now())

    best_distance=float('inf')
    best_edge = None

    for i in range(64):
        edge=merge_greedy(edges, i, XY)
        distance = calculate_distance(edge, XY)
        print(i, distance)
        if distance < best_distance:
            best_distance=distance
            best_edge=edge

    if verbose:
        print('\rFinish %ds' % (time() - start))
        print(datetime.now())

    return best_edge


def optimize_4sector(point, edges, XY):
    best_distance = float('inf')
    best_path = None
    best_edge = None
    for starting in range(4):

        visited = [False] * 4
        visited[starting] = True
        edge = edges[point[starting]]
        Q = deque([(edge, visited, [point[starting]])])

        while Q:
            edge, visited, path = Q.popleft()
            finished = True
            for i in range(4):
                if not visited[i]:
                    finished = False
                    new_visited = visited[:]
                    new_visited[i] = True
                    new_path = path[:]
                    new_path.append(point[i])

                    new_edge = merge_two_sector(edge, edges[point[i]], XY)
                    Q.append((new_edge, new_visited, new_path))
            if finished:
                distance = calculate_distance(edge, XY)
                if distance < best_distance:
                    best_distance = distance
                    best_path = path
                    best_edge = edge

            # local_best_distance = float('inf')
            # local_best_idx = 0
            # local_best_edge = None
            #
            # for i in range(4):
            #     if not visited[i]:
            #         new_edge = merge_two_sector(edge, edges[point[i]], XY)
            #         distance = calculate_distance(new_edge, XY)
            #         if distance < local_best_distance:
            #             local_best_distance = distance
            #             local_best_idx = i
            #             local_best_edge = new_edge
            #
            # if local_best_edge:
            #     new_visited = visited[:]
            #     new_visited[local_best_idx] = True
            #     new_path = path[:]
            #     new_path.append(point[local_best_idx])
            #     Q.append((local_best_edge, new_visited, new_path))
            # else:
            #     distance = calculate_distance(edge, XY)
            #     if distance < best_distance:
            #         best_distance = distance
            #         best_path = path
            #         best_edge = edge

    return best_edge, best_path, best_distance


def merge_4way(edges, XY, verbose=1):
    if verbose:
        start = time()
        print(datetime.now())

    while len(edges) != 1:
        N = int(math.sqrt(len(edges)))
        new_edges = []

        for i in range((N // 2) ** 2):
            base = (i // (N // 2)) * N
            point = [i * 2 + base, i * 2 + 1 + base,
                     (i + N // 2) * 2 + base, (i + N // 2) * 2 + 1 + base]
            print(point)
            new_edge, _, _ = optimize_4sector(point, edges, XY)
            new_edges.append(new_edge)
        edges = new_edges

        if verbose:
            print(f'\r#{len(edges)}', end='')

    if verbose:
        print('\rFinish %ds' % (time() - start))
        print(datetime.now())
    return edges[0]


if __name__ == '__main__':
    print(solveSections.solve(8))
    '''
    XY, courses = init('TSP.csv', 'TSP_sol_16x16.csv', is_json=True)
    edges = list(map(make_edge, courses))

    # edge = merge_with_tree(edges, XY)
    # distance = calculate_distance(edge, XY)
    # print('Total Distance: %f' % distance)
    # with open(f'merged_{distance}.txt', 'wt') as f:
    #     f.write(json.dumps(make_course(edge)) + '\n')

    edge = merge(edges, XY)
    distance = calculate_distance(edge, XY)
    print('Total Distance: %f' % distance)
    with open(f'merged_{distance}.txt', 'wt') as f:
        f.write(json.dumps(make_course(edge)) + '\n')

    # edge = merge_4way(edges, XY)
    # distance = calculate_distance(edge, XY)
    # print('Total Distance: %f' % distance)
    # with open(f'merged_{distance}.txt', 'wt') as f:
    #     f.write(json.dumps(make_course(edge)) + '\n')

    print(len(make_course(edge)))

    import cv2

    image = np.full((1100, 1100, 3), 255, np.uint8)
    convert_xy = lambda x, y: (int(x * 10) + 50, int(y * 10) + 50)

    for x, y in XY:
        cv2.circle(image, center=convert_xy(x, y), radius=3, color=(0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)

    for start, end in edge:
        cv2.line(
            image,
            pt1=convert_xy(XY[start][0], XY[start][1]),
            pt2=convert_xy(XY[end][0], XY[end][1]),
            color=(255, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA
        )
    cv2.imshow('image', image)
    cv2.waitKey(0)
'''