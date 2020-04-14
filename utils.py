# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 14:29:37 2020

@author: ohadary
"""
import numpy as np

def compress_path(path_coordinates,epsilon=1e-1):
    i,j,k = 0,1,2
    compressed = path_coordinates.copy()
    pc = path_coordinates
    while k < len(pc):
        if utils_collinear_int(pc[i],pc[j],pc[k],epsilon):
            compressed.remove(path_coordinates[j])
        else:
            i = j
        j += 1
        k += 1
    return compressed

def compress_path_bresenhem(path_coordinates):
    waypoints_removed = True
    while(waypoints_removed):
        waypoints_removed = False
        triplets = list(zip(path_coordinates[::3],path_coordinates[1::3],path_coordinates[2::3]))
        compressed = path_coordinates.copy()
        for t in triplets:
            intersected_cells = utils_bresenham(t[0],t[2])
            #print(intersected_cells)
            if t[2] in intersected_cells:
                waypoints_removed = True
                compressed.remove(t[2])
        path_coordinates = compressed.copy()
    return compressed

def utils_bresenham(p1, p2, conservative=False):
    # if slope is infinite (vertical line segment) return trivial set of segments
    if p1[0] == p2[0]:
        lower = p1[1] if p1[1] < p2[1] else p2[1]
        upper = p2[1] if lower == p1[1] else p1[1]
        cells = [(p1[0],_) for _ in range(lower, upper)]
        [cells.append((p1[0]-1, b)) for b in range(lower,upper)]
        return np.array(cells)

    swap = p1[0] > p2[0]
    if swap:
        x1,y1 = p2
        x2,y2 = p1
    else:
        x1, y1 = p1
        x2, y2 = p2

    m  = (y2-y1)/(x2-x1)

    # if slope is zero (horizontal line segment) return trivial set of segments
    if m == 0:
        cells = [(_,y1) for _ in range(x1,x2)]
        [cells.append((_, y1-1)) for _ in range(x1,x2)]
        return cells

    # if slope is negative, reflect about x-axis and use a positive slope
    # this allows us to use the same algorithm below for finding intersected cells
    # and the y coordinate of an appended cell is adjusted for a line
    # segment with negative slope (flip == True).
    flip = True if m < 0 else False
    if flip:
        y1,y2 = -y1,-y2
        m  = (y2-y1)/(x2-x1)

    cells = []

    def test_conservative(x,y):
        return x < x2 and y < y2

    def test_nonconservative(x,y):
        return x < x2 and y < y2

    test = test_conservative if conservative else test_nonconservative

    # -------------------------------------------------------
    # Determine set of grid cells intersected by line segment
    # -------------------------------------------------------

    # need dx, dy for algorithm to work correctly when one of points is not at origin

    dx = 1
    dy = 0
    while(test(x1 + dx -1, y1 + dy)):
        # the appended cell coordinate is assumed to be the bottom left corner of the cell
        # hence we need to translate the coordinate one unit down for negatively sloped (flipped) line segments
        cells.append((x1 + dx - 1, -y1 -dy - 1 if flip else y1 + dy))
        if m == 1 and conservative:
            cells.append((x1 + dx, -y1 -dy - 1 if flip else y1 + dy)) # to the right
            cells.append((x1 + dx - 1, -y1 -dy -2 if flip else y1 + dy + 1)) # below (negative slope), above (+ve slope)

        if dy + 1 < m*(dx):
            dy += 1
        elif dy + 1 == m*(dx):
            dy += 1
            dx += 1
        else:
            dx += 1

    return np.array(cells)

def utils_collinear_int(p1, p2, p3, epsilon = 1e-2):
    """
    Determine if p1, p2, and p3 are approximately collinear by evaluating the
    determinant using the simplified version for the 2D case.
    """
    x1,y1 = p1
    x2,y2 = p2
    x3,y3 = p3
    return (abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))  < epsilon)
