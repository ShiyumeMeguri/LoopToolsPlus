"""
Best-fit plane extraction, used by the curve flatten operator.

The covariance matrix of the point cloud is inverted and then power-iterated, so the
dominant eigenvector of the inverse matches the smallest eigenvector of the covariance
matrix, which is the plane normal.
"""

import math
import mathutils


def matrix_determinant(matrix):
    return matrix[0][0] * matrix[1][1] * matrix[2][2] + matrix[0][1] * matrix[1][2] * matrix[2][0] \
        + matrix[0][2] * matrix[1][0] * matrix[2][1] - matrix[0][2] * matrix[1][1] * matrix[2][0] \
        - matrix[0][1] * matrix[1][0] * matrix[2][2] - matrix[0][0] * matrix[1][2] * matrix[2][1]


def matrix_invert(matrix):
    determinant = matrix_determinant(matrix)
    if determinant == 0:
        return None
    adjugate = mathutils.Matrix((
        (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1],
         matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2],
         matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]),
        (matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2],
         matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0],
         matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2]),
        (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0],
         matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1],
         matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0])))
    return adjugate * (1 / determinant)


def calculate_plane(locations, method="best", view_matrix=None):
    center = mathutils.Vector()
    for location in locations:
        center += location
    center /= len(locations)
    x, y, z = center

    if method == 'best':
        matrix = mathutils.Matrix(((0.0, 0.0, 0.0),
                                   (0.0, 0.0, 0.0),
                                   (0.0, 0.0, 0.0)))
        for location in locations:
            matrix[0][0] += (location[0] - x) ** 2
            matrix[1][0] += (location[0] - x) * (location[1] - y)
            matrix[2][0] += (location[0] - x) * (location[2] - z)
            matrix[0][1] += (location[1] - y) * (location[0] - x)
            matrix[1][1] += (location[1] - y) ** 2
            matrix[2][1] += (location[1] - y) * (location[2] - z)
            matrix[0][2] += (location[2] - z) * (location[0] - x)
            matrix[1][2] += (location[2] - z) * (location[1] - y)
            matrix[2][2] += (location[2] - z) ** 2

        inverted = matrix_invert(matrix)
        if inverted:
            matrix = inverted

        axis = 2
        if math.fabs(sum(matrix[0])) < math.fabs(sum(matrix[1])):
            if math.fabs(sum(matrix[0])) < math.fabs(sum(matrix[2])):
                axis = 0
        elif math.fabs(sum(matrix[1])) < math.fabs(sum(matrix[2])):
            axis = 1
        if axis == 0:
            normal = mathutils.Vector((1.0, 0.0, 0.0))
        elif axis == 1:
            normal = mathutils.Vector((0.0, 1.0, 0.0))
        else:
            normal = mathutils.Vector((0.0, 0.0, 1.0))

        iterations = 500
        current = mathutils.Vector((1.0, 1.0, 1.0))
        for _ in range(iterations):
            previous = current
            current = matrix @ previous
            length = math.sqrt(current[0] ** 2 + current[1] ** 2 + current[2] ** 2)
            if length != 0:
                current /= length
            if current == previous:
                break
        if current.length == 0:
            current = mathutils.Vector((1.0, 1.0, 1.0))
        normal = current

    elif method == 'view':
        if view_matrix:
            normal = view_matrix.to_3x3().inverted() @ mathutils.Vector((0.0, 0.0, 1.0))
        else:
            normal = mathutils.Vector((0.0, 0.0, 1.0))
    else:
        normal = mathutils.Vector((0.0, 0.0, 1.0))

    return center, normal
