import numpy as np
import json
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
import numpy as np
import plotly.graph_objects as go
import numpy as np


class initiate_kc:
    def __init__(self, size):
        self.size = size
        self.vertices, self.hex_faces = self.generate_kelvin_cell()

    def generate_kelvin_cell(self):
        # 定义四边形的顶点坐标
        vertices = np.array([
            # +z方向 (XY平面)
            [1, 0, 2], [0, 1, 2], [-1, 0, 2], [0, -1, 2],
            # +x方向 (YZ平面)
            [2, 1, 0], [2, 0, 1], [2, -1, 0], [2, 0, -1],
            # +y方向 (XZ平面)
            [0, 2, 1], [1, 2, 0], [0, 2, -1], [-1, 2, 0],
            # -z方向 (XY平面)
            [1, 0, -2], [0, 1, -2], [-1, 0, -2], [0, -1, -2],
            # -x方向 (YZ平面)
            [-2, 1, 0], [-2, 0, 1], [-2, -1, 0], [-2, 0, -1],
            # -y方向 (XZ平面)
            [0, -2, 1], [1, -2, 0], [0, -2, -1], [-1, -2, 0]
        ]) 

        # 初始化六边形列表
        hex_faces = []

        # 定义每个象限的六边形
        for i, (x_cond, y_cond, z_cond) in enumerate([
            (1, 1, 1),   # 第一象限 (+X, +Y, +Z)
            (-1, 1, 1),  # 第二象限 (-X, +Y, +Z)
            (-1, -1, 1), # 第三象限 (-X, -Y, +Z)
            (1, -1, 1),  # 第四象限 (+X, -Y, +Z)
            (1, 1, -1),  # 第五象限 (+X, +Y, -Z)
            (-1, 1, -1), # 第六象限 (-X, +Y, -Z)
            (-1, -1, -1),# 第七象限 (-X, -Y, -Z)
            (1, -1, -1)  # 第八象限 (+X, -Y, -Z)
        ]):
            hexagon = []
            for idx, v in enumerate(vertices):
                x, y, z = v
                if (x == 2*x_cond) or \
                (y == 2*y_cond) or \
                (z == 2*z_cond):
                    if np.sum(v*[x_cond,y_cond,z_cond])==3:
                        hexagon.append(idx)

            # 排序逻辑：对每个六边形进行顺时针排序
            x, y, z = vertices[hexagon].T

            # 计算每个点的相对于中心点的角度
            angles = np.arctan2(y - y_cond, x - x_cond)

            # 根据角度对顶点进行排序
            sorted_indices = np.argsort(angles)

            # 根据排序后的顺序调整顶点
            hexagon = [hexagon[idx] for idx in sorted_indices]

            hex_faces.append(hexagon)
            # print(f"六边形 {i + 1}（排序后）: {hexagon}")

        return vertices* self.size, hex_faces


class Deformation:
    def apply(self, vertices):
        raise NotImplementedError("The deformation method must be implemented by subclasses")

class RotateDeformation(Deformation):
    def __init__(self, axis, direction=1, degree=90):
        self.axis = axis.lower()
        self.direction = direction
        self.degree = np.deg2rad(degree)  # Convert to radians

        if self.axis not in ['x', 'y', 'z']:
            raise ValueError("Axis must be 'x', 'y', or 'z'")

        self.rotation_matrix = self._create_rotation_matrix()

    def _create_rotation_matrix(self):
        cos_theta = np.cos(self.degree)
        sin_theta = np.sin(self.degree) * self.direction

        if self.axis == 'x':
            return np.array([
                [1, 0, 0],
                [0, cos_theta, -sin_theta],
                [0, sin_theta, cos_theta]
            ])
        elif self.axis == 'y':
            return np.array([
                [cos_theta, 0, sin_theta],
                [0, 1, 0],
                [-sin_theta, 0, cos_theta]
            ])
        elif self.axis == 'z':
            return np.array([
                [cos_theta, -sin_theta, 0],
                [sin_theta, cos_theta, 0],
                [0, 0, 1]
            ])

    def apply(self, vertices, indices_to_deform):
        vertices[indices_to_deform] = np.dot(vertices[indices_to_deform], self.rotation_matrix)
        return vertices

class ExtendDeformation(Deformation):
    def __init__(self, axis, distance=1):
        self.axis = axis.lower()
        self.distance = distance

        if self.axis not in ['x', 'y', 'z']:
            raise ValueError("Axis must be 'x', 'y', or 'z'")

    def apply(self, vertices, indices_to_deform):
        axis_index = {'x': 0, 'y': 1, 'z': 2}[self.axis]
        vertices[indices_to_deform, axis_index] += self.distance
        return vertices

class ShiftDeformation(Deformation):
    def __init__(self, axis, direction, distance=0.5):
        self.axis = axis.lower()
        self.direction = np.deg2rad(direction)  # Convert direction angle to radians
        self.distance = distance

        if self.axis not in ['x', 'y', 'z']:
            raise ValueError("Axis must be 'x', 'y', or 'z'")

    def apply(self, vertices, indices_to_deform):
        # axis_index is the index of the axis that we want to shift along (e.g., 0 for x, 1 for y, 2 for z)
        axis_index = {'x': 0, 'y': 1, 'z': 2}[self.axis]

        # Calculate the shift components along the two axes orthogonal to the specified axis
        shift_vector = np.zeros(vertices.shape[1])
        orthogonal_axis_1 = (axis_index + 1) % 3
        orthogonal_axis_2 = (axis_index + 2) % 3

        # Apply trigonometric functions to determine the components of the shift
        shift_vector[orthogonal_axis_1] = np.cos(self.direction) * self.distance
        shift_vector[orthogonal_axis_2] = np.sin(self.direction) * self.distance

        # Apply the shift to the vertices
        vertices[indices_to_deform] += shift_vector

        return vertices

class DeformManager:
    def __init__(self, vertices):
        self.vertices = vertices

    def _get_direction_from_code(self, axis_code):
        # Convert axis_code back to direction string
        direction_map = {
            1: "+x", -1: "-x",
            2: "+y", -2: "-y",
            3: "+z", -3: "-z"
        }
        return direction_map[axis_code]


    def _get_indices_for_direction(self, direction):
        # Mapping from direction to indices
        direction_map = {
            "+x": [4, 5, 6, 7],
            "-x": [16, 17, 18, 19],
            "+y": [8, 9, 10, 11],
            "-y": [20, 21, 22, 23],
            "+z": [0, 1, 2, 3],
            "-z": [12, 13, 14, 15],
        }
        return direction_map[direction]

    def matrix(self, deformation_matrix):
        for deformation in deformation_matrix:
            operation = int(deformation[0])
            axis_code = int(deformation[1])
            distance = deformation[2]
            degree = deformation[3]

            # Decode axis_code back to direction string (e.g., "+x")
            direction = self._get_direction_from_code(axis_code)

            # Get indices for this direction
            indices_to_deform = self._get_indices_for_direction(direction)

            if operation == 0:  # Rotate
                self.rotate(direction, degree)
            elif operation == 1:  # Extend
                self.extend(direction, distance)
            elif operation == 2:  # Shift
                self.shift(direction, degree, distance)

        return self.vertices



    def rotate(self, direction, degree=90):
        axis = direction[1]  # 'x', 'y', or 'z'
        sign = 1 if direction[0] == '+' else -1  # Determine direction of rotation

        rotation = RotateDeformation(axis, direction=sign, degree=degree)
        indices_to_deform = self._get_indices_for_direction(direction)
        self.vertices = rotation.apply(self.vertices.copy(), indices_to_deform)  # Apply on a copy to maintain continuous deformation
        return self.vertices

    def extend(self, direction, distance=1):
        axis = direction[1]  # 'x', 'y', or 'z'
        sign = 1 if direction[0] == '+' else -1  # Determine direction of extension

        extend = ExtendDeformation(axis, distance * sign)
        indices_to_deform = self._get_indices_for_direction(direction)
        self.vertices = extend.apply(self.vertices.copy(), indices_to_deform)
        return self.vertices

    def shift(self, direction, degree, distance=0.5):
        axis = direction[1]  # 'x', 'y', or 'z'

        shift = ShiftDeformation(axis, degree, distance)
        indices_to_deform = self._get_indices_for_direction(direction)
        self.vertices = shift.apply(self.vertices.copy(), indices_to_deform)
        return self.vertices

class KelvinCell:
    def __init__(self, size):
        ini_kc = initiate_kc(size)
        self.vertices = ini_kc.vertices.astype(np.float64)
        self.hex_faces = ini_kc.hex_faces
        self.deform = DeformManager(self.vertices)  # 初始化DeformManager
        self.center = [0,0,0]
        self.beam_radius = 0.03
        self.connectivity = self._generate_connectivity()

    def _generate_connectivity(self):
        connectivity = []
        for hexagon in self.hex_faces:
            for i in range(len(hexagon)):
                start_point = hexagon[i]
                end_point = hexagon[(i + 1) % len(hexagon)]
                connectivity.append((start_point, end_point))
        return connectivity

    def calculate_stiffness(self):
        # Placeholder: Implement stiffness calculation
        pass

    def to_xml(self, vertices_file, connectivity_file):
        # Export vertices to XML
        pass

    def generate_mesh(self, file_name="kc_beam_mesh.vtk"):
        points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()

        # Add points to the VTK Points object
        for i, vertex in enumerate(self.vertices):
            points.InsertNextPoint(vertex[0], vertex[1], vertex[2])

        # Create and add lines to the VTK CellArray
        for start, end in self.connectivity:
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, start)
            line.GetPointIds().SetId(1, end)
            cells.InsertNextCell(line)

        # Create an unstructured grid
        unstructured_grid = vtk.vtkUnstructuredGrid()
        unstructured_grid.SetPoints(points)
        unstructured_grid.SetCells(vtk.VTK_LINE, cells)

        # Write the grid to a file
        writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileName(file_name)
        writer.SetInputData(unstructured_grid)
        writer.Write()

        print(f"Mesh saved to {file_name}")

    def show_mesh(self):
        fig = go.Figure()

        for start, end in self.connectivity:
            start_coords = self.vertices[start]
            end_coords = self.vertices[end]

            x = [start_coords[0], end_coords[0]]
            y = [start_coords[1], end_coords[1]]
            z = [start_coords[2], end_coords[2]]

            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(width=self.beam_radius*10, color='green')
            ))

        fig.update_layout(scene=dict(
            xaxis=dict(nticks=4, range=[-3, 3], showbackground=True, backgroundcolor="lightgrey"),
            yaxis=dict(nticks=4, range=[-3, 3], showbackground=True, backgroundcolor="lightgrey"),
            zaxis=dict(nticks=4, range=[-3, 3], showbackground=True, backgroundcolor="lightgrey"),
        ),
        width=700,
        margin=dict(r=20, l=10, b=10, t=10))

        fig.show()

    def visualize(self):
        fig = go.Figure()

        # Draw the beams (connectivity)
        for start, end in self.connectivity:
            x = [self.vertices[start][0], self.vertices[end][0]]
            y = [self.vertices[start][1], self.vertices[end][1]]
            z = [self.vertices[start][2], self.vertices[end][2]]

            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(color='blue', width=20),
                showlegend=False
            ))

        # Draw the vertices
        fig.add_trace(go.Scatter3d(
            x=self.vertices[:, 0], y=self.vertices[:, 1], z=self.vertices[:, 2],
            mode='markers',
            marker=dict(size=4, color='red')
        ))

        # Set the layout for better visualization
        fig.update_layout(scene=dict(
            xaxis=dict(nticks=4, showbackground=True, backgroundcolor="lightgrey"),
            yaxis=dict(nticks=4, showbackground=True, backgroundcolor="lightgrey"),
            zaxis=dict(nticks=4, showbackground=True, backgroundcolor="lightgrey"),
        ),
        width=700,
        margin=dict(r=20, l=10, b=10, t=10))

        fig.show()
