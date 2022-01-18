import torch
import math
import numpy as np


class PointCloudGenerator(object):
    def __init__(self, camera_fovy, image_height, image_width, device, cam_matrix=None, rot_matrix=None, position=None):
        super(PointCloudGenerator, self).__init__()

        self.fovy = math.radians(camera_fovy)
        self.height = image_height
        self.width = image_width
        self.device = device

        if rot_matrix != None:
            self.rot_matrix = torch.tensor(rot_matrix, dtype=torch.float32, device=device, requires_grad=False)
            
        if position != None:
            self.position = torch.tensor(position, dtype=torch.float32, device=device, requires_grad=False)

        if cam_matrix != None:
            self.cam_matrix = cam_matrix
        else:
            self.cam_matrix = self.get_cam_matrix()

        self.fx = self.cam_matrix[0, 0]
        self.fy = self.cam_matrix[1, 1]
        self.cx = self.cam_matrix[0, 2]
        self.cy = self.cam_matrix[1, 2]

        self.uv1 = torch.ones((self.height, self.width, 3), dtype=torch.float32, device=device, requires_grad=False)
        for i in range(self.height):
            for j in range(self.width):
                self.uv1[i][j][0] = ((i + 1) - self.cx)/self.fx
                self.uv1[i][j][1] = ((j + 1) - self.cy)/self.fy
        print(self.uv1.shape)
        self.uv1 = self.uv1.reshape(-1, 3)
        print(self.uv1.shape)



    def get_cam_matrix(self):
        f = self.height / (2 * math.tan(self.fovy / 2))

        return torch.tensor(((f, 0, self.width / 2), (0, f, self.height / 2), (0, 0, 1)),
                            dtype=torch.float32, device=self.device, requires_grad=False)



    def reshape_depth(self, depth):
        depth = torch.tensor(np.flip(depth, axis=0), dtype=torch.float32, device=self.device, requires_grad=False)
        depth = depth.reshape(-1, 1)
        return torch.cat((depth, depth, depth), dim=-1)

    def get_PC(self, depth): 
        depth = self.reshape_depth(depth)
        xyz = depth * self.uv1
        return xyz