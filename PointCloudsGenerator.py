import torch
import math
import numpy as np



class PointCloudGenerator(object):
    def __init__(self, camera_fovy, image_height, image_width, device, cam_matrix=None):
        super(PointCloudGenerator, self).__init__()


        self.device = device
    
        self.fovy = [math.radians(i) for i in camera_fovy] 
        self.height = image_height
        self.width = image_width
        

        if isinstance(cam_matrix, type(None)):
            self.cam_matrix = []
            for i in range(len(camera_fovy)):
                self.cam_matrix.append(self.get_cam_matrix(self.height, self.width, self.fovy[i]))
        
            self.cam_matrix = torch.stack(self.cam_matrix)

        else:
            self.cam_matrix = torch.tensor(self.cam_matrix, dtype=torch.float32, device=self.device, requires_grad=False)


        self.uv1 = torch.ones((self.height, self.width, 3), dtype=torch.float32, device=self.device, requires_grad=False)
        for i in range(self.height):
            for j in range(self.width):
                self.uv1[i][j][0] = (j + 1)
                self.uv1[i][j][1] = (i + 1)

        self.uv1 = self.uv1.flip(0)        
        self.uv1 = self.uv1.reshape(-1, 3)



        self.inverse_cam_matrix = []

        self.swap_x = torch.tensor([[1, 0, 0],
                                    [0, -1, 0],
                                    [0, 0, -1]],
                                    dtype=torch.float32, device=self.device, requires_grad=False)
        
        self.res_mat = []

        for i, cam_mat in enumerate(self.cam_matrix):
            self.inverse_cam_matrix.append(self.get_inverse_cam_matrix(cam_mat))
            self.res_mat.append(torch.matmul(self.uv1, self.inverse_cam_matrix[i].T))
            self.res_mat[i] = torch.matmul(self.res_mat[i], self.swap_x.T)
        
        



    def get_cam_matrix(self, height, width, fovy):
        f = height / (2 * math.tan(fovy / 2))

        return torch.tensor(((f, 0, width / 2), 
                             (0, f, height / 2), 
                             (0, 0, 1)),
                            dtype=torch.float32, device=self.device, requires_grad=False)

    def get_inverse_cam_matrix(self, cam_matrix):
        cx = cam_matrix[0,2]
        fx = cam_matrix[0,0]
        cy = cam_matrix[1,2]
        fy = cam_matrix[1,1]

        return torch.tensor(((1/fx, 0, -cx/fx), 
                             (0, 1/fy, -cy/fy), 
                             (0, 0, 1)),
                             dtype=torch.float32, device=self.device, requires_grad=False)


    def reshape_depth(self, depth):
        depth = torch.tensor(np.flip(depth, axis=0), dtype=torch.float32, device=self.device, requires_grad=False)
        depth = depth.reshape(-1, 1)
        return torch.cat((depth, depth, depth), dim=-1)

    def get_PC(self, depth_maps, rot_matrix=None, position=None): 
        
        rot_matrix = torch.tensor(rot_matrix, dtype=torch.float32, device=self.device).reshape((-1,3,3))
        position = torch.tensor(position, dtype=torch.float32, device=self.device)
        xyz_mas = []

        for i, depth in enumerate(depth_maps):
            depth = self.reshape_depth(depth)
            xyz = depth * self.res_mat[i]


            if not isinstance(rot_matrix, type(None)):
                xyz = torch.matmul(xyz, rot_matrix[i].T)

            if not isinstance(position, type(None)):
                xyz = xyz + position[i]

            xyz_mas.append(xyz)

        return torch.cat(xyz_mas, dim=0)
