import os
import numpy as np
import random
import math

from scipy.spatial.transform import Rotation as spRotation
from skimage import io
from skimage import color
from skimage.transform import warp
from skimage.transform import rotate, resize
from skimage.transform import ProjectiveTransform

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from network import Network

from camrot_warp_utils import radial_tan_transform

class CamLocDataset(Dataset):
        """Camera localization dataset.

        Access to image, calibration and ground truth data given a dataset directory.
        """

        def __init__(self, root_dir, 
                mode=1, 
                sparse=False, 
                augment=False, 
                warp=False,
                aug_inplane_rotation=30,
                aug_tilt_rotation=20,
                aug_scale_min=2/3, 
                aug_scale_max=3/2, 
                aug_contrast=0.1, 
                aug_brightness=0.1, 
                image_height=480):
                '''Constructor.

                Parameters:
                        root_dir: Folder of the data (training or test).
                        mode: 
                                0 = RGB only, load no initialization targets, 
                                1 = RGB + ground truth scene coordinates, load or generate ground truth scene coordinate targets
                                2 = RGB-D, load camera coordinates instead of scene coordinates
                        sparse: for mode = 1 (RGB+GT SC), load sparse initialization targets when True, load dense depth maps and generate initialization targets when False
                        augment: Use random data augmentation, note: not supported for mode = 2 (RGB-D) since pre-generateed eye coordinates cannot be agumented
                        warp: Use the warping to azimuthal equidistant projection
                        aug_inplane_rotation: Max 2D image rotation angle, sampled uniformly around 0, both directions
                        aug_tilt_rotation: Max tilt rotation angle, sampled uniformly around 0, both directions. This is a rotation around a (random) axis in the principal plane.
                        aug_scale_min: Lower limit of image scale factor for uniform sampling
                        aug_scale_min: Upper limit of image scale factor for uniform sampling
                        aug_contrast: Max relative scale factor for image contrast sampling, e.g. 0.1 -> [0.9,1.1]
                        aug_brightness: Max relative scale factor for image brightness sampling, e.g. 0.1 -> [0.9,1.1]
                        image_height: RGB images are rescaled to this maximum height
                '''

                self.init = (mode == 1)
                self.sparse = sparse
                self.eye = (mode == 2)

                self.warp = warp

                self.image_height = image_height

                self.augment = augment
                self.aug_inplane_rotation = aug_inplane_rotation
                self.aug_tilt_rotation = aug_tilt_rotation
                self.aug_scale_min = aug_scale_min
                self.aug_scale_max = aug_scale_max
                self.aug_contrast = aug_contrast
                self.aug_brightness = aug_brightness
                
                #if self.eye and self.augment and (self.aug_inplane_rotation > 0 or self.aug_tilt_rotation > 0 or self.aug_scale_min != 1 or self.aug_scale_max != 1):
                if self.eye and self.sparse and self.augment and (self.aug_inplane_rotation > 0 or self.aug_tilt_rotation > 0 or self.aug_scale_min != 1 or self.aug_scale_max != 1):
                        print("WARNING: Check your augmentation settings. Camera coordinates will not be augmented.")


                rgb_dir = root_dir + '/rgb/'
                pose_dir =  root_dir + '/poses/'
                calibration_dir = root_dir + '/calibration/'
                if self.eye and self.sparse:
                        coord_dir =  root_dir + '/eye/'
                elif self.sparse:
                        coord_dir =  root_dir + '/init/'
                else:
                        coord_dir =  root_dir + '/depth/'

                self.rgb_files = os.listdir(rgb_dir)
                self.rgb_files = [rgb_dir + f for f in self.rgb_files]
                self.rgb_files.sort()

                self.image_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(self.image_height),
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4], # statistics calculated over 7scenes training set, should generalize fairly well
                        std=[0.25]
                    )
                ])

                self.pose_files = os.listdir(pose_dir)
                self.pose_files = [pose_dir + f for f in self.pose_files]
                self.pose_files.sort()

                self.pose_transform = transforms.Compose([
                        transforms.ToTensor()
                        ])

                self.calibration_files = os.listdir(calibration_dir)
                self.calibration_files = [calibration_dir + f for f in self.calibration_files]
                self.calibration_files.sort()           

                if self.init or self.eye:
                        self.coord_files = os.listdir(coord_dir)
                        self.coord_files = [coord_dir + f for f in self.coord_files]
                        self.coord_files.sort()

                if len(self.rgb_files) != len(self.pose_files):
                        raise Exception('RGB file count does not match pose file count!')

                if not sparse:

                        #create grid of 2D pixel positions when generating scene coordinates from depth
                        self.prediction_grid = np.zeros((2, 
                                math.ceil(5000 / Network.OUTPUT_SUBSAMPLE), 
                                math.ceil(5000 / Network.OUTPUT_SUBSAMPLE)))

                        for x in range(0, self.prediction_grid.shape[2]):
                                for y in range(0, self.prediction_grid.shape[1]):
                                        self.prediction_grid[0, y, x] = x * Network.OUTPUT_SUBSAMPLE
                                        self.prediction_grid[1, y, x] = y * Network.OUTPUT_SUBSAMPLE            

        def __len__(self):
                return len(self.rgb_files)

        def __getitem__(self, idx):

                image = io.imread(self.rgb_files[idx])

                if len(image.shape) < 3:
                        image = color.gray2rgb(image)

                focal_length = float(np.loadtxt(self.calibration_files[idx]))

                # image will be normalized to standard height, adjust focal length as well
                f_scale_factor = self.image_height / image.shape[0]
                focal_length *= f_scale_factor

                pose = np.loadtxt(self.pose_files[idx])
                pose = torch.from_numpy(pose).float()

                if self.init or self.eye:
                        if self.sparse:
                                coords = torch.load(self.coord_files[idx])
                        else:
                                depth = io.imread(self.coord_files[idx])
                                depth = depth.astype(np.float64)
                                depth /= 1000 # from millimeters to meters
                # instead of loading precomputed camera coords, we compute them online to allow
                # data augmentation
                #elif self.eye: 
                #        coords = torch.load(self.coord_files[idx])
                else:
                        coords = 0

                if self.augment:

                        scale_factor = random.uniform(self.aug_scale_min, self.aug_scale_max)
                        inplane_angle = random.uniform(-self.aug_inplane_rotation, self.aug_inplane_rotation)
                        tilt_angle = random.uniform(-self.aug_tilt_rotation, self.aug_tilt_rotation)
                        tmp_inplane_alpha = random.uniform(0, 2*math.pi)
                        tilt_axis = np.array([np.cos(tmp_inplane_alpha), np.sin(tmp_inplane_alpha), 0])

                        tilt_enabled = not np.isclose(tilt_angle, 0)

                        # augment input image
                        cur_image_transform = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize(int(self.image_height * scale_factor)),
                                transforms.Grayscale(),
                                transforms.ColorJitter(brightness=self.aug_brightness, contrast=self.aug_contrast),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        mean=[0.4],
                                        std=[0.25]
                                        )
                        ])
                        image = cur_image_transform(image)      

                        # actual scale factor used:
                        scale_factor = int(self.image_height * scale_factor) / self.image_height

                        # scale focal length
                        focal_length *= scale_factor

                        # Image has now been resized to another resolution, and focal length has been rescaled accordingly.
                        # The principal point has also been effectively rescaled, such that it is still at the center of the image at the new resolution.

                        # The rotation is counter-clockwise around the negative z-axis, as this corresponds to counter-clockwise 2D rotation in the image plane.
                        # Thus, the following defines the inverse rotation - a counter-clockwise rotation around the positive z-axis.
                        R_inplane_inv = np.eye(3)
                        R_inplane_inv[0, 0] = math.cos(inplane_angle * math.pi / 180)
                        R_inplane_inv[0, 1] = -math.sin(inplane_angle * math.pi / 180)
                        R_inplane_inv[1, 0] = math.sin(inplane_angle * math.pi / 180)
                        R_inplane_inv[1, 1] = math.cos(inplane_angle * math.pi / 180)
                        R_inplane = R_inplane_inv.T

                        if tilt_enabled:
                                # Define camera calibration matrix, taking into account the new focal length and principal point after image resize:
                                K = np.eye(3)
                                K[0,0] = focal_length
                                K[1,1] = focal_length
                                K[0,2] = image.size(2) / 2 # px
                                K[1,2] = image.size(1) / 2 # py

                                R_tilt = spRotation.from_rotvec(tilt_axis * tilt_angle / 180. * math.pi).as_matrix()

                                H = K @ R_tilt @ R_inplane @ np.linalg.inv(K)
                                H_transform = ProjectiveTransform(H)

                        # rotate input image
                        def my_rot(t, inplane_angle, order, mode='constant', img_is_pt_tensor=True):
                                if img_is_pt_tensor:
                                        t = t.permute(1,2,0).numpy()
                                if tilt_enabled:
                                        t = warp(t, H_transform.inverse, order=order, mode=mode)
                                else:
                                        t = rotate(t, inplane_angle, order=order, mode=mode)
                                if img_is_pt_tensor:
                                        t = torch.from_numpy(t).permute(2, 0, 1).float()
                                return t

                        image = my_rot(image, inplane_angle, 1, mode='reflect', img_is_pt_tensor=True)

                        if self.init or self.eye:

                                if self.sparse and self.init:
                                        #rotate and scale initalization targets
                                        coords_w = math.ceil(image.size(2) / Network.OUTPUT_SUBSAMPLE)
                                        coords_h = math.ceil(image.size(1) / Network.OUTPUT_SUBSAMPLE)
                                        coords = F.interpolate(coords.unsqueeze(0), size=(coords_h, coords_w))[0]

                                        coords = my_rot(coords, inplane_angle, 0, mode='constant', img_is_pt_tensor=True)
                                else:
                                        #rotate and scale depth maps
                                        depth = resize(depth, image.shape[1:], order=0)
                                        depth = my_rot(depth, inplane_angle, order=0, mode='constant', img_is_pt_tensor=False)
                                        # depth = rotate(depth, inplane_angle, order=0, mode='constant')

                        # rotate ground truth camera pose
                        inplane_angle = inplane_angle * math.pi / 180
                        # Define a 4x4 matrix for the inverse rotation:
                        T_inplane_inv = torch.eye(4)
                        T_inplane_inv[:3, :3] = torch.from_numpy(R_inplane.T).float()

                        if tilt_enabled:
                                # Define a 4x4 matrix for the inverse rotation:
                                T_tilt_inv = torch.eye(4)
                                T_tilt_inv[:3, :3] = torch.from_numpy(R_tilt.T).float()

                        # "pose" is a 4x4 Euclidean transformation which maps camera coordinates to scene coordinates, i.e. it is the inverse of the extrinsic camera parameters.
                        # In order to integrate the transformations of the augmentations, these should be applied in inverse as well, and multiplied from the right.
                        pose = torch.matmul(pose, T_inplane_inv)
                        if tilt_enabled:
                                pose = torch.matmul(pose, T_tilt_inv)

                else:
                        image = self.image_transform(image)     

                # warp input image
                def my_warp(t, order, mode='constant'):
                        def warp_inverse_map(arr):
                                """
                                        Transforms coordinates from the warped image to the
                                        input image.
                                        arr is a (M, 2) array of (col, row) coordinates in
                                        the warped image.
                                """
                                x, y = radial_tan_transform(arr[:, 0],
                                                            arr[:, 1],
                                                            focal_length,
                                                            focal_length,
                                                            image.shape[2]/2,
                                                            image.shape[1]/2,
                                                            False,
                                                            image.shape[1:])
                                arr[:, 0] = x
                                arr[:, 1] = y
                                return arr


                        t = t.permute(1,2,0).numpy()
                        t = warp(t, warp_inverse_map, order=order, mode=mode)
                        t = torch.from_numpy(t).permute(2, 0, 1).float()
                        return t

                if self.warp:
                        # reflect-padding chosen for the case that the predicted scene_coords are warped back
                        # - this requires some feasible output from the network at the edges of the
                        # warped image
                        image = my_warp(image, 1, mode='reflect')  # constant? reflect?

                if (self.init or self.eye) and not self.sparse:
                        # init: generate initialization targets from depth map
                        # eye: generate camera coords from depth map

                        
                        # offsetX = int(Network.OUTPUT_SUBSAMPLE/2)
                        # offsetY = int(Network.OUTPUT_SUBSAMPLE/2)
                        offsetX = 0
                        offsetY = 0

                        coords = torch.zeros((
                                3, 
                                math.ceil(image.shape[1] / Network.OUTPUT_SUBSAMPLE), 
                                math.ceil(image.shape[2] / Network.OUTPUT_SUBSAMPLE)))

                        # subsample to network output size
                        depth = depth[offsetY::Network.OUTPUT_SUBSAMPLE,offsetX::Network.OUTPUT_SUBSAMPLE] 

                        # construct x and y coordinates of camera coordinate
                        xy = self.prediction_grid[:,:depth.shape[0],:depth.shape[1]].copy()
                        # add random pixel shift
                        xy[0] += offsetX
                        xy[1] += offsetY
                        # substract principal point (assume image center)
                        xy[0] -= image.shape[2] / 2
                        xy[1] -= image.shape[1] / 2
                        # reproject
                        xy /= focal_length
                        xy[0] *= depth
                        xy[1] *= depth

                        #assemble camera coordinates trensor
                        eye = np.ndarray((4, depth.shape[0], depth.shape[1]))
                        eye[0:2] = xy
                        eye[2] = depth
                        
                        if self.init:
                                eye[3] = 1
                                # eye to scene coordinates
                                sc = np.matmul(pose.numpy(), eye.reshape(4,-1))
                                sc = sc.reshape(4, depth.shape[0], depth.shape[1])

                                # mind pixels with invalid depth
                                sc[:, depth == 0] = 0
                                sc[:, depth > 1000] = 0
                                sc = torch.from_numpy(sc[0:3])

                                # coords[:,:sc.shape[1],:sc.shape[2]] = sc
                                assert sc.shape == coords.shape, 'sc.shape {} not consistent with coords.shape {}'.format(tuple(sc.shape), tuple(coords.shape))
                                coords[:,:,:] = sc
                        elif self.eye:
                                # don't map to scene coordinates in this case!

                                # mind pixels with invalid depth
                                eye[:, depth == 0] = 0
                                eye[:, depth > 1000] = 0
                                eye = torch.from_numpy(eye[0:3])

                                assert eye.shape == coords.shape, 'eye.shape {} not consistent with coords.shape {}'.format(tuple(eye.shape), tuple(coords.shape))
                                coords[:,:,:] = eye

                return image, pose, coords, focal_length, self.rgb_files[idx]
