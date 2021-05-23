import torch
import torch.nn.functional as F
import numpy as np
import cv2

import dsacstar

import time
import argparse
import math

from dataset import CamLocDataset
from network import Network

from camrot_warp_utils import radial_arctan_transform_torch
from pytorch_interpolate import interp_bilinear, interp_nearest_nb

parser = argparse.ArgumentParser(
        description='Test a trained network on a specific scene.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('scene', help='name of a scene in the dataset folder, e.g. Cambridge_GreatCourt')

parser.add_argument('network', help='file name of a network trained for the scene')

parser.add_argument('--hypotheses', '-hyps', type=int, default=64, 
        help='number of hypotheses, i.e. number of RANSAC iterations')

parser.add_argument('--threshold', '-t', type=float, default=10, 
        help='inlier threshold in pixels (RGB) or centimeters (RGB-D)')

parser.add_argument('--inlieralpha', '-ia', type=float, default=100, 
        help='alpha parameter of the soft inlier count; controls the softness of the hypotheses score distribution; lower means softer')

parser.add_argument('--maxpixelerror', '-maxerrr', type=float, default=100, 
        help='maximum reprojection (RGB, in px) or 3D distance (RGB-D, in cm) error when checking pose consistency towards all measurements; error is clamped to this value for stability')

parser.add_argument('--mode', '-m', type=int, default=1, choices=[0,1,2],
        help='test mode: 0,1 = RGB, 2 = RGB-D')

parser.add_argument('--tiny', '-tiny', action='store_true',
        help='Load a model with massively reduced capacity for a low memory footprint.')

parser.add_argument('--session', '-sid', default='',
        help='custom session name appended to output files, useful to separate different runs of a script')

parser.add_argument('--num_workers', '-numwork', type=int, default=4,
        help='number of workers in dataloader')

parser.add_argument('--warp', '-warp', action='store_true',
        help='Process the images by warping them to Azimuthal Equidistant projection in the application of the CNN.')

parser.add_argument('--unwarp_interp', default='bilinear',
        help='interpolation type for unwarping - bilinear or nearest_nb')

opt = parser.parse_args()

# setup dataset
if opt.mode < 2: opt.mode = 0 # we do not load ground truth scene coordinates when testing
testset = CamLocDataset("./datasets/" + opt.scene + "/test", mode = opt.mode, warp=opt.warp)
testset_loader = torch.utils.data.DataLoader(testset, shuffle=False, num_workers=opt.num_workers)

# load network
network = Network(torch.zeros((3)), opt.tiny)
network.load_state_dict(torch.load(opt.network))
network = network.cuda()
network.eval()

test_log = open('test_%s_%s.txt' % (opt.scene, opt.session), 'w', 1)
pose_log = open('poses_%s_%s.txt' % (opt.scene, opt.session), 'w', 1)

print('Test images found: ', len(testset))

# keep track of rotation and translation errors for calculation of the median error
rErrs = []
tErrs = []
avg_time = 0

pct5 = 0
pct2 = 0
pct1 = 0

# generate grid of target rewarping pixel positions
pixel_grid = torch.zeros((2, 
        math.ceil(5000 / network.OUTPUT_SUBSAMPLE),             # 5000px is max limit of image size, increase if needed
        math.ceil(5000 / network.OUTPUT_SUBSAMPLE)))

for x in range(0, pixel_grid.size(2)):
        for y in range(0, pixel_grid.size(1)):
                # pixel_grid[0, y, x] = x * network.OUTPUT_SUBSAMPLE + network.OUTPUT_SUBSAMPLE / 2
                # pixel_grid[1, y, x] = y * network.OUTPUT_SUBSAMPLE + network.OUTPUT_SUBSAMPLE / 2
                pixel_grid[0, y, x] = x * network.OUTPUT_SUBSAMPLE
                pixel_grid[1, y, x] = y * network.OUTPUT_SUBSAMPLE

pixel_grid = pixel_grid.cuda()

with torch.no_grad():   

        for image, gt_pose, init, focal_length, file in testset_loader:

                focal_length = float(focal_length[0])
                file = file[0].split('/')[-1] # remove path from file name
                gt_pose = gt_pose[0]
                image = image.cuda()

                cam_mat = torch.eye(3)
                cam_mat[0, 0] = focal_length
                cam_mat[1, 1] = focal_length
                cam_mat[0, 2] = image.size(3) / 2
                cam_mat[1, 2] = image.size(2) / 2
                cam_mat = cam_mat.cuda()

                start_time = time.time()

                # predict scene coordinates and neural guidance
                scene_coordinates = network(image)
                if opt.warp:
                        # for pixel_coords corresponding to gt_coords/pixel_grid_crop 
                        # interpolate scene_coordinates to the original
                        # pixel coords
                        
                        # crop pixel grid to gt_coords-size
                        pixel_grid_crop = pixel_grid[:,0:scene_coordinates.size(2),0:scene_coordinates.size(3)].clone()
                        # find the corresponding indices in the warped image
                        idx_x, idx_y = radial_arctan_transform_torch(pixel_grid_crop[0],
                                                                     pixel_grid_crop[1],
                                                                     cam_mat[0, 0],
                                                                     cam_mat[1, 1],
                                                                     cam_mat[0, 2],
                                                                     cam_mat[1, 2],
                                                                     False,
                                                                     image.shape[2:])  # image has shape [1,1,H,W]
                        # indices corresponding to the original warped image must be subsampled
                        # as the network subsamples

                        # idx_x = (idx_x - network.OUTPUT_SUBSAMPLE / 2) / network.OUTPUT_SUBSAMPLE
                        # idx_y = (idx_y - network.OUTPUT_SUBSAMPLE / 2) / network.OUTPUT_SUBSAMPLE
                        idx_x = idx_x / network.OUTPUT_SUBSAMPLE
                        idx_y = idx_y / network.OUTPUT_SUBSAMPLE

                        # truncate too large values (seem to be small deviations)
                        idx_x[idx_x > scene_coordinates.shape[3]-1] = scene_coordinates.shape[3]-1.00001
                        idx_y[idx_y > scene_coordinates.shape[2]-1] = scene_coordinates.shape[2]-1.00001


                        # interpolate from the output of the network
                        if opt.unwarp_interp == "bilinear":
                                scene_coordinates = interp_bilinear(scene_coordinates, idx_x, idx_y)
                        elif opt.unwarp_interp == "nearest_nb":
                                scene_coordinates = interp_nearest_nb(scene_coordinates, idx_x, idx_y)
                        else:
                                raise ValueError("opt.unwarp_interp must be bilinear or nearest_nb")


                scene_coordinates = scene_coordinates.cpu()

                out_pose = torch.zeros((4, 4))

                if opt.mode < 2:
                        # pose from RGB
                        dsacstar.forward_rgb(
                                scene_coordinates, 
                                out_pose, 
                                opt.hypotheses, 
                                opt.threshold,
                                focal_length, 
                                float(image.size(3) / 2), #principal point assumed in image center
                                float(image.size(2) / 2), 
                                opt.inlieralpha,
                                opt.maxpixelerror,
                                network.OUTPUT_SUBSAMPLE)
                else:
                        # pose from RGB-D
                        dsacstar.forward_rgbd(
                                scene_coordinates, 
                                init, #contains precalculated camera coordinates 
                                out_pose, 
                                opt.hypotheses, 
                                opt.threshold,
                                opt.inlieralpha,
                                opt.maxpixelerror)



                avg_time += time.time()-start_time

                # calculate pose errors
                t_err = float(torch.norm(gt_pose[0:3, 3] - out_pose[0:3, 3]))

                gt_R = gt_pose[0:3,0:3].numpy()
                out_R = out_pose[0:3,0:3].numpy()

                r_err = np.matmul(out_R, np.transpose(gt_R))
                r_err = cv2.Rodrigues(r_err)[0]
                r_err = np.linalg.norm(r_err) * 180 / math.pi

                print("\nRotation Error: %.2fdeg, Translation Error: %.1fcm" % (r_err, t_err*100))

                rErrs.append(r_err)
                tErrs.append(t_err * 100)

                if r_err < 5 and t_err < 0.05:
                        pct5 += 1
                if r_err < 2 and t_err < 0.02:
                        pct2 += 1
                if r_err < 1 and t_err < 0.01:
                        pct1 += 1

                # write estimated pose to pose file
                out_pose = out_pose.inverse()

                t = out_pose[0:3, 3]

                # rotation to axis angle
                rot, _ = cv2.Rodrigues(out_pose[0:3,0:3].numpy())
                angle = np.linalg.norm(rot)
                axis = rot / angle

                # axis angle to quaternion
                q_w = math.cos(angle * 0.5)
                q_xyz = math.sin(angle * 0.5) * axis

                pose_log.write("%s %f %f %f %f %f %f %f %f %f\n" % (
                        file,
                        q_w, q_xyz[0], q_xyz[1], q_xyz[2],
                        t[0], t[1], t[2],
                        r_err, t_err))  

median_idx = int(len(rErrs)/2)
tErrs.sort()
rErrs.sort()
avg_time /= len(rErrs)

print("\n===================================================")
print("\nTest complete.")

print('\nAccuracy:')
print('\n5cm5deg: %.1f%%' %(pct5 / len(rErrs) * 100))
print('2cm2deg: %.1f%%' % (pct2 / len(rErrs) * 100))
print('1cm1deg: %.1f%%' % (pct1 / len(rErrs) * 100))

print("\nMedian Error: %.1fdeg, %.1fcm" % (rErrs[median_idx], tErrs[median_idx]))
print("Avg. processing time: %4.1fms" % (avg_time * 1000))
test_log.write('%f %f %f\n' % (rErrs[median_idx], tErrs[median_idx], avg_time))

test_log.close()
pose_log.close()
