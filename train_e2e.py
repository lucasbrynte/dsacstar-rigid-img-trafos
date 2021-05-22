import torch
import torch.optim as optim

import argparse
import time
import math
import random

import dsacstar

from dataset import CamLocDataset
from network import Network

from camrot_warp_utils import radial_arctan_transform_torch
from pytorch_interpolate import interp_bilinear, interp_nearest_nb

parser = argparse.ArgumentParser(
        description='Train scene coordinate regression in an end-to-end fashion.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('scene', help='name of a scene in the dataset folder')

parser.add_argument('network_in', help='file name of a network initialized for the scene')

parser.add_argument('network_out', help='output file name for the new network')

parser.add_argument('--hypotheses', '-hyps', type=int, default=64, 
        help='number of hypotheses, i.e. number of RANSAC iterations')

parser.add_argument('--threshold', '-t', type=float, default=10, 
        help='inlier threshold in pixels (RGB) or centimeters (RGB-D)')

parser.add_argument('--inlieralpha', '-ia', type=float, default=100, 
        help='alpha parameter of the soft inlier count; controls the softness of the hypotheses score distribution; lower means softer')

parser.add_argument('--learningrate', '-lr', type=float, default=0.000001, 
        help='learning rate')

parser.add_argument('--iterations', '-it', type=int, default=100000, 
        help='number of training iterations, i.e. network parameter updates')

parser.add_argument('--weightrot', '-wr', type=float, default=1.0, 
        help='weight of rotation part of pose loss')

parser.add_argument('--weighttrans', '-wt', type=float, default=100.0, 
        help='weight of translation part of pose loss')

parser.add_argument('--softclamp', '-sc', type=float, default=100, 
        help='robust square root loss after this threshold')

parser.add_argument('--maxpixelerror', '-maxerrr', type=float, default=100, 
        help='maximum reprojection (RGB, in px) or 3D distance (RGB-D, in cm) error when checking pose consistency towards all measurements; error is clamped to this value for stability')

parser.add_argument('--mode', '-m', type=int, default=1, choices=[0,1,2],
        help='test mode: 0,1 = RGB, 2 = RGB-D')

parser.add_argument('--tiny', '-tiny', action='store_true',
        help='Train a model with massively reduced capacity for a low memory footprint.')

parser.add_argument('--session', '-sid', default='',
        help='custom session name appended to output files. Useful to separate different runs of the program')

parser.add_argument('--num_workers', '-numwork', type=int, default=4,
        help='number of workers in dataloader')

parser.add_argument('--warp', '-warp', action='store_true',
        help='Process the images by warping them to Azimuthal Equidistant projection in the application of the CNN.')

parser.add_argument('--no-aug', action='store_true',
        help='Disable data augmentation.')

# parser.add_argument('--no-geometric-aug', action='store_true',
#         help='Disable geometric data augmentation.')

parser.add_argument('--aug-scale-range', type=float, nargs=2, default=(2/3, 3/2),
        help='Maximum angle for inplane rotation augmentation.')

parser.add_argument('--aug-inplane-rot-max', type=float, default=30,
        help='Maximum angle for in-plane rotation augmentation.')

parser.add_argument('--unwarp_interp', default='bilinear',
        help='interpolation type for unwarping - bilinear or nearest_nb')

parser.add_argument('--aug-tilt-rot-max', type=float, default=20,
        help='Maximum angle for tilt rotation augmentation.')
opt = parser.parse_args()

trainset = CamLocDataset(
        "./datasets/" + opt.scene + "/train",
        mode=(0 if opt.mode < 2 else opt.mode),
        augment=not opt.no_aug,
        warp=opt.warp,
        aug_inplane_rotation=opt.aug_inplane_rot_max,
        aug_tilt_rotation=opt.aug_tilt_rot_max,
        aug_scale_min=opt.aug_scale_range[0],
        aug_scale_max=opt.aug_scale_range[1],
)
#        augment=True,
#        aug_inplane_rotation=0,
#        aug_scale_min=1,
#        aug_scale_max=1,
#) # use only photometric augmentation, not rotation and scaling
trainset_loader = torch.utils.data.DataLoader(trainset, shuffle=True, num_workers=opt.num_workers)

print("Found %d training images for %s." % (len(trainset), opt.scene))

# load network
network = Network(torch.zeros((3)), opt.tiny)
network.load_state_dict(torch.load(opt.network_in))
network = network.cuda()
network.train()

print("Successfully loaded %s." % opt.network_in)

optimizer = optim.Adam(network.parameters(), lr=opt.learningrate)

iteration = 0
epochs = int(opt.iterations / len(trainset))

# keep track of training progress
train_log = open('log_e2e_%s_%s.txt' % (opt.scene, opt.session), 'w', 1)

training_start = time.time()

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

for epoch in range(epochs):     

        print("=== Epoch: %7d ======================================" % epoch)

        for image, pose, camera_coordinates, focal_length, file in trainset_loader:

                start_time = time.time()

                focal_length = float(focal_length[0])
                pose = pose[0]

                cam_mat = torch.eye(3)
                cam_mat[0, 0] = focal_length
                cam_mat[1, 1] = focal_length
                cam_mat[0, 2] = image.size(3) / 2
                cam_mat[1, 2] = image.size(2) / 2
                cam_mat = cam_mat.cuda()

                # predict scene coordinates
                scene_coordinates = network(image.cuda()) 

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

                # tensor for gradients
                scene_coordinate_gradients = torch.zeros(scene_coordinates.size())

                if opt.mode == 2:
                        # RGB-D mode
                        loss = dsacstar.backward_rgbd(
                                scene_coordinates.cpu(), 
                                camera_coordinates,
                                scene_coordinate_gradients,
                                pose, 
                                opt.hypotheses, 
                                opt.threshold,
                                opt.weightrot,
                                opt.weighttrans,
                                opt.softclamp,
                                opt.inlieralpha,
                                opt.maxpixelerror,
                                random.randint(0,1000000)) # used to initialize random number generator in C++

                else:
                        # RGB mode
                        loss = dsacstar.backward_rgb(
                                scene_coordinates.cpu(), 
                                scene_coordinate_gradients,
                                pose, 
                                opt.hypotheses, 
                                opt.threshold,
                                focal_length, 
                                float(image.size(3) / 2), #principal point assumed in image center
                                float(image.size(2) / 2),
                                opt.weightrot,
                                opt.weighttrans,
                                opt.softclamp,
                                opt.inlieralpha,
                                opt.maxpixelerror,
                                network.OUTPUT_SUBSAMPLE,
                                random.randint(0,1000000)) # used to initialize random number generator in C++

                # update network parameters
                torch.autograd.backward((scene_coordinates), (scene_coordinate_gradients.cuda()))
                optimizer.step()
                optimizer.zero_grad()
                
                end_time = time.time()-start_time
                print('Iteration: %6d, Loss: %.2f, Time: %.2fs \n' % (iteration, loss, end_time), flush=True)

                train_log.write('%d %f\n' % (iteration, loss))
                iteration = iteration + 1

        if iteration % max(1, round(opt.iterations / 10)) == 0:
                print('Saving snapshot of the network to %s.' % opt.network_out)
                torch.save(network.state_dict(), opt.network_out)
        

print('Saving snapshot of the network to %s.' % opt.network_out)
torch.save(network.state_dict(), opt.network_out)

print('Done without errors. Time: %.1f minutes.' % ((time.time() - training_start) / 60))
train_log.close()
