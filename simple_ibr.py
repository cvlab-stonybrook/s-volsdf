""" simple image-based rendering """

#--------------------------args-------------------------------------#
from helpers.help import logger
from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(version_base=None, config_path="config", config_name="ours")
def get_config(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    global args
    args = cfg

get_config()
#------------------------------------------------------------------#

from PIL import Image
from pathlib import Path
import os
import numpy as np
import cv2
import copy
import torch
import torch.nn.functional as F
from scipy.special import softmax

from volsdf.datasets.scene_dataset import get_trains_ids, get_eval_ids
from datasets.data_io import read_pfm
from helpers.utils import read_camera_parameters, read_img, check_geometric_consistency

def lift(x, y, z, intrinsics):
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    sk = intrinsics[:, 0, 1]

    x_lift = (x - cx.unsqueeze(-1) + cy.unsqueeze(-1)*sk.unsqueeze(-1)/fy.unsqueeze(-1) - sk.unsqueeze(-1)*y/fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z
    y_lift = (y - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z

    # homogeneous
    return torch.stack((x_lift, y_lift, z, torch.ones_like(z)), dim=-1)

def get_camera_params(uv, pose, intrinsics):
    """
    uv: (B, N, 2)
    pose, intrinsics: (B, 4, 4)
    """
    uv = torch.from_numpy(uv[None, :, :]).float()
    pose = torch.from_numpy(pose[None, :, :]).float()
    intrinsics = torch.from_numpy(intrinsics[None, :, :]).float()

    if pose.shape[1] == 7:
        raise NotImplementedError
    else: # In case of pose matrix representation
        cam_loc = pose[:, :3, 3]
        p = pose

    batch_size, num_samples, _ = uv.shape

    depth = torch.ones((batch_size, num_samples))
    x_cam = uv[:, :, 0].view(batch_size, -1) # (B, N)
    y_cam = uv[:, :, 1].view(batch_size, -1) # (B, N)
    z_cam = depth.view(batch_size, -1) # (B, N)

    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics) # (B, N, 4)

    # permute for batch matrix product
    pixel_points_cam = pixel_points_cam.permute(0, 2, 1) # (B, 4, N)

    # world_coords = (p[:, :3, :3] @ pixel_points_cam[:, :3, :] + p[:, :3, 3:]).permute(0, 2, 1) # (B, N, 3)
    world_coords = (torch.bmm(p[:, :3, :3], pixel_points_cam[:, :3, :]) + p[:, :3, 3:]).permute(0, 2, 1) # (B, N, 3)
    # assert torch.abs(world_coords - torch.bmm(p, pixel_points_cam).permute(0, 2, 1)[:, :, :3]).mean() < 0.0004
    
    ray_dirs = world_coords - cam_loc[:, None, :]
    ray_dirs = F.normalize(ray_dirs, dim=2)

    return ray_dirs, cam_loc

def get_dir_loc(_intrinsics, extrinsics, hw):
    h, w = hw[0], hw[1]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = _intrinsics
    pose = np.linalg.inv(extrinsics)
    uv = np.mgrid[0:h, 0:w].astype(np.int32)
    uv = np.flip(uv, axis=0).copy() # (2, 576, 768)
    uv = uv.reshape(2, -1).transpose(1, 0) # (442368, 2)
    ray_dirs, cam_loc = get_camera_params(uv, pose, intrinsics)
    ray_dirs, cam_loc = ray_dirs.squeeze(), cam_loc.squeeze() # (442368, 3) # (3)
    ray_dirs = ray_dirs.reshape(h, w, 3)
    return ray_dirs.numpy(), cam_loc.numpy()

def get_lpIMG(img_A, num_levels=4, is_mask=False):
    # generate Gaussian pyramid for A,B and mask
    G = img_A.copy().astype("float")
    gpA = [G]
    for i in range(num_levels):
        G = cv2.pyrDown(G)  
        gpA.append(G)
    
    # generate Laplacian Pyramids for A,B and masks
    if is_mask:
        lpA = [gpA[num_levels-1]]
        for i in range(num_levels-2,-1,-1):
            GE = gpA[i]
            lpA.append(GE) 
    else:
        lpA = [gpA[num_levels-1]]
        for i in range(num_levels-1,0,-1):
            GE = cv2.pyrUp(gpA[i])  
            L = cv2.subtract(gpA[i-1],GE)
            lpA.append(L)

    return lpA

def Laplacian_Blending(imgs, masks, num_levels=4):
    # Implement Laplacian_blending
    # assume mask is float32 [0,1], it has the same size to img_A and img_B
    # the mask indicates which parts of img_A or img_B are blended together
    # num_levels is the number of levels in the pyramid
    assert imgs.shape == masks.shape

    lp_imgs = []
    for img_i in imgs:
        lp_img_i = get_lpIMG(img_i, num_levels=num_levels)
        lp_imgs.append(lp_img_i)
    
    lp_masks = []
    for mask_i in masks:
        lp_mask_i = get_lpIMG(mask_i, num_levels=num_levels, is_mask=True)
        lp_masks.append(lp_mask_i)
    
    # Now blend images according to mask in each level
    LS = []
    for i in range(num_levels):
        ls = 0
        for j in range(len(masks)):
            ls += lp_masks[j][i] * lp_imgs[j][i]
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in range(1,num_levels):
      ls_ = cv2.pyrUp(ls_)
      ls_ = cv2.add(ls_, LS[i])

    return np.clip(ls_, 0.0, 1.0)

def image_based_render(scan_folder, out_folder):    
    trains_i = get_trains_ids(args.vol.dataset.data_dir, Path(scan_folder).name, args.num_view)
    evals_i = get_eval_ids(args.vol.dataset.data_dir, int(Path(scan_folder).name[4:]))
    logger.debug(f'trains_i {trains_i}')
    logger.debug(f'evals_i {evals_i}')

    pair_data = [(idx, trains_i) for idx in evals_i]
    # for each reference view and the corresponding source views
    for ref_view, src_views in pair_data:
        # load the camera parameters
        ref_intrinsics, ref_extrinsics = read_camera_parameters(
            os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(ref_view)))
        # load the estimated depth of the reference view
        pred_ref_img = read_img(os.path.join(out_folder, 'eval_{:0>3}.png'.format(ref_view)))
        ref_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(ref_view)))[0]
        ref_dir, ref_loc = get_dir_loc(ref_intrinsics, ref_extrinsics, ref_depth_est.shape)

        all_srcview_depth_ests = []
        all_srcview_x = []
        all_srcview_y = []
        all_srcview_geomask = []

        weight_mask_sum = 0
        weight_masks = []
        sampled_img_srcs = []

        # 0. compute the geometric mask
        for src_view in src_views:
            # camera parameters of the source view
            src_intrinsics, src_extrinsics = read_camera_parameters(
                os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(src_view)))
            # the estimated depth of the source view
            src_img = read_img(os.path.join(scan_folder, 'images/{:0>8}.png'.format(src_view)))
            src_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(src_view)))[0]
            assert src_depth_est.shape == ref_depth_est.shape
            geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(ref_depth_est, ref_intrinsics, ref_extrinsics,
                                                                      src_depth_est,
                                                                      src_intrinsics, src_extrinsics,
                                                                      filter_dist=2,
                                                                      )
            # x2d_src, y2d_src (576, 768) src_img (576, 768, 3)
            sampled_img_src = cv2.remap(src_img, x2d_src, y2d_src, interpolation=cv2.INTER_CUBIC)
            sampled_img_srcs.append(sampled_img_src)

            # per point direction
            src_dir, src_loc = get_dir_loc(src_intrinsics, src_extrinsics, src_depth_est.shape)
            sampled_src_dir = cv2.remap(src_dir, x2d_src, y2d_src, interpolation=cv2.INTER_CUBIC)
            sampled_src_dir /= np.linalg.norm(sampled_src_dir, axis=2, keepdims=True) # nan will be replaced by 0 later
            cos_dir = (sampled_src_dir*ref_dir).sum(axis=2) # -1 .. 1

            weight_mask = cos_dir
            weight_mask = np.nan_to_num(weight_mask)
            weight_mask *= geo_mask.astype(np.int32)
            weight_masks.append(weight_mask)

            weight_mask_sum += weight_mask
            all_srcview_depth_ests.append(depth_reprojected)
            all_srcview_x.append(x2d_src)
            all_srcview_y.append(y2d_src)
            all_srcview_geomask.append(geo_mask)

        weight_mask = 0.2 * np.ones_like(ref_depth_est)
        weight_masks.append(weight_mask)
        sampled_img_srcs.append(pred_ref_img)

        weight_masks = np.stack(weight_masks)
        weight_masks = softmax(20 * weight_masks, axis=0)
        weight_masks = weight_masks[..., None].repeat(3, -1) # (N, H, W) -> （N, H, W, 3)
        sampled_img_srcs = np.stack(sampled_img_srcs)

        # 1. fill undefined pixels
        sampled_img_srcs_4lap = sampled_img_srcs * weight_masks + sampled_img_srcs[-1:] * (1-weight_masks)

        # 2. erode mask, so that when bluring, undefined pixels won't come in
        weight_masks_4lap = copy.deepcopy(weight_masks)
        kernel = np.ones((5, 5), np.uint8)
        for i in range(weight_masks_4lap.shape[0]-1):
            erode_mask = cv2.erode((weight_masks_4lap[i]>0.2)*1.0, kernel) * 1.0
            weight_masks_4lap[i] = erode_mask * weight_masks_4lap[i]
        weight_masks_4lap[-1] += 1e-2
        weight_masks_4lap /= weight_masks_4lap.sum(0, keepdims=True)

        # 3. laplacian blend
        blend_image = Laplacian_Blending(sampled_img_srcs_4lap, weight_masks_4lap, num_levels=4)
        Image.fromarray((blend_image * 255).astype(np.uint8)).save(
            os.path.join(out_folder, 'eval_blend_{:0>3}.png'.format(ref_view))
        )


if __name__ == '__main__':
    # python simple_ibr.py          testlist='config/lists/dtu.txt'  outdir=exps_ibr +evals_folder=exps_result
    # python simple_ibr.py vol=bmvs testlist='config/lists/bmvs.txt' outdir=exps_ibr +evals_folder=exps_result

    if 'txt' in args.testlist:
        with open(args.testlist) as f:
            content = f.readlines()
            testlist = [line.rstrip() for line in content] # ["scan1", "scan2"]
    else:
        testlist = [x for x in args.testlist.replace(' ', '').split(',') if x] # ["scan1",]
    scan_ids = [int(x[4:]) for x in testlist]
    logger.warning(scan_ids)

    for scan_id in scan_ids:
        # use the latest epoch's rendering results
        evaldir = f'{args.evals_folder}/{args.vol.train.expname}_{scan_id}' # exps_result/ours_106
        epoch = 0
        for renderdir in os.listdir(evaldir):
            if renderdir.startswith('rendering_'):
                epoch = max(epoch, int(renderdir.replace('rendering_', '')))
        
        out_folder = os.path.join(evaldir, f'rendering_{epoch}') # exps_result/ours_106/rendering_1562
        scan_folder = os.path.join(args.outdir, f'scan{scan_id}') # data_ibr/scan106
        assert os.path.exists(scan_folder) and os.path.exists(out_folder)
        logger.warning(f'use cam & src_imgs in {scan_folder}')
        logger.warning(f'     add new images to {out_folder}')

        image_based_render(scan_folder, out_folder)