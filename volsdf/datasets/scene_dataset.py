import os
import torch
import numpy as np
import random
import cv2
from pathlib import Path
from PIL import Image

import volsdf.utils.general as utils
from volsdf.utils import rend_util

def scan2hash(scan):
    """
    for BlendedMVS dataset
    """
    scan2hash_dict ={
        'scan1': '5a3ca9cb270f0e3f14d0eddb',
        'scan2': '5a6464143d809f1d8208c43c',
        'scan3': '5ab85f1dac4291329b17cb50',
        'scan4': '5b4933abf2b5f44e95de482a',
        'scan5': '5b22269758e2823a67a3bd03',
        'scan6': '5c0d13b795da9479e12e2ee9',
        'scan7': '5c1af2e2bee9a723c963d019',
        'scan8': '5c1dbf200843bc542d8ef8c4',
        'scan9': '5c34300a73a8df509add216d',
    }
    return scan2hash_dict[scan]

def get_trains_ids(data_dir, scan, num_views=0, for_interp=False):
    if num_views <= 0: raise NotImplementedError
    
    if num_views == 49:
        return list(range(49))

    if data_dir == 'DTU':
        train_ids_all = [25, 22, 28, 40, 44, 48, 0, 8, 13]
        return train_ids_all[:num_views]
        
    elif data_dir == 'BlendedMVS':
        assert num_views == 3
        bmvs_train_ids = {
            1: [9, 10, 55],
            2: [59, 9, 52], 
            3: [26, 27, 22], 
            4: [11, 39, 53], 
            5: [32, 42, 47], 
            6: [28, 34, 57],
            7: [5, 25, 2], 
            8: [16, 21, 33], 
            9: [16, 60, 10], 
        }
        bmvs_train_ids_4_interp = {
            1 : [9, 10, 55],
            2 : [59, 52, 9],
            3 : [27, 26, 22],
            4 : [11, 39, 53],
            5 : [42, 32, 47],
            6 : [28, 34, 57],
            7 : [5, 25, 2],
            8 : [16, 33, 21],
            9 : [16, 60, 10],
        }
        if for_interp:
            train_ids_all = bmvs_train_ids_4_interp[int(scan[4:])][:num_views]
        else:
            train_ids_all = bmvs_train_ids[int(scan[4:])][:num_views]
        return train_ids_all
    
    else:
        raise NotImplementedError

def get_eval_ids(data_dir, scan_id=None):
    if 'DTU' == data_dir:
        # from regnerf/pixelnerf
        train_idx = [25, 22, 28, 40, 44, 48, 0, 8, 13]
        exclude_idx = [3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39]
        test_idx = [i for i in range(49) if i not in train_idx + exclude_idx]
        return test_idx
    elif 'BlendedMVS' == data_dir:
        bmvs_test_ids = {1: [19, 35, 58, 0, 33, 37, 31, 20, 61, 22, 36, 13], 2: [14, 49, 37, 1, 27, 8, 12, 39, 65, 23, 71, 68], 3: [33, 0, 28, 11, 6, 7, 15, 25, 13, 31, 16, 1], 4: [30, 48, 9, 68, 50, 59, 23, 29, 0, 46, 2, 71], 5: [58, 55, 24, 57, 17, 16, 41, 44, 0, 13, 20, 26], 6: [55, 25, 13, 75, 73, 2, 88, 22, 10, 80, 40, 1], 7: [17, 10, 22, 8, 23, 0, 27, 6, 14, 13, 29, 7], 8: [27, 25, 47, 28, 10, 17, 30, 18, 8, 26, 24, 43], 9: [118, 18, 136, 92, 68, 89, 133, 83, 25, 65, 94, 59]}
        return bmvs_test_ids[scan_id][:12]
    else:
        raise NotImplementedError

def get_near_id(data_dir, scan_id, idx):
    bmvs_dict = {'scan1': {0: 9, 1: 9, 2: 9, 3: 10, 4: 9, 5: 9, 6: 9, 7: 55, 8: 9, 9: 9, 10: 10, 11: 9, 12: 10, 13: 10, 14: 9, 15: 9, 16: 9, 17: 9, 18: 10, 19: 55, 20: 55, 21: 9, 22: 9, 23: 10, 24: 10, 25: 10, 26: 10, 27: 9, 28: 10, 29: 9, 30: 10, 31: 55, 32: 55, 33: 9, 34: 9, 35: 55, 36: 9, 37: 9, 38: 55, 39: 10, 40: 10, 41: 10, 42: 10, 43: 10, 44: 10, 45: 9, 46: 10, 47: 9, 48: 9, 49: 9, 50: 9, 51: 9, 52: 10, 53: 10, 54: 10, 55: 55, 56: 9, 57: 55, 58: 9, 59: 9, 60: 9, 61: 10, 62: 10, 63: 9}, 'scan2': {0: 52, 1: 59, 2: 59, 3: 59, 4: 9, 5: 9, 6: 59, 7: 52, 8: 59, 9: 9, 10: 9, 11: 52, 12: 59, 13: 52, 14: 52, 15: 52, 16: 52, 17: 52, 18: 52, 19: 9, 20: 52, 21: 52, 22: 9, 23: 9, 24: 9, 25: 9, 26: 52, 27: 52, 28: 52, 29: 52, 30: 59, 31: 59, 32: 9, 33: 52, 34: 59, 35: 52, 36: 52, 37: 59, 38: 52, 39: 59, 40: 9, 41: 59, 42: 52, 43: 59, 44: 59, 45: 9, 46: 52, 47: 52, 48: 9, 49: 52, 50: 52, 51: 52, 52: 52, 53: 52, 54: 9, 55: 9, 56: 52, 57: 52, 58: 52, 59: 59, 60: 59, 61: 52, 62: 52, 63: 9, 64: 9, 65: 52, 66: 52, 67: 52, 68: 52, 69: 9, 70: 52, 71: 9, 72: 9, 73: 52}, 'scan3': {0: 27, 1: 22, 2: 26, 3: 26, 4: 27, 5: 27, 6: 22, 7: 27, 8: 22, 9: 22, 10: 22, 11: 26, 12: 22, 13: 22, 14: 27, 15: 26, 16: 26, 17: 26, 18: 27, 19: 27, 20: 22, 21: 27, 22: 22, 23: 27, 24: 22, 25: 26, 26: 26, 27: 27, 28: 26, 29: 26, 30: 27, 31: 26, 32: 27, 33: 27, 34: 26, 35: 27}, 'scan4': {0: 53, 1: 11, 2: 53, 3: 53, 4: 11, 5: 39, 6: 53, 7: 53, 8: 11, 9: 53, 10: 11, 11: 11, 12: 53, 13: 11, 14: 39, 15: 11, 16: 53, 17: 39, 18: 11, 19: 53, 20: 53, 21: 11, 22: 11, 23: 53, 24: 11, 25: 39, 26: 53, 27: 11, 28: 39, 29: 53, 30: 53, 31: 11, 32: 11, 33: 11, 34: 11, 35: 11, 36: 39, 37: 39, 38: 11, 39: 39, 40: 11, 41: 39, 42: 53, 43: 11, 44: 11, 45: 11, 46: 11, 47: 39, 48: 53, 49: 11, 50: 53, 51: 11, 52: 53, 53: 53, 54: 53, 55: 11, 56: 11, 57: 53, 58: 53, 59: 53, 60: 11, 61: 53, 62: 53, 63: 11, 64: 11, 65: 53, 66: 53, 67: 53, 68: 11, 69: 53, 70: 53, 71: 11, 72: 53, 73: 53, 74: 11, 75: 53, 76: 39, 77: 53, 78: 53, 79: 11, 80: 53, 81: 11, 82: 53}, 'scan5': {0: 47, 1: 47, 2: 47, 3: 42, 4: 47, 5: 32, 6: 42, 7: 32, 8: 42, 9: 32, 10: 32, 11: 32, 12: 42, 13: 42, 14: 42, 15: 42, 16: 47, 17: 42, 18: 32, 19: 42, 20: 32, 21: 42, 22: 47, 23: 47, 24: 32, 25: 47, 26: 47, 27: 47, 28: 47, 29: 42, 30: 47, 31: 32, 32: 32, 33: 42, 34: 42, 35: 42, 36: 42, 37: 42, 38: 32, 39: 47, 40: 42, 41: 47, 42: 42, 43: 42, 44: 47, 45: 42, 46: 32, 47: 47, 48: 47, 49: 47, 50: 42, 51: 47, 52: 42, 53: 42, 54: 42, 55: 42, 56: 42, 57: 42, 58: 32, 59: 42, 60: 42, 61: 32, 62: 42, 63: 32}, 'scan6': {0: 57, 1: 34, 2: 57, 3: 28, 4: 28, 5: 28, 6: 57, 7: 57, 8: 57, 9: 28, 10: 57, 11: 28, 12: 34, 13: 28, 14: 57, 15: 28, 16: 57, 17: 28, 18: 28, 19: 57, 20: 57, 21: 34, 22: 57, 23: 28, 24: 34, 25: 28, 26: 57, 27: 34, 28: 28, 29: 34, 30: 57, 31: 34, 32: 34, 33: 57, 34: 34, 35: 57, 36: 57, 37: 57, 38: 34, 39: 57, 40: 28, 41: 34, 42: 28, 43: 28, 44: 28, 45: 28, 46: 57, 47: 28, 48: 57, 49: 34, 50: 34, 51: 34, 52: 57, 53: 28, 54: 28, 55: 28, 56: 57, 57: 57, 58: 57, 59: 28, 60: 28, 61: 34, 62: 28, 63: 34, 64: 28, 65: 28, 66: 57, 67: 57, 68: 57, 69: 57, 70: 28, 71: 57, 72: 57, 73: 28, 74: 34, 75: 28, 76: 57, 77: 34, 78: 28, 79: 28, 80: 57, 81: 57, 82: 28, 83: 28, 84: 28, 85: 28, 86: 28, 87: 57, 88: 57, 89: 28}, 'scan7': {0: 5, 1: 2, 2: 2, 3: 25, 4: 2, 5: 5, 6: 5, 7: 5, 8: 5, 9: 2, 10: 25, 11: 25, 12: 5, 13: 5, 14: 5, 15: 2, 16: 2, 17: 2, 18: 2, 19: 2, 20: 2, 21: 2, 22: 2, 23: 2, 24: 5, 25: 25, 26: 5, 27: 2, 28: 5, 29: 5, 30: 5}, 'scan8': {0: 21, 1: 16, 2: 21, 3: 21, 4: 16, 5: 16, 6: 16, 7: 16, 8: 16, 9: 33, 10: 16, 11: 21, 12: 21, 13: 16, 14: 16, 15: 16, 16: 16, 17: 16, 18: 33, 19: 33, 20: 33, 21: 21, 22: 16, 23: 16, 24: 16, 25: 33, 26: 16, 27: 16, 28: 21, 29: 21, 30: 16, 31: 16, 32: 16, 33: 33, 34: 16, 35: 16, 36: 16, 37: 33, 38: 33, 39: 16, 40: 21, 41: 16, 42: 16, 43: 33, 44: 21, 45: 16, 46: 21, 47: 21, 48: 16}, 'scan9': {0: 60, 1: 10, 2: 16, 3: 10, 4: 16, 5: 10, 6: 16, 7: 16, 8: 10, 9: 16, 10: 10, 11: 16, 12: 10, 13: 10, 14: 16, 15: 16, 16: 16, 17: 16, 18: 16, 19: 16, 20: 10, 21: 16, 22: 16, 23: 16, 24: 16, 25: 60, 26: 16, 27: 16, 28: 16, 29: 16, 30: 60, 31: 16, 32: 16, 33: 10, 34: 10, 35: 16, 36: 10, 37: 16, 38: 60, 39: 16, 40: 16, 41: 16, 42: 16, 43: 16, 44: 10, 45: 16, 46: 10, 47: 10, 48: 10, 49: 10, 50: 10, 51: 16, 52: 16, 53: 16, 54: 16, 55: 10, 56: 60, 57: 16, 58: 16, 59: 10, 60: 60, 61: 10, 62: 10, 63: 16, 64: 16, 65: 60, 66: 16, 67: 16, 68: 16, 69: 16, 70: 10, 71: 16, 72: 16, 73: 10, 74: 10, 75: 60, 76: 16, 77: 16, 78: 10, 79: 16, 80: 10, 81: 10, 82: 16, 83: 60, 84: 10, 85: 16, 86: 16, 87: 16, 88: 16, 89: 16, 90: 16, 91: 10, 92: 16, 93: 16, 94: 16, 95: 10, 96: 16, 97: 16, 98: 16, 99: 16, 100: 10, 101: 10, 102: 16, 103: 16, 104: 60, 105: 10, 106: 16, 107: 16, 108: 16, 109: 10, 110: 16, 111: 10, 112: 16, 113: 60, 114: 10, 115: 60, 116: 16, 117: 10, 118: 60, 119: 10, 120: 16, 121: 16, 122: 10, 123: 16, 124: 16, 125: 16, 126: 10, 127: 16, 128: 60, 129: 10, 130: 60, 131: 10, 132: 16, 133: 60, 134: 10, 135: 16, 136: 60, 137: 10, 138: 16, 139: 10, 140: 16, 141: 60, 142: 10, 143: 16}}
    if 'BlendedMVS' == data_dir:
        return bmvs_dict[f'scan{scan_id}'][idx]
    else:
        raise NotImplementedError



class SceneDataset(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 img_res,
                 scan_id=0,
                 num_views=-1,
                 data_dir_root=None,
                 ):

        self.data_dir, self.scan_id, self.num_views = data_dir, scan_id, num_views
        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res
        assert num_views in [-1, 3, 4, 5, 6, 9]
        self.mode, self.plot_id = 'train', 0
        self.sampling_idx = None
        self.use_pixel_centers = False

        instance_dir = os.path.join(data_dir_root, data_dir, 'scan{0}'.format(scan_id))
        image_dir = '{0}/image'.format(instance_dir)
        self.cam_file = '{0}/cameras.npz'.format(instance_dir)
        if not os.path.exists(self.cam_file) and int(scan_id) < 200:
            self.cam_file = os.path.join(data_dir_root, data_dir, 'scan114', 'cameras.npz')
        assert os.path.exists(image_dir), "Data directory is empty"
        assert os.path.exists(self.cam_file), "Data directory is empty"
        image_paths = sorted(utils.glob_imgs(image_dir))
        self.n_images = len(image_paths)
        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        # low res
        rgb = rend_util.load_rgb(image_paths[0]) # (C, H, W)
        scale_h, scale_w = self.img_res[0]*1./rgb.shape[0], self.img_res[1]*1./rgb.shape[1]

        # mask
        mask_path = os.path.join(data_dir_root, data_dir, 'eval_mask')
        if data_dir == 'DTU':
            maskf_fn = lambda x: os.path.join(mask_path, f'scan{scan_id}', 'mask', f'{x:03d}.png')
            if not os.path.exists(maskf_fn(0)):
                maskf_fn = lambda x: os.path.join(mask_path, f'scan{scan_id}', f'{x:03d}.png')
        elif data_dir == 'BlendedMVS':
            maskf_fn = lambda x: os.path.join(mask_path, f'scan{scan_id}', 'mask', f'{x:08d}.png')
        else:
            raise NotImplementedError

        self.rgb_images = []
        self.rgb_smooth = []
        self.masks = []
        self.intrinsics_all = []
        self.pose_all = []
        self.scale_factor = scale_mats[0][0,0]
        if int(self.scan_id) == 5 and data_dir == 'BlendedMVS': # scan5only
            # scale_mat for scan5 is wrong, set it to 1 instead
            self.scale_factor = 1.0

        for id_, path in enumerate(image_paths):
            # K, pose
            scale_mat, world_mat = scale_mats[id_], world_mats[id_]
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
            # pose - scale
            intrinsics[0, :] *= scale_w
            intrinsics[1, :] *= scale_h
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

            # rgb
            img = rend_util.load_rgb(path) # (H, W, 3)
            # rgb - scale
            if scale_h != 1 or scale_w != 1:
                img = cv2.resize(img, (self.img_res[1], self.img_res[0]), interpolation=cv2.INTER_CUBIC)
            rgb = img.transpose(2, 0, 1) # (3, H, W)
            rgb = rgb.reshape(3, -1).transpose(1, 0) # -> (3, H*W) -> (H*W, 3)
            self.rgb_images.append(torch.from_numpy(rgb).float())

            # rgb smooth
            img = cv2.GaussianBlur(img, (31,31), 90)
            rgb = img.transpose(2, 0, 1) # (3, H, W)
            rgb = rgb.reshape(3, -1).transpose(1, 0) # -> (3, H*W) -> (H*W, 3)
            self.rgb_smooth.append(torch.from_numpy(rgb).float())

            # mask
            if data_dir == 'DTU' and id_ in get_eval_ids(data_dir=data_dir) and scan_id not in [1, 4, 11, 13, 48]:
                fname = maskf_fn(id_)
                with open(fname, 'rb') as imgin:
                    mask_image = np.array(Image.open(imgin), dtype=np.float32)[:, :, :3] / 255.
                    mask_image = (mask_image == 1).astype(np.float32)
                if scale_h != 1 or scale_w != 1:
                    mask_image = cv2.resize(mask_image, (self.img_res[1], self.img_res[0]), cv2.INTER_NEAREST)
                    mask_image = (mask_image > 0.5).astype(np.float32)
                mask_image = mask_image.transpose(2, 0, 1) # (3, H, W)
                mask_image = mask_image.reshape(3, -1).transpose(1, 0) # -> (3, H*W) -> (H*W, 3)
                self.masks.append(torch.from_numpy(mask_image).float())
            elif data_dir == 'BlendedMVS':
                idxs_with_mask = get_eval_ids(data_dir=data_dir, scan_id=scan_id) + get_trains_ids(data_dir=data_dir, scan=f'scan{scan_id}', num_views=3)
                if id_ in idxs_with_mask:
                    fname = maskf_fn(id_)
                    assert os.path.exists(fname)
                    with open(fname, 'rb') as imgin:
                        mask_image = np.array(Image.open(imgin), dtype=np.float32)
                        assert len(mask_image.shape) == 3 and mask_image.shape[2] == 4
                        mask_image = mask_image[:, :, -1] * 1. / 255.
                        mask_image = np.stack([mask_image,mask_image,mask_image], -1)
                        mask_image = cv2.resize(mask_image, (self.img_res[1], self.img_res[0]), cv2.INTER_NEAREST)
                        mask_image = (mask_image > 0.5).astype(np.float32)
                        mask_image = mask_image.reshape(-1, 3)
                    self.masks.append(torch.from_numpy(mask_image).float())
                else:
                    self.masks.append(torch.ones_like(self.rgb_images[0]).float())
            else:
                self.masks.append(torch.ones_like(self.rgb_images[0]).float())
          
    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        """
        select an image with random N rays/pixels
        """
        if self.num_views >= 1:
            train_ids = self.trains_ids()
            unobs_ids = [x for x in range(self.n_images) if x not in train_ids]
            if self.mode == 'train':
                idx = train_ids[random.randint(0, self.num_views - 1)]
            elif self.mode == 'plot':
                eval_ids = get_eval_ids(data_dir=self.data_dir, scan_id=self.scan_id)
                if len(eval_ids) == 0:
                    eval_ids = unobs_ids
                idx = eval_ids[self.plot_id]
                self.plot_id = (self.plot_id + 1) % len(eval_ids)

        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        if self.use_pixel_centers:
            uv += 0.5

        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.pose_all[idx],
        }
        if self.data_dir in ['BlendedMVS', ]:
            sample["near_pose"] = self.pose_all[get_near_id(data_dir=self.data_dir, scan_id=self.scan_id, idx=idx)]

        ground_truth = {
            "rgb": self.rgb_images[idx],
            "rgb_smooth": self.rgb_smooth[idx],
            "mask": self.masks[idx]
        }

        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            ground_truth["rgb_smooth"] = self.rgb_smooth[idx][self.sampling_idx, :]
            sample["uv"] = uv[self.sampling_idx, :]

        return idx, sample, ground_truth

    def trains_ids(self):
        return get_trains_ids(data_dir=self.data_dir, scan=f'scan{self.scan_id}', num_views=self.num_views)

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']
