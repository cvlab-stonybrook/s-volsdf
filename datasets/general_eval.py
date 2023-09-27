from torch.utils.data import Dataset
import numpy as np
import os, cv2, time
from PIL import Image
from datasets.data_io import *
import copy
from helpers.utils import load_K_Rt_from_P, glob_imgs, read_img
from volsdf.datasets.scene_dataset import scan2hash
from helpers.help import logger

s_h, s_w = 0, 0
class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, data_dir, ndepths=192, interval_scale=1.06, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.nviews_max = 5
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.data_dir = data_dir
        self._max_h, self._max_w = kwargs["max_h"], kwargs["max_w"]
        self.hparams = kwargs.get("args", dict())
        self.trains_i = kwargs.get("trains_i", None)
        self.fix_wh = False

        # check args
        logger.debug(f"max views used in MVS={self.nviews_max}")
        assert len(listfile) == 1 # one scan at a time
        assert self.mode == "test"
        if self.data_dir != 'DTU': assert interval_scale == 1
        assert self.data_dir in ['BlendedMVS', 'DTU']
        assert self.trains_i is not None
        
        self.meta_from_idr(listfile[0], self.data_dir)
        self.metas = self.build_list()
        assert self.trains_i == [x[1] for x in self.metas]

    def meta_from_idr(self, scan, data_dir):
        """
        read metadata in IDR format, see https://github.com/lioryariv/idr
        camera matrix, scale matrix, image paths
        """
        scan_id = scan[4:]
        
        instance_dir = os.path.join(self.hparams.data_dir_root, data_dir, 'scan{0}'.format(scan_id))
        image_dir = '{0}/image'.format(instance_dir)
        cam_file = '{0}/cameras.npz'.format(instance_dir)
        if not os.path.exists(cam_file) and int(scan_id) < 200:
            cam_file = os.path.join(self.hparams.data_dir_root, data_dir, 'scan114', 'cameras.npz')
        assert os.path.exists(image_dir), f"{image_dir} is empty"
        assert os.path.exists(cam_file), f"{cam_file} is empty"

        self.image_paths_idr = sorted(glob_imgs(image_dir))
        n_images = len(self.image_paths_idr)

        camera_dict = np.load(cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]

        self.intrinsics_idr = []
        self.pose_idr = []

        if scan == 'scan5': # scan5only
            # scale_mat for scan5 is wrong, set it to 1 instead
            for scale_mat, world_mat in zip(scale_mats, world_mats):
                P = world_mat @ scale_mat
                intrinsics, pose = load_K_Rt_from_P(None, P[:3, :4])
                self.intrinsics_idr.append(intrinsics)
                self.pose_idr.append(pose)
            self.scale_mat = None
            self.scale_factor = 1.0
            self._scale_mvs = scale_mats[0][0,0]
        else:
            for scale_mat, world_mat in zip(scale_mats, world_mats):
                intrinsics, pose = load_K_Rt_from_P(None, world_mat[:3, :4])
                self.intrinsics_idr.append(intrinsics)
                self.pose_idr.append(pose)
            self.scale_mat = scale_mats[0] # first image of the scene
            self.scale_factor = scale_mats[0][0,0]

    def build_list(self):
        """
        get  (reference view, a list of source views with order from pair.txt)
        """
        metas = []
        scans = self.listfile

        interval_scale_dict = {}
        # scans
        for scan in scans:
            # determine the interval scale of each scene. default is 1.06
            if isinstance(self.interval_scale, float):
                interval_scale_dict[scan] = self.interval_scale
            else:
                interval_scale_dict[scan] = self.interval_scale[scan]

            if self.data_dir == 'DTU':
                pair_file = "{}/pair.txt".format(scan)
                if not os.path.exists(os.path.join(self.datapath, pair_file)): pair_file = "scan1/pair.txt"
            elif self.data_dir == 'BlendedMVS':
                pair_file = "{}/cams/pair.txt".format(scan2hash(scan))
            else:
                raise NotImplementedError
            if self.data_dir in ['DTU', 'BlendedMVS', None]:
                assert os.path.exists(os.path.join(self.datapath, pair_file))     
                # read the pair file
                with open(os.path.join(self.datapath, pair_file)) as f:
                    num_viewpoint = int(f.readline())
                    # viewpoints
                    for view_idx in range(num_viewpoint):
                        ref_view = int(f.readline().rstrip())
                        src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                        # filter by no src view and fill to nviews
                        if len(src_views) > 0:
                            if ref_view not in self.trains_i:
                                continue
                            metas.append((scan, ref_view, src_views, scan))

        metas = [metas[[x[1] for x in metas].index(y)] for y in self.trains_i]
        logger.debug(f"metadata from pair.txt: {metas}")
        self.interval_scale = interval_scale_dict
        logger.debug(f"len_metas: {len(metas)}, interval_scale: {self.interval_scale}")
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename, interval_scale):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        intrinsics[:2, :] /= 4.0
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1])

        if self.data_dir == 'BlendedMVS':
            depth_max = float(lines[11].split()[-1])
            depth_interval = float(depth_max - depth_min) / self.ndepths
            return intrinsics, extrinsics, depth_min, depth_interval
        elif len(lines[11].split()) >= 3:
            # num_depth != 192 (default value)
            num_depth = lines[11].split()[2]
            depth_max = depth_min + int(float(num_depth)) * depth_interval
            depth_interval = (depth_max - depth_min) / self.ndepths

        depth_interval *= interval_scale

        return intrinsics, extrinsics, depth_min, depth_interval

    def scale_mvs_input(self, img, intrinsics, max_w, max_h, base=32):
        intrinsics_resize = copy.deepcopy(intrinsics)

        h, w = img.shape[:2]
        if h != max_h or w != max_w:
            scale = 1.0 * max_h / h
            if scale * w > max_w:
                scale = 1.0 * max_w / w
            new_w, new_h = scale * w // base * base, scale * h // base * base
        else:
            new_w, new_h = 1.0 * w // base * base, 1.0 * h // base * base

        scale_w = 1.0 * new_w / w
        scale_h = 1.0 * new_h / h
        intrinsics_resize[0, :] *= scale_w
        intrinsics_resize[1, :] *= scale_h

        img_resize = cv2.resize(img, (int(new_w), int(new_h)), interpolation=cv2.INTER_CUBIC)

        return img_resize, intrinsics_resize

    def __getitem__(self, idx):
        global s_h, s_w
        meta = self.metas[idx]
        scan, ref_view, src_views, scene_name = meta
        if self.trains_i is not None:
            _srcs = [x for x in src_views if x in self.trains_i]
            view_ids = [ref_view] + _srcs
            view_ids += [x for x in self.trains_i if x not in view_ids]
            assert ref_view not in _srcs and set(view_ids)==set(self.trains_i)
        else:
            # use only the reference view and first nviews-1 source views
            view_ids = [ref_view] + src_views[:self.nviews - 1]

        if len(view_ids) > self.nviews_max: view_ids = view_ids[:self.nviews_max]

        imgs = []
        depth_values = None
        proj_matrices = []

        for i, vid in enumerate(view_ids):
            if self.data_dir == 'BlendedMVS':
                proj_mat_filename = os.path.join(self.datapath, '{}/cams/{:0>8}_cam.txt'.format(scan2hash(scan), vid))
                _intrinsics, _extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename, interval_scale=
                                                                                    self.interval_scale[scene_name])
                # adjust depth range for scan 4, 5
                if scan == 'scan5': # scan5only
                    # scale_mat for scan5 is wrong, set it to 1 instead
                    depth_min, depth_interval = depth_min/self._scale_mvs, depth_interval/self._scale_mvs
                if scan in ['scan4', 'scan5']:
                    depth_max = depth_min + self.ndepths * depth_interval
                    depth_max = min(depth_max, depth_min * 2.197)
                    depth_interval = float(depth_max - depth_min) / self.ndepths
            elif self.data_dir == 'DTU':
                depth_min = 425
                depth_interval = 2.5 * self.interval_scale[scene_name]
            else:
                raise NotImplementedError

            intrinsics = copy.deepcopy(self.intrinsics_idr[vid][:3, :3])
            intrinsics[:2, :] /= 4.0
            extrinsics = np.linalg.inv(self.pose_idr[vid])

            img_filename = os.path.join(self.image_paths_idr[vid])
            img = read_img(img_filename)

            # interpolate the images -> bigger
            # Feat_extract -> (288, 384) (576, 768) (1152, 1536)
            if self.hparams.get("x2_mvsres", False):
                _s_hw = 1536 / self._max_w
                assert self._max_w * _s_hw == 1536 and self._max_h * _s_hw == 1152 
                img, intrinsics = self.scale_mvs_input(img, intrinsics, self._max_w, self._max_h, base=1)
                img, intrinsics = self.scale_mvs_input(img, intrinsics, 1536, 1152)
            else:
                # scale input
                img, intrinsics = self.scale_mvs_input(img, intrinsics, self._max_w, self._max_h)

            # resize to standard height or width
            c_h, c_w = img.shape[:2]
            # assume all images have same size
            if i != 0: assert (c_h == s_h) and (c_w == s_w) 

            imgs.append(img)

            # extrinsics, intrinsics
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                depth_values = np.arange(depth_min, depth_interval * (self.ndepths - 0.5) + depth_min, depth_interval,
                                         dtype=np.float32)
                cam_near_far = np.array([depth_min, depth_interval, self.ndepths, depth_interval * self.ndepths + depth_min])
                s_h, s_w = img.shape[:2]

        # all
        imgs = np.stack(imgs).transpose([0, 3, 1, 2])

        proj_matrices = np.stack(proj_matrices)
        stage2_pjmats = proj_matrices.copy()
        stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 2
        stage3_pjmats = proj_matrices.copy()
        stage3_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 4
        proj_matrices_ms = {
            "stage1": proj_matrices,
            "stage2": stage2_pjmats,
            "stage3": stage3_pjmats
        }

        return {"imgs": imgs[:,:3]*imgs[:,3:] if imgs.shape[1]==4 else imgs,
                "masks": imgs[:,3:] if imgs.shape[1]==4 else np.ones_like(imgs[:,:1]),
                "proj_matrices": proj_matrices_ms,
                "depth_values": depth_values,
                "cam_near_far": cam_near_far,
                "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}",
                }
