from helpers.help import logger
import os
from datetime import datetime
from pyhocon import ConfigFactory
import copy
import itertools
from omegaconf import OmegaConf
import torch
from tqdm import tqdm
from torch.nn.functional import grid_sample
from torch.utils.tensorboard import SummaryWriter

import volsdf.utils.general as utils
import volsdf.utils.plots as plt
from volsdf.utils import rend_util
from volsdf.datasets.scene_dataset import SceneDataset

class VolOpt():
    def __init__(self, **kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        # get configs
        self.hparams = copy.deepcopy(kwargs['args'])
        resolved = OmegaConf.to_container(self.hparams['vol'], resolve=True, throw_on_missing=True)
        self.conf = ConfigFactory.from_dict(resolved)
        self.batch_size = kwargs['batch_size']
        self.exps_folder_name = self.hparams.exps_folder

        root = './'
        self.expname = self.conf.get_string('train.expname')
        kwargs_scan_id = int(kwargs['scan'][4:])
        scan_id = kwargs_scan_id if kwargs_scan_id != -1 else self.conf.get_int('dataset.scan_id', default=-1)
        self.scan_id = scan_id
        if scan_id != -1:
            self.expname = self.expname + '_{0}'.format(scan_id)

        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join(root, self.hparams.exps_folder, self.expname)):
                timestamps = os.listdir(os.path.join(root, self.hparams.exps_folder, self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']

        # create exps dirs
        utils.mkdir_ifnotexists(os.path.join(root, self.exps_folder_name))
        self.expdir = os.path.join(root, self.exps_folder_name, self.expname)
        utils.mkdir_ifnotexists(self.expdir)
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))
        self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
        utils.mkdir_ifnotexists(self.plots_dir)
        # create checkpoints dirs
        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)
        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))

        # save configs
        with open(os.path.join(self.expdir, self.timestamp, 'run.yaml'), "w") as f:
            OmegaConf.save(self.hparams, f)
        
        # dataset config
        logger.info('Loading data ...')
        dataset_conf = self.conf.get_config('dataset')
        if kwargs_scan_id != -1:
            dataset_conf['scan_id'] = kwargs_scan_id
        dataset_conf['data_dir_root'] = self.hparams.data_dir_root
        logger.info(f"    full resolution in VolOpt {dataset_conf['img_res']}")
        assert self.conf.get_string('train.dataset_class') == "volsdf.datasets.scene_dataset.SceneDataset"
        assert [self.hparams.max_h, self.hparams.max_w] == dataset_conf['img_res']

        # generate dataset
        self.data_confs = [copy.deepcopy(dataset_conf) for _ in range(3)]
        self.gen_dataset(stg = 2) # full resolution
        self.gen_plot_dataset()
        self.stg = 2
        self.ds_len = len(self.train_dataset) # number of training images

        # model
        conf_model = self.conf.get_config('model')
        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=conf_model)
        if torch.cuda.is_available():
            self.model.cuda()

        # loss
        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(**self.conf.get_config('loss'))

        # optimizer
        self.lr = self.conf.get_float('train.learning_rate')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # load ckpt
        self.start_epoch = 0
        self.iter_step, self.total_step = 0, 0
        ckpt_dir = self.conf.get_string('train.ckpt_dir', '')
        if is_continue:
            self.load_from_dir(dir=os.path.join(self.expdir, timestamp), checkpoint=kwargs['checkpoint'])
        elif ckpt_dir != '':
            self.load_from_dir(dir=ckpt_dir, checkpoint='latest')

        # some parameters
        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.render_freq = self.conf.get_int('train.render_freq')
        self.checkpoint_freq = self.conf.get_int('train.checkpoint_freq', default=100)
        self.split_n_pixels = self.conf.get_int('train.split_n_pixels', default=10000)
        self.plot_conf = self.conf.get_config('plot')

        # logs
        self.writer = SummaryWriter(log_dir=os.path.join(self.plots_dir, 'logs'))

        # copy hparams to every module
        self.model.hparams = self.hparams
        self.loss.hparams = self.hparams

    def load_from_dir(self, dir, checkpoint='latest'):
        """
        e.g. dir = exps_debug/ours_114/2022_10_21_12_46_36
        """
        old_checkpnts_dir = os.path.join(dir, 'checkpoints')
        logger.info(f"Load from {old_checkpnts_dir} at {checkpoint}.pth")

        saved_model_state = torch.load(
            os.path.join(old_checkpnts_dir, 'ModelParameters', str(checkpoint) + ".pth"))
        self.model.load_state_dict(saved_model_state["model_state_dict"])

        self.start_epoch = saved_model_state['epoch']
        self.iter_step = saved_model_state['iter_step']

        data = torch.load(
            os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(checkpoint) + ".pth"))
        self.optimizer.load_state_dict(data["optimizer_state_dict"])

    def gen_plot_dataset(self):
        data_conf = copy.deepcopy(self.conf.get_config('dataset'))
        data_conf['img_res'] = [int(_/4.) for _ in data_conf['img_res']]
        self.plot_dataset = SceneDataset(**data_conf)
        self.plot_dataloader = torch.utils.data.DataLoader(self.plot_dataset,
                                                        batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                        shuffle=False,
                                                        collate_fn=self.plot_dataset.collate_fn
                                                        )

    def gen_dataset(self, stg):
        data_conf_stg = self.data_confs[stg]

        self.train_dataset = SceneDataset(**data_conf_stg)
        logger.info('Finish loading data. Data-set size: {0}'.format(len(self.train_dataset)))

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=self.train_dataset.collate_fn
                                                            )
        self.eval_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                        batch_size=1,
                                                        shuffle=False,
                                                        collate_fn=self.train_dataset.collate_fn
                                                        )

        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.train_dataset.img_res
        self.scale_factor = self.train_dataset.scale_factor
        self.n_batches = len(self.train_dataloader)

    def save_checkpoints(self, epoch, latest_only=False):
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict(), "iter_step": self.iter_step},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))

        if latest_only:
            return 0

        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict(), "iter_step": self.iter_step},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))

    def train_step(self, batch, use_mvs=False):
        self.model.train()

        indices, model_input, ground_truth = batch
        model_input["intrinsics"] = model_input["intrinsics"].cuda()
        model_input["uv"] = model_input["uv"].cuda() # [1, 512, 2]           
        model_input['pose'] = model_input['pose'].cuda()
        model_input['iter_step'] = self.iter_step
        
        fast = 1
        model_outputs = self.model(model_input, fast=fast)
        if use_mvs:
            model_outputs['pj'], model_outputs['pi'], _ = self.cost_mapping(
                z_vals=model_outputs['depth_vals'], ts=indices, xyz_raw=model_outputs['xyz'])
        loss_output = self.loss(model_outputs, ground_truth)

        loss = loss_output['loss']

        self.optimizer.zero_grad()
        loss.backward()
        if self.hparams.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.on_after_backward()
        self.optimizer.step()

        psnr = rend_util.get_psnr(model_outputs['rgb_values'],
                                    ground_truth['rgb'].cuda().reshape(-1,3))
           
        if self.total_step % 50 == 0:
            for k, v in loss_output.items():
                self.writer.add_scalar('t/'+k, v, self.total_step)
            
            self.writer.add_scalar('t/beta', self.model.density.get_beta().item(), self.total_step)
            self.writer.add_scalar('t/alpha', 1. / self.model.density.get_beta().item(), self.total_step)
            self.writer.add_scalar('t/psnr', psnr.item(), self.total_step)

        self.train_dataset.change_sampling_idx(self.num_pixels)

        self.iter_step += 1
        self.total_step += 1

    def render_step(self, batch, epoch, dataset, fast=-1):
        self.model.eval()

        indices, model_input, ground_truth = batch
        model_input["intrinsics"] = model_input["intrinsics"].cuda()
        model_input["uv"] = model_input["uv"].cuda()
        model_input['pose'] = model_input['pose'].cuda()
        model_input['iter_step'] = self.iter_step

        split = utils.split_input(model_input, dataset.total_pixels, n_pixels=self.split_n_pixels)
        res = []
        for s in tqdm(split, leave=False):
            out = self.model(s, fast=fast)
            d = {'rgb_values': out['rgb_values'].detach().cpu(),
                    'normal_map': out['normal_map'].detach().cpu(),
                    'depth_values': out['depth_values'].detach().cpu(),
                    'depth_vals': out['depth_vals'].detach().cpu(),
                    'weights': out['weights'].detach().cpu(),
                    'xyz': out['xyz'].detach().cpu(),
                    }
            res.append(d)
        
        del out

        batch_size = ground_truth['rgb'].shape[0]
        model_outputs = utils.merge_output(res, dataset.total_pixels, batch_size)
        stack = []

        plot_data = self.get_plot_data(model_input, model_outputs, model_input['pose'], ground_truth['rgb'])

        depth_cuda = plt.lin2img(plot_data['depth_map'][..., None], dataset.img_res)
        depth_cuda = depth_cuda[0].cuda() * self.scale_factor
        acc = model_outputs['weights'].sum(1).reshape(depth_cuda.shape)
        depth_cuda[acc<0.2] = depth_cuda.max()

        mask = ground_truth['mask'].reshape(-1, 3)
        mask_bin = (mask == 1.)
        mse = torch.mean((model_outputs['rgb_values'] - ground_truth['rgb'].reshape(-1,3))[mask_bin] ** 2)
        psnr = -10. * torch.log(mse) / torch.log(torch.Tensor([10.]))
        self.writer.add_scalar('val/psnr', psnr.item(), self.total_step)

        stack += plt.stacked_plot(indices, plot_data, self.plots_dir, epoch, dataset.img_res, **self.plot_conf)
        stack[0][~mask_bin.reshape(dataset.img_res+[3,]).permute(2, 0, 1)] = 0
        stack = torch.stack(stack, dim=0) # (B, 3, H, W)
        self.writer.add_images('val/vis', stack, self.total_step)

        photo_conf = None

        self.total_step += 1

        return depth_cuda, photo_conf

    def get_plot_data(self, model_input, model_outputs, pose, rgb_gt):
        batch_size, num_samples, _ = rgb_gt.shape

        rgb_eval = model_outputs['rgb_values'].reshape(batch_size, num_samples, 3)
        normal_map = model_outputs['normal_map'].reshape(batch_size, num_samples, 3)
        normal_map = (normal_map + 1.) / 2.
      
        depth_map = model_outputs['depth_values'].reshape(batch_size, num_samples)
        acc = model_outputs['weights'].sum(1).reshape(batch_size, num_samples)
        
        plot_data = {
            'rgb_gt': rgb_gt,
            'pose': pose,
            'rgb_eval': rgb_eval,
            'normal_map': normal_map,
            'depth_map': depth_map,
            'acc': acc
        }

        return plot_data

    def render_mvs(self, id_k, epoch):
        self.train_dataset.mode = 'test'

        self.train_dataset.change_sampling_idx(-1)

        batch = next(itertools.islice(self.eval_dataloader, id_k, None))
        depth_cuda, depth_confi = self.render_step(batch, epoch, self.train_dataset, fast=-1)

        self.train_dataset.mode = 'train'

        return depth_cuda, depth_confi

    def run(self, opt_stepN=1e8):
        # set everything here
        start_iter_step = self.iter_step
        logger.info(f"train volsdf at {self.checkpoints_path} ..")
        logger.info(f"[NOW] total_step={self.total_step}, iter_step={self.iter_step} stg={self.stg} opt_stepN={opt_stepN}")
        
        # start pbar
        pbar = tqdm(total=opt_stepN-start_iter_step, desc="Train", ncols=60)

        # optimization: VolSDF uses 2000 epoch with 50 images, 2000*50 / 3 images -> ?
        epoch = self.start_epoch
        while True:
            if epoch % self.checkpoint_freq == 0:
                self.save_checkpoints(epoch)

            # render during training to check what's wrong
            early_render = (self.iter_step-start_iter_step <= 120*50) and epoch % (20*50//self.ds_len) == 0
            if epoch % self.render_freq == 0 or early_render:
                self.plot_dataset.change_sampling_idx(-1)
                self.plot_dataset.mode = 'plot'
                batch = next(iter(self.plot_dataloader))
                self.plot_dataset.mode = 'train'
                self.render_step(batch, epoch, self.plot_dataset, fast=-1)

                self.save_checkpoints(epoch, latest_only=True)
                torch.cuda.empty_cache()

            self.train_dataset.change_sampling_idx(self.num_pixels)

            for data_index, batch in enumerate(self.train_dataloader):
                self.train_step(batch, self.hparams.use_mvs)
                pbar.update(1)
            
            if self.iter_step-start_iter_step > opt_stepN:
                break

            epoch += 1

        # close pbar
        pbar.close()

        # save
        self.save_checkpoints(epoch)
        self.start_epoch = epoch

        return epoch

    def get_mvs_input(self, outs_samples):
        self.costs, self.z_mvs, self.bd_mvs = dict(), dict(), dict()
        for i in range(len(outs_samples)): # i = 0, 1, 2, id_k = 25, 22, 28
            prob_volume = outs_samples[i]['prob_volume']
            depth_values = outs_samples[i]['depth_values'] / self.scale_factor
            self.costs[i] = prob_volume # (1, 48, 288, 384)
            self.z_mvs[i] = depth_values # (1, 48, 288, 384)
            bd_mvs_i = depth_values[:, [0, -1], :, :] # (1, 2, 288, 384)
            bd_mvs_i[:, 0, :, :] = torch.minimum(bd_mvs_i[:, 0, :, :], torch.ones_like(bd_mvs_i[:, 0, :, :])*self.conf.get_float('model.scene_bounding_sphere'))
            self.bd_mvs[i] = bd_mvs_i

        del prob_volume, depth_values

    @torch.no_grad()
    def cost_mapping(self, z_vals, ts, xyz_raw):
        """
        xyz_w = p @ xy_lift
        x = x_lift/z * fx + cx + (y - cy)*sk/fy
        y = y_lift/z * fy + cy
        """
        results_cost_j = torch.zeros_like(z_vals, requires_grad=False) # (N=N_rays, D=N_samples)
        results_cost_mvs = torch.zeros_like(z_vals, requires_grad=False) # (N=N_rays, D=N_samples)
        valid_mask = torch.zeros_like(z_vals, requires_grad=False, dtype=bool)
        for i, id_k in enumerate(self.trains_i):
            same_view = ts[0] == id_k # (N_rays,)

            xyz_j = xyz_raw.detach().clone() # N, D, 3
            cost, z_mvs = self.costs[i].to(xyz_raw.device), self.z_mvs[i].to(xyz_raw.device) # (1, 48, H, W)
            _h, _w = self.train_dataset.img_res
            K = self.train_dataset.intrinsics_all[id_k] # (4, 4)
            c2w = self.train_dataset.pose_all[id_k].to(xyz_raw.device) # Ps[id_]
            # assert list(cost.shape[2:4]) == self.train_dataset.img_res # check K == cost.K
            c2w = c2w[:3] # (3, 4)
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            sk = K[0, 1]         

            xyz_j -= c2w[:, 3].view(1, 1, 3) # 1, 1, 3 rays_o_j
            xyz_j = xyz_j @ c2w[:, :3] # N, D, 3
            # xyz_j[..., 1:] *= -1 # "right down front"
            # assert (xyz_j[...,2] - uvw[...,2]).sum() < 0.001 # (xyz_j[...,2] - z_vals).sum()
            # assert xyz_j[..., 2:].min() > 0
            xyz_j[..., :2] /= xyz_j[..., 2:] # N, D, 1
            xyz_j[..., 1] = xyz_j[..., 1] * fy + cy
            xyz_j[..., 0] = xyz_j[..., 0] * fx + cx + (xyz_j[..., 1] - cy)*sk/fy
            # assert (xyz_j[..., :2] - uvw[...,:2]).sum() < 0.5
            xyz_j[..., 0] = xyz_j[..., 0] / ((_w - 1) / 2) - 1 # K[0, 2]*2
            xyz_j[..., 1] = xyz_j[..., 1] / ((_h - 1) / 2) - 1 # K[1, 2]*2
            # interpolat near,far, then normalize depth
            Hj, Dj, _ = xyz_j.shape
            xyz_j = xyz_j.view(1,Hj,Dj,3) # 1, H', D', 3
            mvs_near, mvs_far = z_mvs[:,:1,:,:], z_mvs[:,-1:,:,:] # 1, C=1, H, W
            # 1, C=1, H, W    1, H', D', 2  ->  1, C=1, H', D'
            bound_hw = 1.001
            invalid_j = (xyz_j[..., 2]<1e-5)|(xyz_j[..., 0]>bound_hw)|(xyz_j[..., 0]<-bound_hw)|(xyz_j[..., 1]>bound_hw)|(xyz_j[..., 1]<-bound_hw) # 1, H', D'
            xyz_j[invalid_j, :] = -99.
            near_j = grid_sample(mvs_near, xyz_j[...,:2], mode='bilinear', padding_mode='zeros', align_corners=True)[:,0,:,:]
            far_j = grid_sample(mvs_far, xyz_j[...,:2], mode='bilinear', padding_mode='zeros', align_corners=True)[:,0,:,:]
            if self.hparams.inverse_depth and self.stg == 0:
                far_j[invalid_j] = 1e-8
                xyz_j[..., 2] = 2 * (1. - near_j/xyz_j[..., 2]) / (1. - near_j/far_j) - 1 # 1, H', D'
            elif self.hparams.inverse_depth and self.stg >= 1:
                raise NotImplementedError
            else:
                xyz_j[..., 2] = 2 * (xyz_j[..., 2]-near_j) / (far_j-near_j) - 1
            bound_z = 1.01
            invalid_j = (near_j<1e-5)|(far_j<1e-5)|(xyz_j[..., 2]>bound_z)|(xyz_j[..., 2]<-bound_z)|invalid_j # 1, H', D'
            xyz_j[invalid_j, :] = -99.
            # interpolate cost
            xyz_j = xyz_j.view(1,Hj,Dj,1,3) # 1, H', D', 1, 3
            xyz_j = xyz_j.permute(0,2,1,3,4) # 1, D', H', 1, 3
            # B=1,C=1,D=48,H=144,W=192    1, D', H', W'=1, 3   ->  1, C=1, D', H', W'=1
            cost_j = grid_sample(cost[None,...], xyz_j, mode='bilinear', padding_mode='zeros', align_corners=True)
            cost_j = cost_j.squeeze_().permute(1,0) # (N_rays, N_samples)
            # add to cost
            if same_view:
                results_cost_mvs = cost_j
            else:
                results_cost_j += cost_j
                # mask out points not inside at least 1 another view's frustum
                valid_mask = valid_mask | (~invalid_j[0, :, :])
        # mask out points..
        results_cost_mvs[~valid_mask] = 0.
        del cost_j, z_mvs, cost, xyz_j
        return results_cost_j, results_cost_mvs, valid_mask

    def on_after_backward(self) -> None:
        valid_gradients = True
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break

        if not valid_gradients:
            logger.warning(f'detected inf or nan values in gradients. not updating model parameters')
            self.optimizer.zero_grad()