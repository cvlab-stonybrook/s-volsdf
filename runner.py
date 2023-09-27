#--------------------------args & GPU-------------------------------------#
from omegaconf import DictConfig, OmegaConf
import hydra
from helpers.help import logger
from helpers.help import run_help

# init args
@hydra.main(version_base=None, config_path="config", config_name="ours")
def get_config(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    global args
    args = cfg
get_config()

# update args & set GPU
args = run_help(args)
#------------------------------------------------------------------------#

import gc, cv2, signal
import os
import time
import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from plyfile import PlyData, PlyElement
from multiprocessing import Pool
from functools import partial
from pathlib import Path
from skimage.morphology import binary_dilation, disk
from models.TransMVSNet import TransMVSNet
from models.CasMVSNet import CascadeMVSNet
from models.ucsnet import UCSNet
from helpers.utils import *
from datasets.data_io import read_pfm, save_pfm
from datasets.general_eval import MVSDataset
from volsdf.datasets.scene_dataset import get_trains_ids, get_eval_ids
from volsdf.vsdf import VolOpt

cudnn.benchmark = True
torch.set_printoptions(sci_mode=False, precision=4)
np.set_printoptions(suppress=True, precision=4)

def save_depth(testlist):
    for scene in testlist:
        logger.info(f"parameter adjust - {scene}")
        _sparse_weight, _inverse_depth = args.vol.loss.sparse_weight, args.inverse_depth

        # sparsity regularization, slightly better performance with following settings
        if args.vol.dataset.data_dir == 'DTU':
            if scene in ['scan37']:
                args.vol.loss.sparse_weight = 0.1
            elif scene in ['scan24']:
                args.vol.loss.sparse_weight = 0
        elif args.vol.dataset.data_dir == 'BlendedMVS':
            if scene in ['scan2', 'scan3', 'scan7', 'scan9']:
                args.vol.loss.sparse_weight = 0

        # unbounded scenes, use inverse depth sampling
        if args.vol.dataset.data_dir == 'BlendedMVS':
            if scene in ['scan1', 'scan2', 'scan5', 'scan6', 'scan8', 'scan9']:
                args.inverse_depth = True
                logger.info("    inverse_D=[True,False,False]")

        logger.info(f"    inverse depth={args.inverse_depth} sparse_weight={args.vol.loss.sparse_weight}")

        save_scene_depth(scene)

        args.vol.loss.sparse_weight, args.inverse_depth = _sparse_weight, _inverse_depth

# prepare data for image-based rendering
def create_scene(scene):
    os.makedirs(os.path.join(args.outdir, scene), exist_ok=True)

    # dataset, dataloader
    trains_i = get_trains_ids(args.vol.dataset.data_dir, scene, args.num_view)
    evals_i = get_eval_ids(args.vol.dataset.data_dir, int(scene[4:])) # only save cam, no need for images
    trains_i += evals_i
    mvs_datapath = os.path.join(args.data_dir_root, args.vol.dataset.data_dir, 'mvs_data')
    test_dataset = MVSDataset(mvs_datapath, [scene], "test", args.num_view, args.vol.dataset.data_dir, args.numdepth, args.interval_scale,
                            max_h=args.max_h, max_w=args.max_w, trains_i=trains_i, args=args)
    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    for batch_idx, sample in enumerate(TestImgLoader):
        filenames = sample["filename"]
        cams = sample["proj_matrices"]["stage{}".format(3)].numpy()
        imgs = sample["imgs"].numpy()
        cam_near_far = sample["cam_near_far"].numpy()[0]
        del sample

        # save depth maps and confidence maps
        for filename, cam, img in zip(filenames, cams, imgs):
            img = img[0]  #ref view
            cam = cam[0]  #ref cam
            id_ = int(filename.split('/')[-1][:8])
            cam_filename = os.path.join(args.outdir, filename.format('cams', '_cam.txt'))
            img_filename = os.path.join(args.outdir, filename.format('images', '.png'))
            os.makedirs(cam_filename.rsplit('/', 1)[0], exist_ok=True)
            os.makedirs(img_filename.rsplit('/', 1)[0], exist_ok=True)
            # cam
            write_cam(cam_filename, cam, cam_near_far)
            # img
            if id_ not in evals_i:
                img = np.clip(np.transpose(img, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_filename, img_bgr)

# run CasMVS model to save depth maps and confidence maps
def save_scene_depth(scene):
    # save args for individual scan (scene)
    os.makedirs(os.path.join(args.outdir, scene), exist_ok=True)
    with open(os.path.join(args.outdir, scene, 'args.yaml'), "w") as f:
        OmegaConf.save(args, f)

    # dataset, dataloader
    trains_i = get_trains_ids(args.vol.dataset.data_dir, scene, args.num_view)
    mvs_datapath = os.path.join(args.data_dir_root, args.vol.dataset.data_dir, 'mvs_data')
    test_dataset = MVSDataset(mvs_datapath, [scene], "test", args.num_view, args.vol.dataset.data_dir, args.numdepth, args.interval_scale,
                            max_h=args.max_h, max_w=args.max_w, trains_i=trains_i, args=args)
    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    # Let's begin
    torch.cuda.empty_cache()

    # MVS model
    if args.mvs_model_name == 'casmvsnet':
        mvs_model_ckpt_path = os.path.join(args.data_dir_root, 'mvs_models', 'casmvsnet.ckpt')
        model = CascadeMVSNet(refine=False, ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                              depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                              share_cr=args.share_cr,
                              cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                              grad_method=args.grad_method)
    elif args.mvs_model_name == 'ucsnet':
        mvs_model_ckpt_path = os.path.join(args.data_dir_root, 'mvs_models', 'ucsnet.ckpt')
        model = UCSNet(stage_configs=[int(nd) for nd in args.ndepths.split(",") if nd], lamb=1.5)
    elif args.mvs_model_name == 'transmvsnet':
        mvs_model_ckpt_path = os.path.join(args.data_dir_root, 'mvs_models', 'model_dtu.ckpt')
        model = TransMVSNet(refine=False, ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                            depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                            share_cr=args.share_cr,
                            cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                            grad_method=args.grad_method)
    else:
        mvs_model_ckpt_path = None
        raise NotImplementedError

    # load MVS model
    logger.info("loading model {}".format(args.mvs_model_name))
    state_dict = torch.load(mvs_model_ckpt_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict['model'], strict=True)
    model.cuda()
    model.eval()

    # set some args for easy use later
    use_nerf_d = [d_i for d_i in args.use_nerf_d]
    opt_stepNs = [d_i for d_i in args.opt_stepNs]
    ndepths = [int(nd) for nd in args.ndepths.split(",") if nd]

    logger.debug(", ".join([f"{id_}: {sample['filename'][0]}" for id_, sample in enumerate(TestImgLoader)]))

    # volume optimizer
    vol_opt = VolOpt(args=args,
                    batch_size=1,
                    is_continue=args.get('is_continue', False),
                    timestamp='latest',
                    checkpoint='latest',
                    scan=scene)
    vol_opt.trains_i = trains_i
    assert vol_opt.trains_i == vol_opt.train_dataset.trains_ids()

    #-----------------------------------------------------------#
    img_n = len(TestImgLoader)
    view_extra_samples, outs_samples = [None,]*img_n, [None,]*img_n
    depths, confis = [None,]*img_n, [None,]*img_n

    for stage_idx in range(3):
        int_r = None if args.mvs_model_name =='ucsnet' else model.depth_interals_ratio[stage_idx]

        # (a) ---------- cost volume ----------
        start_time = time.time()
        outs, view_extras = [None,]*img_n, [None,]*img_n
        for i, sample in enumerate(TestImgLoader):
            sample_cuda = tocuda(sample)
            with torch.no_grad():
                # step 1. feature extraction
                imgs = sample_cuda["imgs"]
                features = []
                for nview_idx in range(imgs.size(1)): #imgs shape (B, N, C, H, W)
                    img = imgs[:, nview_idx]
                    if args.mvs_model_name =='ucsnet':
                        features.append(model.feature_extraction(img))
                    else:
                        features.append(model.feature(img))
                if args.mvs_model_name == 'transmvsnet':
                    features = model.FMT_with_pathway(features)
                del imgs
                torch.cuda.empty_cache()
                # step 2. cost volume
                outs[i], view_extras[i] = model(stage_idx, 
                    sample_cuda, features=features, extra=view_extra_samples[i], outputs=outs_samples[i],
                    int_r=int_r, prevent_oom=args.prevent_oom, inverse_depth=args.inverse_depth)
                if args.prevent_oom:
                    outs[i], view_extras[i] = tocpu(outs[i]), tocpu(view_extras[i])
                del features
                torch.cuda.empty_cache()

        if args.prevent_oom:
            outs, view_extras = tocuda(outs), tocuda(view_extras)
        torch.cuda.empty_cache()
        logger.debug(f"time(gen cost volume)={time.time()-start_time:.2f}")
        if args.ablate:
            for i in range(img_n):
                outs_samples[i], view_extra_samples[i] = outs[i], view_extras[i]
            continue

        # (b) ----------Volume optimization ----------
        do_volopt = opt_stepNs[stage_idx] > 0 and use_nerf_d[stage_idx] > 0
        if do_volopt:
            # create multi-scale P - data loader
            vol_opt.gen_dataset(stage_idx)
            vol_opt.stg = stage_idx
            vol_opt.loss.set_stg(stage_idx)
            # mvs inputs: P, depth_range
            vol_opt.get_mvs_input(outs)

            # noise-tolerant optimization
            epoch = 0
            if opt_stepNs[stage_idx] > 1:
                epoch = vol_opt.run(opt_stepNs[stage_idx])

            # Vol render depth
            logger.info(f"render volsdf at {vol_opt.plots_dir} ..")
            for i, id_k in enumerate(trains_i):
                depths[i], confis[i] = vol_opt.render_mvs(id_k, epoch)

            # next stage depth
            logger.info(f"mvs_depth replaced by vol_depth at stg={stage_idx} in 0,1,2")
            for i, sample in enumerate(TestImgLoader):
                # assert use_nerf_d[stage_idx] == 1
                outs[i][f"stage{stage_idx+1}"]['depth'] = depths[i]
                outs[i]['depth'] = depths[i]
        
        # update result
        for i in range(img_n):
            outs_samples[i], view_extra_samples[i] = outs[i], view_extras[i]
        del outs, view_extras, sample_cuda
        torch.cuda.empty_cache()
    #-----------------------------------------------------------#
    for batch_idx, sample in enumerate(TestImgLoader):
        outputs = outs_samples[batch_idx]
        dep_min, dep_max = sample["depth_values"].numpy().min(), sample["depth_values"].numpy().max()
        outputs = tensor2numpy(outputs)
        filenames = sample["filename"]
        cams = sample["proj_matrices"]["stage{}".format(3)].numpy()
        imgs = sample["imgs"].numpy()
        del sample

        # save depth maps and confidence maps
        for filename, cam, img, depth_est, photometric_confidence, conf_1, conf_2 in zip(filenames, cams, imgs, \
                                                        outputs["depth"], outputs["photometric_confidence"], outputs['stage1']["photometric_confidence"], outputs['stage2']["photometric_confidence"]):
            #ref view
            img = img[0]
            cam = cam[0]

            H,W = depth_est.shape
            conf_1 = cv2.resize(conf_1, (W,H))
            conf_2 = cv2.resize(conf_2, (W,H))
            photometric_confidence = cv2.resize(photometric_confidence, (W,H))
            conf_final = conf_1 * conf_2 * photometric_confidence

            depth_filename = os.path.join(args.outdir, filename.format('depth_est', '.pfm'))
            confidence_filename = os.path.join(args.outdir, filename.format('confidence', '.pfm'))
            cam_filename = os.path.join(args.outdir, filename.format('cams', '_cam.txt'))
            img_filename = os.path.join(args.outdir, filename.format('images', '.jpg'))
            os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
            os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)
            os.makedirs(cam_filename.rsplit('/', 1)[0], exist_ok=True)
            os.makedirs(img_filename.rsplit('/', 1)[0], exist_ok=True)
            #save depth maps
            save_pfm(depth_filename, depth_est) # (1152, 1536) float32
            _depth_min, _depth_max = np.quantile(depth_est, 0.01), dep_max
            depth_color = visualize_depth(depth_est, depth_min=_depth_min, depth_max=_depth_max)
            cv2.imwrite(os.path.join(args.outdir, filename.format('depth_est', '.png')), depth_color)
            cv2.imwrite(os.path.join(args.outdir, filename.format('depth_est', '_1.png')), visualize_depth(outputs['stage1']["depth"][0], depth_min=_depth_min, depth_max=_depth_max))
            cv2.imwrite(os.path.join(args.outdir, filename.format('depth_est', '_2.png')), visualize_depth(outputs['stage2']["depth"][0], depth_min=_depth_min, depth_max=_depth_max))
            #save confidence maps
            save_pfm(confidence_filename, conf_final)
            cv2.imwrite(os.path.join(args.outdir, filename.format('confidence', '_final.png')),visualize_depth(conf_final, direct=True))
            #save cams, img
            write_cam(cam_filename, cam)
            img = np.clip(np.transpose(img, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_filename, img_bgr)

    del outs_samples
    torch.cuda.empty_cache()
    gc.collect()

def filter_depth(scan_folder, out_folder, plyfilename):
    trains_i = get_trains_ids(args.vol.dataset.data_dir, Path(scan_folder).name, args.num_view)
    pair_data = [
        (trains_i[i], [x for x in trains_i if x != trains_i[i]]) for i in range(len(trains_i))
    ]
    logger.debug(pair_data)

    vertexs, vertex_colors = [], []
    # for each reference view and the corresponding source views
    for ref_view, src_views in pair_data:
        # load the camera parameters
        ref_intrinsics, ref_extrinsics = read_camera_parameters(
            os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(ref_view)))
        # load the reference image
        ref_img = read_img(os.path.join(scan_folder, 'images/{:0>8}.jpg'.format(ref_view)))
        # load the estimated depth of the reference view
        ref_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(ref_view)))[0]
        # load the photometric mask of the reference view
        confidence = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}.pfm'.format(ref_view)))[0]
        assert ref_depth_est.shape == ref_img.shape[:2]
        photo_mask = confidence > args.conf

        all_srcview_depth_ests = []
        all_srcview_x = []
        all_srcview_y = []
        all_srcview_geomask = []

        # compute the geometric mask
        geo_mask_sum = 0
        for src_view in src_views:
            # camera parameters of the source view
            src_intrinsics, src_extrinsics = read_camera_parameters(
                os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(src_view)))
            # the estimated depth of the source view
            src_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(src_view)))[0]
            assert src_depth_est.shape == ref_img.shape[:2]
            geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(ref_depth_est, ref_intrinsics, ref_extrinsics,
                                                                      src_depth_est,
                                                                      src_intrinsics, src_extrinsics,
                                                                      args.filter_dist, args.filter_diff)
            geo_mask_sum += geo_mask.astype(np.int32)
            all_srcview_depth_ests.append(depth_reprojected)
            all_srcview_x.append(x2d_src)
            all_srcview_y.append(y2d_src)
            all_srcview_geomask.append(geo_mask)

        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)
        geo_mask = geo_mask_sum >= args.thres_view
        final_mask = np.logical_and(photo_mask, geo_mask)
        if args.eval_mask:
            eval_mask_dir = os.path.join(args.data_dir_root, args.vol.dataset.data_dir, 'eval_mask')
            if args.vol.dataset.data_dir == 'BlendedMVS':
                eval_mask_path = os.path.join(eval_mask_dir, out_folder.split('/')[-1], "mask/{:0>8}.png".format(ref_view))
            elif args.vol.dataset.data_dir == 'DTU':
                eval_mask_path = os.path.join(eval_mask_dir, out_folder.split('/')[-1], "mask/{:0>3}.png".format(ref_view))
                if not os.path.exists(eval_mask_path):  
                    eval_mask_path = os.path.join(eval_mask_dir, out_folder.split('/')[-1], "{:0>3}.png".format(ref_view))
            else:
                raise NotImplementedError
            assert os.path.exists(eval_mask_path)

            eval_mask = read_img(eval_mask_path)
            if len(eval_mask.shape) == 3:
                eval_mask = eval_mask[:, :, -1]
            eval_mask = binary_dilation(eval_mask, disk(12))
            eval_mask = eval_mask * 1.
            eval_mask = cv2.resize(eval_mask, geo_mask.shape[::-1])
            final_mask = np.logical_and(final_mask, eval_mask > 0.)

        os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_photo.png".format(ref_view)), photo_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_geo.png".format(ref_view)), geo_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_final.png".format(ref_view)), final_mask)

        logger.info("processing {}, ref-view{:0>2}, photo/geo/final-mask:{:.3f}/{:.3f}/{:.3f}".format(scan_folder, ref_view,
                                                                                    photo_mask.mean(),
                                                                                    geo_mask.mean(), final_mask.mean()))

        height, width = depth_est_averaged.shape[:2]
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        valid_points = final_mask
        logger.debug(f"valid_points {valid_points.mean():.3f}")
        x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]
        color = ref_img[valid_points]
        xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                            np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                              np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color * 255).astype(np.uint8))

    # final point cloud
    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]
    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    logger.info(f"saving the final MVS result to {plyfilename}")

def init_worker():
    '''
    Catch Ctrl+C signal to termiante workers
    '''
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def pcd_filter_worker(scan):
    assert 'scan' in scan
    scan_id = int(scan[4:])
    save_name = 'mvsnet{:0>3}_l3.ply'.format(scan_id)
    scan_folder = os.path.join(args.outdir, scan)
    out_folder = os.path.join(args.outdir, scan)
    filter_depth(scan_folder, out_folder, os.path.join(args.outdir, save_name))

def pcd_filter(testlist, number_worker):

    partial_func = partial(pcd_filter_worker)

    p = Pool(number_worker, init_worker)
    try:
        p.map(partial_func, testlist)
    except KeyboardInterrupt:
        print("....\nCaught KeyboardInterrupt, terminating workers")
        p.terminate()
    else:
        p.close()
    p.join()


if __name__ == '__main__':
    if 'txt' in args.testlist:
        with open(args.testlist) as f:
            content = f.readlines()
            testlist = [line.rstrip() for line in content] # ["scan1", "scan2"]
    else:
        testlist = [x for x in args.testlist.replace(' ', '').split(',') if x] # ["scan1",]

    logger.warning(f"{testlist} {args.outdir} {args.exps_folder}")
    time.sleep(5)

    # prepare data for image-based rendering
    if args.get("create_scene", False):
        args.x2_mvsres = False
        assert args.num_view == 3
        for scene in testlist:
            create_scene(scene)
        exit()

    # step1. save all the depth maps and the masks in outputs directory
    if not args.filter_only:
        save_depth(testlist)
    
    # step2. filter saved depth maps with photometric confidence maps and geometric constraints:
    pcd_filter(testlist, args.num_worker)