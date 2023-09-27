from loguru import logger
import sys
logger.remove()  # All configured handlers are removed
logger.add(sys.stderr, format="<level>{level: <8}</level> | - <level>{message}</level>", level="INFO")
# logger.add(sys.stderr, format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="DEBUG") 
# logger.add(sys.stderr, format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="INFO") 

from omegaconf import OmegaConf
import os 
import GPUtil

def check_CUDA_OOM(gpu_id, min_cuda_memory=12):
    import nvidia_smi
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_id)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    prevent_oom = info.free/1024/1024/1024 < min_cuda_memory
    nvidia_smi.nvmlShutdown()
    return prevent_oom

def run_help(args):
    if args.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=1.0, maxMemory=0.8, includeNan=False, excludeID=[], excludeUUID=[])
        args.gpu = deviceIDs[0]
    logger.info(f"gpu -> {args.gpu}")
    os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(args.gpu)

    if not args.prevent_oom:
        try:
            args.prevent_oom = check_CUDA_OOM(gpu_id=args.gpu, min_cuda_memory=12)
        except:
            args.prevent_oom = False
            logger.warning(f"nvidia-smi is down.")

    args.vol.dataset.img_res = [args.max_h, args.max_w]
    args.vol.dataset.num_views = args.num_view
    if args.vol.dataset.data_dir != 'DTU':
        args.interval_scale = 1.0

    logger.debug(f"filter dist -> {args.filter_dist}, diff -> {args.filter_diff}")
    logger.debug(f"interval_scale for MVS -> {args.interval_scale}")
    logger.debug(f"img size -> {args.vol.dataset.img_res}")
    logger.debug(f"num_view -> {args.vol.dataset.num_views}")
    logger.debug(f"depth_inter_r > {args.depth_inter_r}")
    logger.debug(f"n_depth -> {args.ndepths}")
    
    # some imoprtant args
    assert len([int(nd) for nd in args.ndepths.split(",") if nd]) == 3
    assert args.depth_inter_r == "1,0.5,0.5"
    assert args.ndepths == "192,32,8"
    assert args.use_nerf_d == [1,0,0]
    assert args.x2_mvsres 
    assert args.vol.dataset.data_dir in ["BlendedMVS", "DTU"]

    os.makedirs(args.outdir, exist_ok=True)
    if not args.filter_only:
        # save global args for all scans
        with open(os.path.join(args.outdir, 'all_scans.yaml'), "w") as f:
            OmegaConf.save(args, f)
    logger.info('{0}'.format(' '.join(sys.argv)))
    
    return args 