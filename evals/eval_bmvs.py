# adapted from: https://github.com/jzhangbs/DTUeval-python

import numpy as np
import open3d as o3d
from tqdm import tqdm
import argparse
import os
import glob
import sklearn.neighbors as skln
import trimesh

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

def apply_transform(vertices, matrix):
    """
    Transform mesh by a homogeneous transformation matrix.

    Does the bookkeeping to avoid recomputing things so this function
    should be used rather than directly modifying self.vertices
    if possible.

    Parameters
    ------------
    matrix : (4, 4) float
        Homogeneous transformation matrix
    """
    # get c-order float64 matrix
    matrix = np.asanyarray(
        matrix, order='C', dtype=np.float64)

    # only support homogeneous transformations
    if matrix.shape != (4, 4):
        raise ValueError('Transformation matrix must be (4, 4)!')

    # new vertex positions
    new_vertices = trimesh.transformations.transform_points(
        vertices,
        matrix=matrix)
    return new_vertices

def get_scales(data_dir_root):
    data_dir = 'DTU'
    scan_id = 114
    instance_dir = os.path.join(data_dir_root, data_dir, 'scan{0}'.format(scan_id))
    cam_file = '{0}/cameras.npz'.format(instance_dir)
    camera_dict = np.load(cam_file)
    assert camera_dict['scale_mat_0'].astype(np.float32)[0,0] == camera_dict['scale_mat_1'].astype(np.float32)[0,0]
    DTU_scale = camera_dict['scale_mat_0'].astype(np.float64)[0,0]
    print('DTU_scale:', DTU_scale)

    BMVS_scale = {}
    relative_scale = {}
    data_dir = 'BlendedMVS'
    for scan_id in range(1, 10):
        instance_dir = os.path.join(data_dir_root, data_dir, 'scan{0}'.format(scan_id))
        cam_file = '{0}/cameras.npz'.format(instance_dir)
        camera_dict = np.load(cam_file)
        assert camera_dict['scale_mat_0'].astype(np.float32)[0,0] == camera_dict['scale_mat_1'].astype(np.float32)[0,0]
        BMVS_scale[scan_id] = camera_dict['scale_mat_0'].astype(np.float64)[0,0]
        relative_scale[scan_id] = BMVS_scale[scan_id] / DTU_scale
    print('BMVS_scale:')
    print(BMVS_scale)
    print('relative_scale:')
    print(relative_scale)

def write_vis_pcd(file, points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(file, pcd)

if __name__ == "__main__":
    # python evals/eval_bmvs.py --data_dir_root data_s_volsdf --datadir exps_mvs --scan 4
    # python evals/eval_bmvs.py --data_dir_root data_s_volsdf --datadir exps_mvs > exps_mvs/bmvs_chamfer.txt

    """
    sample = 100K
        See VolSDF, "To measure the Chamfer l1 distance we used 100K random point samples from each surface."

    First, we set point cloud scale to be the same as the DTU dataset
        relative_scale = cam_scale_mat[scan_id] / cam_scale_DTU

    Then, we follow DTU dataset settings
        max_dist = 20, radius = 0.2
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=int, default=100000)
    parser.add_argument('--scan', type=int, default=-1)
    parser.add_argument('--datadir', type=str, default='', help='pred point cloud')
    parser.add_argument('--dataset_dir', type=str, default='bmvs/dataset_textured_meshes', help='GT mesh')
    parser.add_argument('--data_dir_root', type=str, default='data_s_volsdf', help='GT data dir')
    parser.add_argument('--save_gt', action='store_true')
    parser.add_argument('-ve', '--visualize_error', action='store_true')
    parser.add_argument('--no_crop', action='store_true', help='NOT [eval only above the ground plane & using object masks]')
    
    args = parser.parse_args()

    gt_pcd_dir = os.path.join(args.data_dir_root, 'BlendedMVS', 'stl') # GT point cloud
    relative_scale = {1: 0.0010051393651899145, 2: 0.0015733906993148704, 3: 0.0012326845045689896, 4: 0.0015294108512811993, 5: 0.007349738091050388, 6: 0.01192223325424887, 7: 0.001284409757598681, 8: 0.0014762879597404273, 9: 0.022978406132555827}
    scans = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    if args.scan in scans:
        scans = [args.scan]

    print("ply_name, chamfer(mm)")
    for scan in scans:
        pbar = tqdm(total=6)

        pbar.set_description('read pred pcd')
        pcd_file = os.path.join(args.datadir, 'mvsnet{:0>3}_l3.ply'.format(scan))
        data_pcd_o3d = o3d.io.read_point_cloud(pcd_file)
        data_pcd = np.asarray(data_pcd_o3d.points).astype('float32') # (N, 3)

        if scan == 5: # scan5only
            # scale_mat for scan5 is wrong, set it to 1 instead
            instance_dir = os.path.join(args.data_dir_root, 'BlendedMVS', f'scan{scan}')
            cam_file = '{0}/cameras.npz'.format(instance_dir)
            scale_mat = np.load(cam_file)['scale_mat_0']
            data_pcd = apply_transform(data_pcd, scale_mat)

        pbar.update(1)
        pbar.set_description('read gt mesh')

        gt_pointcloud_path = os.path.join(gt_pcd_dir, f'scan{scan}.ply')
        if args.save_gt:
            hash_name = scan2hash(f'scan{scan}')
            gt_dir = os.path.join(args.dataset_dir, hash_name, 'textured_mesh')

            # Find all '.obj' files in the current directory
            gt_files = glob.glob(os.path.join(gt_dir, '*.obj'))

            pbar.update(1)
            pbar.set_description('sample pcd from mesh')

            vertices_lis, triangles_lis = [], []
            tri_idx = 0
            # Load the first file as the initial mesh
            mesh = o3d.io.read_triangle_mesh(gt_files[0])
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles) #  0.. 731
            tri_idx += vertices.shape[0]
            vertices_lis.append(vertices)
            triangles_lis.append(triangles)

            # # Loop through the remaining files and add them to the mesh
            for file in gt_files[1:]:
                mesh = o3d.io.read_triangle_mesh(file)
                vertices = np.asarray(mesh.vertices)
                triangles = np.asarray(mesh.triangles) + tri_idx 
                tri_idx += vertices.shape[0]
                vertices_lis.append(vertices)
                triangles_lis.append(triangles)

            vertices = np.concatenate(vertices_lis, axis=0)
            triangles = np.concatenate(triangles_lis, axis=0)

            # GT pcd
            gt_mesh = o3d.geometry.TriangleMesh()
            gt_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            gt_mesh.triangles = o3d.utility.Vector3iVector(triangles)
            sampled_pcd = gt_mesh.sample_points_uniformly(number_of_points=args.sample)
            gt_pointcloud = sampled_pcd
            pbar.set_description(f'{args.sample}')

            o3d.io.write_point_cloud(gt_pointcloud_path, gt_pointcloud)
            continue
        else:
            if not args.no_crop:
                gt_pointcloud_path = os.path.join(gt_pcd_dir, f'scan{scan}_crop.ply')
            assert os.path.exists(gt_pointcloud_path)
            gt_pointcloud = o3d.io.read_point_cloud(gt_pointcloud_path)
            gt_pcd = np.asarray(gt_pointcloud.points).astype('float32')
            pbar.set_description(f'{gt_pcd.shape[0]}')

        # ----------------------chamfer distance---------------------- #
        max_dist = 20
        radius = 0.2
        # x' = a*x + b
        # |x1'-x2'| = |(a*x1+b) - (a*x2+b)| = a|x1-x2|
        # |x1'/a-x2'/a| = |(x1+b/a) - (x2+b/a)| = |x1-x2|
        gt_pcd /= relative_scale[scan]
        data_pcd /= relative_scale[scan]

        pbar.update(1)
        pbar.set_description('random shuffle pcd index')
        shuffle_rng = np.random.default_rng()
        shuffle_rng.shuffle(data_pcd, axis=0)

        nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=radius, algorithm='kd_tree', n_jobs=-1)
        # Find the closest neighbor within the radius of a point.

        pbar.update(1)
        pbar.set_description('compute data2stl')
        try:
            nn_engine.fit(gt_pcd)
            dist_d2s, idx_d2s = nn_engine.kneighbors(data_pcd, n_neighbors=1, return_distance=True)
        except:
            continue
        mean_d2s = dist_d2s[dist_d2s < max_dist].mean()

        pbar.update(1)
        pbar.set_description('compute stl2data')
        try:
            nn_engine.fit(data_pcd)
            dist_s2d, idx_s2d = nn_engine.kneighbors(gt_pcd, n_neighbors=1, return_distance=True)
        except:
            continue
        mean_s2d = dist_s2d[dist_s2d < max_dist].mean()

        pbar.update(1)
        if args.visualize_error:
            # Green: Errors larger than threshold (20)
            # White to Red: Errors counted in the reported statistics
            pbar.set_description('visualize error')
            vis_out_dir = os.path.join(args.datadir, 'result')
            os.makedirs(vis_out_dir, exist_ok=True)
            vis_dist = 10
            R = np.array([[1,0,0]], dtype=np.float64)
            G = np.array([[0,1,0]], dtype=np.float64)
            B = np.array([[0,0,1]], dtype=np.float64)
            W = np.array([[1,1,1]], dtype=np.float64)
            data_color = np.tile(B, (data_pcd.shape[0], 1))
            data_alpha = dist_d2s.clip(max=vis_dist) / vis_dist
            data_color = R * data_alpha + W * (1-data_alpha)
            data_color[dist_d2s[:,0] >= max_dist] = G
            write_vis_pcd(f'{vis_out_dir}/{scan}_d2s.ply', data_pcd, data_color)
            stl_color = np.tile(B, (gt_pcd.shape[0], 1))
            stl_alpha = dist_s2d.clip(max=vis_dist) / vis_dist
            stl_color = R * stl_alpha + W * (1-stl_alpha)
            stl_color[dist_s2d[:,0] >= max_dist] = G
            write_vis_pcd(f'{vis_out_dir}/{scan}_s2d.ply', gt_pcd, stl_color)

        pbar.update(1)
        pbar.set_description('done')
        pbar.close()
        over_all = (mean_d2s + mean_s2d) / 2
        print('scan{:0>3} {:.2f} {:.2f} {:.2f}'.format(scan, mean_d2s, mean_s2d, over_all))