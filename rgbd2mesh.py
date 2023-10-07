import numpy as np
import cv2
import torch
from collections import defaultdict
import mediapipe as mp
import potpourri3d as pp3d
import time
from tqdm import tqdm

def save_img(pred_dict):
    rendered_img = pred_dict['rendered_img']
    rendered_img = rendered_img.cpu().numpy().squeeze()
    depth_img = pred_dict['depth_img']
    depth_img = depth_img.cpu().numpy().squeeze()
    content = (depth_img > 0)
    depth_img[depth_img < 0] = 0
    m, s = np.mean(depth_img[content]), np.std(depth_img[content])
    depth_img[content] = np.clip(depth_img[content], m-s*2, m+s*2)
    mi, ma = depth_img[content].min(), depth_img[content].max()
    print(mi, ma)
    depth_img[content] = (depth_img[content] - mi) / (ma - mi) * 65534 + 1 # [0, 1] -> [0, 65534] -> [1, 65535]
    depth_img = (depth_img).astype(np.uint16)

    out_img = rendered_img[:, :, :3].astype(np.uint8)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('test_bfm_color.png', out_img)
    cv2.imwrite('test_bfm_depth.png', depth_img)

def transform(pred_dict):
    xyz_in_world = pred_dict['vs']
    camera = pred_dict['camera']
    # project from world to screen
    X, Y, Z = xyz_in_world[:, :, 0], xyz_in_world[:, :, 1], xyz_in_world[:, :, 2]
    XYZ1 = torch.stack([X, Y, Z, torch.ones_like(X)], dim=2)
    XYZ1 = XYZ1.view(-1, 4)
    world_to_view_transform = camera.get_world_to_view_transform()
    view_to_ndc_transform = camera.get_projection_transform()
    xyz_in_view = world_to_view_transform.transform_points(xyz_in_world)
    world2ndc = camera.get_full_projection_transform().get_matrix() 
    xyz1 = XYZ1 @ world_to_view_transform.get_matrix() @ view_to_ndc_transform.get_matrix() # world -> ndc
    xyz = xyz1[:, :, :3] / xyz1[:, :, 3:] # world -> ndc
    xyz[:, :, :2] = (1.-xyz[:, :, :2]) /2. * 255. # ndc -> screen
    xyz_in_screen = camera.transform_points_screen(xyz_in_world, image_size=((256, 256),))
    print(abs(xyz[0, :, :] - xyz_in_screen).max())

    # unproject from screen to world
    XYZ = screen_to_world(xyz.squeeze(0), world2ndc)
    print(abs(XYZ - xyz_in_world.squeeze(0)).max())

def screen_to_world(xyz_in_screen, world2ndc):
    xyz_in_ndc = xyz_in_screen.clone()
    xyz_in_ndc[:, :2] = 1 - xyz_in_screen[:, :2] / 255. * 2. # screen -> ndc
    xyz_in_ndc = torch.cat([xyz_in_ndc, torch.ones_like(xyz_in_ndc[:, :1])], dim=1)
    xyz_in_ndc = xyz_in_ndc / xyz_in_ndc[:, 2:3]
    xyz_in_world = xyz_in_ndc @ world2ndc.inverse() # ndc -> world
    xyz_in_world = xyz_in_world.squeeze(0)
    xyz_in_world = xyz_in_world[:, :3] / xyz_in_world[:, 3:]
    del xyz_in_ndc
    return xyz_in_world

lipsUpperOuter = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
lipsLowerOuter = [375, 321, 405, 314, 17, 84, 181, 91, 146, 61]
lipsUpperInner = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
lipsLowerInner = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324]
lips_all = [76, 184, 74, 73, 72, 11, 302, 303, 304, 408, 306,
            62, 183, 42, 41, 38, 12, 268, 271, 272, 407, 292,
            62, 96, 89, 179, 86, 15, 316, 403, 319, 325, 292,
            76, 77, 90, 180, 85, 16, 315, 404, 320, 307, 306,
            ] + lipsUpperOuter + lipsLowerOuter + lipsUpperInner + lipsLowerInner

def get_keypoints(rgb_img, depth_img):
    
    bone_points_xyz_in_screen = []
    bone_points_type = {}
    DELTA = 0.05 * (depth_img.max() - depth_img[depth_img>0].min())
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(rgb_img)
        if not results.multi_face_landmarks:
            print('no face detected')
            return None
        cnt = 0
        for index in lipsUpperOuter + lipsLowerOuter:
            landmark = results.multi_face_landmarks[0].landmark[index]
            x, y, z = landmark.x, landmark.y, landmark.z
            x, y = int(x*256), int(y*256)
            if depth_img[y, x] <= 0:
                continue
            z = 1./depth_img[y, x]
            if [x, y, z] in bone_points_xyz_in_screen:
                continue
            bone_points_xyz_in_screen.append([x, y, z])
            if index in lipsUpperOuter:
                bone_points_type[cnt] = 'lipsUpperOuter'
            elif index in lipsLowerOuter:
                bone_points_type[cnt] = 'lipsLowerOuter'
            cnt += 1
        for index, landmark in enumerate(results.multi_face_landmarks[0].landmark):
            if index in lips_all:
                continue
            x, y, z = landmark.x, landmark.y, landmark.z
            x, y = int(x*256), int(y*256)
            if depth_img[y, x] <= 0 or x > 205:
                continue
            z = 1./depth_img[y, x]
            # filter out the points that are too close to each other
            distance = [abs(xyz[0]-x) + abs(xyz[1]-y) for xyz in bone_points_xyz_in_screen]
            if len(distance) > 0 and min(distance) < 10:
                continue
            # filter out the points that are too sharp
            if x-3 > 0 and abs(depth_img[y, x-3] - depth_img[y, x]) > DELTA:
                continue
            if x+3 < 256 and abs(depth_img[y, x+3] - depth_img[y, x]) > DELTA:
                continue
            if y-3 > 0 and abs(depth_img[y-3, x] - depth_img[y, x]) > DELTA:
                continue
            if y+3 < 256 and abs(depth_img[y+3, x] - depth_img[y, x]) > DELTA:
                continue
            # if 100 < x and x < 160 and 165 < y and y < 178:
            #     continue
            bone_points_xyz_in_screen.append([x, y, z])
            bone_points_type[cnt] = 'other'
            cnt += 1
    bone_points_xyz_in_screen = np.array(bone_points_xyz_in_screen)
    return bone_points_xyz_in_screen, bone_points_type

def get_keypoints_as_bone(rgb_img, depth_img):
    ''' get keypoints from rgb image and depth image
        rgb_img: (256, 256, 3)
        depth_img: (256, 256)
    '''
    bone_points_xyz_in_screen, bone_points_type = get_keypoints(rgb_img, depth_img)
    visualize_bone_points(rgb_img, bone_points_xyz_in_screen, bone_points_type)
    return bone_points_xyz_in_screen, bone_points_type

def visualize_bone_points(img, bone_points_xyz_in_screen, bone_points_type, output_name='test_bfm_color_bones.png'):
    out_img = img.copy()
    for i, (x, y, z) in enumerate(bone_points_xyz_in_screen):
        if i in [ 255]:
            out_img[int(y), int(x)] = [255, 0, 255]
        elif i in bone_points_type:
            if bone_points_type[i] == 'lipsUpperOuter':
                out_img[int(y), int(x)] = [0, 0, 255]
            elif bone_points_type[i] == 'lipsLowerOuter':
                out_img[int(y), int(x)] = [0, 255, 0]
            elif bone_points_type[i] == 'lipsUpperInner':
                out_img[int(y), int(x)] = [255, 0, 0]
            elif bone_points_type[i] == 'lipsLowerInner':
                out_img[int(y), int(x)] = [255, 255, 0]
            else:
                out_img[int(y), int(x)] = [0, 255, 255]
    cv2.imwrite(output_name, out_img)

def calculate_weights_with_geodist(xyz_in_world, tri, pix_to_face, bone_points_xyz_in_screen, k=8):
    xyz_in_world = xyz_in_world.squeeze(0).cpu().numpy()
    tri = tri.cpu().numpy()
    pix_to_face = pix_to_face.squeeze(0).cpu().numpy()

    start_time = time.time()
    solver = pp3d.MeshHeatMethodDistanceSolver(xyz_in_world, tri)
    weights, index = [], []
    weights_lap, index_lap = [], []
    for i in tqdm(range(xyz_in_world.shape[0])):
        dist = []
        distances_to_all = solver.compute_distance(i)
        for (x, y, z) in bone_points_xyz_in_screen:
            face = pix_to_face[int(y), int(x)]
            if face == -1:
                continue
            source = tri[face]
            dist.append(distances_to_all[source].min())
        dist = np.array(dist)
        topk_index = np.argpartition(dist, k)[:k]
        topk_dist = dist[topk_index]
        weights_i = 1 / (topk_dist ** 2 + 1e-3)
        weights_i = weights_i / np.sum(weights_i)
        index.append(topk_index)
        weights.append(weights_i)

        topk_index_lap = np.argpartition(distances_to_all, k)[:k]
        topk_dist_lap = distances_to_all[topk_index_lap]
        weights_i_lap = 1 / (topk_dist_lap ** 2 + 1e-3)
        weights_i_lap = weights_i_lap / np.sum(weights_i_lap)
        weights_lap.append(weights_i_lap)
        index_lap.append(topk_index_lap)
    print('calculate geodesic distance time: ', time.time() - start_time)
    weights = torch.tensor(weights, device='cuda:0', dtype=torch.float32)
    weights_lap = torch.tensor(weights_lap, device='cuda:0', dtype=torch.float32)
    return weights, index, weights_lap, index_lap


def convert_flow(flow_xy):
    forward_pos = defaultdict(list)
    forward_weight = defaultdict(list)
    for i in range(256):
        for j in range(256):
            x, y = flow_xy[j, i]
            y_, x_ = int(y), int(x)
            forward_pos[(x_, y_)].append((i, j))
            forward_pos[(x_+1, y_)].append((i, j))
            forward_pos[(x_, y_+1)].append((i, j))
            forward_pos[(x_+1, y_+1)].append((i, j))

            forward_weight[(x_, y_)].append((x_+1-x) * (y_+1-y))
            forward_weight[(x_+1, y_)].append((x-x_) * (y_+1-y))
            forward_weight[(x_, y_+1)].append((x_+1-x) * (y-y_))
            forward_weight[(x_+1, y_+1)].append((x-x_) * (y-y_))
    return forward_pos, forward_weight
    
def source_to_driving(flow_xy, depth_res, bone_points_xyz_in_screen, bone_points_type, camera, depth_mi=7.5038233, depth_ma=8.115526, vis=False):
    
    flow_xy = (flow_xy + 1.) / 2. * 256.
    # flow_xy = np.around(flow_xy+0.5).astype(np.int)
    flow_xy = flow_xy.reshape(256, 256, 2) 

    depth_res = depth_res.astype(np.float32) / np.max(depth_res) # normalize to [0, 1]
    content = (depth_res > 0)
    depth_res[content] = depth_res[content] * (depth_ma - depth_mi) + depth_mi # map to [mi, ma]

    world2ndc = camera.get_full_projection_transform().get_matrix()

    # convert backward flow to forward flow
    forward_pos, forward_weight = convert_flow(flow_xy)
    
    translation_in_world = np.zeros_like(bone_points_xyz_in_screen)
    bone_points_new_xyz_in_screen = np.zeros_like(bone_points_xyz_in_screen)
    untrack_points = []
    lll = []
    for i, v in enumerate(bone_points_xyz_in_screen):
        x, y = int(v[0]), int(v[1])
        if (x, y) in forward_pos:
            weights = np.array(forward_weight[(x, y)])
            weights = weights / weights.sum()
            pos = np.array(forward_pos[(x, y)])
            _x, _y = (pos * weights[:, None]).sum(0)
            if _x == 0 or _x == 255 or _y == 0 or _y == 255:
                continue
            lll.append(abs(depth_res[int(_y), int(_x)] - depth_res[y, x]))
    DELTA = np.mean(lll) + np.std(lll) * 2
    for i, v in enumerate(bone_points_xyz_in_screen):
        x, y = int(v[0]), int(v[1])
        if (x, y) in forward_pos:
            weights = np.array(forward_weight[(x, y)])
            weights = weights / weights.sum()
            pos = np.array(forward_pos[(x, y)])
            _x, _y = (pos * weights[:, None]).sum(0)
            if _x == 0 or _x == 255 or _y == 0 or _y == 255:
                untrack_points.append(i)
                continue
            if abs(depth_res[int(_y), int(_x)] - depth_res[y, x]) > DELTA:
                untrack_points.append(i)
                continue
            _xyz_in_screen = np.array([_x, _y, 1./depth_res[int(_y), int(_x)]])
            bone_points_new_xyz_in_screen[i] = _xyz_in_screen
            _xyz_in_world = screen_to_world(torch.tensor(_xyz_in_screen, device='cuda:0', dtype=torch.float32).unsqueeze(0), world2ndc)
            xyz_in_world = screen_to_world(torch.tensor(v, device='cuda:0', dtype=torch.float32).unsqueeze(0), world2ndc)
            translation_in_world[i] = (_xyz_in_world - xyz_in_world).cpu().numpy()
        else:
            untrack_points.append(i)

    return translation_in_world, untrack_points

def delete_untrack_points_from_weights(weights, index, untrack_points):
    weights[np.isin(index, untrack_points)] = 0.
    weights = weights / weights.sum(dim=1, keepdims=True)
    return weights

def save_obj(path, v, f, c=None):
    with open(path, 'w') as file:
        if c is not None:
            for i in range(len(v)):
                file.write('v %f %f %f %f %f %f\n' %
                        (v[i, 0], v[i, 1], v[i, 2], c[i, 0]/255., c[i, 1]/255., c[i, 2]/255.))
        else:
            for i in range(len(v)):
                file.write('v %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2]))

        file.write('\n')

        for i in range(len(f)):
            file.write('f %d %d %d\n' % (f[i, 0], f[i, 1], f[i, 2]))

    file.close()

def warp_mesh(pred_dict, translation_in_world, recon_model, weights, index, angles=None):
    xyz_in_world = pred_dict['vs'].squeeze(0)
    xyz_in_world += torch.bmm(weights.unsqueeze(1), torch.tensor(translation_in_world[index], device='cuda:0', dtype=torch.float32)).squeeze(1)
    xyz_in_world = xyz_in_world.unsqueeze(0)
    rendered_img = recon_model.render_mesh(xyz_in_world, pred_dict['color'], angles)
    rendered_img = rendered_img.cpu().numpy().squeeze()
    out_img = rendered_img[:, :, :3].astype(np.uint8)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('test_warp_color.png', out_img)
