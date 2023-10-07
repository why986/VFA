import argparse
import os
import torch
import torch.nn as nn
from networks.generator import Generator
import cv2
import numpy as np
from rgbd2mesh import *
import torchvision
from BFM.BFM09Model import BFM09ReconModel
from scipy.io import loadmat
from pytorch3d.io import load_objs_as_meshes

from pytorch3d.structures import Meshes

from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    blending
)

def norm_image(color, depth, save_path=None):
    color = np.array(color, dtype='float32')
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    if save_path is not None:
        cv2.imwrite(os.path.join(save_path, 'color.png'), color)

    depth = np.array(depth, dtype='float32')
    content = (depth > 0)
    bkg = (depth <= 0)
    color = color / 255. # [0, 1]
    color = color * 2. - 1. # [-1, 1]

    depth[bkg] = 0
    mi, ma = np.min(depth[content]), np.max(depth[content])
    depth[content] = (depth[content] - mi) / (ma - mi)
    depth = depth * 2. - 1. # [-1, 1]
    img = np.concatenate([color, depth[:, :, None]], axis=2) # 256 x 256 x 4
    if save_path is not None:
        depth = (depth + 1) * 65535. / 2.
        depth = depth.astype('uint16')
        cv2.imwrite(os.path.join(save_path, 'depth.png'), depth)

    img = np.transpose(img, (2, 0, 1)) # C x 256 x 256
    img = torch.from_numpy(img).unsqueeze(0).float()  
    return img
import ffmpeg
def load_video(color_video_path, length, offset=0):
    depth_video_path = color_video_path.replace('color', 'depth').replace('.mp4', '.nut')
    try:
        out_depth_byte, _ = (
            ffmpeg
                .input(depth_video_path)
                .output('pipe:', format='rawvideo', pix_fmt='gray16le', loglevel="quiet")
                .run(capture_stdout=True)
            )
        out_color_bytes, _ = (
            ffmpeg
                .input(color_video_path)
                .output('pipe:', format='rawvideo', pix_fmt='rgb24', loglevel="quiet")
                .run(capture_stdout=True)
            )
    except Exception:
        print('Error in loading video %s %s' % (color_video_path, depth_video_path))
        return None, None
    width, height = 256, 256
    video_depth = np.frombuffer(out_depth_byte, np.uint16).reshape([-1, height, width])
    video_color = np.frombuffer(out_color_bytes, np.uint8).reshape([-1, height, width, 3])
    color0, depth0 = video_color[0, ...], video_depth[0, ...]
    driving0 = norm_image(color0, depth0)
    T = video_color.shape[0]
    T = min(T, offset+length)
    vid = []
    for i in range(offset, T):
        color = video_color[i, ...]
        depth = video_depth[i, ...]
        vid.append(norm_image(color, depth))
    vid = torch.cat(vid, dim=0)
    vid = vid.unsqueeze(0) # [1, T, C, H, W]
    return vid, driving0, 30

def load_photos(png_path, length, offset=0):
    photo_list = []
    for i in range(length):
        color = cv2.imread(os.path.join(png_path, 'color', str(i+offset)+".png"))
        depth = cv2.imread(os.path.join(png_path, 'depth', str(i+offset)+".png"), -1)
        photo_list.append(norm_image(color, depth))
    vid = torch.cat(photo_list, dim=0)
    vid = vid.unsqueeze(0) # [1, T, C, H, W]
    return vid, 30

def save_img_from_tensor(tensor, save_path, save_name):
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min()) * 255
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(save_path, save_name), img)

def save_video(vid_source, vid_target_recon, save_path, save_name, fps, use_depth, vid_depth=None):
    vid = vid_target_recon.permute(0, 2, 3, 4, 1) # [B, T, H, W, C]
    print(vid.shape)
    vid = vid.clamp(-1, 1).cpu()
    if use_depth :
        vid_depth = vid_depth.permute(0, 2, 3, 4, 1) # [B, T, H, W, C]
        vid_depth_16 = ((vid_depth - vid_depth.min()) / (vid_depth.max() - vid_depth.min()) * 65535).numpy().astype(np.uint16)
    vid = ((vid - vid.min()) / (vid.max() - vid.min()) * 255).type('torch.ByteTensor')

    vid_source = vid_source.permute(0, 2, 3, 4, 1) 
    print(vid_source.shape)
    vid_source = vid_source.clamp(-1, 1).cpu()
    vid_source = ((vid_source - vid_source.min()) / (vid_source.max() - vid_source.min()) * 255).type('torch.ByteTensor')

    vid_rgb = vid[0, :, :, :, :3]
    if use_depth :
        vid_depth = vid_depth[0, :, :, :, :]
        vid_depth = ((vid_depth - vid_depth.min()) / (vid_depth.max() - vid_depth.min()) * 255).type('torch.ByteTensor')
        vid_depth = vid_depth.repeat(1, 1, 1, 3)
        torchvision.io.write_video(save_name, torch.cat([vid_source[0, :, :, :, :3], vid_rgb, vid_depth], dim=1), fps=fps)
    else:
        torchvision.io.write_video(save_name, torch.cat([vid_source[0, :, :, :, :3], vid_rgb], dim=1), fps=fps)



import face_alignment
from scipy.spatial import ConvexHull

def _normalize_kp(kp):
    kp = kp - kp.mean(axis=0, keepdims=True)
    area = ConvexHull(kp[:, :2]).volume
    area = np.sqrt(area)
    kp[:, :2] = kp[:, :2] / area
    return kp
def find_best_frame(source, driving, fa, kp_source=None):
    if kp_source is None:
        img_source = (source + 1) / 2 * 255
        img_source = img_source.squeeze(0).permute(1, 2, 0).cpu().numpy()
        kp_source = fa.get_landmarks(img_source)[0]
        kp_source = _normalize_kp(kp_source)
    norm  = float('inf')
    frame_num = 0
    _, T, C, H, W = driving.shape
    for i in tqdm(range(T)):
        image = driving[0, i, :3, :, :].permute(1, 2, 0).cpu().numpy()
        image = (image + 1) / 2 * 255
        kp_driving = fa.get_landmarks(image)
        if kp_driving is None:
            continue
        kp_driving = kp_driving[0]
        kp_driving = _normalize_kp(kp_driving)
        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i
    return frame_num

class Demo(nn.Module):
    def __init__(self, args):
        super(Demo, self).__init__()
        self.args = args
        self.calc_weights = args.calc_weights
        self.weights_k = args.weights_k
        self.img_size = 256
        self.source_path = args.source_path
        self.model_name = args.checkpoint.split('/')[1]
        self.save_path = args.save_folder + '/%s' % self.model_name
        self.device = 'cuda:0'
        self.render = self._get_renderer('cuda:0')


        self.avatar_name = args.source_path.split('/')[-1].split('.')[0]
        self.save_path = os.path.join(self.save_path, self.avatar_name + '_' + args.driving_path.split('/')[1] + '_' + str(args.offset) )
        os.makedirs(self.save_path, exist_ok=True)

        self.load_mesh(self.source_path)

        print('==> loading model')
        self.gen = Generator(size=256, in_channels=4, depth_motion_dim=5, distilling=True).cuda()
        weight = torch.load(args.checkpoint, map_location=torch.device('cpu'))['gen']
        self.gen.load_state_dict(weight, strict=False)
        self.gen.eval()

        self.save_name = os.path.join(self.save_path, 'inference.mp4')
        if '.mp4' in args.driving_path:
            self.vid_target, self.driving0, self.fps = load_video(args.driving_path, args.length, args.offset)
        else:
            color = cv2.imread(os.path.join(args.driving_path, 'color', "3500.png"))
            depth = cv2.imread(os.path.join(args.driving_path, 'depth', "3500.png"), -1)
            self.driving0 = norm_image(color, depth)
            self.vid_target, self.fps = load_photos(args.driving_path, args.length, args.offset)
    
    def compute_rotation_matrix(self, angles):
        n_b = angles.size(0)
        sinx = torch.sin(angles[:, 0])
        siny = torch.sin(angles[:, 1])
        sinz = torch.sin(angles[:, 2])
        cosx = torch.cos(angles[:, 0])
        cosy = torch.cos(angles[:, 1])
        cosz = torch.cos(angles[:, 2])

        rotXYZ = torch.eye(3).view(1, 3, 3).repeat(
            n_b * 3, 1, 1).view(3, n_b, 3, 3).to(angles.device)

        rotXYZ[0, :, 1, 1] = cosx
        rotXYZ[0, :, 1, 2] = -sinx
        rotXYZ[0, :, 2, 1] = sinx
        rotXYZ[0, :, 2, 2] = cosx
        rotXYZ[1, :, 0, 0] = cosy
        rotXYZ[1, :, 0, 2] = siny
        rotXYZ[1, :, 2, 0] = -siny
        rotXYZ[1, :, 2, 2] = cosy
        rotXYZ[2, :, 0, 0] = cosz
        rotXYZ[2, :, 0, 1] = -sinz
        rotXYZ[2, :, 1, 0] = sinz
        rotXYZ[2, :, 1, 1] = cosz

        rotation = rotXYZ[2].bmm(rotXYZ[1]).bmm(rotXYZ[0])

        return rotation.permute(0, 2, 1)
    
    def rigid_transform(self, vs, rot, trans):

        vs_r = torch.matmul(vs, rot)
        vs_t = vs_r + trans.view(-1, 1, 3)

        return vs_t

    def render_mesh(self, vs, tri, face_texture, angles=None, trans=[0., 0., 0.]):
        if angles is not None:
            rotation = self.compute_rotation_matrix(angles)
            vs = self.rigid_transform(vs, rotation, torch.tensor(trans).to(self.device)) # 000007
        mesh = Meshes(verts=vs, faces=tri.unsqueeze(0), textures=face_texture)
        render_img = self.render(mesh)
        return render_img

    
    def _get_renderer(self, device):
        R, T = look_at_view_transform(eye=torch.tensor([self.args.eye], device=self.device).float()) # YOU MIGHT NEED TO CHANGE VIEW POINT TO GET BETTER RESULTS
        cameras = PerspectiveCameras(device=device, R=R, T=T, image_size=((self.img_size, self.img_size),), focal_length=(1000), principal_point=((self.img_size//2, self.img_size//2),))
        self.camera = cameras
        lights = PointLights(device=device, location=[[0.0, 0.0, 1e5]],
                             ambient_color=[[.5, .5, .5]],
                             specular_color=[[0., 0., 0.]], diffuse_color=[[.5, .5, .5]]
                             )

        raster_settings = RasterizationSettings(
            image_size=self.img_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        blend_params = blending.BlendParams(background_color=[0, 0, 0])

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device,
                cameras=cameras,
                lights=lights,
                blend_params=blend_params
            )
        )
        return renderer
    
    def load_bfm_mesh(self, coef_path):
        model_dict = loadmat('BFM/BFM09_model_info.mat')
        recon_model = BFM09ReconModel(model_dict, batch_size=1, device='cuda:0', img_size=256)
        coeff = np.load(coef_path)
        coeff[80:144] = np.zeros(64)
        coeff[224:227] = np.zeros(3)
        coeff[254:] = np.array((0, 0, 2))
        coeff = torch.tensor(coeff[None, :], device='cuda:0', dtype=torch.float32)
        pred_dict = recon_model(coeff, render=True)
        xyz_in_world = pred_dict['vs']
        xyz_in_world = xyz_in_world.squeeze(0)
        tri = pred_dict['tri']
        face_texture = TexturesVertex(pred_dict['color'])
        return xyz_in_world, tri, face_texture, pred_dict['color']
    
    def load_mesh(self, source_path):
        print('==> loading data')
        if '.obj' in source_path:
            meshes = load_objs_as_meshes([source_path], device=self.device, load_textures=True)
            xyz_in_world = meshes.verts_packed()
            tri = meshes.faces_packed()
            face_texture = meshes.textures
            if face_texture is None:
                face_texture = TexturesVertex(verts_features=torch.ones_like(xyz_in_world).unsqueeze(0))
            xyz_in_world = xyz_in_world.cuda()
            face_color = None
        elif '.npy' in source_path:
            self.avatar_name = source_path.split('/')[-1].split('.')[0].split('_')[0]
            xyz_in_world, tri, face_texture, face_color = self.load_bfm_mesh(source_path)
        Translation = -xyz_in_world.mean(0) + torch.tensor(self.args.translation, device=self.device) # YOU MIGHT NEED TO CHANGE VIEW POINT TO GET BETTER RESULTS
        xyz_in_world = self.rigid_transform(xyz_in_world.unsqueeze(0), torch.eye(3).to(self.device), Translation).cuda()

        mesh = Meshes(verts=xyz_in_world, faces=tri.unsqueeze(0), textures=face_texture)
        color_img = self.render(mesh)
        color_img = color_img[0, ..., :3]
        color_img = torch.clamp(color_img, 0, 255)
        color_img = (color_img - color_img.min()) / (color_img.max() - color_img.min()) * 255
        color_img = color_img[..., [2, 1, 0]]
        self.color_source = color_img.cpu().numpy().astype(np.uint8)
        depth_img = self.render.rasterizer(mesh).zbuf[0, ..., 0:]
        self.depth_source = depth_img.squeeze(-1).cpu().numpy()
        depth_mi, depth_ma = depth_img[depth_img > 0].min(), depth_img.max()
        self.img_source = norm_image(color_img.cpu(), depth_img.cpu().squeeze(-1), save_path=self.save_path) # [1, 4, 256, 256]
        self.img_source = self.img_source.to(self.device)
        pix_to_face = self.render.rasterizer(mesh).pix_to_face
        return xyz_in_world, tri, face_texture, face_color, pix_to_face, depth_mi.item(), depth_ma.item()
    
    def run(self):
        xyz_in_world, tri, face_texture, face_color, pix_to_face, depth_mi, depth_ma = self.load_mesh(self.source_path)
        print('==> loading data done')
        print(depth_mi, depth_ma)
        if self.args.adjust_camera:
            return
        avatar_name = self.avatar_name
        bone_points_xyz_in_screen, bone_points_type = get_keypoints_as_bone(self.color_source, self.depth_source)
        print(bone_points_xyz_in_screen.shape)
        lambda_lap = args.lambda_lap
        if not os.path.exists('weights/'+avatar_name+'_weights.npy') or self.calc_weights:
            if not os.path.exists('weights'):
                os.makedirs('weights')
            weights, index, weights_lap, index_lap = calculate_weights_with_geodist(xyz_in_world, tri, pix_to_face, bone_points_xyz_in_screen, k=self.weights_k)
            np.save('weights/'+avatar_name+'_weights.npy', weights.cpu().numpy())
            np.save('weights/'+avatar_name+'_index.npy', index)
            np.save('weights/'+avatar_name+'_weights_lap.npy', weights_lap.cpu().numpy())
            np.save('weights/'+avatar_name+'_index_lap.npy', index_lap)
        else:
            weights = torch.tensor(np.load('weights/'+avatar_name+'_weights.npy'), device='cuda:0', dtype=torch.float32)
            index = np.load('weights/'+avatar_name+'_index.npy')
            weights_lap = torch.tensor(np.load('weights/'+avatar_name+'_weights_lap.npy'), device='cuda:0', dtype=torch.float32)
            index_lap = np.load('weights/'+avatar_name+'_index_lap.npy')
            # print(weights.shape, index.shape)
        angles = torch.tensor([[ 0.19897363, -0.32896024, -0.03197825]], device=self.device, dtype=torch.float32)
        with torch.no_grad():
            if args.find_best:
                fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True, device='cuda')
                best_frame_num = find_best_frame(self.img_source[:, :3, :, :], self.vid_target, fa)
                print('best_frame_num: ', best_frame_num)
                h_start = self.gen.enc.enc_motion(self.vid_target[:, best_frame_num, :, :, :].cuda())
            elif args.best_frame > 0:
                h_start = self.gen.enc.enc_motion(self.vid_target[:, args.best_frame-args.offset, :, :, :].cuda())
            else:
                h_start = self.gen.enc.enc_motion(self.driving0.cuda())
            vid_recon_results = self.vid_target.permute(0, 2, 1, 3, 4) # [B, C, T, H, W]
            vid_recon_results = vid_recon_results[:, :3, :, :, :]
            vid_source = torch.zeros_like(vid_recon_results) - 1.
            
            N = self.vid_target.shape[1]
            vid_target_recon = []
            vid_target_recon_angles = []
            vid_rgb_recon = []
            source_img = self.color_source / 255. * 2 - 1 
            source_img = source_img[..., [2, 1, 0]]
            source_img = torch.tensor(source_img, dtype=torch.float32).cpu().permute(2, 0, 1).unsqueeze(0).unsqueeze(2) # [B, C, 1, H, W]
            vid_source = torch.cat([vid_source, source_img.repeat(1, 1, N, 1, 1), source_img.repeat(1, 1, N, 1, 1), source_img.repeat(1, 1, N, 1, 1)], dim=4)
            delta_xyz_in_world_list = []
            for j in tqdm(range(N)):
                if args.target_frame != -1 and j != args.target_frame:
                    continue
                img_target = self.vid_target[:, j, :, :, :].cuda()
                img_recon, res = self.gen(self.img_source, img_target, h_start=h_start, testing=True)
                depth_recon = img_recon[0, 3, :, :].cpu().numpy() # [H, W]
                rgb_recon = img_recon[:, :3, :, :] # [B, C, H, W]
                rgb_motion_flow = res['rgb_motion_flow'].cpu().numpy() # [H, W, 2]
                vid_rgb_recon.append(rgb_recon.unsqueeze(2)) # [B, C, T, H, W]
                
                # directly warp the 3D mesh and visualize it
                depth_recon = depth_recon.clip(-1, 1)
                depth_recon = ((depth_recon - depth_recon.min()) / (depth_recon.max() - depth_recon.min()) * 65535).astype(np.uint16)
                translation_in_world, untrack_points = source_to_driving(rgb_motion_flow, depth_recon, bone_points_xyz_in_screen, bone_points_type, self.camera, depth_mi, depth_ma, vis=True)
                if len(untrack_points) > 0:
                    new_weights = weights.clone()
                    new_weights = delete_untrack_points_from_weights(new_weights, index, untrack_points)
                    if torch.isnan(new_weights).any() or torch.isinf(new_weights).any():
                        delta_xyz_in_world = 0.
                    else:
                        delta_xyz_in_world = torch.bmm(new_weights.unsqueeze(1), torch.tensor(translation_in_world[index], device='cuda:0', dtype=torch.float32)).squeeze(1)
                else:
                    delta_xyz_in_world = torch.bmm(weights.unsqueeze(1), torch.tensor(translation_in_world[index], device='cuda:0', dtype=torch.float32)).squeeze(1)
                # laplacian smooth
                for k in range(args.lap_times):
                    delta_xyz_in_world = torch.bmm(weights_lap.unsqueeze(1), delta_xyz_in_world[index_lap]).squeeze(1) * lambda_lap + delta_xyz_in_world * (1 - lambda_lap)
                delta_xyz_in_world_list.append(delta_xyz_in_world)
            for j in tqdm(range(N)):
                if args.target_frame != -1 and j != args.target_frame:
                    continue
                if args.target_frame == -1:
                    delta_xyz_in_world = delta_xyz_in_world_list[j]
                new_xyz_in_world = xyz_in_world + delta_xyz_in_world
                rendered_img = self.render_mesh(new_xyz_in_world, tri, face_texture, None)
                rendered_img = rendered_img[:, :, :, :3]
                rendered_img = torch.clamp(rendered_img, 0., 255.)
                rendered_img = (rendered_img - rendered_img.min()) / (rendered_img.max() - rendered_img.min()) * 2. - 1. # [-1, 1]
                vid_target_recon.append(rendered_img.permute(0, 3, 1, 2).unsqueeze(2)) # [B, C, T, H, W]

                rendered_img2 = self.render_mesh(new_xyz_in_world, tri, face_texture, angles, trans=[-0.1, 0., 0])
                rendered_img2 = rendered_img2[:, :, :, :3]
                rendered_img2 = torch.clamp(rendered_img2, 0., 255.)
                rendered_img2 = (rendered_img2 - rendered_img2.min()) / (rendered_img2.max() - rendered_img2.min()) * 2. - 1. # [-1, 1]
                vid_target_recon_angles.append(rendered_img2.permute(0, 3, 1, 2).unsqueeze(2)) # [B, C, T, H, W]
            if args.target_frame != -1:
                cv2.imwrite(os.path.join(self.save_path, str(args.target_frame)+'_depth_img.png'), depth_recon)
                save_img_from_tensor(self.img_source[0, :3, :, :], self.save_path, 'source_img.png')
                save_img_from_tensor(self.vid_target[0, args.target_frame, :3, :, :], self.save_path, str(args.target_frame)+'_driving_img.png')
                save_img_from_tensor(vid_target_recon[0][0, :3, 0, :, :], self.save_path, str(args.target_frame)+'_recon_img.png')
                save_img_from_tensor(vid_target_recon_angles[0][0, :3, 0, :, :], self.save_path, str(args.target_frame)+'_recon_img_angles.png')
                save_img_from_tensor(vid_rgb_recon[0][0, :3, 0, :, :], self.save_path, str(args.target_frame)+'_recon_img_rgb.png')
            else:
                vid_target_recon = torch.cat(vid_target_recon, dim=2).cpu()
                vid_target_recon_angles = torch.cat(vid_target_recon_angles, dim=2).cpu()
                vid_rgb_recon = torch.cat(vid_rgb_recon, dim=2).cpu()
                vid_recon_results = torch.cat([vid_recon_results, vid_rgb_recon, vid_target_recon, vid_target_recon_angles], dim=4)
            
                save_video(vid_source, vid_recon_results, self.save_path, self.save_name, self.fps, use_depth=False)
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_folder", type=str, default='./results')
    parser.add_argument("--checkpoint", type=str, default='./saved_models/VFA.pt')

    parser.add_argument("--length", type=int, default=1000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--driving_path", type=str, default='')
    parser.add_argument("--source_path", type=str, default='./dataset/demo_source/woman.npy')
    
    # change camera position, view render result, no warping
    parser.add_argument("--adjust_camera", action='store_true', default=False)
    parser.add_argument("--translation", type=list, default=[0., 0., 0.])
    parser.add_argument("--eye", type=list, default=[0., 0., 8.])

    # calculate controlling weights
    parser.add_argument("--calc_weights", action='store_true', default=False)
    parser.add_argument("--weights_k", type=int, default=10)

    # find best frame as driving initial frame
    parser.add_argument("--find_best", action='store_true', default=False)
    parser.add_argument("--best_frame", type=int, default=0)

    # generate result with one target frame
    parser.add_argument("--target_frame", type=int, default=-1)

    # laplacian smooth config
    parser.add_argument("--lambda_lap", type=float, default=0.6)
    parser.add_argument("--lap_times", type=int, default=5)
    args = parser.parse_args()

    demo = Demo(args)
    demo.run()