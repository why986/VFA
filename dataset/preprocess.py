import mediapipe as mp
import cv2
import numpy as np
import os, time
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
from PIL import Image
import ffmpeg
import argparse

class Worker():
    def __init__(self, shape=(256, 256), save_dir='./'):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.shape = shape
        self.output_color_dir = os.path.join(save_dir, 'color')
        self.output_depth_dir = os.path.join(save_dir, 'depth')
        os.makedirs(self.output_color_dir, exist_ok=True)
        os.makedirs(self.output_depth_dir, exist_ok=True)
    
    def detect_face(self, face_mesh, color_image):
        result = face_mesh.process(color_image)
        if result is not None and result.multi_face_landmarks is not None and result.multi_face_landmarks[0] is not None:
            detect_res = np.array([[res.x, res.y, res.z] for res in result.multi_face_landmarks[0].landmark])
            x, y, _ = np.min(detect_res, axis=0)
            _x, _y, _ = np.max(detect_res, axis=0)
            x, y, _x, _y = int(x*color_image.shape[1]), int(y*color_image.shape[0]), int(_x*color_image.shape[1]), int(_y*color_image.shape[0])
            w, h = _x - x, _y - y
            if w < h:
                h = h + 15
                x = x - (h-w)//2
                w = h
            return x, y, w, h
        else:
            return None, None, None, None

    def normalize_image(self, color_image, depth_image):
        content = (depth_image > 0)
        # remove outliers (2 sigma)
        m, s = np.mean(depth_image[content], axis=0), np.std(depth_image[content], axis=0)
        depth_image[content] = np.clip(depth_image[content], m-s*2, m+s*2)
        # normalize depth image
        mi, ma = np.min(depth_image[content]), np.max(depth_image[content])
        depth_image[content] = (depth_image[content] - mi) / (ma - mi) * 65534 + 1 # [0, 1] -> [0, 65534] -> [1, 65535]
        depth_image = (depth_image).astype(np.uint16)
        depth_image = depth_image[:, :, np.newaxis]

        return color_image, depth_image
    
    def crop_image(self, color_image, depth_image, x, y, w, h):
        # crop image
        color_image = color_image[y:y+h, x:x+w] 
        depth_image = depth_image[y:y+h, x:x+w]
        # resize image
        color_image = cv2.resize(np.array(color_image, dtype=np.float32), self.shape, interpolation = cv2.INTER_NEAREST) 
        depth_image = cv2.resize(np.array(depth_image, dtype=np.float32), self.shape, interpolation = cv2.INTER_NEAREST)
        color_image = np.array(color_image, dtype=np.uint8)
        return color_image, depth_image

    def save_image(self, color_image, depth_image, index):
        color_image = Image.fromarray(np.uint8(color_image))
        color_image.save(os.path.join(self.output_color_dir, 'color_'+str(index)+'.png'))
        depth_image = np.float32(depth_image) / 256.0
        depth_image = np.clip(depth_image, 0, 255)
        depth_image = np.repeat(depth_image, 3, axis=2)
        depth_image = Image.fromarray(np.uint8(depth_image))
        depth_image.save(os.path.join(self.output_depth_dir, 'depth_'+str(index)+'.png'))
    
    def load_data_from_rgbd_video(self, video_name):
        if not os.path.exists(video_name):
            print('Video does not exist.')
            return None, None
        probe = ffmpeg.probe(video_name)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        out_byte, _ = (
            ffmpeg
                .input(video_name)
                .output('pipe:', format='rawvideo', pix_fmt='rgb24', loglevel="quiet")
                .run(capture_stdout=True)
            )

        video = np.frombuffer(out_byte, np.uint8).reshape([-1, height, width, 3])
        video_depth = video[:, :, :width//2, 0]
        video_depth = np.array(video_depth, dtype=np.float32) / 256.0
        video_color = video[:, :, width//2:, :]
        return video_color, video_depth

    def load_data_from_dir(self, path):
        video_color, video_depth = [], []
        color_path = os.path.join(path, 'rgb')
        depth_path = os.path.join(path, 'depth')
        cnt = len(os.listdir(color_path))
        for index in range(cnt):
            color = cv2.imread(os.path.join(color_path, str(index)+'.jpg'))
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            depth = cv2.imread(os.path.join(depth_path, str(index)+'.exr'), cv2.IMREAD_ANYDEPTH)
            video_color.append(color)
            video_depth.append(depth)
        video_color = np.array(video_color)
        video_depth = np.array(video_depth)
        return video_color, video_depth
    
    def loop_video(self, video_color, video_depth):
        N = video_color.shape[0]
        process_tmp = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', framerate=30, s='{}x{}'.format(256, 256))
            .output('tmp.mp4', pix_fmt='yuv420p', vcodec='libx264', r=30, loglevel="quiet", crf=3)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
        with self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
            for index in range(N):
                x, y, w, h = self.detect_face(face_mesh, video_color[index].copy())
                if x is not None:
                    break
            for index in range(N):
                color_image, depth_image = video_color[index].copy(), video_depth[index].copy()
                # remove background
                bkg = np.isnan(depth_image) | (depth_image <= 0) | (depth_image >= 0.5) # assume foreground is closer than 0.5m
                color_image[bkg] *= 0
                depth_image[bkg] *= 0

                # x, y, w, h = self.detect_face(face_mesh, color_image)
                if x is not None:
                    # crop image
                    color_image, depth_image = self.crop_image(color_image, depth_image, x, y, w, h)

                    # remove outliers and normalize depth image
                    color_image, depth_image = self.normalize_image(color_image, depth_image)

                    self.process_color.stdin.write(color_image.tobytes())
                    self.process_depth.stdin.write(depth_image.tobytes())
                    tmp_image = np.asarray(depth_image, dtype=np.float32) / 256.0
                    tmp_image = np.clip(tmp_image, 0, 255)
                    tmp_image = np.repeat(tmp_image, 3, axis=2)
                    tmp_image = np.asarray(tmp_image, dtype=np.uint8)
                    process_tmp.stdin.write(tmp_image.tobytes())
                    # if index == 0:
                    #     self.save_image(color_image, depth_image, index)
                    
        self.process_color.stdin.close()
        self.process_color.wait()
        self.process_depth.stdin.close()
        self.process_depth.wait()
        process_tmp.stdin.close()
        process_tmp.wait()
    
    def build_video_writer(self, video_name, video_fps=30):
        color_video_path = os.path.join(self.output_color_dir, video_name+'.mp4')
        depth_video_path = os.path.join(self.output_depth_dir, video_name+'.nut')
        self.process_color = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', framerate=video_fps, s='{}x{}'.format(256, 256))
            .output(color_video_path, pix_fmt='yuv420p', vcodec='libx264', r=video_fps, loglevel="quiet", crf=3)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

        self.process_depth = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='gray16le', framerate=video_fps, s='{}x{}'.format(256, 256))
            .output(depth_video_path, pix_fmt='gray16le', vcodec='ffv1', r=video_fps, loglevel="quiet")
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_name', type=str, default='test')
    parser.add_argument('--output_name', type=str, default='output')
    args = parser.parse_args()
    if os.path.isdir(args.video_name):
        worker = Worker(save_dir=args.video_name+'_output')
        video_color, video_depth = worker.load_data_from_dir(args.video_name)
    elif args.video_name.endswith('.mp4'):
        worker = Worker(save_dir=args.video_name[:-4])
        video_color, video_depth = worker.load_data_from_rgbd_video(args.video_name)
    worker.build_video_writer(args.output_name)
    worker.loop_video(video_color, video_depth)

