import random
from PIL import Image
from torch.utils.data import Dataset
import glob
import os
from augmentations import AugmentationTransform
from PIL import ImageFile
from imageio import mimread
from random import choice, randint
import numpy as np
import cv2
import ffmpeg

ImageFile.LOAD_TRUNCATED_IMAGES = True

def color_norm(color, depth=None):
    if depth is not None:
        content = (depth > 0)
        bkg = (depth == 0)
        m, s = np.mean(color[content], axis=0), np.std(color[content], axis=0)
        color[content] = (color[content] - m) / (s*2) 
        color[content] = np.clip(color[content], -1., 1.) # [-1, 1]
        color[bkg] = -1.
        depth = np.array(depth, dtype='float32') / np.max(depth) # [0, 1]
        depth = depth * 2. - 1. # [-1, 1]
        color = np.concatenate([color, depth[:, :, None]], axis=2)
    else:
        color = np.array(color, dtype='float32') / 255.
        color = color * 2. - 1. # [-1, 1]
    return color

class AvatarDataset(Dataset):
    def __init__(self, dataset_path, in_channels=3, resample=1, is_train=True) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.in_channels = in_channels
        self.resample = resample
        self.is_train = is_train
        self.source = self.read_photo('static.png')
        self.size_of_total = (len(os.listdir(os.path.join(self.dataset_path, 'color'))) - 1) // self.resample
        self.size_of_trainset = self.size_of_total * 8 // 10
        # print(self.dataset_path, self.size_of_total, self.size_of_trainset)
    
    def read_photo(self, photo_name):
        color = np.array(cv2.imread(os.path.join(self.dataset_path, 'color', photo_name)), dtype='float32')[:, :, [2, 1, 0]]
        if self.in_channels == 4:
            depth = cv2.imread(os.path.join(self.dataset_path, 'depth', photo_name), -1)
            photo = color_norm(color, depth)
        else:
            photo = color_norm(color)
        photo = photo.transpose((2, 0, 1))
        return photo

    def __len__(self):
        if self.is_train:
            return self.size_of_trainset
        else:
            return self.size_of_total - self.size_of_trainset
    
    def __getitem__(self, index):
        if self.is_train:
            driving = self.read_photo(str(index * self.resample)+'.png')
        else:
            driving = self.read_photo(str((self.size_of_trainset+index) * self.resample)+'.png')
        return self.source, driving

class VideoDataset(Dataset):
    def __init__(self, dataset_path, in_channels=3, is_train=True) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.in_channels = in_channels
        self.is_train = is_train
        self.video_list = os.listdir(os.path.join(self.dataset_path, 'color'))
        self.size_of_total = (len(self.video_list)) 
        self.size_of_trainset = self.size_of_total * 8 // 10
    
    def load_data_from_video(self, video_name):
        depth_video_path = os.path.join(self.dataset_path, 'depth', video_name[:-4]+'.nut')
        color_video_path = os.path.join(self.dataset_path, 'color', video_name)
        # print(color_video_path, depth_video_path)
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
        return video_color, video_depth
    
    def load_data(self, video_name, index=None):
        video_color, video_depth = self.load_data_from_video(video_name)
        if index is None:
            index = np.random.choice(len(video_color), size=1)
        color = video_color[index].squeeze(0).astype('float32')
        if self.in_channels == 4:
            depth = video_depth[index].squeeze(0)
            # print(index, color.shape, depth.shape, video_color.shape, video_depth.shape)
            color = color_norm(color, depth)
        else:
            color = color_norm(color)
        color = color.transpose((2, 0, 1))
        return color
    
    def __len__(self):
        if self.is_train:
            return self.size_of_trainset
        else:
            return self.size_of_total - self.size_of_trainset
    
    def __getitem__(self, index):
        if self.is_train:
            video_name = self.video_list[index]
        elif self.size_of_trainset+index < self.size_of_total:
            video_name = self.video_list[self.size_of_trainset+index]
        else:
            video_name = self.video_list[-1]
        source = self.load_data(video_name, index=[0])
        driving = self.load_data(video_name)
        return source, driving

class MMFace4D(Dataset):
    def __init__(self, split, transform=None, augmentation=False, in_channels=3) -> None:
        super().__init__()
        self.split = split
        self.augmentation = augmentation

        if self.augmentation:
            self.aug = AugmentationTransform(False, True, True)

        self.transform = transform
        self.in_channels = in_channels
        self.real_rgbd_path = os.listdir('./datasets/MMFace4D')
        if self.split == 'test':
            self.real_rgbd_path = os.listdir('./datasets/MMFace4D_test')
        self.img_path = [ os.path.join('./datasets', 'kinect_data')]
        self.avatar_dataset = []
        self.end_of_dataset = []
        sum = 0
        if self.split != 'test':
            for k, v in enumerate(self.img_path):
                if k == 0 and self.split == 'train':
                    self.avatar_dataset.append(AvatarDataset(dataset_path=v, in_channels=in_channels, resample=4))
                else:
                    self.avatar_dataset.append(AvatarDataset(dataset_path=v, in_channels=in_channels, resample=1, is_train=(self.split == 'train')))
                sum += len(self.avatar_dataset[k])
                self.end_of_dataset.append(sum)
        
            for v in self.real_rgbd_path:
                if not os.path.exists(os.path.join('./datasets/MMFace4D', v, 'color')):
                    continue
                self.avatar_dataset.append(VideoDataset(dataset_path=os.path.join('./datasets/MMFace4D', v), in_channels=in_channels, is_train=(self.split == 'train')))
                sum += len(self.avatar_dataset[-1])
                self.end_of_dataset.append(sum)
        else:
            for v in self.real_rgbd_path:
                if not os.path.exists(os.path.join('./datasets/MMFace4D_test', v, 'color')):
                    continue
                self.avatar_dataset.append(VideoDataset(dataset_path=os.path.join('./datasets/MMFace4D_test', v), in_channels=in_channels, is_train=(self.split == 'train')))
                sum += len(self.avatar_dataset[-1])
                self.end_of_dataset.append(sum)

    
    def __len__(self):
        if self.split == 'train':
            return sum([len(v) for v in self.avatar_dataset])
        elif self.split == 'test':
            return sum([len(v) for v in self.avatar_dataset])
        elif self.split == 'raw':
            return len(self.avatar_dataset[0])
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        if self.split == 'raw':
            return self.avatar_dataset[0][idx]
        else:
            for k in range(len(self.avatar_dataset)):
                if idx < self.end_of_dataset[k]:
                    if k == 0:
                        return self.avatar_dataset[k][idx]
                    else:
                        return self.avatar_dataset[k][idx-self.end_of_dataset[k-1]]
    


class Vox256(Dataset):
    def __init__(self, split, transform=None, augmentation=False):

        if split == 'train':
            self.ds_path = './datasets/vox/train'
        elif split == 'test':
            self.ds_path = './datasets/vox/test'
        else:
            raise NotImplementedError

        self.videos = os.listdir(self.ds_path)
        self.augmentation = augmentation

        if self.augmentation:
            self.aug = AugmentationTransform(False, True, True)
        else:
            self.aug = None

        self.transform = transform

    def __getitem__(self, idx):
        video_path = os.path.join(self.ds_path, self.videos[idx])
        frames_paths = sorted(glob.glob(video_path + '/*.png'))
        nframes = len(frames_paths)

        items = random.sample(list(range(nframes)), 2)

        img_source = Image.open(frames_paths[items[0]]).convert('RGB')
        img_target = Image.open(frames_paths[items[1]]).convert('RGB')

        if self.augmentation:
            img_source, img_target = self.aug(img_source, img_target)

        if self.transform is not None:
            img_source = self.transform(img_source)
            img_target = self.transform(img_target)

            return img_source, img_target

    def __len__(self):
        return len(self.videos)


class Vox256_vox2german(Dataset):
    def __init__(self, transform=None):
        self.source_root = './datasets/german/'
        self.driving_root = './datasets/vox/test/'

        self.anno = pd.read_csv('pairs_annotations/german_vox.csv')

        self.source_imgs = os.listdir(self.source_root)
        self.transform = transform

    def __getitem__(self, idx):
        source_name = str('%03d' % self.anno['source'][idx])
        driving_name = self.anno['driving'][idx]

        source_vid_path = self.source_root + source_name
        driving_vid_path = self.driving_root + driving_name

        source_frame_path = sorted(glob.glob(source_vid_path + '/*.png'))[0]
        driving_frames_path = sorted(glob.glob(driving_vid_path + '/*.png'))[:100]

        source_img = self.transform(Image.open(source_frame_path).convert('RGB'))
        driving_vid = [self.transform(Image.open(p).convert('RGB')) for p in driving_frames_path]

        return source_img, driving_vid, source_name, driving_name

    def __len__(self):
        return len(self.source_imgs)


class Taichi(Dataset):
    def __init__(self, split, transform=None, augmentation=False):

        if split == 'train':
            self.ds_path = './datasets/taichi/train/'
        else:
            self.ds_path = './datasets/taichi/test/'

        self.videos = os.listdir(self.ds_path)
        self.augmentation = augmentation

        if self.augmentation:
            self.aug = AugmentationTransform(True, True, True)
        else:
            self.aug = None

        self.transform = transform

    def __getitem__(self, idx):

        video_path = self.ds_path + self.videos[idx]
        frames_paths = sorted(glob.glob(video_path + '/*.png'))
        nframes = len(frames_paths)

        items = random.sample(list(range(nframes)), 2)

        img_source = Image.open(frames_paths[items[0]]).convert('RGB')
        img_target = Image.open(frames_paths[items[1]]).convert('RGB')

        if self.augmentation:
            img_source, img_target = self.aug(img_source, img_target)

        if self.transform is not None:
            img_source = self.transform(img_source)
            img_target = self.transform(img_target)

        return img_source, img_target

    def __len__(self):
        return len(self.videos)


class Taichi_eval(Dataset):
    def __init__(self, transform=None):
        self.ds_path = './datasets/taichi/test/'
        self.videos = os.listdir(self.ds_path)
        self.transform = transform

    def __getitem__(self, idx):
        vid_name = self.videos[idx]
        video_path = os.path.join(self.ds_path, vid_name)

        frames_paths = sorted(glob.glob(video_path + '/*.png'))
        vid_target = [self.transform(Image.open(p).convert('RGB')) for p in frames_paths]

        return vid_name, vid_target

    def __len__(self):
        return len(self.videos)


class TED(Dataset):
    def __init__(self, split, transform=None, augmentation=False):

        if split == 'train':
            self.ds_path = './datasets/ted/train/'
        else:
            self.ds_path = './datasets/ted/test/'

        self.videos = os.listdir(self.ds_path)
        self.augmentation = augmentation

        if self.augmentation:
            self.aug = AugmentationTransform(False, True, True)
        else:
            self.aug = None

        self.transform = transform

    def __getitem__(self, idx):
        video_path = os.path.join(self.ds_path, self.videos[idx])
        frames_paths = sorted(glob.glob(video_path + '/*.png'))
        nframes = len(frames_paths)

        items = random.sample(list(range(nframes)), 2)

        img_source = Image.open(frames_paths[items[0]]).convert('RGB')
        img_target = Image.open(frames_paths[items[1]]).convert('RGB')

        if self.augmentation:
            img_source, img_target = self.aug(img_source, img_target)

        if self.transform is not None:
            img_source = self.transform(img_source)
            img_target = self.transform(img_target)

        return img_source, img_target

    def __len__(self):
        return len(self.videos)


class TED_eval(Dataset):
    def __init__(self, transform=None):
        self.ds_path = './datasets/ted/test/'
        self.videos = os.listdir(self.ds_path)
        self.transform = transform

    def __getitem__(self, idx):
        vid_name = self.videos[idx]
        video_path = os.path.join(self.ds_path, vid_name)

        frames_paths = sorted(glob.glob(video_path + '/*.png'))
        vid_target = [self.transform(Image.open(p).convert('RGB')) for p in frames_paths]

        return vid_name, vid_target

    def __len__(self):
        return len(self.videos)

if __name__ == '__main__':
    test = Vox256_eval()
    # print(len(test))
    # print(len(test[0]))
    # _, vid = test[0]
    # print(vid[0])
    # print(vid[0].shape)