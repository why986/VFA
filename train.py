import argparse
import os
import torch
from torch.utils import data
from dataset import MMFace4D, Vox256, Taichi, TED
import torchvision
import torchvision.transforms as transforms
from trainer import Trainer
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def data_sampler(dataset, shuffle):
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def display_img(idx, img, name, writer):
    img = img.clamp(-1, 1)
    img = ((img - img.min()) / (img.max() - img.min())).data
    
    writer.add_images(tag='%s' % (name), global_step=idx, img_tensor=img[:, :3, :, :])
    if img.shape[1] > 3:
        img_depth = img[:, 3, :, :]
        img_depth = img_depth.unsqueeze(1).repeat((1, 3, 1, 1))
        writer.add_images(tag='%s' % (name+'_depth'), global_step=idx, img_tensor=img_depth)


def write_loss(i, losses, writer):
    if 'vgg_loss' in losses:
        writer.add_scalar('vgg_loss', losses['vgg_loss'].item(), i)
    if 'l1_loss' in losses:
        writer.add_scalar('l1_loss', losses['l1_loss'].item(), i)
    if 'gradient_loss' in losses:
        writer.add_scalar('gradient_loss', losses['gradient_loss'].item(), i)
    if 'smooth_loss' in losses:
        writer.add_scalar('smooth_loss', losses['smooth_loss'].item(), i)
    if 'structure_preserve_loss' in losses:
        writer.add_scalar('structure_preserve_loss', losses['structure_preserve_loss'].item(), i)
    writer.flush()


def ddp_setup(args, rank, world_size):
    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = args.port

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def main(rank, world_size, args):
    # init distributed computing
    ddp_setup(args, rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda")

    # make logging folder
    log_path = os.path.join(args.exp_path, args.exp_name + '/log')
    checkpoint_path = os.path.join(args.exp_path, args.exp_name + '/checkpoint')

    os.makedirs(log_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    writer = SummaryWriter(log_path)

    print('==> preparing dataset')
    transform = torchvision.transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    )

    if args.dataset == 'ted':
        dataset = TED('train', transform, True)
        dataset_test = TED('test', transform)
    elif args.dataset == 'vox':
        dataset = Vox256('train', transform, False)
        dataset_test = Vox256('test', transform)
    elif args.dataset == 'taichi':
        dataset = Taichi('train', transform, True)
        dataset_test = Taichi('test', transform)
    elif args.dataset == 'MMFace4D':
        dataset = MMFace4D('train', augmentation=False, in_channels=args.in_channels)
        dataset_test = MMFace4D('test', in_channels=args.in_channels)
    elif args.dataset == 'raw':
        dataset = MMFace4D('raw', augmentation=False, in_channels=args.in_channels)
        dataset_test = MMFace4D('test', in_channels=args.in_channels)
    else:
        raise NotImplementedError

    loader = data.DataLoader(
        dataset,
        num_workers=8,
        batch_size=args.batch_size // world_size,
        sampler=data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True),
        pin_memory=True,
        drop_last=False,
    )

    loader_test = data.DataLoader(
        dataset_test,
        num_workers=8,
        batch_size=4,
        sampler=data.distributed.DistributedSampler(dataset_test, num_replicas=world_size, rank=rank, shuffle=False),
        pin_memory=True,
        drop_last=False,
    )

    loader = sample_data(loader)
    loader_test = sample_data(loader_test)

    print('==> initializing trainer')
    # Trainer
    trainer = Trainer(args, device, rank)

    # resume
    if args.resume_ckpt is not None:
        args.start_iter = trainer.resume(args.resume_ckpt)
        print('==> resume from iteration %d' % (args.start_iter))

    print('==> training')
    pbar = range(args.iter-args.start_iter)
    for idx in pbar:
        i = idx + args.start_iter

        # laoding data
        img_source, img_target = next(loader)
        img_source = img_source.to(rank, non_blocking=True)
        img_target = img_target.to(rank, non_blocking=True)

        # update generator
        losses, img_recon = trainer.gen_update(img_source, img_target, distilling=args.distilling)


        if rank == 0:
            # write to log
            write_loss(idx, losses, writer)

        # display
        if i % args.display_freq == 0 and rank == 0:
            loss_values = {key: val.detach().item() for key, val in losses.items()}
            print("[Iter %d/%d]"%(i, args.iter) + str(loss_values))

            if rank == 0:
                img_test_source, img_test_target = next(loader_test)
                img_test_source = img_test_source.to(rank, non_blocking=True)
                img_test_target = img_test_target.to(rank, non_blocking=True)

                img_recon, img_source_ref = trainer.sample(img_test_source, img_test_target, distilling=args.distilling)
                display_img(i, img_test_source, 'source', writer)
                display_img(i, img_test_target, 'target', writer)
                if isinstance(img_recon, dict):
                    display_img(i, img_recon['out_warp'], 'recon_warp', writer)
                    display_img(i, img_recon['out_inpaint'], 'recon_inpaint', writer)
                else:
                    display_img(i, img_recon, 'recon', writer)
                display_img(i, img_source_ref, 'source_ref', writer)
                writer.flush()
                print("==> Display finished")

        # save model
        if i % args.save_freq == 0 and rank == 0:
            trainer.save(i, checkpoint_path)
            print("==> Model saved")

    return


if __name__ == "__main__":
    # training params
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=800000)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--in_channels", type=int, default=4)
    parser.add_argument("--lambda_loss_l1", type=float, default=200.0)
    parser.add_argument("--lambda_loss_sm", type=float, default=200.0)
    parser.add_argument("--lambda_loss_gr", type=float, default=100.0)
    parser.add_argument("--lambda_loss_sp", type=float, default=50.0)
    parser.add_argument("--g_reg_every", type=int, default=4)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--lr_freq", type=int, default=5000)
    parser.add_argument("--display_freq", type=int, default=2000)
    parser.add_argument("--save_freq", type=int, default=1000)
    parser.add_argument("--latent_dim_style", type=int, default=512)
    parser.add_argument("--latent_dim_motion", type=int, default=20)
    parser.add_argument("--latent_dim_depth_motion", type=int, default=5)
    parser.add_argument("--dataset", type=str, default='MMFace4D')
    parser.add_argument("--exp_path", type=str, default='./saved_models/')
    parser.add_argument("--exp_name", type=str, default='v1')
    parser.add_argument("--addr", type=str, default='localhost')
    parser.add_argument("--port", type=str, default='12345')

    parser.add_argument("--distilling", action='store_true', default=False)
    opts = parser.parse_args()

    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2

    world_size = n_gpus
    print('==> training on %d gpus' % n_gpus)
    mp.spawn(main, args=(world_size, opts,), nprocs=world_size, join=True)
