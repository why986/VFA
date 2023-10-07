import torch
from networks.discriminator import Discriminator
from networks.generator import Generator
import torch.nn.functional as F
from torch import nn, optim
import os
from vgg19 import VGGLoss
from torch.nn.parallel import DistributedDataParallel as DDP


def requires_grad(net, flag=True):
    for p in net.parameters():
        p.requires_grad = flag
 
def update_v(state_dict, k, v):
    if state_dict[k].shape == v.shape:
        state_dict.update({k: v})
    elif state_dict[k].shape[0] == 4:
        state_dict[k][:3] = v
    elif state_dict[k].shape[1] == 4:
        state_dict[k][:, :3] = v
    elif v.shape[1] == 3:
        state_dict[k][:, :3] = v
    else:
        print(k, state_dict[k].shape, v.shape)


class Trainer(nn.Module):
    def __init__(self, args, device, rank):
        super(Trainer, self).__init__()

        self.args = args
        self.batch_size = args.batch_size

        self.gen = Generator(args.size, args.latent_dim_style, args.latent_dim_motion, args.channel_multiplier, args.in_channels, args.latent_dim_depth_motion, distilling=args.distilling).to(
            device)

        # distributed computing
        self.gen = DDP(self.gen, device_ids=[rank], find_unused_parameters=True)

        g_reg_ratio = 1 

        self.g_optim = optim.Adam(
            filter(lambda p: p.requires_grad, self.gen.parameters()),
            lr=args.lr * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio)
        )

        self.g_scheduler = torch.optim.lr_scheduler.StepLR(self.g_optim, step_size=args.lr_freq, gamma=0.2)


        self.criterion_vgg = VGGLoss().to(rank)

        self.lambda_loss_l1 = args.lambda_loss_l1
        self.lambda_loss_sm = args.lambda_loss_sm
        self.lambda_loss_gr = args.lambda_loss_gr
        self.lambda_loss_sp = args.lambda_loss_sp

        self.gradient_x_weight = torch.Tensor([[0., 0., 0.], [1., 0., -1.], [0., 0., 0.]]).view(1, 1, 3, 3).to(device)
        self.gradient_y_weight = torch.Tensor([[0., 1., 0.], [0., 0., 0.], [0., -1., 0.]]).view(1, 1, 3, 3).to(device)
        self.smooth_loss_weight = torch.Tensor([[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]]).view(1, 1, 3, 3).to(device)

        self.eps = 1e-6

    def g_nonsaturating_loss(self, fake_pred):
        return F.softplus(-fake_pred).mean()

    
    def calc_charbonnier_loss(self, X, Y):
        diff = X - Y
        error = torch.sqrt(diff * diff + self.eps)
        return torch.mean(error)
    
    def calc_depth_loss(self, img_recon, img_target):
        l1_loss = F.l1_loss(img_recon, img_target) * self.lambda_loss_l1

        x_grad_recon = F.conv2d(img_recon, weight=self.gradient_x_weight, padding=1)
        y_grad_recon = F.conv2d(img_recon, weight=self.gradient_y_weight, padding=1)
        x_grad_target = F.conv2d(img_target, weight=self.gradient_x_weight, padding=1)
        y_grad_target = F.conv2d(img_target, weight=self.gradient_y_weight, padding=1)

        # Reconstruction-based Pairwise Depth Dataset for Depth Image Enhancement Using CNN
        x_mask = (x_grad_target.abs() > 0.1).float()
        y_mask = (y_grad_target.abs() > 0.1).float()
        x_grad_recon *= x_mask
        y_grad_recon *= y_mask
        x_grad_target *= x_mask
        y_grad_target *= y_mask
        xl = self.calc_charbonnier_loss(F.max_pool2d(x_grad_recon.abs(), kernel_size=5, padding=2, stride=1),
                                        F.max_pool2d(x_grad_target.abs(), kernel_size=5, padding=2, stride=1))
        yl = self.calc_charbonnier_loss(F.max_pool2d(y_grad_recon.abs(), kernel_size=5, padding=2, stride=1),
                                        F.max_pool2d(y_grad_target.abs(), kernel_size=5, padding=2, stride=1))
        structure_preserve_loss =  (xl + yl) * self.lambda_loss_sp

        
        lap_recon = F.conv2d(img_recon, weight=self.smooth_loss_weight, padding=1) * (x_grad_target.abs() < 0.1).float() * (y_grad_target.abs() < 0.1).float()
        lap_target = F.conv2d(img_target, weight=self.smooth_loss_weight, padding=1) * (x_grad_target.abs() < 0.1).float() * (y_grad_target.abs() < 0.1).float()
        smooth_loss = self.calc_charbonnier_loss(lap_recon, lap_target) * self.lambda_loss_sm
        
        depth_loss = {"l1_loss": l1_loss,
                    # "gradient_loss": gradient_loss,
                    "smooth_loss": smooth_loss,
                    "structure_preserve_loss": structure_preserve_loss}
        return depth_loss

    def gen_update(self, img_source, img_target, distilling=False):
        self.gen.train()
        self.gen.zero_grad()

        if distilling:
            img_target_recon, student_recon_result = self.gen(img_source, img_target)
            student_img_target_recon = student_recon_result['out_inpaint']
            img_target_recon = img_target_recon[:, :3, :, :]
            vgg_loss = self.criterion_vgg(student_img_target_recon, img_target_recon).mean()
            l1_loss = F.l1_loss(student_img_target_recon, img_target_recon) * self.lambda_loss_l1

            student_img_target_warp = student_recon_result['out_warp']
            mask = student_recon_result['mask']
            vgg_loss_warp = self.criterion_vgg(student_img_target_warp*mask, img_target_recon*mask).mean()
            l1_loss_warp = F.l1_loss(student_img_target_warp*mask, img_target_recon*mask) * self.lambda_loss_l1

            losses = {'l1_loss': l1_loss, 'vgg_loss': vgg_loss, 'l1_loss_warp': l1_loss_warp, 'vgg_loss_warp': vgg_loss_warp}
            img_target_recon = student_img_target_recon
        else:
            img_target_recon = self.gen(img_source, img_target)

            if img_target.shape[1] > 3:
                vgg_loss = self.criterion_vgg(img_target_recon[:, :3, :, :], img_target[:, :3, :, :]).mean()
                losses = self.calc_depth_loss(img_target_recon[:, 3, :, :].unsqueeze(1), img_target[:, 3, :, :].unsqueeze(1))
                losses['vgg_loss'] = vgg_loss
            else:
                vgg_loss = self.criterion_vgg(img_target_recon, img_target).mean()
                l1_loss = F.l1_loss(img_target_recon, img_target) * self.lambda_loss_l1
                losses = {'l1_loss': l1_loss, 'vgg_loss': vgg_loss}

        g_loss = sum(losses.values())
        g_loss.backward()
        self.g_optim.step()
        self.g_scheduler.step()

        return losses, img_target_recon

    def sample(self, img_source, img_target, distilling=False):
        with torch.no_grad():
            self.gen.eval()

            if distilling:
                _, student_img_recon = self.gen(img_source, img_target)
                img_recon = student_img_recon
                img_source_ref, _ = self.gen(img_source, None)
                img_source_ref = img_source_ref[:, :3, :, :]
            else:
                img_recon = self.gen(img_source, img_target)
                img_source_ref = self.gen(img_source, None)

        return img_recon, img_source_ref
    

    def resume(self, resume_ckpt):
        print("load model:", resume_ckpt)
        ckpt = torch.load(resume_ckpt, map_location=torch.device('cpu'))
        ckpt_name = os.path.basename(resume_ckpt)
        start_iter = int(os.path.splitext(ckpt_name)[0])

        model_weights = self.gen.module.state_dict().copy()
        for k, v in ckpt["gen"].items():
            if k in model_weights:
                update_v(model_weights, k, v)
            else:
                print(k, v.shape)
        self.gen.module.load_state_dict(model_weights, strict=False)

        return start_iter

    def save(self, idx, checkpoint_path):
        torch.save(
            {
                "gen": self.gen.module.state_dict(),
                "g_optim": self.g_optim.state_dict(),
                "args": self.args
            },
            f"{checkpoint_path}/{str(idx).zfill(6)}.pt"
        )
