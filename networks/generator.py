from torch import nn
from .encoder import Encoder
from .styledecoder import Synthesis, StudentSynthesis


class Generator(nn.Module):
    def __init__(self, size, style_dim=512, motion_dim=20, channel_multiplier=1, in_channels=3, depth_motion_dim=5, blur_kernel=[1, 3, 3, 1], distilling=False):
        super(Generator, self).__init__()

        # encoder
        self.enc = Encoder(size, style_dim, motion_dim, in_channels)
        self.dec = Synthesis(size, style_dim, motion_dim, blur_kernel, channel_multiplier, in_channels, depth_motion_dim, distilling=distilling)

        self.distilling = distilling
        if distilling:
            for p in self.enc.parameters():
                p.requires_grad = False
            for p in self.dec.parameters():
                p.requires_grad = False
            self.student_dec = StudentSynthesis()

    def get_direction(self):
        return self.dec.direction(None)

    def synthesis(self, wa, alpha, feat):
        img = self.dec(wa, alpha, feat)

        return img

    def forward(self, img_source, img_drive, h_start=None, testing=False):
        wa, alpha, feats = self.enc(img_source, img_drive, h_start)
        if testing:
            img_recon, skip_flow, style, res = self.dec(wa, alpha, feats, testing=True)
            if self.distilling:
                student_img_recon = self.student_dec(img_source, skip_flow, style, testing=True)
                res.update(student_img_recon)
            return img_recon, res
        elif self.distilling:
            img_recon, skip_flow, style = self.dec(wa, alpha, feats, testing=False)
            student_img_recon = self.student_dec(img_source, skip_flow, style)
            return img_recon, student_img_recon
        else:
            img_recon = self.dec(wa, alpha, feats)
            return img_recon
