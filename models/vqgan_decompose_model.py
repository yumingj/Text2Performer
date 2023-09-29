import os
from collections import OrderedDict

import lpips
import torch
from torchvision.utils import save_image

from models.archs.vqgan_arch import (
    DecoderUpOthersDoubleIdentity, Discriminator,
    EncoderDecomposeBaseDownOthersDoubleIdentity, VectorQuantizer)
from models.base_model import BaseModel
from models.losses.vqgan_loss import (DiffAugment, adopt_weight,
                                      calculate_adaptive_weight, hinge_d_loss)
from utils.dist_util import master_only


class VQGANDecomposeModel(BaseModel):

    def __init__(self, opt):
        super().__init__(opt)

        self.encoder = self.model_to_device(
            EncoderDecomposeBaseDownOthersDoubleIdentity(
                ch=opt['ch'],
                num_res_blocks=opt['num_res_blocks'],
                attn_resolutions=opt['attn_resolutions'],
                ch_mult=opt['ch_mult'],
                other_ch_mult=opt['other_ch_mult'],
                in_channels=opt['in_channels'],
                resolution=opt['resolution'],
                z_channels=opt['z_channels'],
                double_z=opt['double_z'],
                dropout=opt['dropout']))
        self.decoder = self.model_to_device(
            DecoderUpOthersDoubleIdentity(
                in_channels=opt['in_channels'],
                resolution=opt['resolution'],
                z_channels=opt['z_channels'],
                ch=opt['ch'],
                out_ch=opt['out_ch'],
                num_res_blocks=opt['num_res_blocks'],
                attn_resolutions=opt['attn_resolutions'],
                ch_mult=opt['ch_mult'],
                other_ch_mult=opt['other_ch_mult'],
                dropout=opt['dropout'],
                resamp_with_conv=True,
                give_pre_end=False))
        self.quantize_identity = self.model_to_device(
            VectorQuantizer(opt['n_embed'], opt['embed_dim'], beta=0.25))
        self.quant_conv_identity = self.model_to_device(
            torch.nn.Conv2d(opt["z_channels"], opt['embed_dim'], 1))
        self.post_quant_conv_identity = self.model_to_device(
            torch.nn.Conv2d(opt['embed_dim'], opt["z_channels"], 1))

        self.quantize_others = self.model_to_device(
            VectorQuantizer(opt['n_embed'], opt['embed_dim'] // 2, beta=0.25))
        self.quant_conv_others = self.model_to_device(
            torch.nn.Conv2d(opt["z_channels"] // 2, opt['embed_dim'] // 2, 1))
        self.post_quant_conv_others = self.model_to_device(
            torch.nn.Conv2d(opt['embed_dim'] // 2, opt["z_channels"] // 2, 1))

        self.disc = self.model_to_device(
            Discriminator(
                opt['n_channels'], opt['ndf'], n_layers=opt['disc_layers']))
        self.perceptual = lpips.LPIPS(net="vgg").to(self.device)
        self.perceptual_weight = opt['perceptual_weight']
        self.disc_start_step = opt['disc_start_step']
        self.disc_weight_max = opt['disc_weight_max']
        self.diff_aug = opt['diff_aug']
        self.policy = "color,translation"

        self.disc.train()

        if self.opt['pretrained_models'] is not None:
            self.load_pretrained_network()

        self.init_training_settings()

    def init_training_settings(self):
        self.configure_optimizers()

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()) +
            list(self.quantize_identity.parameters()) +
            list(self.quant_conv_identity.parameters()) +
            list(self.post_quant_conv_identity.parameters()) +
            list(self.quantize_others.parameters()) +
            list(self.quant_conv_others.parameters()) +
            list(self.post_quant_conv_others.parameters()),
            lr=self.opt['lr'])

        self.disc_optimizer = torch.optim.Adam(
            self.disc.parameters(), lr=self.opt['lr'])

    @master_only
    def save_network(self, save_path):
        """Save networks.

        Args:
            net (nn.Module): Network to be saved.
            net_label (str): Network label.
            current_iter (int): Current iter number.
        """

        save_dict = {}
        save_dict['encoder'] = self.get_bare_model(self.encoder).state_dict()
        save_dict['decoder'] = self.get_bare_model(self.decoder).state_dict()
        save_dict['quantize_identity'] = self.get_bare_model(
            self.quantize_identity).state_dict()
        save_dict['quant_conv_identity'] = self.get_bare_model(
            self.quant_conv_identity).state_dict()
        save_dict['post_quant_conv_identity'] = self.get_bare_model(
            self.post_quant_conv_identity).state_dict()
        save_dict['quantize_others'] = self.get_bare_model(
            self.quantize_others).state_dict()
        save_dict['quant_conv_others'] = self.get_bare_model(
            self.quant_conv_others).state_dict()
        save_dict['post_quant_conv_others'] = self.get_bare_model(
            self.post_quant_conv_others).state_dict()
        save_dict['disc'] = self.get_bare_model(self.disc).state_dict()
        torch.save(save_dict, save_path)

    def load_pretrained_network(self):

        self.load_network(
            self.encoder, self.opt['pretrained_models'], param_key='encoder')
        self.load_network(
            self.decoder, self.opt['pretrained_models'], param_key='decoder')
        self.load_network(
            self.quantize_identity,
            self.opt['pretrained_models'],
            param_key='quantize_identity')
        self.load_network(
            self.quant_conv_identity,
            self.opt['pretrained_models'],
            param_key='quant_conv_identity')
        self.load_network(
            self.post_quant_conv_identity,
            self.opt['pretrained_models'],
            param_key='post_quant_conv_identity')
        self.load_network(
            self.quantize_others,
            self.opt['pretrained_models'],
            param_key='quantize_others')
        self.load_network(
            self.quant_conv_others,
            self.opt['pretrained_models'],
            param_key='quant_conv_others')
        self.load_network(
            self.post_quant_conv_others,
            self.opt['pretrained_models'],
            param_key='post_quant_conv_others')

    def optimize_parameters(self, data, current_iter):
        self.encoder.train()
        self.decoder.train()
        self.quantize_identity.train()
        self.quant_conv_identity.train()
        self.post_quant_conv_identity.train()
        self.quantize_others.train()
        self.quant_conv_others.train()
        self.post_quant_conv_others.train()

        loss, d_loss = self.training_step(data, current_iter)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if current_iter > self.disc_start_step:
            self.disc_optimizer.zero_grad()
            d_loss.backward()
            self.disc_optimizer.step()

    def feed_data(self, data):
        x_identity = data['identity_image']
        x_frame_aug = data['frame_img_aug']
        x_frame = data['frame_img']

        return x_identity.float().to(self.device), x_frame_aug.float().to(
            self.device), x_frame.float().to(self.device)

    def encode(self, x_identity, x_frame):
        h_identity, _ = self.encoder(x_identity)
        h_identity = self.quant_conv_identity(h_identity)
        quant_identity, emb_loss_identity, _ = self.quantize_identity(
            h_identity)

        _, h_frame = self.encoder(x_frame)
        h_frame = self.quant_conv_others(h_frame)
        quant_frame, emb_loss_frame, _ = self.quantize_others(h_frame)
        return [quant_identity,
                quant_frame], emb_loss_identity + emb_loss_frame

    def decode(self, quant_list):
        quant_identity = self.post_quant_conv_identity(quant_list[0])
        quant_frame = self.post_quant_conv_others(quant_list[1])
        dec = self.decoder(quant_identity, quant_frame)
        return dec

    def forward_step(self, x_identity, x_frame_aug):
        quant_list, diff = self.encode(x_identity, x_frame_aug)
        dec = self.decode(quant_list)
        return dec, diff

    def training_step(self, data, step):
        x_identity, x_frame_aug, x_frame = self.feed_data(data)
        xrec, codebook_loss = self.forward_step(x_identity, x_frame_aug)

        # get recon/perceptual loss
        recon_loss = torch.abs(x_frame.contiguous() - xrec.contiguous())
        p_loss = self.perceptual(x_frame.contiguous(), xrec.contiguous())
        nll_loss = recon_loss + self.perceptual_weight * p_loss
        nll_loss = torch.mean(nll_loss)

        # augment for input to discriminator
        if self.diff_aug:
            xrec = DiffAugment(xrec, policy=self.policy)

        # update generator
        logits_fake = self.disc(xrec)
        g_loss = -torch.mean(logits_fake)
        last_layer = self.get_bare_model(self.decoder).conv_out.weight
        d_weight = calculate_adaptive_weight(nll_loss, g_loss, last_layer,
                                             self.disc_weight_max)
        d_weight *= adopt_weight(1, step, self.disc_start_step)
        loss = nll_loss + d_weight * g_loss + codebook_loss

        loss_dict = OrderedDict()

        loss_dict["loss"] = loss
        loss_dict["l1"] = recon_loss.mean()
        loss_dict["perceptual"] = p_loss.mean()
        loss_dict["nll_loss"] = nll_loss
        loss_dict["g_loss"] = g_loss
        loss_dict["d_weight"] = d_weight
        loss_dict["codebook_loss"] = codebook_loss

        if step > self.disc_start_step:
            if self.diff_aug:
                logits_real = self.disc(
                    DiffAugment(
                        x_frame.contiguous().detach(), policy=self.policy))
            else:
                logits_real = self.disc(x_frame.contiguous().detach())
            logits_fake = self.disc(xrec.contiguous().detach(
            ))  # detach so that generator isn"t also updated
            d_loss, _, _ = hinge_d_loss(logits_real, logits_fake)
            loss_dict["d_loss"] = d_loss
        else:
            d_loss = None

        self.log_dict = self.reduce_loss_dict(loss_dict)

        return loss, d_loss

    @master_only
    def get_vis(self, x_identity, x_frame_aug, x_frame, xrec, save_dir,
                img_name):
        os.makedirs(save_dir, exist_ok=True)
        img_cat = torch.cat([x_identity, x_frame_aug, x_frame, xrec],
                            dim=3).detach()
        img_cat = ((img_cat + 1) / 2)
        img_cat = img_cat.clamp_(0, 1)
        save_image(img_cat, f'{save_dir}/{img_name}.png', nrow=1, padding=4)

    @torch.no_grad()
    def inference(self, data_loader, save_dir):
        self.encoder.eval()
        self.decoder.eval()
        self.quantize_identity.eval()
        self.quant_conv_identity.eval()
        self.post_quant_conv_identity.eval()
        self.quantize_others.eval()
        self.quant_conv_others.eval()
        self.post_quant_conv_others.eval()

        loss_total = 0
        num = 0

        for _, data in enumerate(data_loader):
            img_name = data['video_name'][0]
            x_identity, x_frame_aug, x_frame = self.feed_data(data)
            xrec, _ = self.forward_step(x_identity, x_frame_aug)

            recon_loss = torch.abs(x_frame.contiguous() - xrec.contiguous())
            p_loss = self.perceptual(x_frame.contiguous(), xrec.contiguous())
            nll_loss = recon_loss + self.perceptual_weight * p_loss
            nll_loss = torch.mean(nll_loss)
            loss_total += nll_loss

            num += x_frame.size(0)

            self.get_vis(x_identity, x_frame_aug, x_frame, xrec, save_dir,
                         img_name)

        return (loss_total / num).item() * (-1)
