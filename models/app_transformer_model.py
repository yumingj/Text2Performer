import logging
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.distributions as dists
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torchvision.utils import save_image

from models.archs.transformer_arch import TransformerLanguage
from models.archs.vqgan_arch import (
    DecoderUpOthersDoubleIdentity,
    EncoderDecomposeBaseDownOthersDoubleIdentity, VectorQuantizer)

logger = logging.getLogger('base')


class AppTransformerModel():
    """Texture-Aware Diffusion based Transformer model.
    """

    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda')
        self.is_train = opt['is_train']

        # VQVAE for image
        self.img_encoder = EncoderDecomposeBaseDownOthersDoubleIdentity(
            ch=opt['img_ch'],
            num_res_blocks=opt['img_num_res_blocks'],
            attn_resolutions=opt['img_attn_resolutions'],
            ch_mult=opt['img_ch_mult'],
            other_ch_mult=opt['img_other_ch_mult'],
            in_channels=opt['img_in_channels'],
            resolution=opt['img_resolution'],
            z_channels=opt['img_z_channels'],
            double_z=opt['img_double_z'],
            dropout=opt['img_dropout']).to(self.device)
        self.img_decoder = DecoderUpOthersDoubleIdentity(
            in_channels=opt['img_in_channels'],
            resolution=opt['img_resolution'],
            z_channels=opt['img_z_channels'],
            ch=opt['img_ch'],
            out_ch=opt['img_out_ch'],
            num_res_blocks=opt['img_num_res_blocks'],
            attn_resolutions=opt['img_attn_resolutions'],
            ch_mult=opt['img_ch_mult'],
            other_ch_mult=opt['img_other_ch_mult'],
            dropout=opt['img_dropout'],
            resamp_with_conv=True,
            give_pre_end=False).to(self.device)
        self.quantize_identity = VectorQuantizer(
            opt['img_n_embed'], opt['img_embed_dim'],
            beta=0.25).to(self.device)
        self.quant_conv_identity = torch.nn.Conv2d(opt["img_z_channels"],
                                                   opt['img_embed_dim'],
                                                   1).to(self.device)
        self.post_quant_conv_identity = torch.nn.Conv2d(
            opt['img_embed_dim'], opt["img_z_channels"], 1).to(self.device)

        self.quantize_others = VectorQuantizer(
            opt['img_n_embed'], opt['img_embed_dim'] // 2,
            beta=0.25).to(self.device)
        self.quant_conv_others = torch.nn.Conv2d(opt["img_z_channels"] // 2,
                                                 opt['img_embed_dim'] // 2,
                                                 1).to(self.device)
        self.post_quant_conv_others = torch.nn.Conv2d(
            opt['img_embed_dim'] // 2, opt["img_z_channels"] // 2,
            1).to(self.device)
        self.load_pretrained_image_vae()

        # define sampler
        self._denoise_fn = TransformerLanguage(
            codebook_size=opt['codebook_size'],
            bert_n_emb=opt['bert_n_emb'],
            bert_n_layers=opt['bert_n_layers'],
            bert_n_head=opt['bert_n_head'],
            block_size=opt['block_size'] * 2,
            embd_pdrop=opt['embd_pdrop'],
            resid_pdrop=opt['resid_pdrop'],
            attn_pdrop=opt['attn_pdrop']).to(self.device)

        self.num_classes = opt['codebook_size']
        self.shape = tuple(opt['latent_shape'])
        self.num_timesteps = 1000

        self.mask_id = opt['codebook_size']
        self.loss_type = opt['loss_type']
        self.mask_schedule = opt['mask_schedule']

        self.sample_steps = opt['sample_steps']

        self.text_seq_len = opt['text_seq_len']

        self.init_training_settings()

        self.get_fixed_language_model()

    def load_pretrained_image_vae(self):
        # load pretrained vqgan for segmentation mask
        img_ae_checkpoint = torch.load(self.opt['img_ae_path'])
        self.img_encoder.load_state_dict(
            img_ae_checkpoint['encoder'], strict=True)
        self.img_decoder.load_state_dict(
            img_ae_checkpoint['decoder'], strict=True)
        self.quantize_identity.load_state_dict(
            img_ae_checkpoint['quantize_identity'], strict=True)
        self.quant_conv_identity.load_state_dict(
            img_ae_checkpoint['quant_conv_identity'], strict=True)
        self.post_quant_conv_identity.load_state_dict(
            img_ae_checkpoint['post_quant_conv_identity'], strict=True)
        self.quantize_others.load_state_dict(
            img_ae_checkpoint['quantize_others'], strict=True)
        self.quant_conv_others.load_state_dict(
            img_ae_checkpoint['quant_conv_others'], strict=True)
        self.post_quant_conv_others.load_state_dict(
            img_ae_checkpoint['post_quant_conv_others'], strict=True)
        self.img_encoder.eval()
        self.img_decoder.eval()
        self.quantize_identity.eval()
        self.quant_conv_identity.eval()
        self.post_quant_conv_identity.eval()
        self.quantize_others.eval()
        self.quant_conv_others.eval()
        self.post_quant_conv_others.eval()

    def init_training_settings(self):
        optim_params = []
        for v in self._denoise_fn.parameters():
            if v.requires_grad:
                optim_params.append(v)
        # set up optimizer
        self.optimizer = torch.optim.Adam(
            optim_params,
            self.opt['lr'],
            weight_decay=self.opt['weight_decay'])
        self.log_dict = OrderedDict()

    @torch.no_grad()
    def get_quantized_img(self, image):
        h_identity, _ = self.img_encoder(image)
        h_identity = self.quant_conv_identity(h_identity)
        _, _, [_, _, identity_tokens] = self.quantize_identity(h_identity)

        _, h_frame = self.img_encoder(image)
        h_frame = self.quant_conv_others(h_frame)
        _, _, [_, _, pose_tokens] = self.quantize_others(h_frame)

        # reshape the tokens
        b = image.size(0)
        identity_tokens = identity_tokens.view(b, -1)
        pose_tokens = pose_tokens.view(b, -1)

        return identity_tokens, pose_tokens

    @torch.no_grad()
    def decode(self, quant_list):
        quant_identity = self.post_quant_conv_identity(quant_list[0])
        quant_frame = self.post_quant_conv_others(quant_list[1])
        dec = self.img_decoder(quant_identity, quant_frame)
        return dec

    @torch.no_grad()
    def decode_image_indices(self, quant_identity, quant_frame):
        quant_identity = self.quantize_identity.get_codebook_entry(
            quant_identity, (quant_identity.size(0), self.shape[0],
                             self.shape[1], self.opt["img_z_channels"]))
        quant_frame = self.quantize_others.get_codebook_entry(
            quant_frame, (quant_frame.size(0), self.shape[0] // 4,
                          self.shape[1] // 4, self.opt["img_z_channels"] // 2))
        dec = self.decode([quant_identity, quant_frame])

        return dec

    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(
                1, self.num_timesteps + 1, (b, ), device=device).long()
            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt

        else:
            raise ValueError

    def q_sample(self, x_0, x_0_gt, t):
        # samples q(x_t | x_0)
        # randomly set token to mask with probability t/T
        # x_t, x_0_ignore = x_0.clone(), x_0.clone()
        x_t = x_0.clone()

        mask = torch.rand_like(x_t.float()) < (
            t.float().unsqueeze(-1) / self.num_timesteps)
        x_t[mask] = self.mask_id
        # x_0_ignore[torch.bitwise_not(mask)] = -1

        # for every gt token list, we also need to do the mask
        x_0_gt_ignore = x_0_gt.clone()
        x_0_gt_ignore[torch.bitwise_not(mask)] = -1

        return x_t, x_0_gt_ignore, mask

    def _train_loss(self, x_identity_0, x_pose_0):
        b, device = x_identity_0.size(0), x_identity_0.device

        # choose what time steps to compute loss at
        t, pt = self.sample_time(b, device, 'uniform')

        # make x noisy and denoise
        if self.mask_schedule == 'random':
            x_identity_t, x_identity_0_gt_ignore, mask = self.q_sample(
                x_0=x_identity_0, x_0_gt=x_identity_0, t=t)
            x_pose_t, x_pose_0_gt_ignore, mask = self.q_sample(
                x_0=x_pose_0, x_0_gt=x_pose_0, t=t)
        else:
            raise NotImplementedError

        # sample p(x_0 | x_t)
        x_identity_0_hat_logits, x_pose_0_hat_logits = self._denoise_fn(
            self.text_embedding, x_identity_t, x_pose_t, t=t)

        x_identity_0_hat_logits = x_identity_0_hat_logits[:,
                                                          1:1 + self.shape[0] *
                                                          self.shape[1], :]
        x_pose_0_hat_logits = x_pose_0_hat_logits[:, 1 + self.shape[0] *
                                                  self.shape[1]:]

        # Always compute ELBO for comparison purposes
        cross_entropy_loss = 0

        cross_entropy_loss = F.cross_entropy(
            x_identity_0_hat_logits.permute(0, 2, 1),
            x_identity_0_gt_ignore,
            ignore_index=-1,
            reduction='none').sum(1) + F.cross_entropy(
                x_pose_0_hat_logits.permute(0, 2, 1),
                x_pose_0_gt_ignore,
                ignore_index=-1,
                reduction='none').sum(1)
        vb_loss = cross_entropy_loss / t
        vb_loss = vb_loss / pt
        vb_loss = vb_loss / (math.log(2) * x_identity_0.shape[1:].numel())
        if self.loss_type == 'elbo':
            loss = vb_loss
        elif self.loss_type == 'mlm':
            denom = mask.float().sum(1)
            denom[denom == 0] = 1  # prevent divide by 0 errors.
            loss = cross_entropy_loss / denom
        elif self.loss_type == 'reweighted_elbo':
            weight = (1 - (t / self.num_timesteps))
            loss = weight * cross_entropy_loss
            loss = loss / (math.log(2) * x_identity_0.shape[1:].numel())
        else:
            raise ValueError

        return loss.mean(), vb_loss.mean()

    def feed_data(self, data):
        self.image = data['image'].to(self.device)
        self.text = data['text']  #.to(self.device)
        self.get_text_embedding()

        self.identity_tokens, self.pose_tokens = self.get_quantized_img(
            self.image)

    def get_fixed_language_model(self):
        self.language_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.text_feature_dim = 384

    @torch.no_grad()
    def get_text_embedding(self):
        self.text_embedding = self.language_model.encode(self.text, show_progress_bar=False)
        self.text_embedding = torch.Tensor(self.text_embedding).to(
            self.device).unsqueeze(1)

    def optimize_parameters(self):
        self._denoise_fn.train()

        loss, vb_loss = self._train_loss(self.identity_tokens,
                                         self.pose_tokens)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_dict['loss'] = loss
        self.log_dict['vb_loss'] = vb_loss

        self._denoise_fn.eval()

    def sample_fn(self, temp=1.0, sample_steps=None):
        self._denoise_fn.eval()

        b, device = self.image.size(0), 'cuda'
        x_identity_t = torch.ones(
            (b, np.prod(self.shape)), device=device).long() * self.mask_id
        x_pose_t = torch.ones((b, np.prod(self.shape) // 16),
                              device=device).long() * self.mask_id
        unmasked_identity = torch.zeros_like(
            x_identity_t, device=device).bool()
        unmasked_pose = torch.zeros_like(x_pose_t, device=device).bool()
        sample_steps = list(range(1, sample_steps + 1))

        for t in reversed(sample_steps):
            print(f'Sample timestep {t:4d}', end='\r')
            t = torch.full((b, ), t, device=device, dtype=torch.long)

            # where to unmask
            changes_identity = torch.rand(
                x_identity_t.shape,
                device=device) < 1 / t.float().unsqueeze(-1)
            # don't unmask somewhere already unmasked
            changes_identity = torch.bitwise_xor(
                changes_identity,
                torch.bitwise_and(changes_identity, unmasked_identity))
            # update mask with changes
            unmasked_identity = torch.bitwise_or(unmasked_identity,
                                                 changes_identity)

            changes_pose = torch.rand(
                x_pose_t.shape, device=device) < 1 / t.float().unsqueeze(-1)
            # don't unmask somewhere already unmasked
            changes_pose = torch.bitwise_xor(
                changes_pose, torch.bitwise_and(changes_pose, unmasked_pose))
            # update mask with changes
            unmasked_pose = torch.bitwise_or(unmasked_pose, changes_pose)

            x_identity_0_hat_logits, x_pose_0_hat_logits = self._denoise_fn(
                self.text_embedding, x_identity_t, x_pose_t, t=t)

            x_identity_0_hat_logits = x_identity_0_hat_logits[:, 1:1 +
                                                              self.shape[0] *
                                                              self.shape[1], :]
            x_pose_0_hat_logits = x_pose_0_hat_logits[:, 1 + self.shape[0] *
                                                      self.shape[1]:]

            # scale by temperature
            x_identity_0_hat_logits = x_identity_0_hat_logits / temp
            x_identity_0_dist = dists.Categorical(
                logits=x_identity_0_hat_logits)
            x_identity_0_hat = x_identity_0_dist.sample().long()

            x_pose_0_hat_logits = x_pose_0_hat_logits / temp
            x_pose_0_dist = dists.Categorical(logits=x_pose_0_hat_logits)
            x_pose_0_hat = x_pose_0_dist.sample().long()

            # x_t would be the input to the transformer, so the index range should be continual one
            x_identity_t[changes_identity] = x_identity_0_hat[changes_identity]
            x_pose_t[changes_pose] = x_pose_0_hat[changes_pose]

        self._denoise_fn.train()

        return x_identity_t, x_pose_t

    def get_vis(self, image, gt_quant_identity, gt_quant_frame, quant_identity,
                quant_frame, save_path):
        # original image
        ori_img = self.decode_image_indices(gt_quant_identity, gt_quant_frame)
        # pred image
        pred_img = self.decode_image_indices(quant_identity, quant_frame)
        img_cat = torch.cat([
            image,
            ori_img,
            pred_img,
        ], dim=3).detach()
        img_cat = ((img_cat + 1) / 2)
        img_cat = img_cat.clamp_(0, 1)
        save_image(img_cat, save_path, nrow=1, padding=4)

    def inference(self, data_loader, save_dir):
        self._denoise_fn.eval()

        for _, data in enumerate(data_loader):
            img_name = data['img_name']
            self.feed_data(data)
            b = self.image.size(0)
            with torch.no_grad():
                x_identity_t, x_pose_t = self.sample_fn(
                    temp=1, sample_steps=self.sample_steps)
            for idx in range(b):
                self.get_vis(self.image[idx:idx + 1],
                             self.identity_tokens[idx:idx + 1],
                             self.pose_tokens[idx:idx + 1],
                             x_identity_t[idx:idx + 1], x_pose_t[idx:idx + 1],
                             f'{save_dir}/{img_name[idx]}.png')

        self._denoise_fn.train()

    def sample_appearance(self, text, save_path, shape=[256, 128]):
        self._denoise_fn.eval()

        self.text = text
        self.image = torch.zeros([1, 3, shape[0], shape[1]]).to(self.device)
        self.get_text_embedding()

        with torch.no_grad():
            x_identity_t, x_pose_t = self.sample_fn(
                temp=1, sample_steps=self.sample_steps)

        self.get_vis_generated_only(x_identity_t, x_pose_t, save_path)

        quant_identity = self.quantize_identity.get_codebook_entry(
            x_identity_t, (x_identity_t.size(0), self.shape[0], self.shape[1],
                           self.opt["img_z_channels"]))
        quant_frame = self.quantize_others.get_codebook_entry(
            x_pose_t,
            (x_pose_t.size(0), self.shape[0] // 4, self.shape[1] // 4,
             self.opt["img_z_channels"] // 2)).view(
                 x_pose_t.size(0), self.opt["img_z_channels"] // 2,
                 -1).permute(0, 2, 1)

        self._denoise_fn.train()

        return quant_identity, quant_frame

    def get_vis_generated_only(self, quant_identity, quant_frame, save_path):
        # pred image
        pred_img = self.decode_image_indices(quant_identity, quant_frame)
        img_cat = ((pred_img.detach() + 1) / 2)
        img_cat = img_cat.clamp_(0, 1)
        save_image(img_cat, save_path, nrow=1, padding=4)

    def get_current_log(self):
        return self.log_dict

    def update_learning_rate(self, epoch, iters=None):
        """Update learning rate.

        Args:
            current_iter (int): Current iteration.
            warmup_iter (int): Warmup iter numbers. -1 for no warmup.
                Default: -1.
        """
        lr = self.optimizer.param_groups[0]['lr']

        if self.opt['lr_decay'] == 'step':
            lr = self.opt['lr'] * (
                self.opt['gamma']**(epoch // self.opt['step']))
        elif self.opt['lr_decay'] == 'cos':
            lr = self.opt['lr'] * (
                1 + math.cos(math.pi * epoch / self.opt['num_epochs'])) / 2
        elif self.opt['lr_decay'] == 'linear':
            lr = self.opt['lr'] * (1 - epoch / self.opt['num_epochs'])
        elif self.opt['lr_decay'] == 'linear2exp':
            if epoch < self.opt['turning_point'] + 1:
                # learning rate decay as 95%
                # at the turning point (1 / 95% = 1.0526)
                lr = self.opt['lr'] * (
                    1 - epoch / int(self.opt['turning_point'] * 1.0526))
            else:
                lr *= self.opt['gamma']
        elif self.opt['lr_decay'] == 'schedule':
            if epoch in self.opt['schedule']:
                lr *= self.opt['gamma']
        elif self.opt['lr_decay'] == 'warm_up':
            if iters <= self.opt['warmup_iters']:
                lr = self.opt['lr'] * float(iters) / self.opt['warmup_iters']
            else:
                lr = self.opt['lr']
        else:
            raise ValueError('Unknown lr mode {}'.format(self.opt['lr_decay']))
        # set learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def save_network(self, net, save_path):
        """Save networks.

        Args:
            net (nn.Module): Network to be saved.
            net_label (str): Network label.
            current_iter (int): Current iter number.
        """
        state_dict = net.state_dict()
        torch.save(state_dict, save_path)

    def load_network(self):
        checkpoint = torch.load(self.opt['pretrained_sampler'])
        self._denoise_fn.load_state_dict(checkpoint, strict=True)
        self._denoise_fn.eval()
