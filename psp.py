import math

import torch
from torch import nn
from encoders import psp_encoders
from model import Generator

class pSp(nn.Module):

    def __init__(self, encoder_type='BackboneEncoderUsingLastLayerIntoWPlus', ckpt_path='stylegan2-ffhq-config-f.pt'):
        super(pSp, self).__init__()

        d = dict()
        d['output_size'] = 256 # TODO: check parameters
        d['encoder_type'] = 'GradualStyleEncoder'
        d['latent_avg'] = None
        d['input_nc'] = 3
        d['label_nc'] = 0

        d['start_from_latent_avg'] = False
        d['learn_in_w'] = False

        self.opts = dotdict(d)
        
        

        # compute number of style inputs based on the output resolution
        self.opts.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2
        # Define architecture
        self.encoder = self.set_encoder()
        self.decoder = Generator(self.opts.output_size, 512, 8)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        # Load weights if needed
        self.load_weights(ckpt_path)

    def forward(self, x, resize=True, latent_mask=None, input_code=False, randomize_noise=True, 
                inject_latent=None, return_latents=False, alpha=None):

        if input_code:
            codes = x
        else:
            codes = self.encoder(x)
            
            # TODO: test out the commented code
            
            # normalize with respect to the center of an average face (TODO)
            # if self.opts.start_from_latent_avg:
            #     if self.opts.learn_in_w:
            #         codes = codes + self.opts.latent_avg.repeat(codes.shape[0], 1)
            #     else:
            #         codes = codes + self.opts.latent_avg.repeat(codes.shape[0], 1, 1)
        # if latent_mask is not None:
        #     for i in latent_mask:
        #         if inject_latent is not None:
        #             if alpha is not None:
        #                 codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
        #             else:
        #                 codes[:, i] = inject_latent[:, i]
        #         else:
        #             codes[:, i] = 0

        input_is_latent = not input_code
        images, result_latent = self.decoder([codes],
                                             input_is_latent=input_is_latent,
                                             randomize_noise=randomize_noise,
                                             return_latents=return_latents)

        if resize:
            images = self.face_pool(images)

        if return_latents:
            return images, result_latent
        else:
            return images

    def set_encoder(self):
        if self.opts.encoder_type == 'GradualStyleEncoder':
            encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoW':
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoWPlus':
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoWPlus(50, 'ir_se', self.opts)
        else:
            raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
        return encoder

    def load_weights(self, ckpt_path):
        ckpt = torch.load(ckpt_path)
        self.decoder.load_state_dict(ckpt, strict=False)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
