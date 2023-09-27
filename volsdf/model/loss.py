import torch
from torch import nn
import volsdf.utils.general as utils
import math
import numpy as np
from helpers.help import logger

def anneal_linearly(t, val0, val1):
    if t >= 1:
        return val1
    elif t <= 0:
        return val0
    return val0 + (val1 - val0) * np.minimum(t, 1.)

class VolSDFLoss(nn.Module):
    def __init__(self, rgb_loss, 
            eikonal_weight, rgb_weight=1., mvs_weight=0., 
            sparse_weight=0., anneal_rgb=0,
            gce=1, confi=0):
        super().__init__()
        self.eikonal_weight = eikonal_weight
        self.rgb_weight = rgb_weight
        self.mvs_weight = mvs_weight
        self.sparse_weight = sparse_weight
        self.gce = gce
        self.anneal_rgb = anneal_rgb
        self.confi = confi
        logger.info(f"loss lambda: RGB_{rgb_weight} EK_{eikonal_weight} MVS_{mvs_weight} SP_{sparse_weight}")
        self.rgb_loss = utils.get_class(rgb_loss)(reduction='mean')

    def set_stg(self, stg):
        self.iter_step = 0
        if stg >= 1:
            self.anneal_rgb = 0
            self.sparse_weight = 0
            raise NotImplementedError

    def get_rgb_loss(self,rgb_values, rgb_gt, model_outputs=None, t=0):
        rgb_gt = rgb_gt.reshape(-1, 3)
        if t > 0:
            pi, pj = model_outputs['pi'], model_outputs['pj'] # (512, 98)
            confi = (pi*pj).sum(-1)
            loss = torch.abs(rgb_values-rgb_gt).mean(-1)
            loss = loss * (confi<t) # confi<0.01, uncertainty
            return loss.mean()
        rgb_loss = self.rgb_loss(rgb_values, rgb_gt)
        return rgb_loss

    def get_eikonal_loss(self, grad_theta):
        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def get_mvs_loss(self, model_outputs):
        pi, pj = model_outputs['pi'], model_outputs['pj'] # (512, 98)
        w = model_outputs['weights'] # (512, 98)
        pw = pi * pj

        if self.gce == 1:
            loss = - pw * w
        elif self.gce == 0:
            loss = - pw * torch.log(w+1e-8)
        else:
            loss = - pw * w.detach() ** self.gce
            loss = loss * torch.log(w+1e-8)
        loss = loss.sum(1)
        loss = 1. * (pw.sum(1) > self.confi) * loss
        return loss.mean()

    def get_sparse_loss(self, model_outputs):
        pi, pj = model_outputs['pi'], model_outputs['pj'] # (512, 98)
        confi = (pi*pj).sum(-1)
        if 'depth_values_all' in model_outputs:
            dep = model_outputs['depth_values_all'].squeeze()
        else:
            dep = model_outputs['depth_values'].squeeze()
        loss = 1. / (dep + 1e-3)
        loss = loss * (confi<self.confi)
        return loss.mean()

    def forward(self, model_outputs, ground_truth):
        rgb_gt = ground_truth['rgb'].cuda()

        output = {}
        output['rgb_loss'] = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt)
        if 'grad_theta' in model_outputs:
            output['eikonal_loss'] = self.get_eikonal_loss(model_outputs['grad_theta'])
        else:
            output['eikonal_loss'] = torch.tensor(0.0).cuda().float()

        # generalized cross entropy loss
        if 'pi' in model_outputs and self.mvs_weight > 0:
            output['mvs_loss'] = self.get_mvs_loss(model_outputs)
        else:
            output['mvs_loss'] = torch.tensor(0.0).cuda().float()

        # sparsity regularization
        if 'pi' in model_outputs and self.sparse_weight > 0 and self.anneal_rgb > 0 and self.iter_step < self.anneal_rgb:
            output['sparse_loss'] = self.get_sparse_loss(model_outputs)
        else:
            output['sparse_loss'] = torch.tensor(0.0).cuda().float()

        anneal_sparse = 0
        if self.sparse_weight > 0 and self.anneal_rgb > 0 and self.iter_step < self.anneal_rgb:
            anneal_sparse = anneal_linearly(self.iter_step/self.anneal_rgb, 1.0, 0.)
            output['rgb_loss'] = self.get_rgb_loss(model_outputs['rgb_values'], ground_truth['rgb_smooth'].cuda(), model_outputs, t=1e-8)

        # total loss
        output['loss'] = self.rgb_weight * output['rgb_loss'] +\
            self.eikonal_weight * output['eikonal_loss'] +\
            self.mvs_weight * output['mvs_loss'] +\
            self.sparse_weight * anneal_sparse * output['sparse_loss']

        self.iter_step += 1

        return output