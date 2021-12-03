
from model.xvector_PLDA_helper import xvector_PLDA_helper
import torch
import numpy as np
import torch.nn as nn

class xvector_PLDA(nn.Module):

    def __init__(self, model_file, xv_extractor_file="./xv_system/xvecTDNN_origin.ckpt",
                plda_file="./xv_system/plda.txt", ivector_meanfile="./xv_system/mean.vec", 
                transform_mat_file="./xv_system/transform.txt", device="cpu",
                transform_layer=None, transform_param=None, threshold=None):

        super().__init__()

        self.device = device

        self.model_file = model_file
        model_info = np.loadtxt(self.model_file, dtype=str)
        if len(model_info.shape) == 1:
            model_info = model_info[np.newaxis, :] # for SV
        self.num_spks = model_info.shape[0]
        self.spk_ids = list(model_info[:, 0])
        self.identity_locations = list(model_info[:, 1])
        
        self.z_norm_means = (model_info[:, 2]).astype(
            np.float32)  # float32, make consistency
        self.z_norm_stds = (model_info[:, 3]).astype(
            np.float32)  # float32, make consistency
        self.z_norm_means = torch.tensor(self.z_norm_means, device=self.device)
        self.z_norm_stds = torch.tensor(self.z_norm_stds, device=self.device)

        self.enroll_ivectors = None
        for index, path in enumerate(self.identity_locations):
            iv = torch.load(path, map_location=self.device).unsqueeze(0)
            if index == 0:
                self.enroll_ivectors = iv
            else:
                self.enroll_ivectors = torch.cat([self.enroll_ivectors, iv], dim=0)
        
        self.helper = xvector_PLDA_helper(xv_extractor_file, plda_file, 
                ivector_meanfile, transform_mat_file, device=self.device,
                transform_layer=transform_layer, transform_param=transform_param)
        
        # If you need SV or OSI, must input threshold
        self.threshold = threshold if threshold else -np.infty # Valid for SV and OSI tasks; CSI: infty

    
    def forward(self, audios):
        return self.helper.score(audios, self.enroll_ivectors)
        
    def score(self, audios, z_norm=False):
        scores = self.forward(audios)
        # when set z_norm to True, caution that some adversarial examples may become ineffective 
        # since attack calls 'foward' which does not use z_norm
        if z_norm:
            scores = (scores - self.z_norm_means) / self.z_norm_stds
        return scores
    
    def make_decision(self, audios, z_norm=False): # -1: reject
        scores = self.score(audios, z_norm=z_norm)
        decisions = torch.argmax(scores, dim=1)
        max_scores = torch.max(scores, dim=1)[0]
        decisions = torch.where(max_scores > self.threshold , decisions, 
                        torch.tensor([-1] * decisions.shape[0], dtype=torch.int64, device=decisions.device))

        return decisions, scores
    
    def to(self, device):
        if self.device == device:
            return
        self.device = device
        self.z_norm_means = self.z_norm_means.to(self.device)
        self.z_norm_stds = self.z_norm_stds.to(self.device)
        self.enroll_ivectors = self.enroll_ivectors.to(self.device)
        self.helper.to(self.device)
    
    def clone(self, device=None):
        import copy
        copy_model = copy.deepcopy(self)
        if device is None or device == self.device:
            pass
        else:
            copy_model.to(device)
        return copy_model
