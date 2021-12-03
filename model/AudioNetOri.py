
'''
Part of the code is drawn from 
https://github.com/usc-sail/gard-adversarial-speaker-id
Paper: 
Jati et al. Adversarial attack and defense strategies for deep speaker recognition systems
'''
import torch.nn as nn
import time
import sys

from model.Preprocessor import Preprocessor

from defense.defense import *
from defense.time_domain import *
from defense.frequency_domain import *
from defense.speech_compression import *
from defense.feature_level import *

BITS = 16

class AudioNetOri(nn.Module):
    """Adaption of AudioNet (arXiv:1807.03418)."""
    def __init__(self, num_class, transform_layer=None, transform_param=None):
        super().__init__()
        self.prep = Preprocessor()
        self.num_spks = num_class
        
        assert transform_layer in (Input_Transformation + [None])
        self.wav_transform = False
        self.feat_transform = False
        self.transform_layer = None
        self.param = None
        self.other_param = None
        
        if transform_layer == 'FEATURE_COMPRESSION' or transform_layer == 'FeCo':
            self.transform_layer = FEATURE_COMPRESSION
            self.feat_transform = True
            assert isinstance(transform_param, list) and len(transform_param) == 4
            self.cl_m, self.feat_point, self.param, self.other_param = transform_param
            assert self.cl_m in ['kmeans', 'warped_kmeans']
            assert self.feat_point in ['raw'] # AudioNet not uses delta, cmvn and final
            if self.cl_m == 'kmeans':
                assert self.other_param in ["L2", "cos"]
            elif self.cl_m == 'warped_kmeans':
                assert self.other_param in ['ts', 'random']
            else:
                raise NotImplementedError('Currently FEATURE COMPRESSION only suppots kmeans and warped_kmeans')
            assert 0 < self.param <= 1
        elif transform_layer:
            self.wav_transform = True
            if transform_layer == 'BPF':
                assert isinstance(transform_param, list) and len(transform_param) == 2
            self.param = transform_param
            self.transform_layer = getattr(sys.modules[__name__], transform_layer)
        
        print(self.wav_transform,
        self.feat_transform,
        self.transform_layer,
        self.param,
        self.other_param)

        # =========== EXPERIMENTAL pre-filtering ======
        # 32 x 100
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=[5, 5], stride=1, padding=[2, 2]),
            nn.BatchNorm2d(1),
        )
        # =========== ============= ======

        # 32 x 100
        self.conv2 = nn.Sequential( 
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        )
        # 64 x 100
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        # 128 x 100
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        # 128 x 50
        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        )
        # 128 x 50
        self.conv6 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        # 128 x 25
        self.conv7 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        )
        self.conv8 = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        # 32 x 30
        self.fc = nn.Linear(32, num_class)
    

    def make_feature(self, x):

        if self.wav_transform:
            x = self.transform_layer(x.squeeze(1), param=self.param).unsqueeze(1)
        
        x = self.prep(x.squeeze(1))
        if self.feat_transform:
            x = self.apply_feat_filter(x)
        
        return x
    
    def apply_feat_filter(self, x_batch):
        
        y_batch = None
        start_t = time.time()
        #### Naive Loop, since it is hard to parallel ###
        for index, x in enumerate(x_batch):
            t1 = time.time()
            # y = self.transform_layer(x.T, param=self.param, other_param=self.other_param)
            y = self.transform_layer(x.T, self.cl_m, param=self.param, other_param=self.other_param)
            t2 = time.time()
            if index == 0:
                y_batch = y.T.view(1, y.shape[1], -1) 
            else:
                y_batch = torch.cat([y_batch, y.T.view(1, y.shape[1], -1)], dim=0)
        end_t = time.time()
        return y_batch 
    
    def encode_feat(self, x):
        # ===== pre-filtering ========
        # [B, F, T]
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = x.squeeze(1)
        # ===== pre-filtering ========

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        target_len = 3
        real_len = x.shape[2]
        if real_len < target_len:
            n = target_len // real_len
            if target_len % real_len == 0:
                n = n
            else:
                n = n + 1
            x = x.repeat(1, 1, n)

        x = self.conv8(x)
        x, _ = x.max(2)
        return x

    def encode(self, x):
        x = self.make_feature(x)
        return self.encode_feat(x)

    def predict_from_embeddings(self, x):
        return self.fc(x)

    def forward(self, x):
        """
        Inputs:
            x: [B, 1, T] waveform
        Outputs:
            x: [B, 1, T] waveform
        """
        # 
        lower = -1
        upper = 1
        if not (x.max() <= 2 * upper and x.min() >= 2 * lower): # 2*lower and 2*upper due to floating point issue, e.g., sometimes will have 1.0002
            x = x / (2 ** (BITS-1)) 
        embedding = self.encode(x)
        logits = self.predict_from_embeddings(embedding)
        return logits
    
    def score(self, x):
        logits = self.forward(x)
        scores = F.softmax(logits, dim=1)
        return scores
    
    def make_decision(self, x):
        scores = self.score(x)
        decisions = torch.argmax(scores, dim=1)
        return decisions, scores