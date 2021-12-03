
import torch
import copy
import numpy as np
import sys

from model.xvector_extract import xvectorExtractor
from model.plda import PLDA

from defense.defense import *
from defense.time_domain import *
from defense.frequency_domain import *
from defense.speech_compression import *
from defense.feature_level import * 

BITS = 16

class xvector_PLDA_helper(object):

    def __init__(self, xv_extractor_file, plda_file, xvector_meanfile, transform_mat_file, device="cpu",
                transform_layer=None, transform_param=None):
        super().__init__()

        self.device = device

        self.xv_extractor_file = xv_extractor_file
        self.plda_file = plda_file

        self.extractor = xvectorExtractor(self.xv_extractor_file)
        self.plda = PLDA(self.plda_file)

        # for quick load and debug
        # import pickle
        # with open("./system/plda.pickle", "wb") as writer:
        #     pickle.dump(self.plda, writer, -1)
        # with open("./system/plda.pickle", "rb") as reader:
        #     self.plda = pickle.load(reader)

        if self.extractor.device != self.device:
            self.extractor.to(self.device)
        if self.plda.device != self.device:
            self.plda.to(self.device)

        self.embedding_dim = self.extractor.xvector_dim

        rfile = open(xvector_meanfile, 'r')
        line = rfile.readline()
        data = line.split()[1:-1]
        for i in range(self.embedding_dim):
            data[i] = float(data[i])
        self.xvector_mean = torch.tensor(data, device=self.device)
        rfile.close()

        with open(transform_mat_file, "r") as reader:
            lines = reader.readlines()
            lines = lines[1:]
            n_rows = len(lines)
            n_cols = len(lines[0][:-1].lstrip().rstrip().split(" "))
            transform_mat = np.zeros((n_rows, n_cols))
            for i, line in enumerate(lines):
                if i < n_rows - 1:
                    line = lines[i][:-1].lstrip().rstrip().split(" ")
                else:
                    line = line = lines[i][:-2].lstrip().rstrip().split(" ")
                line = np.array([float(item) for item in line]) 
                transform_mat[i] = line
            self.transform_mat = torch.tensor(transform_mat, device=self.device, dtype=torch.float)

        assert transform_layer in (Input_Transformation + [None])
        self.wav_transform = False
        self.feat_transform = False
        self.mask = False
        self.transform_layer = None
        self.param = None
        self.other_param = None
        if transform_layer == 'FEATURE_COMPRESSION' or transform_layer == 'FeCo':
            self.transform_layer = FEATURE_COMPRESSION
            self.feat_transform = True
            assert isinstance(transform_param, list) and len(transform_param) == 4
            self.cl_m, self.feat_point, self.param, self.other_param = transform_param
            assert self.cl_m in ['kmeans', 'warped_kmeans']
            # assert self.feat_point in ['raw', 'delta', 'cmvn', 'final']
            assert self.feat_point in ['raw', 'cmvn', 'final'] # no delta
            if self.cl_m == 'kmeans':
                assert self.other_param in ["L2", "cos"]
            elif self.cl_m == 'warped_kmeans':
                assert self.other_param in ['ts', 'random']
            else:
                raise NotImplementedError('Currently FEATURE COMPRESSION only suppots kmeans and warped_kmeans')
            assert 0 < self.param <= 1
        elif transform_layer is not None:
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

    def score(self, audios, enroll_ivectors):
        """
        """
        lower = -1
        upper = 1
        if audios.max() <= 2 * upper and audios.min() >= 2 * lower: # 2*lower and 2*upper due to floating point issue, e.g., sometimes will have 1.0002
            audios = audios * (2 ** (BITS-1))
        n_audios = audios.shape[0]
        n_spks = enroll_ivectors.size()[0]
        scores = torch.zeros((n_audios, n_spks), device=self.device)
        if self.wav_transform:
            audios = self.transform_layer(audios.squeeze(1), param=self.param).unsqueeze(1)
        for index in range(n_audios):
            ivector = self.compute_ivector(audios[index])
            scores[index] = self.ComputeScores(enroll_ivectors, ivector)
        
        return scores
    
    def compute_ivector(self, audio, num_utt=1, simple_length_norm=False, normalize_length=True):
        ivector = self.Extract(audio)
        return self.process_ivector(ivector, num_utt=num_utt, 
                        simple_length_norm=simple_length_norm, normalize_length=normalize_length)

    def Extract(self, audio):
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)
        mfcc = self.mfcc(audio)
        return self.extractor.Extract(mfcc) 
    
    def process_ivector(self, xvector, num_utt=1, simple_length_norm=False, normalize_length=True):
        """ xvector: (batch, xv_dim)
        """
        xvector = self.SubtractGlobalMean(xvector)
        xvector = self.lda_reduce_dim(xvector)
        xvector = self.LengthNormalization(xvector,
                        torch.sqrt(torch.tensor(xvector.size()[0], dtype=torch.float, device=self.device)))
        xvector = self.TransformIvector(xvector, num_utt, simple_length_norm, normalize_length)
        return xvector  
    
    def SubtractGlobalMean(self, xvector): 
        return self.extractor.SubtractGlobalMean(xvector, self.xvector_mean)
    
    def lda_reduce_dim(self, ivector):
        _, transform_cols = self.transform_mat.size()
        vec_dim = ivector.size()[0]
        reduced_dim_vec = None
        if transform_cols == vec_dim:
            pass # not our case, just skip
        else:
            assert transform_cols == vec_dim + 1
            reduced_dim_vec = self.transform_mat[:, vec_dim:vec_dim+1]
            reduced_dim_vec = reduced_dim_vec.clone() # avoid influcing self.transform_mat  
            reduced_dim_vec.add_(1. * torch.matmul(self.transform_mat[:, :-1], torch.unsqueeze(ivector, 1)))

        return torch.squeeze(reduced_dim_vec)

    def LengthNormalization(self, xvector, dim):
        return self.extractor.LengthNormalization(xvector, dim)
    
    def TransformIvector(self, xvector, num_utt, simple_length_norm, normalize_length):
        """
        xvector must be (xv_dim,)
        plda not support batch here
        """
        return self.plda.TransformIvector(xvector, num_utt, simple_length_norm, normalize_length)
    
    def ComputeScores(self, enroll_ivectors, xvector):
        """
        xvector must be (xv_dim,)
        plda not support batch here
        """
        return self.plda.ComputeScores(enroll_ivectors, 1, xvector)

    def mfcc(self, audio):
        if audio.device != self.device:
            audio = audio.to(self.device)
        raw_feat = self.raw(audio)

        if not self.feat_transform:
            delta_feat = raw_feat # no delta feature
            cmvn_feat = self.cmvn(delta_feat)
            vad_result = self.vad(raw_feat)
            final_feat = self.select_voiced_frames(cmvn_feat, vad_result)
        elif self.feat_point == 'raw':
            raw_feat = self.transform_layer(raw_feat, self.cl_m, param=self.param, other_param=self.other_param)
            delta_feat = raw_feat
            cmvn_feat = self.cmvn(delta_feat)
            vad_result = self.vad(raw_feat)
            final_feat = self.select_voiced_frames(cmvn_feat, vad_result)
        # elif self.feat_point == 'delta': # no delta feature
        #     delta_feat = self.transform_layer(self.add_delta(raw_feat), self.cl_m, param=self.param, other_param=self.other_param)
        #     cmvn_feat = self.cmvn(delta_feat)
        #     raw_feat = self.transform_layer(raw_feat, self.cl_m, param=self.param, other_param=self.other_param)
        #     vad_result = self.vad(raw_feat)
        #     final_feat = self.select_voiced_frames(cmvn_feat, vad_result)
        elif self.feat_point == 'cmvn':
            delta_feat = raw_feat
            cmvn_feat = self.transform_layer(self.cmvn(delta_feat), self.cl_m, param=self.param, other_param=self.other_param)
            raw_feat = self.transform_layer(raw_feat, self.cl_m, param=self.param, other_param=self.other_param)
            vad_result = self.vad(raw_feat)
            final_feat = self.select_voiced_frames(cmvn_feat, vad_result)
        elif self.feat_point == 'final':
             delta_feat = raw_feat
             cmvn_feat = self.cmvn(delta_feat)
             vad_result = self.vad(raw_feat)
             final_feat = self.transform_layer(self.select_voiced_frames(cmvn_feat, vad_result), self.cl_m,
                            param=self.param, other_param=self.other_param)
        else:
            raise NotImplementedError('Not Supported Feat Point')
        
        return final_feat 

    def raw(self, audio):
        raw_feat = torchaudio.compliance.kaldi.mfcc(audio,

                                    sample_frequency=16000, 
                                    frame_shift=10,
                                    frame_length=25,
                                    dither=1.0,  
                                    # dither=0.0, 
                                    preemphasis_coefficient=0.97,
                                    remove_dc_offset=True,
                                    window_type="povey",
                                    round_to_power_of_two=True,
                                    blackman_coeff=0.42,
                                    snip_edges=False,
                                    # allow_downsample=False,
                                    # allow_upsample=False,
                                    # max_feature_vectors=-1,

                                    num_mel_bins=30,
                                    low_freq=20,
                                    high_freq=7600,
                                    vtln_low=100,
                                    vtln_high=-500,
                                    vtln_warp=1.0,
                                    # debug_mel=False,
                                    # htk_mode=False,

                                    # num_ceps=24, # egs/voxceleb/v1
                                    num_ceps=30, # egs/voxceleb/v2
                                    use_energy=True,
                                    energy_floor=0.0, 
                                    # energy_floor=1.0, 
                                    # energy_floor=0.1,   
                                    raw_energy=True, 
                                    cepstral_lifter=22.0,
                                    htk_compat=False) 
        
        return raw_feat

    def cmvn(self, delta_feat):

        opts = {}
        CENTER = "center"
        NORMALIZE_VARIANCE = "normalize_variance"
        CMN_WINDOW = "cmn_window"
        opts[CENTER] = True
        opts[NORMALIZE_VARIANCE] = False
        opts[CMN_WINDOW] = 300

        num_frames, dim = delta_feat.size()
        last_window_start = -1
        last_window_end = -1
        cur_sum = torch.zeros((dim, ), device=delta_feat.device)
        cur_sumsq = torch.zeros((dim, ), device=delta_feat.device)

        cmvn_feat = delta_feat.clone()
        for t in range(num_frames):

            window_start = 0
            window_end = 0

            if opts[CENTER]:
                window_start = t - (opts[CMN_WINDOW] / 2)
                window_end = window_start + opts[CMN_WINDOW]
            else:
                pass

            if window_start < 0:
                window_end -= window_start
                window_start = 0
            
            if not opts[CENTER]:
                pass

            if window_end > num_frames:
                window_start -= (window_end - num_frames)
                window_end = num_frames
                if window_start < 0:
                    window_start = 0
            
            if last_window_start == -1:
                delta_feat_part = delta_feat[int(window_start):int(window_end), :]
                cur_sum.fill_(0.)
                cur_sum.add_(torch.sum(delta_feat_part, 0, keepdim=False), alpha=1.)
                if opts[NORMALIZE_VARIANCE]:
                    pass
            else:
                if window_start > last_window_start:
                    assert window_start == last_window_start + 1
                    frame_to_remove = delta_feat[int(last_window_start), :]
                    cur_sum.add_(frame_to_remove, alpha=-1.)

                    if opts[NORMALIZE_VARIANCE]:
                        pass
                
                if window_end > last_window_end:
                    assert window_end == last_window_end + 1
                    frame_to_add = delta_feat[int(last_window_end), :] 
                    cur_sum.add_(frame_to_add, alpha=1.)

                    if opts[NORMALIZE_VARIANCE]:
                        pass
            
            window_frames = window_end - window_start
            last_window_start = window_start
            last_window_end = window_end

            assert window_frames > 0 
            cmvn_feat[t].add_(cur_sum, alpha=-1. / window_frames)

            if opts[NORMALIZE_VARIANCE]:
                pass
        
        return cmvn_feat

    def vad(self, raw_feat): #Note: different from the vad method in ivector_PLDA_helper.py

        vad_energy_threshold = 5.5 # default 5.0
        vad_energy_mean_scale = 0.5 # default 0.5
        # vad_frames_context = 0 # default 0, egs/voxceleb/v1
        vad_frames_context = 2 # egs/voxceleb/v2
        # vad_proportion_threshold = 0.6 # default 0.6, egs/voxceleb/v1
        vad_proportion_threshold = 0.12 # egs/voxceleb/v2

        assert vad_energy_mean_scale >= 0.0
        assert vad_frames_context >= 0
        assert vad_proportion_threshold > 0.0
        assert vad_proportion_threshold < 1.0

        T = raw_feat.size()[0]
        log_energy = raw_feat[:, 0]

        energy_threshold = vad_energy_threshold
        energy_threshold += vad_energy_mean_scale * torch.mean(log_energy)

        binary = (log_energy > energy_threshold).float()
        vad_result = torch.zeros_like(log_energy)
        for i in range(T):
            binary_part = binary[max(0, i-vad_frames_context):min(T, i+vad_frames_context+1)]
            if torch.mean(binary_part) >= vad_proportion_threshold:
                vad_result[i] = 1.0 

        return vad_result

    def select_voiced_frames(self, cmvn_feat, vad_result):
        num_frames = cmvn_feat.size()[0]
        assert num_frames == vad_result.size()[0]

        voiced_index = (vad_result == 1.0).nonzero().view(-1)
        assert voiced_index.shape[0] > 0
        final_feats = cmvn_feat[voiced_index, :]
        return final_feats


    def to(self, device):
        if device == self.device:
            return
        self.device = device
        self.extractor.to(self.device)
        self.plda.to(self.device)
        self.xvector_mean = self.xvector_mean.to(self.device)
        self.transform_mat = self.transform_mat.to(self.device)

    def copy(self, device):
        copy_helper = copy.deepcopy(self)  
        copy_helper.to(device)
        return copy_helper