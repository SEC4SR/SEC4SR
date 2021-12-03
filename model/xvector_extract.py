
import torch
import time
from model.xvecTDNN import xvecTDNN


class xvectorExtractor(object):
	def __init__(self, xvnet_ckpt, device="cpu"):

		self.device = device
		if isinstance(xvnet_ckpt, str): # when xvnet_ckpt is the path of the ckpt
			ori_dict = torch.load(xvnet_ckpt)
			num_spks = ori_dict['fc3.bias'].shape[0]
			self.extractor = xvecTDNN(numSpkrs=num_spks)
			my_dict = self.extractor.state_dict()
			update_dict = {k:v for k, v in ori_dict.items() if k in my_dict}
			self.extractor.load_state_dict(update_dict)
		elif isinstance(xvnet_ckpt, xvecTDNN): # when xvnet_ckpt is directly an instance of xvecTDNN
			self.extractor = xvnet_ckpt
		else:
			raise NotImplementedError('Invalid parameter, plz provide a ckpt or a xvecTDNN instance')
		self.extractor.eval().to(self.device)

		self.xvector_dim = 512 ## HARD CODE HERE

	def Extract(self, feat):
		'''
		feat: num_frames, n_dim
		'''
		return self.extractor.embedding(feat.unsqueeze(0).transpose(1, 2)).squeeze(0)

	def LengthNormalization(self, ivector, expected_length):
		# input_norm = torch.norm(ivector)
		input_norm = torch.norm(ivector).item()
		if input_norm == 0:
			print('Zero ivector!')
			exit(0)
		radio = expected_length / input_norm
		ivector = ivector * radio

		return ivector

	def SubtractGlobalMean(self, ivector, mean):
		ivector = ivector-mean
		return ivector
	
	def to(self, device):

		if device == self.device: 
			return

		self.device = device
		self.extractor.to(self.device)