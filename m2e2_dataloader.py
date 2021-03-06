from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch as th
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd
import os
import numpy as np
import re
import random
import json

from stop_words import ENGLISH_STOP_WORDS

class M2E2DataLoader(Dataset):
	"""M2E2 dataset loader."""

	def __init__(
			self,
			csv,
			sentences,
			we,
			we_dim=300,
			max_words=30
	):
		"""
		Args:
		"""
		self.csv = pd.read_csv(csv)
		self.sentences = json.load(open(sentences))
		self.we = we
		self.we_dim = we_dim
		self.max_words = max_words
		
	def __len__(self):
		return len(self.csv)

	def _zero_pad_tensor(self, tensor, size):
		if len(tensor) >= size:
			return tensor[:size]
		else:
			zero = np.zeros((size - len(tensor), self.we_dim), dtype=np.float32)
			return np.concatenate((tensor, zero), axis=0)

	def _tokenize_text(self, sentence):
		w = re.findall(r"[\w']+", str(sentence))
		return w

	def _words_to_we(self, words):
		words = [word for word in words if word in self.we.vocab and word not in ENGLISH_STOP_WORDS]
		if words:
			we = self._zero_pad_tensor(self.we[words], self.max_words)
			return th.from_numpy(we)
		else:
			return th.zeros(self.max_words, self.we_dim)

	def _get_text(self, sentences):
		rint = random.randint(0,len(sentences)-1)
		return self._words_to_we(self._tokenize_text(sentences[rint]))

	def _get_video(self, feat_2d_path,feat_3d_path):
		feat_2d = np.load(feat_2d_path)
		feat_3d = np.load(feat_3d_path)
		
		feat_2d = th.from_numpy(feat_2d).float()
		feat_2d = F.normalize(th.max(feat_2d, dim=0)[0], dim=0)

		feat_3d = th.from_numpy(feat_3d).float()
		feat_3d = F.normalize(th.max(feat_3d, dim=0)[0], dim=0)

		return th.cat((feat_2d, feat_3d))


	def __getitem__(self, idx):
		video_id = self.csv['video_id'].values[idx]
		feat_2d_path = self.csv['2d'].values[idx]
		feat_3d_path = self.csv['3d'].values[idx]
		video = self._get_video(feat_2d_path,feat_3d_path)
		text = self._get_text(self.sentences[video_id])
		return {'video': video, 'text': text, 'video_id': video_id}

class M2E2ASRDataLoader(Dataset):
	"""M2E2 dataset loader."""

	def __init__(
			self,
			csv,
			videoID2feature_path,
			we,
			we_dim=300,
			max_words=30
	):
		"""
		Args:
		"""
		self.csv = pd.read_csv(csv)
		self.videoID2feature = json.load(open(videoID2feature_path))
		self.we = we
		self.we_dim = we_dim
		self.max_words = max_words
		self.fps = {'2d': 1, '3d': 1.5}
		
	def __len__(self):
		return len(self.csv)

	def _zero_pad_tensor(self, tensor, size):
		if len(tensor) >= size:
			return tensor[:size]
		else:
			zero = np.zeros((size - len(tensor), self.we_dim), dtype=np.float32)
			return np.concatenate((tensor, zero), axis=0)

	def _tokenize_text(self, sentence):
		w = re.findall(r"[\w']+", str(sentence))
		return w

	def _words_to_we(self, words):
		words = [word for word in words if word in self.we.vocab and word not in ENGLISH_STOP_WORDS]
		if words:
			we = self._zero_pad_tensor(self.we[words], self.max_words)
			return th.from_numpy(we)
		else:
			return th.zeros(self.max_words, self.we_dim)

	def _get_text(self, text):
		return self._words_to_we(self._tokenize_text(text))

	def _get_slice(self,feat_path,start,end,dimension):
		feat = np.load(feat_path)
		start = int(start * self.fps[dimension])
		end = int(end * self.fps[dimension]) + 1
		feat_sliced = feat[start:end]
		return feat_sliced


	def _get_video(self, feat_2d_path,feat_3d_path,start,end):
		feat_2d = self._get_slice(feat_2d_path,start,end,'2d')
		feat_3d = self._get_slice(feat_3d_path,start,end,'3d')
		feat_2d = th.from_numpy(feat_2d).float()
		feat_3d = th.from_numpy(feat_3d).float()

		if len(feat_2d) < 1 or len(feat_3d) < 1: 
			print("Something wrong in {}, from {} to {}".format(feat_2d_path,start,end))
		else:
			feat_2d = F.normalize(th.max(feat_2d, dim=0)[0], dim=0)
			feat_3d = F.normalize(th.max(feat_3d, dim=0)[0], dim=0)

		return th.cat((feat_2d, feat_3d))


	def __getitem__(self, idx):
		video_id = self.csv['video_id'].values[idx]
		text = self.csv['text'].values[idx]
		start = self.csv['start'].values[idx]
		end = self.csv['end'].values[idx]

		feat_2d_path = self.videoID2feature[video_id]['2d']
		feat_3d_path = self.videoID2feature[video_id]['3d']

		video = self._get_video(feat_2d_path,feat_3d_path,start,end)
		text = self._get_text(text)
		return {'video': video, 'text': text, 'video_id': video_id}
