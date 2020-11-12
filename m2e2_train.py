# This script is used to finetune AND evaluate on the M2E2 dataset

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch as th
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
from args import get_args
import random
import os
from model import Net
from metrics import compute_metrics, print_computed_metrics
from loss import MaxMarginRankingLoss
from gensim.models.keyedvectors import KeyedVectors
import pickle
from m2e2_dataloader import M2E2DataLoader

args = get_args()
if args.verbose:
	print(args)

# predefining random initial seeds
th.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if args.checkpoint_dir != '' and not(os.path.isdir(args.checkpoint_dir)):
	os.mkdir(args.checkpoint_dir)


print('Loading word vectors: {}'.format(args.word2vec_path))
we = KeyedVectors.load_word2vec_format(args.word2vec_path, binary=True)
print('done')


if args.m2e2:
	dataset = M2E2DataLoader(
		csv=args.m2e2_train_csv_path,
		sentences=arg.m2e2_sentences_path,
		we=we,
		max_words=args.max_words,
		we_dim=args.we_dim,
	)

dataset_size = len(dataset)
dataloader = DataLoader(
	dataset,
	batch_size=args.batch_size,
	num_workers=args.num_thread_reader,
	shuffle=True,
	batch_sampler=None,
	drop_last=True,
)

if args.eval_m2e2:
	dataset_val_m2e2 = M2E2DataLoader(
		csv=args.m2e2_test_csv_path,
		sentences=arg.m2e2_sentences_path,
		we=we,
		max_words=args.max_words,
		we_dim=args.we_dim,
	)
	dataloader_val_m2e2 = DataLoader(
		dataset_val_m2e2,
		batch_size=args.batch_size_val,
		num_workers=args.num_thread_reader,
		shuffle=False,
	)

net = Net(
	video_dim=args.feature_dim,
	embd_dim=args.embd_dim,
	we_dim=args.we_dim,
	n_pair=args.n_pair,
	max_words=args.max_words,
	sentence_dim=args.sentence_dim,
)
net.train()
# Optimizers + Loss
loss_op = MaxMarginRankingLoss(
	margin=args.margin,
	negative_weighting=args.negative_weighting,
	batch_size=args.batch_size,
	n_pair=args.n_pair,
	hard_negative_rate=args.hard_negative_rate,
)

net.cuda()
loss_op.cuda()

if args.pretrain_path != '':
	net.load_checkpoint(args.pretrain_path)

optimizer = optim.Adam(net.parameters(), lr=args.lr)

if args.verbose:
	print('Starting training loop ...')

def TrainOneBatch(model, opt, data, loss_fun):
	text = data['text'].cuda()
	video = data['video'].cuda()
	video = video.view(-1, video.shape[-1])
	text = text.view(-1, text.shape[-2], text.shape[-1])
	opt.zero_grad()
	with th.set_grad_enabled(True):
		sim_matrix = model(video, text)
		loss = loss_fun(sim_matrix)
	loss.backward()
	opt.step()
	return loss.item()

def Eval_retrieval(model, eval_dataloader, dataset_name):
	model.eval()
	print('Evaluating Text-Video retrieval on {} data'.format(dataset_name))
	with th.no_grad():
		for i_batch, data in enumerate(eval_dataloader):
			text = data['text'].cuda()
			video = data['video'].cuda()
			m = model(video, text)
			m  = m.cpu().detach().numpy()
			metrics = compute_metrics(m)
			print_computed_metrics(metrics)

for epoch in range(args.epochs):
	running_loss = 0.0
	if args.eval_m2e2:
		Eval_retrieval(net, dataloader_val_m2e2, 'M2E2')
	if args.verbose:
		print('Epoch: %d' % epoch)
	for i_batch, sample_batch in enumerate(dataloader):
		batch_loss = TrainOneBatch(net, optimizer, sample_batch, loss_op)
		running_loss += batch_loss
		if (i_batch + 1) % args.n_display == 0 and args.verbose:
			print('Epoch %d, Epoch status: %.4f, Training loss: %.4f' %
			(epoch + 1, args.batch_size * float(i_batch) / dataset_size,
			running_loss / args.n_display))
			running_loss = 0.0
	for param_group in optimizer.param_groups:
		param_group['lr'] *= args.lr_decay
	if args.checkpoint_dir != '':
		path = os.path.join(args.checkpoint_dir, 'e{}.pth'.format(epoch + 1))
		net.save_checkpoint(path)

if args.eval_m2e2:
	Eval_retrieval(net, dataloader_val_m2e2, 'M2E2')
