import numpy as np
import pandas as pd
import os
from utils import *
from sklearn.model_selection import StratifiedKFold, KFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, RMSprop
from transformers import BertTokenizer
from model import Eyettention_readerID
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.nn.functional import cross_entropy, softmax
from collections import deque
import pickle
import json
import matplotlib.pyplot as plt
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='run uniform baseline')
	parser.add_argument(
		'--test_mode',
		help='test mode: text',
		type=str,
		default='text'
	)
	parser.add_argument(
		'--atten_type',
		help='attention type',
		type=str,
		default='local-g'
	)
	parser.add_argument(
		'--save_data_folder',
		help='folder path for saving results',
		type=str,
		default='./results/BSC/'
	)
	parser.add_argument(
		'--gpu',
		help='gpu index',
		type=int,
		default=3
	)
	parser.add_argument(
		'--emb_size',
		help='readerID embedding size',
		type=int,
		default=16
	)
	args = parser.parse_args()
	gpu = args.gpu

	torch.set_default_tensor_type('torch.FloatTensor')
	availbl = torch.cuda.is_available()
	print(torch.cuda.is_available())
	if availbl:
		device = f'cuda:{gpu}'
	else:
		device = 'cpu'
	torch.cuda.set_device(gpu)

	cf = {"model_pretrained": "bert-base-chinese",
			"lr": 1e-3,
			"max_grad_norm": 10,
			"n_epochs": 1000,
			"n_folds": 5,
			"dataset": 'BSC',
			"atten_type": args.atten_type,
			"subid_emb_size": args.emb_size,
			"batch_size": 256,
			"max_sn_len": 27, #include start token and end token
			"max_sp_len": 40, #include start token and end token
			"norm_type": "z-score",
			"earlystop_patience": 20,
			}

	#Encode the label into interger categories, setting the exclusive category 'cf["max_sn_len"]-1' as the end sign
	le = LabelEncoder()
	le.fit(np.append(np.arange(-cf["max_sn_len"]+3, cf["max_sn_len"]-1), cf["max_sn_len"]-1))
	#le.classes_

	#load corpus
	word_info_df, pos_info_df, eyemovement_df = load_corpus(cf["dataset"])
	#Make list with sentence index
	sn_list = np.unique(eyemovement_df.sn.values).tolist()
	#Make list with reader index
	reader_list = np.unique(eyemovement_df.id.values).tolist()

	#Split training&test sets by text
	print('Start evaluating on new sentences.')
	split_list = sn_list


	n_folds = cf["n_folds"]
	kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
	fold_indx = 0
	for train_idx, test_idx in kf.split(split_list):
		loss_dict = {'val_loss':[], 'train_loss':[], 'test_ll':[]}
		list_train = [split_list[i] for i in train_idx]
		list_test = [split_list[i] for i in test_idx]

		# create train validation split for training the models:
		kf_val = KFold(n_splits=n_folds, shuffle=True, random_state=0)
		for train_index, val_index in kf_val.split(list_train):
			# we only evaluate a single fold
			break
		list_train_net = [list_train[i] for i in train_index]
		list_val_net = [list_train[i] for i in val_index]

		sn_list_train = list_train_net
		sn_list_val = list_val_net
		sn_list_test = list_test
		reader_list_train, reader_list_val, reader_list_test = reader_list, reader_list, reader_list

		#initialize tokenizer
		tokenizer = BertTokenizer.from_pretrained(cf['model_pretrained'])
		#Preparing batch data
		dataset_train = BSCdataset(word_info_df, eyemovement_df, cf, reader_list_train, sn_list_train, tokenizer)
		train_dataloaderr = DataLoader(dataset_train, batch_size = cf["batch_size"], shuffle = True, drop_last=True)

		dataset_val = BSCdataset(word_info_df, eyemovement_df, cf, reader_list_val, sn_list_val, tokenizer)
		val_dataloaderr = DataLoader(dataset_val, batch_size = cf["batch_size"], shuffle = False, drop_last=True)

		dataset_test = BSCdataset(word_info_df, eyemovement_df, cf, reader_list_test, sn_list_test, tokenizer)
		test_dataloaderr = DataLoader(dataset_test, batch_size = cf["batch_size"], shuffle = False, drop_last=False)

		#z-score normalization for gaze features
		fix_dur_mean, fix_dur_std = calculate_mean_std(dataloader=train_dataloaderr, feat_key="sp_fix_dur", padding_value=0, scale=1000)
		landing_pos_mean, landing_pos_std = calculate_mean_std(dataloader=train_dataloaderr, feat_key="sp_landing_pos", padding_value=0)
		sn_word_len_mean, sn_word_len_std = calculate_mean_std(dataloader=train_dataloaderr, feat_key="sn_word_len")

		# load model here
		dnn = Eyettention_readerID(cf)

		#training
		episode = 0
		optimizer = Adam(dnn.parameters(), lr=cf["lr"])
		dnn.train()
		dnn.to(device)
		av_score = deque(maxlen=100)
		old_score = 1e10
		save_ep_couter = 0
		print('Start training')
		for episode_i in range(episode, cf["n_epochs"]+1):
			dnn.train()
			print('episode:', episode_i)
			counter = 0
			for batchh in train_dataloaderr:
				counter += 1
				batchh.keys()
				sn_input_ids = batchh["sn_input_ids"].to(device)
				sn_attention_mask = batchh["sn_attention_mask"].to(device)
				sp_input_ids = batchh["sp_input_ids"].to(device)
				sp_attention_mask = batchh["sp_attention_mask"].to(device)
				sp_pos = batchh["sp_pos"].to(device)
				sp_landing_pos = batchh["sp_landing_pos"].to(device)
				sp_fix_dur = (batchh["sp_fix_dur"]/1000).to(device)
				sn_word_len = batchh["sn_word_len"].to(device)
				sub_id = batchh["sub_id"].to(device)

				#normalize gaze features
				mask = ~torch.eq(sp_fix_dur, 0)
				sp_fix_dur = (sp_fix_dur-fix_dur_mean)/fix_dur_std * mask
				sp_landing_pos = (sp_landing_pos - landing_pos_mean)/landing_pos_std * mask
				sp_fix_dur = torch.nan_to_num(sp_fix_dur)
				sp_landing_pos = torch.nan_to_num(sp_landing_pos)
				sn_word_len = (sn_word_len - sn_word_len_mean)/sn_word_len_std
				sn_word_len = torch.nan_to_num(sn_word_len)

				# zero old gradients
				optimizer.zero_grad()
				# predict output with DNN
				dnn_out, atten_weights = dnn(sn_emd=sn_input_ids,
											sn_mask=sn_attention_mask,
											sp_emd=sp_input_ids,
											sp_pos=sp_pos,
											word_ids_sn=None,
											word_ids_sp=None,
											sp_fix_dur=sp_fix_dur,
											sp_landing_pos=sp_landing_pos,
											sn_word_len = sn_word_len,
											sub_id = sub_id)

				dnn_out = dnn_out.permute(0,2,1)              #[batch, dec_o_dim, step]

				#prepare label and mask
				pad_mask, label = load_label(sp_pos, cf, le, device)
				loss = nn.CrossEntropyLoss(reduction="none")
				batch_error = torch.mean(torch.masked_select(loss(dnn_out, label), ~pad_mask))

				# backpropagate loss
				batch_error.backward()
				# clip gradients
				gradient_clipping(dnn, cf["max_grad_norm"])

				#learn
				optimizer.step()
				av_score.append(batch_error.to('cpu').detach().numpy())
				print('counter:',counter)
				print('\rSample {}\tAverage Error: {:.10f} '.format(counter, np.mean(av_score)), end=" ")
			loss_dict['train_loss'].append(np.mean(av_score))

			val_loss = []
			dnn.eval()
			for batchh in val_dataloaderr:
				with torch.no_grad():
					sn_input_ids_val = batchh["sn_input_ids"].to(device)
					sn_attention_mask_val = batchh["sn_attention_mask"].to(device)
					sp_input_ids_val = batchh["sp_input_ids"].to(device)
					sp_attention_mask_val = batchh["sp_attention_mask"].to(device)
					sp_pos_val = batchh["sp_pos"].to(device)
					sp_landing_pos_val = batchh["sp_landing_pos"].to(device)
					sp_fix_dur_val = (batchh["sp_fix_dur"]/1000).to(device)
					sn_word_len_val = batchh["sn_word_len"].to(device)
					sub_id_val = batchh["sub_id"].to(device)

					#normalize gaze features
					mask = ~torch.eq(sp_fix_dur_val, 0)
					sp_fix_dur_val = (sp_fix_dur_val-fix_dur_mean)/fix_dur_std * mask
					sp_landing_pos_val = (sp_landing_pos_val - landing_pos_mean)/landing_pos_std * mask
					sp_fix_dur_val = torch.nan_to_num(sp_fix_dur_val)
					sp_landing_pos_val = torch.nan_to_num(sp_landing_pos_val)
					sn_word_len_val = (sn_word_len_val - sn_word_len_mean)/sn_word_len_std
					sn_word_len_val = torch.nan_to_num(sn_word_len_val)

					dnn_out_val, atten_weights_val = dnn(sn_emd=sn_input_ids_val,
														sn_mask=sn_attention_mask_val,
														sp_emd=sp_input_ids_val,
														sp_pos=sp_pos_val,
														word_ids_sn=None,
														word_ids_sp=None,
														sp_fix_dur=sp_fix_dur_val,
														sp_landing_pos=sp_landing_pos_val,
														sn_word_len = sn_word_len_val,
														sub_id = sub_id_val)

					dnn_out_val = dnn_out_val.permute(0,2,1)              #[batch, dec_o_dim, step

					#prepare label and mask
					loss = nn.CrossEntropyLoss(reduction="none")
					pad_mask_val, label_val = load_label(sp_pos_val, cf, le, device)
					batch_error_val = torch.mean(torch.masked_select(loss(dnn_out_val, label_val), ~pad_mask_val))
					val_loss.append(batch_error_val.detach().to('cpu').numpy())
			print('\nvalidation loss is {} \n'.format(np.mean(val_loss)))
			loss_dict['val_loss'].append(np.mean(val_loss))

			if np.mean(val_loss) < old_score:
				# save model if val loss is smallest
				torch.save(dnn.state_dict(), '{}/CELoss_BSC_text_eyettention_readerID_{}_emb{}_newloss_fold{}.pth'.format(args.save_data_folder, args.atten_type, args.emb_size, fold_indx))
				old_score= np.mean(val_loss)
				print('\nsaved model state dict\n')
				save_ep_couter = episode_i
			else:
				#early stopping
				if episode_i - save_ep_couter >= cf["earlystop_patience"]:
					break


		#evaluation
		dnn.eval()
		res_llh=[]
		dnn.load_state_dict(torch.load(os.path.join(args.save_data_folder,f'CELoss_BSC_text_eyettention_readerID_{args.atten_type}_emb{args.emb_size}_newloss_fold{fold_indx}.pth'), map_location='cpu'))
		dnn.to(device)
		batch_indx = 0
		for batchh in test_dataloaderr:
			with torch.no_grad():
				sn_input_ids_test = batchh["sn_input_ids"].to(device)
				sn_attention_mask_test = batchh["sn_attention_mask"].to(device)
				sp_input_ids_test = batchh["sp_input_ids"].to(device)
				sp_attention_mask_test = batchh["sp_attention_mask"].to(device)
				sp_pos_test = batchh["sp_pos"].to(device) # 28: '<Sep>', 29: '<'Pad'>'
				sp_landing_pos_test = batchh["sp_landing_pos"].to(device)
				sp_fix_dur_test = (batchh["sp_fix_dur"]/1000).to(device)
				sn_word_len_test = batchh["sn_word_len"].to(device)
				sub_id_test = batchh["sub_id"].to(device)

				#normalize gaze features
				mask = ~torch.eq(sp_fix_dur_test, 0)
				sp_fix_dur_test = (sp_fix_dur_test-fix_dur_mean)/fix_dur_std * mask
				sp_landing_pos_test = (sp_landing_pos_test - landing_pos_mean)/landing_pos_std * mask
				sp_fix_dur_test = torch.nan_to_num(sp_fix_dur_test)
				sp_landing_pos_test = torch.nan_to_num(sp_landing_pos_test)
				sn_word_len_test = (sn_word_len_test - sn_word_len_mean)/sn_word_len_std
				sn_word_len_test = torch.nan_to_num(sn_word_len_test)

				dnn_out_test, atten_weights_test = dnn(sn_emd=sn_input_ids_test,
														sn_mask=sn_attention_mask_test,
														sp_emd=sp_input_ids_test,
														sp_pos=sp_pos_test,
														word_ids_sn=None,
														word_ids_sp=None,
														sp_fix_dur=sp_fix_dur_test,
														sp_landing_pos=sp_landing_pos_test,
														sn_word_len = sn_word_len_test,
														sub_id = sub_id_test)

				#We do not use nn.CrossEntropyLoss here to calculate the likelihood because it combines nn.LogSoftmax and nn.NLL,
				#while nn.LogSoftmax returns a log value based on e, we want 2 instead
				#m = nn.LogSoftmax(dim=2) -- base e, we want base 2
				m = nn.Softmax(dim=2)
				dnn_out_test = m(dnn_out_test).detach().to('cpu').numpy()

				#prepare label and mask
				pad_mask_test, label_test = load_label(sp_pos_test, cf, le, 'cpu')
				pred = dnn_out_test.argmax(axis=2)
				#compute log likelihood for the batch samples
				res_batch = eval_log_llh(dnn_out_test, label_test, pad_mask_test)
				res_llh.append(np.array(res_batch))

				batch_indx +=1

		res_llh = np.concatenate(res_llh).ravel()
		loss_dict['test_ll'].append(res_llh)
		loss_dict['fix_dur_mean'] = fix_dur_mean
		loss_dict['fix_dur_std'] = fix_dur_std
		loss_dict['landing_pos_mean'] = landing_pos_mean
		loss_dict['landing_pos_std'] = landing_pos_std
		loss_dict['sn_word_len_mean'] = sn_word_len_mean
		loss_dict['sn_word_len_std'] = sn_word_len_std
		print('\nTest likelihood is {} \n'.format(np.mean(res_llh)))
		#save results
		with open('{}/res_BSC_NRS_eyettention_readerID_{}_emb{}_Fold{}.pickle'.format(args.save_data_folder, args.atten_type, args.emb_size, fold_indx), 'wb') as handle:
			pickle.dump(loss_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

		fold_indx += 1
