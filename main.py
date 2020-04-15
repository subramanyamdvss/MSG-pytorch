import torch
import argparse
from utils import projection, get_syn_data, gram_schmidt, incremental_update, rank1_update, stochastic_power_update, msg, objective, device_templ
from torch import nn
import torch.utils.data
from torch.utils.data import TensorDataset, DataLoader
import tqdm

def main():

	parser = argparse.ArgumentParser('PCA with Pytorch')
	parser.add_argument('--method',default='l2rmsg',help="can be among ['l1rmsg','l2rmsg','l12rmsg','msg','incremental','original','sgd']")
	parser.add_argument('--subspace_ratio',type=float,default=0.5,help='k/d ratio')
	parser.add_argument('--beta',type=float,default=0.5,help='regularization const for l1')
	parser.add_argument('--lambda',type=float,default=1e-3,help='regularization const for l2')
	parser.add_argument('--eta',type=float,default=1,help='learning rate')
	parser.add_argument('--eps',type=float,default=1e-6,help='threshold on norm for rank-1 update of non-trivial components')
	parser.add_argument('--nepochs',type=int,default=20,help='no. of epochs')
	parser.add_argument('--cuda',action='store_true',default=False,help='To train on GPU')
	parser.add_argument('--verbose',action='store_true',default=False,help='if true then progress bar gets printed')
	parser.add_argument('--log_interval',type=int,default=1,help='log interval in epochs')
	args = parser.parse_args()
	device = lambda tens: device_templ(tens,args.cuda)

	torch.manual_seed(7)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(7)

	print('-----------------------TRAINING {}--------------------'.format(args.method.upper()))

	X_train, X_val, X_test = get_syn_data(device=device)
	k = int(args.subspace_ratio*X_train.size(1))
	d = X_train.size(1)
	epochs_iter = range(args.nepochs)
	epochs_iter = tqdm.tqdm(epochs_iter) if args.verbose else epochs_iter

	for epoch in epochs_iter:
		iterator = DataLoader(TensorDataset(X_train),shuffle=True)
		iterator = tqdm.tqdm(iterator) if args.verbose else iterator
		for x in iterator:
			x = x[0].squeeze()
			method = args.method
			if method in ['l1rmsg','l2rmsg','l12rmsg','msg']:
				if epoch==0:
					U = device(torch.zeros(d,k).float())
					S = device(torch.zeros(k).float())
				U,S = msg(U,S,k,x,args.eta,args.eps,args.beta)
			elif method in 'incremental':
				if epoch==0:
					U = device(torch.zeros(d,k).float())
					S = device(torch.zeros(k).float())
				U,S = incremental_update(U,S,x,max_rank = None)
				# print(U,S)
				U,S = U[:,:k],S[:k]
			elif method in 'sgd':
				if epoch==0:
					U = gram_schmidt(nn.init.uniform_(device(torch.zeros(d,k))))
				U = stochastic_power_update(U,x,args.eta)
				U = gram_schmidt(U)
			elif method in 'original':
				_,S,V = torch.svd(X_train)
				U = V[:,:k]
				break
		if method in ['l1rmsg','l2rmsg','l12rmsg','msg']:
			finalU = U[:,:k]
		elif method in 'incremental':
			finalU = U
		elif method in 'sgd':
			finalU = U
		elif method in 'original':
			finalU = U
		if method in 'original':
			break
		if epoch %args.log_interval ==0:
			# print(epoch)
			print('Objective(higher is good): TRAIN {:.4f} VALIDATION {:.4f}'.format(objective(X_train,finalU),objective(X_val,finalU)))
	method = args.method
	print('Objective(higher is good): TRAIN {:.4f} VALIDATION {:.4f}'.format(objective(X_train,finalU),objective(X_val,finalU)))

if __name__ == '__main__':
	main()