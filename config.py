import argparse
import os
import sys
parser = argparse.ArgumentParser()
parser.add_argument("--hidden_dim", type=int, default=300)
parser.add_argument("--emb_dim", type=int, default=300)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.00015)
parser.add_argument("--max_grad_norm", type=float, default=5.0)
parser.add_argument("--max_enc_steps", type=int, default=400)
parser.add_argument("--max_dec_steps", type=int, default=20)
parser.add_argument("--min_dec_steps", type=int, default=5)
parser.add_argument("--beam_size", type=int, default=5)
parser.add_argument("--save_path", type=str, default="save/")
parser.add_argument("--save_path_dataset", type=str, default="save/")
parser.add_argument("--cuda", action="store_false")
parser.add_argument("--test", action="store_true")
parser.add_argument("--model", type=str, default='RLtrans')
parser.add_argument("--use_oov_emb", action="store_false")
parser.add_argument("--pretrain_emb", action="store_false")

## transformer 
parser.add_argument("--hop", type=int, default=4)
parser.add_argument("--heads", type=int, default=4)#4
parser.add_argument("--depth", type=int, default=64)#DD=64 Emp=256
parser.add_argument("--filter", type=int, default=2048)#2048 Emp=1024

arg = parser.parse_args()
print(arg)
model = arg.model

# Hyperparameters
hidden_dim= arg.hidden_dim
emb_dim= arg.emb_dim
batch_size= arg.batch_size
lr=arg.lr


max_enc_steps=arg.max_enc_steps
max_dec_step= max_dec_steps=arg.max_dec_steps

min_dec_steps=arg.min_dec_steps 
beam_size=arg.beam_size

adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=arg.max_grad_norm

USE_CUDA = arg.cuda
cov_loss_wt = 1.0
lr_coverage=0.15
eps = 1e-12
epochs = 10000
UNK_idx = 0
PAD_idx = 1
EOS_idx = 2
SOS_idx = 3

json_file = os.path.join(sys.path[0]+'/generated.jsonl')
emb_file = os.path.join(sys.path[0]+"/glove/glove.6B.{}d.txt".format(str(emb_dim)))
preptrained = arg.pretrain_emb

save_path = arg.save_path
save_path_dataset = arg.save_path_dataset

test = arg.test
if(not test):
    save_path_dataset = save_path


### transformer 
hop = arg.hop
heads = arg.heads
depth = arg.depth
filter = arg.filter

#dataset = 'dd'
UNK_idx = 0
PAD_idx = 1
EOS_idx = 2
SOS_idx = 3
USR_idx = 4
SYS_idx = 5
CLS_idx = 6
CLS1_idx = 7
Y_idx = 8
#full_kl_step = 12000#12000
#kl_ceiling = 0.05#0.05
#aux_ceiling = 1
#num_var_layers = arg.num_var_layers
