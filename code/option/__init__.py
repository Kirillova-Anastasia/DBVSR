import argparse
from option import template

parser = argparse.ArgumentParser(description='VideoSR')

parser.add_argument('--template', default='DBVSR',
                    help='You can set various templates in options.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=0,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default='../../Dataset',
                    help='dataset directory')
parser.add_argument('--dir_data_test', type=str, default='../../Dataset',
                    help='dataset directory')
parser.add_argument('--data_train', type=str, default='DIV2K',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='Set5',
                    help='test dataset name')
parser.add_argument('--process', action='store_true',
                    help='if True, load all dataset at once at RAM')
parser.add_argument('--patch_size', type=int, default=64,
                    help='output patch size')
parser.add_argument('--size_must_mode', type=int, default=1,
                    help='the size of the network input must mode this number')
parser.add_argument('--rgb_range', type=int, default=1,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')
parser.add_argument('--test_padding', type=int, default=0,
                    help='the padding when test, then crop the extra part from output')
parser.add_argument('--n_sequences', type=int, default=3,
                    help='the sequence number of every iteration')
parser.add_argument('--n_frames_per_video', type=int, default=100,
                    help='the number of images used in every video')
parser.add_argument('--scale', type=int, default=4,
                    help='scale factor of the model')

# Model specifications
parser.add_argument('--model', default='RCAN',
                    help='model name')
parser.add_argument('--pre_train', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_resgroups', type=int, default=10,
                    help='number of res_groups used')
parser.add_argument('--n_resblocks', type=int, default=20,
                    help='number of res_blocks used')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of channels used in feature')
parser.add_argument('--reduction', type=int, default=16,
                    help='reduction factor')
parser.add_argument('--res_scale', type=int, default=1,
                    help='reduction factor')


# Training specifications
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=500,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=1,
                    help='input batch size for training')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')

# Optimization specifications
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate of reconstruction-net and kernel-net')
parser.add_argument('--pwc_lr', type=float, default=1e-6,
                    help='learning rate of flow-net')
parser.add_argument('--lr_decay', type=int, default=200,  # lr_decay:200->50
                    help='learning rate decay per N epochs')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--mid_loss_weight', type=float, default=1.,
                    help='the weight of mid loss in trainer')

# Log specifications
parser.add_argument('--experiment_dir', type=str, default='../experiment/',
                    help='file name to save')
parser.add_argument('--pretrain_models_dir', type=str, default='../pretrain_models/',
                    help='file name to save')
parser.add_argument('--save', type=str, default='default_save',
                    help='file name to save')
parser.add_argument('--save_middle_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--load', type=str, default='.',
                    help='file name to load')
parser.add_argument('--resume', action='store_true',
                    help='resume from the latest if true')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_images', default=True, action='store_true',
                    help='save images')
parser.add_argument('--test_model_path', type=str, default='',
                    help='path to model')
parser.add_argument('--save_dir_path', type=str, default='',
                    help='where save result')

args = parser.parse_args()
template.set_template(args)

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
