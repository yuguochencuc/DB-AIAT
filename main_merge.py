import argparse
from aia_trans import dual_aia_trans_merge_crm
from train_merge import main
from config_merge import *
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

parser = argparse.ArgumentParser("gated complex convolutional recurrent neural network")
parser.add_argument('--json_dir', type=str, default=json_dir,
                    help='The directory of the dataset feat,json format')
parser.add_argument('--loss_dir', type=str, default=loss_dir,
                    help='The directory to save tr loss and cv loss')
parser.add_argument('--batch_size', type=int, default=batch_size,
                    help='The number of the batch size')
parser.add_argument('--cv_batch_size', type=int, default=batch_size,
                    help='The number of the batch size')
parser.add_argument('--epochs', type=int, default=epochs,
                    help='The number of the training epoch')
parser.add_argument('--lr', type=float, default=lr,
                    help='Learning rate of the network')
parser.add_argument('--early_stop', dest='early_stop', default=1, type=int,
                    help='Early stop training when no improvement for 10 epochs')
parser.add_argument('--half_lr', type=int, default=1,
                    help='Whether to decay learning rate to half scale')
parser.add_argument('--shuffle', type=int, default=1,
                    help='Whether to shuffle within each batch')
parser.add_argument('--num_workers', type=int, default=4,
                    help='Number of workers to generate batch')
parser.add_argument('--l2', type=float, default=1e-7,
                    help='weight decay (L2 penalty)')
parser.add_argument('--best_path', default=model_best_path,
                    help='Location to save best cv model')
parser.add_argument('--cp_path', type=str, default=check_point_path)
parser.add_argument('--print_freq', type=int, default=500,
                    help='The frequency of printing loss infomation')
parser.add_argument('--is_conti', type=bool, default=is_conti)
parser.add_argument('--conti_path', type=str, default=conti_path)

# select GPU device
train_model = dual_aia_trans_merge_crm()

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args, train_model)