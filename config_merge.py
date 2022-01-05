import os

# front-end parameter settings
win_size = 320
fft_num = 320
win_shift = 160
chunk_length = 3*16000
feat_type = 'sqrt' #the compression on magnitude  # normal, sqrt, cubic, log_1x
is_conti = False
conti_path = './CP_dir/aia_merge_new/checkpoint_early_exit_46th.pth.tar'
is_pesq =  True #use pesq criterion for validate or not
# server parameter settings
json_dir = '/home/yuguochen/vbdata/Json'
file_path = '/home/yuguochen/vbdataset'
loss_dir = './LOSS/XXX.mat'
batch_size = 4
epochs = 80
lr = 5e-4
model_best_path = './BEST_MODEL/XXX.pth.tar'
check_point_path = './CP_dir/XXX'

os.makedirs('./BEST_MODEL', exist_ok=True)
os.makedirs('./LOSS', exist_ok=True)
os.makedirs(check_point_path, exist_ok=True)