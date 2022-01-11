import torch
import argparse
import librosa
import os
import numpy as np
from istft import ISTFT
from aia_trans import aia_complex_trans_mag, aia_complex_trans_ri, dual_aia_trans_merge_crm
import soundfile as sf

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
def enhance(args):
    model = dual_aia_trans_merge_crm()
    checkpoint = torch.load(args.Model_path)
    model.load_state_dict(checkpoint)
    print(model)
    model.eval()
    model.cuda()

    with torch.no_grad():
        cnt = 0
        mix_file_path = args.mix_file_path
        esti_file_path = args.esti_file_path        
        file_list = os.listdir(mix_file_path)
        istft = ISTFT(filter_length=320, hop_length=160, window='hanning')
        for file_id in file_list:
            feat_wav, _ = sf.read(os.path.join(mix_file_path, file_id))
            c = np.sqrt(len(feat_wav) / np.sum((feat_wav ** 2.0)))
            feat_wav = feat_wav * c
            wav_len = len(feat_wav)
            frame_num = int(np.ceil((wav_len - 320 + 320) / 160 + 1))
            fake_wav_len = (frame_num - 1) * 160 + 320 - 320
            left_sample = fake_wav_len - wav_len
            feat_wav = torch.FloatTensor(np.concatenate((feat_wav, np.zeros([left_sample])), axis=0))
            feat_x = torch.stft(feat_wav.unsqueeze(dim=0), n_fft=320, hop_length=160, win_length=320,
                                window=torch.hann_window(320)).permute(0, 3, 2, 1)
            noisy_phase = torch.atan2(feat_x[:, -1, :, :], feat_x[:, 0, :, :])
            feat_x_mag = (torch.norm(feat_x, dim=1)) ** 0.5
            feat_x = torch.stack((feat_x_mag * torch.cos(noisy_phase), feat_x_mag * torch.sin(noisy_phase)), dim=1)
            esti_x = model(feat_x.cuda())
            esti_mag, esti_phase = torch.norm(esti_x, dim=1), torch.atan2(esti_x[:, -1, :, :],
                                                                             esti_x[:, 0, :, :])
            esti_mag = esti_mag ** 2
            esti_com = torch.stack((esti_mag * torch.cos(esti_phase), esti_mag * torch.sin(esti_phase)), dim=1)
            esti_com = esti_com.cpu()
            esti_utt = istft(esti_com).squeeze().numpy()
            esti_utt = esti_utt[:wav_len]
            esti_utt = esti_utt / c
            os.makedirs(os.path.join(esti_file_path), exist_ok=True)
            sf.write(os.path.join(esti_file_path, file_id), esti_utt, args.fs)
            print(' The %d utterance has been decoded!' % (cnt + 1))
            cnt += 1

def enhance_ri(args):
    model = aia_complex_trans_ri()
    checkpoint = torch.load(args.Model_path)['model_state_dict']
    model.load_state_dict(checkpoint)
    print(model)
    model.eval()
    model.cuda()

    with torch.no_grad():
        cnt = 0
        mix_file_path = args.mix_file_path
        esti_file_path = args.esti_file_path        
        file_list = os.listdir(mix_file_path)
        istft = ISTFT(filter_length=320, hop_length=160, window='hanning')
        for file_id in file_list:
            feat_wav, _ = sf.read(os.path.join(mix_file_path, file_id))
            c = np.sqrt(len(feat_wav) / np.sum((feat_wav ** 2.0)))
            feat_wav = feat_wav * c
            wav_len = len(feat_wav)
            frame_num = int(np.ceil((wav_len - 320 + 320) / 160 + 1))
            fake_wav_len = (frame_num - 1) * 160 + 320 - 320
            left_sample = fake_wav_len - wav_len
            feat_wav = torch.FloatTensor(np.concatenate((feat_wav, np.zeros([left_sample])), axis=0))
            feat_x = torch.stft(feat_wav.unsqueeze(dim=0), n_fft=320, hop_length=160, win_length=320,
                                window=torch.hann_window(320)).permute(0, 3, 2, 1)
            noisy_phase = torch.atan2(feat_x[:, -1, :, :], feat_x[:, 0, :, :])
            feat_x_mag = (torch.norm(feat_x, dim=1)) ** 0.5
            feat_x = torch.stack((feat_x_mag * torch.cos(noisy_phase), feat_x_mag * torch.sin(noisy_phase)), dim=1)
            esti_x = model(feat_x.cuda())
            esti_mag, esti_phase = torch.norm(esti_x, dim=1), torch.atan2(esti_x[:, -1, :, :],
                                                                             esti_x[:, 0, :, :])
            esti_mag = esti_mag ** 2
            esti_com = torch.stack((esti_mag * torch.cos(esti_phase), esti_mag * torch.sin(esti_phase)), dim=1)
            esti_com = esti_com.cpu()
            esti_utt = istft(esti_com).squeeze().numpy()
            esti_utt = esti_utt[:wav_len]
            esti_utt = esti_utt / c
            os.makedirs(os.path.join(esti_file_path), exist_ok=True)
            sf.write(os.path.join(esti_file_path, file_id), esti_utt, args.fs)
            print(' The %d utterance has been decoded!' % (cnt + 1))
            cnt += 1



if __name__ == '__main__':

    
    
    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str, default='/home/yuguochen/DNS_NONBLIND_TEST/no_reverb_noisy/')
    parser.add_argument('--esti_file_path', type=str, default='./estimated_audio/dns_nonblind_test/aia_merge_dns300_best')
    parser.add_argument('--snr', type=list, default=[-5, 0, 5, 10, 15, 20])     #  -5  -2  0  2  5
    parser.add_argument('--fs', type=int, default=16000,
                        help='The sampling rate of speech')
    parser.add_argument('--Model_path', type=str, default='./BEST_MODEL/aia_merge_dns300.pth.tar',
                        help='The place to save best model')
    args = parser.parse_args()
    print(args)
    enhance(args=args)    
     
