# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md

from pathlib import Path
import cog
import torch
import librosa
import numpy as np
from istft import ISTFT
from aia_trans import (
    dual_aia_trans_merge_crm,
)
import soundfile as sf


SAMPLE_RATE = 16000
CHUNK_LENGTH = SAMPLE_RATE * 10  # 10 seconds
CHUNK_OVERLAP = int(SAMPLE_RATE * .1)  # 100 ms
CHUNK_HOP = CHUNK_LENGTH - CHUNK_OVERLAP


class Predictor(cog.Predictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = dual_aia_trans_merge_crm()
        checkpoint = torch.load("./BEST_MODEL/vb_aia_merge_new.pth.tar")
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.model.cuda()
        self.istft = ISTFT(filter_length=320, hop_length=160, window="hanning")

    @cog.input("audio", type=Path, help="Noisy audio input")
    def predict(self, audio):
        """Run a single prediction on the model"""

        # process audio in chunks to prevent running out of memory
        clean_chunks = []
        noisy, _ = librosa.load(str(audio), sr=SAMPLE_RATE, mono=True)
        for i in range(0, len(noisy), CHUNK_HOP):
            print(f"Processing samples {min(i + CHUNK_LENGTH, len(noisy))} / {len(noisy)}")
            noisy_chunk = noisy[i:i + CHUNK_LENGTH]
            clean_chunk = self.speech_enhance(noisy_chunk)
            clean_chunks.append(clean_chunk)

        last_clean_chunk = clean_chunks[-1]
        if len(clean_chunks) > 1 and len(last_clean_chunk) < CHUNK_OVERLAP:
            clean_chunks = clean_chunks[:-1]

        # recreate clean audio by overlapping windows
        clean = np.zeros(noisy.shape)
        hanning = np.hanning(CHUNK_OVERLAP * 2)
        fade_in = hanning[:CHUNK_OVERLAP]
        fade_out = hanning[CHUNK_OVERLAP:]
        for i, clean_chunk in enumerate(clean_chunks):
            is_first = i == 0
            is_last = i == len(clean_chunks) - 1
            if not is_first:
                clean_chunk[:CHUNK_OVERLAP] *= fade_in
            if not is_last:
                clean_chunk[CHUNK_HOP:] *= fade_out
            clean[i * CHUNK_HOP:(i + 1) * CHUNK_HOP + CHUNK_OVERLAP] += clean_chunk

        out_path = Path("/tmp/out.wav")
        sf.write(str(out_path), clean, SAMPLE_RATE)

        return out_path

    def speech_enhance(self, signal):
        with torch.no_grad():
            c = np.sqrt(len(signal) / np.sum((signal ** 2.0)))
            signal = signal * c
            wav_len = len(signal)
            frame_num = int(np.ceil((wav_len - 320 + 320) / 160 + 1))
            fake_wav_len = (frame_num - 1) * 160 + 320 - 320
            left_sample = fake_wav_len - wav_len
            signal = torch.FloatTensor(
                np.concatenate((signal, np.zeros([left_sample])), axis=0)
            )
            feat_x = torch.stft(
                signal.unsqueeze(dim=0),
                n_fft=320,
                hop_length=160,
                win_length=320,
                window=torch.hann_window(320),
            ).permute(0, 3, 2, 1)
            noisy_phase = torch.atan2(feat_x[:, -1, :, :], feat_x[:, 0, :, :])
            feat_x_mag = (torch.norm(feat_x, dim=1)) ** 0.5
            feat_x = torch.stack(
                (
                    feat_x_mag * torch.cos(noisy_phase),
                    feat_x_mag * torch.sin(noisy_phase),
                ),
                dim=1,
            )
            esti_x = self.model(feat_x.cuda())
            esti_mag, esti_phase = torch.norm(esti_x, dim=1), torch.atan2(
                esti_x[:, -1, :, :], esti_x[:, 0, :, :]
            )
            esti_mag = esti_mag ** 2
            esti_com = torch.stack(
                (esti_mag * torch.cos(esti_phase), esti_mag * torch.sin(esti_phase)),
                dim=1,
            )
            esti_com = esti_com.cpu()
            esti_utt = self.istft(esti_com).squeeze().numpy()
            esti_utt = esti_utt[:wav_len]
            esti_utt = esti_utt / c

        return esti_utt
