import argparse
import numpy as np
import time
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path
from timm.models import create_model
from src.dataset.build import build_plrs_dataset
from src import utils
import src.model.modeling
from tqdm import tqdm
import torchaudio

np.set_printoptions(threshold=np.inf)

@torch.no_grad()
def compute_plrs(model, eval_data_path, wav_path):
    # switch to evaluation mode
    model.eval()
    eval_mesh_name_list = os.listdir(eval_data_path)

    cosine_similarity = 0
    for eval_mesh_name in tqdm(eval_mesh_name_list):
        eval_mesh_path = os.path.join(eval_data_path, eval_mesh_name)

        if 'condition' in eval_mesh_name:
            eval_wav_path = os.path.join(wav_path, "_".join(eval_mesh_name.split("_")[:-5])+'.wav')
        else:
            eval_wav_path = os.path.join(wav_path, eval_mesh_name.replace('npy', 'wav'))
        assert os.path.isfile(eval_wav_path) and os.path.isfile(eval_mesh_path)
        audios = load_audio_mel(eval_wav_path)
        
        ## gt evaluation
        if 'vertices_npy' in eval_data_path:
            vertice = np.load(eval_mesh_path)[::2, :]
        else:
            vertice = np.load(eval_mesh_path)
        clip_num = min(len(audios), vertice.shape[0] - 4)
        audios = audios[:clip_num]
        audio_batch = []
        vertice_batch = []
        for i in range(clip_num//5):
            audio_batch.append(audios[i*5])
            vertice_batch.append(torch.from_numpy(vertice[i*5:i*5 + 5]))

        vertice_batch_tensor = torch.stack(vertice_batch, axis=0).cuda()
        audio_batch_tensor = torch.stack(audio_batch, dim=0).cuda()
        vertex_feature, audio_feature = model(vertice_batch_tensor.float(), audio_batch_tensor.unsqueeze(1))
        cosine_similarity += (vertex_feature @ audio_feature.t()).diag().mean()
        #print(f"{eval_mesh_name} : {(vertex_feature @ audio_feature.t()).diag().mean()}")

    plrs = cosine_similarity/len(eval_mesh_name_list)

    return plrs

@torch.no_grad()
def load_audio_mel(fname):
    audio, sample_rate = torchaudio.load(fname)
    if not sample_rate == 19200:
        audio = torchaudio.transforms.Resample(sample_rate, 19200)(audio)  # vocaset [1, 54400]
    audio_len = audio.shape[1]
    audios = []
    i = 0
    while i*640+3200 <= audio_len:
        audio_start_idx = i *640
        waveform = audio[:, audio_start_idx:audio_start_idx + 3200]
        waveform = waveform - waveform.mean()
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=16000, frame_length=8,
                                                  use_energy=False,
                                                  window_type='hanning', num_mel_bins=128, dither=0.0,
                                                  frame_shift=10)
        fbank = (fbank +4.2677393) / (4.5689974 * 2)
        audios.append(fbank)  # (Time, Freq), i.e., (seq_len, num_mel_bin)
        i += 1
    return audios

def get_args():
    parser = argparse.ArgumentParser('PLRS evaluation script for audio-driven 3D talking head generation',
                                     add_help=False)
    # Model parameters
    parser.add_argument('--model', default='plrs', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size')
    parser.add_argument('--model_path', default='', help='model checkpoint path')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--depth', required=True, type=int,
                        help='depth of video encoder')
    parser.add_argument('--depth_audio', required=True, type=int,
                        help='depth of audio encoder')
    parser.add_argument('--input_size_audio', default=256, type=int,
                        help='audio input size (default: 256, i.e., 256 ms) for backbone')
    parser.add_argument('--num_mel_bins', type=int, default=128)

    # Dataset parameters
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--wav_path', default=None, type=str,
                        help='wav path for evaluation')

    return parser.parse_args()


def main(args):
    print(args)

    device = torch.device(args.device)

    # from AudioMAE
    norm_stats = {'audioset': [-4.2677393, 4.5689974], 'k400': [-4.2677393, 4.5689974],
                  'esc50': [-6.6268077, 5.358466], 'speechcommands': [-6.845978, 5.5654526]}
    args.audio_conf = {'num_mel_bins': args.num_mel_bins,
                      'target_length': args.input_size_audio,
                      'freqm': 0,
                      'timem': 0,
                      'mean': norm_stats['audioset'][0],
                      'std': norm_stats['audioset'][1],
                      'noise': False,
                      }

    model = create_model(
        args.model,
        pretrained=False,
        depth=args.depth,
        depth_audio=args.depth_audio,
    )
    if model.encoder is not None:
        patch_size = 254*3
        print("Patch size = %s" % str(patch_size))
        args.window_size = (args.num_frames // 1, patch_size)
        args.patch_size = patch_size
    # me: audio
    if model.encoder_audio is not None:
        patch_size_audio = model.encoder_audio.patch_embed.patch_size  # (16,16)
        print("Patch size (audio) = %s" % str(patch_size_audio))
        args.window_size_audio = (
        args.input_size_audio // patch_size_audio[0], args.num_mel_bins // patch_size_audio[1])
        args.patch_size_audio = patch_size_audio

    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)

    print("Load ckpt from %s" % args.model_path)
    checkpoint_model = None
    for model_key in args.model_key.split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint

    utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    model.to(device)

    plrs = compute_plrs(model, args.eval_data_path, args.wav_path)
    print(f"Evaluation mesh path: {args.eval_data_path}")
    print(f"PLRS score: {round(plrs.item(),3)}")


if __name__ == '__main__':
    opts = get_args()
    main(opts)