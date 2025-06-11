import os
import numpy as np
import torch
from torchvision import transforms
import warnings
from torch.utils.data import Dataset
import random
import glob
import torchaudio

class SpeechMeshEval(torch.utils.data.Dataset):
    """Load your own video classification dataset.
    Parameters
    ----------
    root : str, required.
        Path to the root folder storing the dataset.
    setting : str, required.
        A text file describing the dataset, each line per video sample.
        There are three items in each line: (1) video path; (2) video length and (3) video label.
    train : bool, default True.
        Whether to load the training or validation set.
    test_mode : bool, default False.
        Whether to perform evaluation on the test set.
        Usually there is three-crop or ten-crop evaluation strategy involved.
    name_pattern : str, default None.
        The naming pattern of the decoded video frames.
        For example, img_00012.jpg.
    video_ext : str, default 'mp4'.
        If video_loader is set to True, please specify the video format accordinly.
    is_color : bool, default True.
        Whether the loaded image is color or grayscale.
    modality : str, default 'rgb'.
        Input modalities, we support only rgb video frames for now.
        Will add support for rgb difference image and optical flow image later.
    num_segments : int, default 1.
        Number of segments to evenly divide the video into clips.
        A useful technique to obtain global video-level information.
        Limin Wang, etal, Temporal Segment Networks: Towards Good Practices for Deep Action Recognition, ECCV 2016.
    num_crop : int, default 1.
        Number of crops for each image. default is 1.
        Common choices are three crops and ten crops during evaluation.
    new_length : int, default 1.
        The length of input video clip. Default is a single image, but it can be multiple video frames.
        For example, new_length=16 means we will extract a video clip of consecutive 16 frames.
    new_step : int, default 1.
        Temporal sampling rate. For example, new_step=1 means we will extract a video clip of consecutive frames.
        new_step=2 means we will extract a video clip of every other frame.
    temporal_jitter : bool, default False.
        Whether to temporally jitter if new_step > 1.
    video_loader : bool, default False.
        Whether to use video loader to load data.
    use_decord : bool, default True.
        Whether to use Decord video loader to load data. Otherwise use mmcv video loader.
    transform : function, default None.
        A function that takes data and label and transforms them.
    data_aug : str, default 'v1'.
        Different types of data augmentation auto. Supports v1, v2, v3 and v4.
    lazy_init : bool, default False.
        If set to True, build a dataset instance without loading any dataset.
    """

    def __init__(self,
                 root,
                 setting,
                 train=True,
                 test_mode=False,
                 name_pattern='img_%05d.jpg',
                 video_ext='mp4',
                 is_color=True,
                 modality='rgb',
                 num_segments=1,
                 num_crop=1,
                 new_length=1,
                 new_step=1,
                 temporal_jitter=False,
                 video_loader=False,
                 use_decord=False,
                 audio_conf=None,
                 roll_mag_aug=False,
                 audio_sample_rate=16000,
                 mask_generator_audio=None,
                 max_length=12,
                 ):

        super(SpeechMeshEval, self).__init__()
    
        self.root = root
        self.setting = setting
        self.train = train
        self.test_mode = test_mode
        self.is_color = is_color
        self.modality = modality
        self.num_segments = num_segments
        self.num_crop = num_crop
        self.new_length = new_length
        self.new_step = new_step
        self.skip_length = self.new_length * self.new_step
        self.temporal_jitter = temporal_jitter
        self.name_pattern = name_pattern
        self.video_loader = video_loader
        self.video_ext = video_ext
        self.use_decord = use_decord
        self.max_length = max_length

        self.clips = self._make_dataset(root, setting)
        if len(self.clips) == 0:
            raise (RuntimeError("Found 0 vertex clips in subfolders of: " + root + "\n"
                                                                                  "Check your data directory (opt.data-dir)."))
        self.crop_idxs = None

        # audio
        self.audio_conf = audio_conf
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        self.roll_mag_aug = roll_mag_aug

        self.audio_sample_rate = audio_sample_rate
        self.mask_generator_audio = mask_generator_audio

    def __getitem__(self, index):
        mesh_directory, audio_directory, _ = self.clips[index]
        vertices = np.load(mesh_directory) # vertices : [5,5023,3]

        process_data = torch.FloatTensor(vertices)  # T*C,H,W
        N = process_data.shape[0]
        process_data = process_data.reshape(N,-1)
        # audio
        try:
            audio_data = self._audio_decord_batch_loader(audio_directory)
        except Exception as e:
            next_idx = random.randint(0, self.__len__() - 1)
            print(
                f"==> Exception '{e}' occurred when processed '{audio_directory}', move to random next one (idx={next_idx}).")
            return self.__getitem__(next_idx)

        return (process_data, audio_data, mesh_directory)

    def __len__(self):
        return len(self.clips)

    def _make_dataset(self, directory, setting):
        if not os.path.exists(setting):
            raise (RuntimeError("Setting file %s doesn't exist. Check opt.train-list and opt.val-list. " % (setting)))
        clips = []

        with open(setting) as split_f:
            data = split_f.readlines()
            for line in data:
                line_info = line.split(' ')
                if len(line_info) < 2:
                    raise (RuntimeError('Video input format is not correct, missing one or more element. %s' % line))
                clip_mesh_path = os.path.join(line_info[0])
                clip_audio_path = os.path.join(line_info[1])
                target = int(line_info[2])
                item = (clip_mesh_path, clip_audio_path, target)
                clips.append(item)
        return clips

    # audio
    def _audio_decord_batch_loader(self, audio_directory):
        # sample matched audio waveform from the corresponding video interval
        audio_start = 0.0
        audio_end = 0.2

        audio_start_idx = int(audio_start * self.audio_sample_rate)
        audio_num_samples = int((audio_end - audio_start) * self.audio_sample_rate)
        audio, sr = torchaudio.load(audio_directory, frame_offset=audio_start_idx, num_frames=audio_num_samples)
        assert sr == self.audio_sample_rate, f'Error: wrong audio sample rate: {sr} (expected {self.audio_sample_rate})!'
        audio = audio.numpy()

        # assert audio.shape[
        #            1] > min_audio_length, f'Error: corrupted audio with length={audio.shape[1]} (min length: {min_audio_length})'

        fbank, _ = self._wav2fbank(audio,
                                   sr=self.audio_sample_rate)  # (Time, Freq), i.e., (seq_len, num_mel_bin)
        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        return fbank.unsqueeze(0)  # (C, T, F), C=1

    def _roll_mag_aug(self, waveform):
        idx = np.random.randint(len(waveform))
        rolled_waveform = np.roll(waveform, idx)
        mag = np.random.beta(10, 10) + 0.5
        return torch.Tensor(rolled_waveform * mag)

    def _wav2fbank(self, waveform1, waveform2=None, sr=None):
        if waveform2 == None:
            waveform = waveform1
            waveform = waveform - waveform.mean()
            if self.roll_mag_aug:
                waveform = self._roll_mag_aug(waveform)
        # mixup
        else:
            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if self.roll_mag_aug:
                waveform1 = self._roll_mag_aug(waveform1)
                waveform2 = self._roll_mag_aug(waveform2)

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, frame_length=8,
                                                  use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0,
                                                  frame_shift=10)
        target_length = self.audio_conf.get('target_length')
        n_frames = fbank.shape[0]
        p = target_length - n_frames
        
        if p < 0:
            fbank = fbank[0:target_length, :]

        if waveform2 == None:
            return fbank, 0  # (Time, Freq), i.e., (seq_len, num_mel_bin)
        else:
            return fbank, mix_lambda