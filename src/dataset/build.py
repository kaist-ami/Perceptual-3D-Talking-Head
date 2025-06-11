from .base import SpeechMeshEval

def build_plrs_dataset(args):
    dataset = SpeechMeshEval(
        root=None,
        setting=args.data_path,
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        new_length=args.num_frames,
        new_step=args.sampling_rate, # change to 1
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        num_segments=1,
        audio_conf=args.audio_conf,
        roll_mag_aug=args.roll_mag_aug,
        audio_sample_rate=16000,
        max_length=args.num_frames
    )
    return dataset