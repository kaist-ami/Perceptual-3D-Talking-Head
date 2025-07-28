#!/bin/zsh
# input
input_size=160
input_size_audio=64

# path to pre-trained model
MODEL_PATH="/path/to/eval/model/checkpoint/"
EVAL_DATA_PATH=$1
WAV_PATH='/path/to/vocaset/wav/'

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
    --master_port 13307 \
    evaluate_PLRS.py \
    --model speech_mesh_rep \
    --model_path ${MODEL_PATH} \
    --input_size ${input_size} \
    --input_size_audio ${input_size_audio} \
    --depth 10 \
    --depth_audio 10 \
    --num_frames 5 \
    --eval_data_path $EVAL_DATA_PATH \
    --wav_path $WAV_PATH
