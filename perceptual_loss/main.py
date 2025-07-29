import re, random, math
import numpy as np
import argparse
from tqdm import tqdm
import os, shutil
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import get_dataloaders
from faceformer import Faceformer

## Perceptual-3D-Talking-Head
from tqdm import tqdm
import math
import random
random_seed=1111
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.model.modeling import SpeechMeshTransformer
from src import utils
from functools import partial
#mp.set_sharing_strategy('file_system')

mouth_map = np.array([
    1576, 1577, 1578, 1579, 1580, 1581, 1582, 1583, 1590, 1590, 1591, 1593, 1593,
    1657, 1658, 1661, 1662, 1663, 1667, 1668, 1669, 1670, 1686, 1687, 1691, 1693,
    1694, 1695, 1696, 1697, 1700, 1702, 1703, 1704, 1709, 1710, 1711, 1712, 1713,
    1714, 1715, 1716, 1717, 1718, 1719, 1720, 1721, 1722, 1723, 1728, 1729, 1730,
    1731, 1732, 1733, 1734, 1735, 1736, 1737, 1738, 1740, 1743, 1748, 1749, 1750,
    1751, 1758, 1763, 1765, 1770, 1771, 1773, 1774, 1775, 1776, 1777, 1778, 1779,
    1780, 1781, 1782, 1787, 1788, 1789, 1791, 1792, 1793, 1794, 1795, 1796, 1801,
    1802, 1803, 1804, 1826, 1827, 1836, 1846, 1847, 1848, 1849, 1850, 1865, 1866,
    2712, 2713, 2714, 2715, 2716, 2717, 2718, 2719, 2726, 2726, 2727, 2729, 2729,
    2774, 2775, 2778, 2779, 2780, 2784, 2785, 2786, 2787, 2803, 2804, 2808, 2810,
    2811, 2812, 2813, 2814, 2817, 2819, 2820, 2821, 2826, 2827, 2828, 2829, 2830,
    2831, 2832, 2833, 2834, 2835, 2836, 2837, 2838, 2839, 2840, 2843, 2844, 2845,
    2846, 2847, 2848, 2849, 2850, 2851, 2852, 2853, 2855, 2858, 2863, 2864, 2865,
    2866, 2869, 2871, 2873, 2878, 2879, 2880, 2881, 2882, 2883, 2884, 2885, 2886,
    2887, 2888, 2889, 2890, 2891, 2892, 2894, 2895, 2896, 2897, 2898, 2899, 2904,
    2905, 2906, 2907, 2928, 2929, 2934, 2935, 2936, 2937, 2938, 2939, 2948, 2949,
    3503, 3504, 3506, 3509, 3511, 3512, 3513, 3531, 3533, 3537, 3541, 3543, 3546,
    3547, 3790, 3791, 3792, 3793, 3794, 3795, 3796, 3797, 3798, 3799, 3800, 3801,
    3802, 3803, 3804, 3805, 3806, 3914, 3915, 3916, 3917, 3918, 3919, 3920, 3921,
    3922, 3923, 3924, 3925, 3926, 3927, 3928
])

def trainer(args, train_loader, dev_loader, model, guidance_model, optimizer, criterion, epoch=100):
    ## Perceptual-3D-Talking-Head
    def rep_output(audios, motion, audio_length):
        clip_num = min(audio_length, motion.shape[0] - 4)
        audios = audios[:clip_num]
        batch_num = clip_num // args.guidance_batch_size
        g_loss_sum = 0

        for b in range(batch_num):
            motion_batch = []
            for i in range(args.guidance_batch_size):
                motion_batch.append(motion[b * args.guidance_batch_size + i:b * args.guidance_batch_size + i + 5])
            motion_batch_tensor = torch.stack(motion_batch, dim=0).cuda()
            audio_batch_tensor = torch.stack(
                audios[b * args.guidance_batch_size:b * args.guidance_batch_size + args.guidance_batch_size], dim=0).cuda()
            g_loss = guidance_model.compute_percp_loss(
                motion_batch_tensor, audio_batch_tensor
            )
            g_loss_sum += g_loss

        remain_num = clip_num % 80
        if remain_num >= 2:
            motion_batch = []
            for i in range(remain_num):
                motion_batch.append(
                    motion[batch_num * args.guidance_batch_size + i:batch_num * args.guidance_batch_size + i + 5])
            motion_batch_tensor = torch.stack(motion_batch, dim=0).cuda()
            audio_batch_tensor = torch.stack(
                audios[batch_num * args.guidance_batch_size:batch_num * args.guidance_batch_size + remain_num],
                dim=0).cuda()
            g_loss = guidance_model.compute_percp_loss(
                motion_batch_tensor, audio_batch_tensor
            )
            g_loss_sum += g_loss
            batch_num = batch_num + 1

        return g_loss_sum / batch_num
    save_path = os.path.join(args.dataset,args.save_path)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    train_subjects_list = [i for i in args.train_subjects.split(" ")]
    iteration = 0
    for e in range(epoch+1):
        all_loss_log = []
        loss_log = []
        perceptual_loss_log = []
        # train
        model.train()
        pbar = tqdm(enumerate(train_loader),total=len(train_loader))
        optimizer.zero_grad()

        for i, (audio, vertice, template, one_hot, file_name, rep_audio) in pbar:
            iteration += 1
            # to gpu
            audio, vertice, template, one_hot  = audio.to(device="cuda"), vertice.to(device="cuda"), template.to(device="cuda"), one_hot.to(device="cuda")
            vertice_out, loss = model(audio, template,  vertice, one_hot, criterion,teacher_forcing=False)

            ## Perceptual-3D-Talking-Head
            if args.model_type == 'ours':
                perceptual_loss = rep_output(
                    audios=rep_audio,
                    motion=vertice_out.squeeze(0),
                    audio_length=len(rep_audio),
                )

            elif args.model_type == 'original':
                perceptual_loss = None

            if perceptual_loss == None:
                if args.model_type == 'original':
                    all_loss = loss
                    perceptual_loss = torch.tensor(0).to(loss.device)
                else:
                    continue
            else:
                all_loss = loss + perceptual_loss * args.guidance_weight

            all_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            all_loss_log.append(all_loss.item())
            loss_log.append(loss.item())

            if args.model_type == 'original':
                pbar.set_description(
                    "(Epoch {}, iteration {}) TRAIN ALL LOSS:{:.7f}, ORIGINAL LOSS{:.7f}"
                        .format((e + 1), iteration, np.mean(all_loss_log), np.mean(loss_log)))
            elif args.model_type == 'baseline' or args.model_type == 'ours':
                perceptual_loss_log.append(perceptual_loss.item())
                pbar.set_description(
                    "(Epoch {}, iteration {}) TRAIN ALL LOSS:{:.7f}, ORIGINAL LOSS{:.7f}, PERCEPTUAL LOSS:{:.7f}"
                        .format((e + 1), iteration, np.mean(all_loss_log), np.mean(loss_log),
                                np.mean(perceptual_loss_log)))
        # validation
        vertices_gt = []
        vertices_pred = []
        valid_all_loss_log = []
        valid_loss_log = []
        valid_perceptual_loss_log = []
        model.eval()
        for audio, vertice, template, one_hot_all,file_name, rep_audio in dev_loader:
            # to gpu
            audio, vertice, template, one_hot_all= audio.to(device="cuda"), vertice.to(device="cuda"), template.to(device="cuda"), one_hot_all.to(device="cuda")
            train_subject = "_".join(file_name[0].split("_")[:-1])
            if train_subject in train_subjects_list:
                condition_subject = train_subject
                iter = train_subjects_list.index(condition_subject)
                one_hot = one_hot_all[:,iter,:]
                vertice_out,loss = model(audio, template,  vertice, one_hot, criterion)

                ## Perceptual-3D-Talking-Head
                if args.model_type == 'ours':
                    perceptual_loss = rep_output(
                        audios=rep_audio,
                        motion=vertice_out.squeeze(0),
                        audio_length=len(rep_audio),
                    )

                elif args.model_type == 'original':
                    perceptual_loss = None

                if perceptual_loss == None:
                    if args.model_type == 'original':
                        all_loss = loss
                        perceptual_loss = torch.tensor(0).to(loss.device)
                    else:
                        continue
                else:
                    all_loss = loss + perceptual_loss * args.guidance_weight

                valid_loss_log.append(loss.item())
            else:
                for iter in range(one_hot_all.shape[-1]):
                    condition_subject = train_subjects_list[iter]
                    one_hot = one_hot_all[:,iter,:]
                    vertice_out,loss = model(audio, template,  vertice, one_hot, criterion)

                    ## Perceptual-3D-Talking-Head
                    if args.model_type == 'ours':
                        perceptual_loss = rep_output(
                            audios=rep_audio,
                            motion=vertice_out.squeeze(0),
                            audio_length=len(rep_audio),
                        )

                    elif args.model_type == 'original':
                        perceptual_loss = None

                    if perceptual_loss == None:
                        if args.model_type == 'original':
                            all_loss = loss
                            perceptual_loss = torch.tensor(0).to(loss.device)
                        else:
                            continue
                    else:
                        all_loss = loss + perceptual_loss * args.guidance_weight

                    valid_all_loss_log.append(all_loss.item())
                    valid_loss_log.append(loss.item())
                    vertices_gt.append(vertice.reshape(-1, 5023, 3).detach().cpu().numpy())
                    vertices_pred.append(vertice_out.reshape(-1, 5023, 3).detach().cpu().numpy())

                    if args.model_type == 'ours':
                        valid_perceptual_loss_log.append(perceptual_loss.item())

        current_all_loss = np.mean(valid_all_loss_log)
        current_loss = np.mean(valid_loss_log)

        vertices_gt = np.concatenate(vertices_gt)
        vertices_pred = np.concatenate(vertices_pred)
        L2_dis_mouth_max = np.array(
            [np.square(vertices_gt[:, v, :] - vertices_pred[:, v, :]) for v in mouth_map]
        )
        L2_dis_mouth_max = np.transpose(L2_dis_mouth_max, (1, 0, 2))
        L2_dis_mouth_max = np.sum(L2_dis_mouth_max, axis=2)
        L2_dis_mouth_max = np.max(L2_dis_mouth_max, axis=1)
        lve = np.mean(L2_dis_mouth_max)

        if (e > 0 and e % 25 == 0) or e == args.max_epoch:
            torch.save(model.state_dict(), os.path.join(save_path, '{}_model.pth'.format(e)))

        if args.model_type == 'original':
            print("epoch: {}, current all loss:{:.7f}, original loss:{:.7f}, lve:{:.7f}".format(
                e + 1,
                current_all_loss,
                current_loss, lve))

        elif args.model_type == 'ours':
            current_perceptual_loss = np.mean(valid_perceptual_loss_log)
            print("epoch: {}, current all loss:{:.7f}, original loss:{:.7f}, perceptual loss:{:.7f}, lve:{:.7f}".format(
                e + 1,
                current_all_loss,
                current_loss,
                current_perceptual_loss, lve))

    return model

@torch.no_grad()
def test(args, model, test_loader,epoch):
    result_path = os.path.join(args.dataset,args.result_path)
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path)

    save_path = os.path.join(args.dataset,args.save_path)
    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    model.load_state_dict(torch.load(os.path.join(save_path, '{}_model.pth'.format(epoch))))
    model = model.to(torch.device("cuda"))
    model.eval()
   
    for audio, vertice, template, one_hot_all, file_name in test_loader:
        # to gpu
        audio, vertice, template, one_hot_all= audio.to(device="cuda"), vertice.to(device="cuda"), template.to(device="cuda"), one_hot_all.to(device="cuda")
        train_subject = "_".join(file_name[0].split("_")[:-1])
        if train_subject in train_subjects_list:
            condition_subject = train_subject
            iter = train_subjects_list.index(condition_subject)
            one_hot = one_hot_all[:,iter,:]
            prediction = model.predict(audio, template, one_hot)
            prediction = prediction.squeeze() # (seq_len, V*3)
            np.save(os.path.join(result_path, file_name[0].split(".")[0]+"_condition_"+condition_subject+".npy"), prediction.detach().cpu().numpy())
        else:
            for iter in range(one_hot_all.shape[-1]):
                condition_subject = train_subjects_list[iter]
                one_hot = one_hot_all[:,iter,:]
                prediction = model.predict(audio, template, one_hot)
                prediction = prediction.squeeze() # (seq_len, V*3)
                np.save(os.path.join(result_path, file_name[0].split(".")[0]+"_condition_"+condition_subject+".npy"), prediction.detach().cpu().numpy())
         
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser(description='FaceFormer: Speech-Driven 3D Facial Animation with Transformers')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--dataset", type=str, default="vocaset", help='vocaset or BIWI')
    parser.add_argument("--vertice_dim", type=int, default=5023*3, help='number of vertices - 5023*3 for vocaset; 23370*3 for BIWI')
    parser.add_argument("--feature_dim", type=int, default=64, help='64 for vocaset; 128 for BIWI')
    parser.add_argument("--period", type=int, default=30, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--wav_path", type=str, default= "wav", help='path of the audio signals')
    parser.add_argument("--vertices_path", type=str, default="vertices_npy", help='path of the ground truth')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--max_epoch", type=int, default=100, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--template_file", type=str, default="templates.pkl", help='path of the personalized templates')
    parser.add_argument("--save_path", type=str, default="save", help='path of the trained models')
    parser.add_argument("--result_path", type=str, default="result", help='path to the predictions')
    parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA"
       " FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA"
       " FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA"
       " FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")
    parser.add_argument("--val_subjects", type=str, default="FaceTalk_170811_03275_TA"
       " FaceTalk_170908_03277_TA")
    parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA"
       " FaceTalk_170731_00024_TA")
    parser.add_argument("--guidance_weight", type=float, default=0.0000001, help='guidance_weight')
    parser.add_argument("--guidance_batch_size", type=int, default=80, help='guidance_batch_size')
    parser.add_argument("--model_type", type=str, default='ours', help='model_type')
    parser.add_argument("--guidance_model_path", type=str, default='/path/to/perceptual/loss/model/checkpoint/', help='guidance_model_path')
    args = parser.parse_args()

    #build model
    model = Faceformer(args)
    print("model parameters: ", count_parameters(model))

    ## Perceptual-3D-Talking-Head
    guidance_model = SpeechMeshTransformer(
        vertex_size=5023 * 3,
        patch_size=5023 * 3,
        embed_dim=512,
        num_heads=8,
        depth=10,
        # audio
        img_size_audio=(64, 128),  # (T, F)
        patch_size_audio=16,
        embed_dim_audio=512,
        num_heads_audio=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        max_length=5,
        depth_audio=10,)

    print("Load guidance model ckpt from %s" % args.guidance_model_path)
    checkpoint = torch.load(args.guidance_model_path, map_location='cpu')

    checkpoint_model = checkpoint['model']
    utils.load_state_dict(guidance_model, checkpoint_model)

    # model.to(device)
    # print("Load ckpt from %s" % guidance_model_checkpoint_path)
    #
    # checkpoint_model = checkpoint['model']
    #
    #
    # guidance_model.load_state_dict(checkpoint_model)

    for name, param in guidance_model.named_parameters():
        param.requires_grad = False

    guidance_model.to(torch.device("cuda"))
    guidance_model.eval()

    # to cuda
    assert torch.cuda.is_available()
    model = model.to(torch.device("cuda"))
    
    #load data
    dataset = get_dataloaders(args)
    # loss
    criterion = nn.MSELoss()

    # Train the model
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=args.lr)
    model = trainer(args, dataset["train"], dataset["valid"], model, guidance_model, optimizer, criterion, epoch=args.max_epoch)
    
    #test(args, model, dataset["test"], epoch=args.max_epoch)
    
if __name__=="__main__":
    main()