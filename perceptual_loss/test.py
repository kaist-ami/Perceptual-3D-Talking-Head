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


@torch.no_grad()
def test(args, model, test_loader, epoch):
    result_path = os.path.join(args.dataset, args.result_path)
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path)

    save_path = os.path.join(args.dataset, args.save_path)
    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    model.load_state_dict(torch.load(os.path.join(save_path, '{}_model.pth'.format(epoch))))
    print(f"Load checkpoint from {os.path.join(save_path, '{}_model.pth'.format(epoch))}")
    model = model.to(torch.device("cuda"))
    model.eval()

    import pickle
    with open("/home/chaeyeon/krafton/FaceFormer/FLAME_masks.pkl", "rb") as f:
        fb = pickle.load(f, encoding="latin")

    output = []
    for r in ["eye_region", "forehead", "nose"]:
        output.extend(fb[r])
    upper_map = list(set(output))

    vertices_gt = []
    vertices_pred = []
    motion_std_difference = []

    for audio, vertice, template, one_hot_all, file_name, _ in test_loader:
        # to gpu
        audio, vertice, template, one_hot_all = audio.to(device="cuda"), vertice.to(device="cuda"), template.to(
            device="cuda"), one_hot_all.to(device="cuda")
        train_subject = "_".join(file_name[0].split("_")[:-1])
        if train_subject in train_subjects_list:
            condition_subject = train_subject
            iter = train_subjects_list.index(condition_subject)
            one_hot = one_hot_all[:, iter, :]
            prediction = model.predict(audio, template, one_hot)
            prediction = prediction.squeeze()  # (seq_len, V*3)
            np.save(os.path.join(result_path, file_name[0].split(".")[0] + "_condition_" + condition_subject + ".npy"),
                    prediction.detach().cpu().numpy())

            vertices_gt.append(vertice.reshape(-1, 5023, 3).detach().cpu().numpy())
            vertices_pred.append(prediction.reshape(-1, 5023, 3).detach().cpu().numpy())
        else:
            for iter in range(one_hot_all.shape[-1]):
                condition_subject = train_subjects_list[iter]
                one_hot = one_hot_all[:, iter, :]
                prediction = model.predict(audio, template, one_hot)
                prediction = prediction.squeeze()  # (seq_len, V*3)
                np.save(
                    os.path.join(result_path, file_name[0].split(".")[0] + "_condition_" + condition_subject + ".npy"),
                    prediction.detach().cpu().numpy())

                vertices_gt.append(vertice[:, :prediction.shape[0]].reshape(-1, 5023, 3).detach().cpu().numpy())
                vertices_pred.append(prediction.reshape(-1, 5023, 3).detach().cpu().numpy())

                motion_pred = prediction.reshape(-1, 5023, 3).detach().cpu().numpy() - template.reshape(1, 5023,
                                                                                                        3).detach().cpu().numpy()
                motion_gt = vertice[:, :prediction.shape[0]].reshape(-1, 5023,
                                                                     3).detach().cpu().numpy() - template.reshape(1,
                                                                                                                  5023,
                                                                                                                  3).detach().cpu().numpy()

                L2_dis_upper = np.array([np.square(motion_gt[:, v, :]) for v in upper_map])
                L2_dis_upper = np.transpose(L2_dis_upper, (1, 0, 2))
                L2_dis_upper = np.sum(L2_dis_upper, axis=2)
                L2_dis_upper = np.std(L2_dis_upper, axis=0)
                gt_motion_std = np.mean(L2_dis_upper)

                L2_dis_upper = np.array([np.square(motion_pred[:, v, :]) for v in upper_map])
                L2_dis_upper = np.transpose(L2_dis_upper, (1, 0, 2))
                L2_dis_upper = np.sum(L2_dis_upper, axis=2)
                L2_dis_upper = np.std(L2_dis_upper, axis=0)
                pred_motion_std = np.mean(L2_dis_upper)

                motion_std_difference.append(gt_motion_std - pred_motion_std)

    vertices_gt = np.concatenate(vertices_gt)
    vertices_pred = np.concatenate(vertices_pred)
    L2_dis_mouth_max = np.array(
        [np.square(vertices_gt[:, v, :] - vertices_pred[:, v, :]) for v in mouth_map]
    )
    L2_dis_mouth_max = np.transpose(L2_dis_mouth_max, (1, 0, 2))
    L2_dis_mouth_max = np.sum(L2_dis_mouth_max, axis=2)
    L2_dis_mouth_max = np.max(L2_dis_mouth_max, axis=1)
    lve = np.mean(L2_dis_mouth_max)
    print('LVE: {:.4e}, FDD: {:.4e}'.format(lve, sum(motion_std_difference) / len(motion_std_difference)))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(description='FaceFormer: Speech-Driven 3D Facial Animation with Transformers')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--dataset", type=str, default="vocaset", help='vocaset or BIWI')
    parser.add_argument("--dataset_dir", type=str, default="vocaset", help='dataset_dir')
    parser.add_argument("--vertice_dim", type=int, default=5023 * 3,
                        help='number of vertices - 5023*3 for vocaset; 23370*3 for BIWI')
    parser.add_argument("--feature_dim", type=int, default=64, help='64 for vocaset; 128 for BIWI')
    parser.add_argument("--period", type=int, default=30, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--wav_path", type=str, default="wav", help='path of the audio signals')
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
    args = parser.parse_args()

    # build model
    model = Faceformer(args)
    print("model parameters: ", count_parameters(model))

    # to cuda
    assert torch.cuda.is_available()
    model = model.to(torch.device("cuda"))
    # load data
    dataset = get_dataloaders(args)

    test(args, model, dataset["test"], epoch=args.max_epoch)


if __name__ == "__main__":
    main()