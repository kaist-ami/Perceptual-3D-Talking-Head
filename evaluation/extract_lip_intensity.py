import os
import numpy as np

# Use the predefined mouth vertex index map
mouth_map = [
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
]

def calculate_facial_movement_intensity(vertice_path):
    # Load the vertex .npy file
    keypoints = np.load(vertice_path, allow_pickle=True)
    keypoints = np.array(keypoints).reshape(-1, 5023, 3)  # Shape: T x 5023 x 3
    keypoints = keypoints[:, mouth_map, :]  # Select only mouth region vertices

    # Compute frame-to-frame displacements
    displacements = keypoints[1:] - keypoints[:-1]  # Shape: (T-1) x N x 3

    # Compute displacement magnitudes for each vertex
    displacement_magnitudes = np.sqrt(np.sum(displacements ** 2, axis=2))  # Shape: (T-1) x N

    # Compute average movement intensity per frame
    movement_intensity_per_frame = np.mean(displacement_magnitudes, axis=1)  # Shape: (T-1)

    # Compute overall facial movement intensity for the clip
    facial_movement_intensity = np.sqrt(np.mean(movement_intensity_per_frame ** 2))

    return facial_movement_intensity

# Set source and destination directories
source_root = './data_SLCC'
destination_root = './lip_disp'

# Traverse all .npy files under the source directory
for dirpath, dirnames, filenames in os.walk(source_root):
    for filename in filenames:
        if filename.endswith('.npy'):
            src_path = os.path.join(dirpath, filename)

            identity = filename.split('_')[0]
            emotion = filename.split('_')[1]
            level = 'level_' + filename.split('_')[3]
            clip = filename.split('_')[4]
            cond = filename.split('condition')[1][1:]

            dst_dir = os.path.join(destination_root, identity, emotion, level, clip)
            dst_path = os.path.join(dst_dir, cond.replace('.npy', '.csv'))

            # Create destination directory if it doesn't exist
            os.makedirs(dst_dir, exist_ok=True)

            # Calculate facial movement intensity
            intensity = calculate_facial_movement_intensity(src_path)

            # Save the result to a CSV file
            np.savetxt(dst_path, [intensity], delimiter=',')

            print(f"Processed {src_path} -> {dst_path}")
