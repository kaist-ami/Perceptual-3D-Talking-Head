import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical analysis libraries
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Set visualization style
sns.set(style="whitegrid")

# --------------------
# 1. Load and Merge Data
# --------------------

# Path to lip intensity data
lip_data_root = './lip_disp'

# Path to speech intensity data
speech_data_root = './MEAD_rms'

# Directory to save results
output_dir = './SLCC_results/'
os.makedirs(output_dir, exist_ok=True)

# Initialize list to collect all data
data_list = []

# Collect lip intensity data
for root, dirs, files in os.walk(lip_data_root):
    for file in files:
        if file.endswith('csv'):

            lip_file_path = os.path.join(root, file)
            # Load lip_disp value
            try:
                lip_df = pd.read_csv(lip_file_path, header=None)
                lip_intensity = lip_df.iloc[0, 0]
                lip_intensity = float(lip_intensity)
            except Exception as e:
                print(f"Error loading lip intensity file: {lip_file_path}, Error: {e}")
                continue  # Skip if any error occurs

            # Extract metadata from path
            path_parts = lip_file_path.split(os.sep)
            try:
                level_index = [i for i, part in enumerate(path_parts) if 'level_' in part][0]
                level = path_parts[level_index]
            except IndexError:
                continue  # Skip if level information is missing

            emotion = path_parts[level_index - 1]
            identity = path_parts[level_index - 2]
            clip = path_parts[level_index + 1]

            # Construct corresponding speech intensity file path
            speech_file_path = os.path.join(
                speech_data_root,
                identity,
                'video',
                'front',
                emotion,
                level,
                clip + '.csv'
            )

            if os.path.exists(speech_file_path):
                # Load speech intensity data
                try:
                    speech_df = pd.read_csv(speech_file_path)
                    speech_intensity = speech_df['rms_value'].values[0]
                    audio_length = speech_df['audio_length_seconds'].values[0]
                except Exception as e:
                    print(f"Error loading speech intensity file: {speech_file_path}, Error: {e}")
                    continue

                # Add data entry
                data_list.append({
                    'LipIntensity': lip_intensity,
                    'SpeechIntensity': speech_intensity,
                    'Level': level,
                    'Emotion': emotion,
                    'Identity': identity,
                    'Clip': clip,
                    'AudioLength': audio_length
                })
            else:
                print(f"Speech intensity file not found: {speech_file_path}")

# Convert list to DataFrame
df = pd.DataFrame(data_list)

# Order levels
df['Level'] = pd.Categorical(df['Level'], categories=['level_1', 'level_2', 'level_3'], ordered=True)

# --------------------
# 2. Identity-wise Z-normalization
# --------------------

# Z-normalize speech intensity per identity
df['Z_SpeechIntensity'] = df.groupby('Identity')['SpeechIntensity'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# Z-normalize lip intensity per identity
df['Z_LipIntensity'] = df.groupby('Identity')['LipIntensity'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# Remove rows with NaNs caused by zero standard deviation
df.dropna(subset=['Z_SpeechIntensity', 'Z_LipIntensity'], inplace=True)

# --------------------
# 3. Correlation Plotting for Each Level
# --------------------

# (3) Correlation between lip and speech intensity (using z-normalized values)
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Z_SpeechIntensity', y='Z_LipIntensity', data=df, alpha=0.5)
sns.regplot(x='Z_SpeechIntensity', y='Z_LipIntensity', data=df, scatter=False, color='red')

corr_coef, p_value = stats.pearsonr(df['Z_SpeechIntensity'], df['Z_LipIntensity'])

plt.title(f'Correlation between Z-normalized Speech and Lip Intensity\nPearson r = {corr_coef:.2f}, p = {p_value:.2e}')
plt.xlabel('Z-normalized Speech Intensity')
plt.ylabel('Z-normalized Lip Intensity')

plt.xlim([-5, 5])  # X-axis range
plt.ylim([-5, 5])  # Y-axis range

plt.savefig(os.path.join(output_dir, 'z_lip_vs_speech_intensity.png'))
plt.close()

# Get unique expression levels
levels = df['Level'].unique()

for level in levels:
    # Filter data by level
    df_level = df[df['Level'] == level]

    # Compute correlation
    corr_coef, p_value = stats.pearsonr(df_level['Z_SpeechIntensity'], df_level['Z_LipIntensity'])

    # Plot with regression line
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='Z_SpeechIntensity', y='Z_LipIntensity', data=df_level, alpha=0.5)
    sns.regplot(x='Z_SpeechIntensity', y='Z_LipIntensity', data=df_level, scatter=False, color='red')

    plt.title(f'Correlation between Z-normalized Speech and Lip Intensity\nLevel: {level}, Pearson r = {corr_coef:.2f}, p = {p_value:.2e}')
    plt.xlabel('Z-normalized Speech Intensity')
    plt.ylabel('Z-normalized Lip Intensity')

    plt.xlim([-5, 5])
    plt.ylim([-5, 5])

    # Save plot
    plot_filename = f'z_lip_vs_speech_intensity_{level}.png'
    plt.savefig(os.path.join(output_dir, plot_filename))
    plt.close()
