import os
import numpy as np
import pandas as pd
from moviepy.editor import VideoFileClip
from scipy.io import wavfile
from tempfile import TemporaryDirectory
import glob
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_audio_rms_and_length(video_path):
    """
    Extracts audio from an MP4 video file and calculates the RMS value and audio length.
    
    Args:
        video_path (str): The path to the MP4 video file.
        
    Returns:
        tuple: (RMS value, audio length in seconds)
    """
    try:
        logging.info(f"Processing video: {video_path}")
        
        # Temporary directory for storing the audio
        with TemporaryDirectory() as tmpdirname:
            audio_file = os.path.join(tmpdirname, "audio.wav")
            
            # Extract audio from video and save it to the temporary file
            logging.info(f"Extracting audio from video: {video_path}")
            video = VideoFileClip(video_path)
            audio = video.audio
            
            if audio is None:
                logging.warning(f"No audio track found in {video_path}")
                return None, None

            logging.info(f"Saving extracted audio to WAV: {audio_file}")
            audio.write_audiofile(audio_file, codec='pcm_s16le')  # Save as WAV file
            
            # Load the WAV file to calculate the RMS
            sample_rate, audio_data = wavfile.read(audio_file)
            
            # Calculate audio length in seconds
            audio_length = len(audio_data) / sample_rate
            logging.info(f"Audio length: {audio_length} seconds")
            
            # If stereo, average the channels
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Normalize audio data
            # audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Calculate RMS value
            rms_value = np.sqrt(np.mean(audio_data**2))
            logging.info(f"RMS value: {rms_value}")
            
            return rms_value, audio_length

    except Exception as e:
        logging.error(f"Error processing {video_path}: {e}")
        return None, None

def save_rms_and_length_to_csv(video_path, output_csv_path, rms_value, audio_length):
    """
    Saves the RMS value and audio length to a CSV file.
    
    Args:
        video_path (str): The path to the MP4 video file.
        output_csv_path (str): The path to save the CSV file.
        rms_value (float): The RMS value of the audio signal.
        audio_length (float): The length of the audio signal in seconds.
    """
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        
        # Save the results to a CSV file
        df = pd.DataFrame({
            "video_path": [video_path],
            "rms_value": [rms_value],
            "audio_length_seconds": [audio_length]
        })
        
        df.to_csv(output_csv_path, index=False)
        logging.info(f"Saved results to {output_csv_path}")
    
    except Exception as e:
        logging.error(f"Error saving CSV file {output_csv_path}: {e}")

def process_directory(video_directory, output_directory):
    """
    Traverses the directory of videos, extracts the RMS and length for each, and saves the results.
    
    Args:
        video_directory (str): The root directory containing the MP4 video files.
        output_directory (str): The root directory to save the resulting CSV files.
    """
    video_paths = glob.glob(os.path.join(video_directory, '**/front/**/*.mp4'), recursive=True)
    
    logging.info(f"Found {len(video_paths)} videos to process.")
    
    # Process each video file
    for video_path in video_paths:
        try:
            # Extract the RMS and audio length
            rms_value, audio_length = extract_audio_rms_and_length(video_path)
            
            if rms_value is None or audio_length is None:
                continue  # Skip if extraction failed
            
            # Generate corresponding CSV file path in the output directory
            relative_path = os.path.relpath(video_path, video_directory)
            csv_output_path = os.path.join(output_directory, os.path.splitext(relative_path)[0] + '.csv')
            
            # Save the RMS and length to the CSV file
            save_rms_and_length_to_csv(video_path, csv_output_path, rms_value, audio_length)
        
        except Exception as e:
            logging.error(f"Error processing video {video_path}: {e}")

# Main function to execute the process
if __name__ == "__main__":
    video_directory = './MEAD'  # Root directory of input videos
    output_directory = './MEAD_rms'  # Root directory for saving CSV files
    
    logging.info("Starting RMS extraction process...")
    process_directory(video_directory, output_directory)
    logging.info("RMS extraction process completed.")
