# This script processes the raw interview recordings as downladed from the server. It: 

# 1. Selects audio files by call center agents, keeping only agents administering the survey, and moves them from subfolders to the main directory
# 2. Converts audio files from .gsm to .wav format and removes those with duration less than one minute 
# 3. Collects phone numbers of eligible participants from the audio files and organizes them in directories
# 4. Removes audio files with duration less than the specified minimum duration
# 5. Concatenates audio files from the subfolders and moves everything to the main directory. 

# The final output is a directory containing audio files named after the phone numbers of eligible participants, with each file containing the full interview recording.

# The script takes the following arguments:
# - directory: Path to the directory containing the audio files
# - dta_path: Path to the .dta file containing the survey data (to extract phone numbers of eligible participants)
# - min_duration: Minimum duration in milliseconds for audio files to be kept (default is 180000 ms)

# Run the script with the following command:
# python3 process_raw_audios.py <directory> <dta_path> --min_duration <min_duration>
# python3 process_raw_audios.py "../data/audio/audios_apr2024/all_files" "../data/working/qualtrics/qualtrics_23mar.dta"

import os
import shutil
import argparse
from pydub import AudioSegment
import pandas as pd

def select_agents(directory):
    """
    Selects and moves audio files of specific agents from subfolders to the main directory.

    Args:
        directory (str): The path to the main directory containing subfolders and audio files.

    Returns:
        None

    Raises:
        None

    Notes:
        This function iterates through all subfolders and files in the given directory.
        It checks if the name of each file is valid and deletes it if not.
        Valid names are those that have the agent's name as the third part of the filename.
        The valid agent names are "Phoebe", "SamDe", "Ayisah", and "Rose".
        If a file is valid, it is moved to the main directory.
        After processing all files, empty subfolders are removed.
    """
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in [n for n in files if n.endswith(".gsm") or n.endswith(".wav")]:
            full_path = os.path.join(root, name)
            parts = name.split('_')
            if parts[2] not in {"Phoebe", "SamDe", "Ayisah", "Rose"}:
                os.remove(full_path)
                continue
            shutil.move(full_path, directory)
        if root != directory:
            shutil.rmtree(root)
    return None

def collect_phone_numbers(directory, dta_path):
    """
    Collects phone numbers from a directory of audio files based on eligibility criteria.

    Parameters:
    - directory (str): The path to the directory containing the audio files in wav format.
    - dta_path (str): The path to the .dta file containing the survey data.

    Returns:
    - None

    Raises:
    - FileNotFoundError: If the directory or .dta file does not exist.
    - ValueError: If the .dta file is not in the correct format.

    Description:
    - This function reads a .dta file containing participant data and filters it based on eligibility criteria.
    - It then collects the phone numbers of eligible participants from the audio files in the specified directory.
    - The phone numbers are stored in a dictionary, where each phone number is a key and the corresponding audio files are the values.
    - If a phone number has multiple audio files, they are moved to a new directory named after the phone number.
    - If a phone number has a single audio file, the file is renamed to the phone number.
    - If a phone number is not eligible, the corresponding audio file is removed.

    Eligibility Criteria:
    - Participants must have given consent to participate (consent == "Yes, I'd like to participate").
    - Participants must have answered more than 50% of the questions (proportion_answered > 0.5).

    Note:
    - The phone numbers are extracted from the audio file names, assuming the format "prefix_phoneNumber_suffix.wav".
    - The prefix '23' is removed from the phone numbers before storing them in the dictionary.
    """

    # Load the .dta file
    data = pd.read_stata(dta_path)
    
    # Filter data for eligible participants
    eligible_data = data[(data['consent'] == "Yes, I'd like to participate") & (data['proportion_answered'] > 0.5)]

    # Create a set of eligible phone numbers (after removing the prefix '23')
    eligible_phones = set(eligible_data['reference_phone'].dropna().apply(lambda x: str(int(x))[2:]))

    phone_dict = {}
    for file in os.listdir(directory):
        if file.endswith(".wav"):
            phone_number = file.split('_')[1]
            if phone_number in eligible_phones:
                if phone_number in phone_dict:
                    phone_dict[phone_number].append(file)
                else:
                    phone_dict[phone_number] = [file]
            else:
                os.remove(os.path.join(directory, file))
    for phone, files in phone_dict.items():
        if len(files) > 1:
            new_dir = os.path.join(directory, f"23{phone}")
            os.makedirs(new_dir, exist_ok=True)
            for file in files:
                shutil.move(os.path.join(directory, file), new_dir)
        else:
            os.rename(os.path.join(directory, files[0]), os.path.join(directory, f"23{phone}.wav"))
    return None

def clean_and_convert(directory, min_duration=60000):
    """
    Cleans and converts audio files from .gms to .wav in the given directory.

    Args:
        directory (str): The directory containing the audio files.
        min_duration (int, optional): The minimum duration in milliseconds. 
            Files with duration less than this value will be removed. 
            Defaults to 60000.

    Returns:
        None
    """
    for file in os.listdir(directory):
        full_path = os.path.join(directory, file)
        if file.endswith(".gsm"):
            sound = AudioSegment.from_file(full_path, format="gsm")
            if len(sound) < min_duration:
                os.remove(full_path)
            else:
                # Convert to WAV format
                wav_path = full_path.replace('.gsm', '.wav')
                sound.export(wav_path, format="wav")
                os.remove(full_path)
        elif file.endswith(".wav"):
            sound = AudioSegment.from_file(full_path, format="wav")
            if len(sound) < min_duration:
                os.remove(full_path)
    return None

def concatenate_audio_in_subfolders(directory):
  """
  This function concatenates audio files in subfolders of a directory,
  excluding the main directory itself.

  Args:
    directory: Path to the directory containing subfolders with audio files.
  """
  for root, _, files in os.walk(directory):
    # Skip the main directory (avoid processing files in it)
    if root == directory:
      continue

    if not files:
      continue

    # Sort files based on date (YYYYMMDD) and time (HHMMSS) in filename
    sorted_files = sorted(files, key=lambda filename: extract_datetime(filename))

   # Create an empty audio segment for concatenation
    combined_audio = AudioSegment.empty()

    for filename in sorted_files:
      filepath = os.path.join(root, filename)
      # Load audio segment
      audio_segment = AudioSegment.from_wav(filepath)
      # Append the segment to the combined audio
      combined_audio += audio_segment

    # Extract subfolder name (assuming it's the last part of the path)
    subfolder_name = os.path.basename(root)

    # Generate output filename with combined format
    output_filename = f"{subfolder_name}.wav"
    output_path = os.path.join(directory, output_filename)

    # Export the combined audio to a new WAV file
    combined_audio.export(output_path, format="wav")

    print(f"Concatenated audio files in subfolder: {subfolder_name}")

def extract_datetime(filename):
    """
    This function extracts date and time from the filename format.

    Args:
    filename: Name of the audio file.

    Returns:
    A string combining date and time (YYYYMMDDHHMMSS).
    """
    parts = filename.split("_")
    date, time = parts[0].split("-")
    return f"{date}{time}"

def main(directory, dta_path, min_duration):
    # Step 1: filter files by agent name and move them to the main directory
    select_agents(directory)
    # Step 2: convert audios to .wav format and remove those with duration less than one minute
    clean_and_convert(directory)
    # Step 3: collect phone numbers of eligible participants and create directories for each phone number with multiple audio files
    collect_phone_numbers(directory, dta_path)
    # Step 4: remove audio files with duration less than the specified minimum duration
    clean_and_convert(directory, min_duration)
    # Step 5: concatenate audio files in subfolders
    concatenate_audio_in_subfolders(directory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raw audio files.")
    parser.add_argument("directory", type=str, help="Path to the directory containing audio files.")
    parser.add_argument("dta_path", type=str, help="Path to the .dta file.")
    parser.add_argument("--min_duration", type=int, default=180000, help="Minimum duration in milliseconds for audio files to be kept.")
    args = parser.parse_args()

    main(args.directory, args.dta_path, args.min_duration)

