import os
import numpy as np
import librosa

def load_audio_in_dict(audio_id, audio_path):
    """
    Load an audio signal and return it as a dictionary.

    Parameters:
    audio_id (str): The ID of the audio file to be loaded.
    audio_path (str): The path to the folder containing the audio file.

    Returns:
    dict: A dictionary containing the audio signal and its sample rate.
    """
    sig, sr = librosa.load(audio_path + audio_id, sr=None)
    return {"signal": sig, "signal_rate": sr}

def select_labelled_audios(df, label, num_audios,audio_path="data/raw/X_train/"):
    """
    Select audio signals based on their label.

    Parameters:
    df (pandas DataFrame): A DataFrame containing audio information.
    label (float): The label value to select audio signals for.
    num_audios (int): The number of audio signals to select.
    audio_path (str): Path for audios 

    Returns:
    dict: A dictionary of selected audio signals, where the keys are the audio IDs.
    """
    audio_ids = df[df['pos_label'] == label].head(num_audios).id.values.tolist()
    audio_signals = {}
    for audio in audio_ids:
        audio_signals[audio] = load_audio_in_dict(audio, audio_path)
    return audio_signals





def group_filenames_by_site(filenames):
    """
    Groups a list of filenames by site based on the site identifier in the filename.
    
    Args:
    - filenames (list): A list of strings representing filenames
    
    Returns:
    - A dictionary where the keys are site identifiers and the values are lists of filenames
    """
    sites = {}
    for filename in filenames:
        site_id = filename.split('-')[-1].split('.')[0]
        if site_id not in sites:
            sites[site_id] = []
        sites[site_id].append(filename)
    return sites


def aggregate_npy_files(root_directory, target_file, save=False, filename='aggregated_spectrograms.npy'):
    """
    Aggregates all numpy (.npy) files with the given target filename within the root directory and its subdirectories.
    
    Args:
    - root_directory (str): The root directory to search for .npy files
    - target_file (str): The target filename to search for (including extension .npy)
    - save (bool): Whether or not to save the concatenated spectrograms to a file
    - filename (str): The filename to use if save=True
    
    Returns:
    - A numpy array containing the concatenated contents of all the target .npy files
    """
    all_spectrograms = []
    for subdir, dirs, files in os.walk(root_directory):
        for file_name in files:
            if file_name == target_file:
                file_path = os.path.join(subdir, file_name)
                spectrograms = np.load(file_path)
                all_spectrograms.append(spectrograms)
           
    concatenated_spectrograms = np.concatenate(all_spectrograms, axis=0)
    
    if save:
        np.save(filename, concatenated_spectrograms)
    
    return concatenated_spectrograms
