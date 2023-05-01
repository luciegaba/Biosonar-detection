import os
import numpy as np
import cv2
import librosa
from scipy.signal import butter, sosfiltfilt
import matplotlib.pyplot as plt

class AudioProcessor:
    
    """
    A class to process audio files and create spectrograms.


    Attributes
    ----------
    audio_dir : str
        The directory containing the audio files.
    output_base_dir : str
        The base directory to save the processed spectrograms and labels.
    frame_size : int, optional
        The number of samples in each frame for the short-time Fourier transform (default is 2048).
    hop_size : int, optional
        The number of samples to advance between frames (default is 128).
    window : str, optional
        The window function to apply to each frame (default is "hamming").
    lowcut : float, optional
        The lower frequency cutoff for the bandpass filter (default is 5000).
    highcut : float, optional
        The upper frequency cutoff for the bandpass filter (default is 100000).
    order_bandpass : int, optional
        The order of the bandpass filter (default is 6).
    size_image : tuple, optional
        The size of the resized spectrogram image (default is (150, 150)).
    grey_option: boolean, optional
        The option to get non-colored images

    Methods
    -------
    generate_signals(audio_path):
        Load the audio file and return the audio signal and its sampling rate.
    butter_bandpass_filter(data, sr):
        Apply a bandpass filter to the audio signal.
    create_spectrogram(audio_path):
        Create a spectrogram from the audio file.
    resize_and_normalize_spectrogram(spectrogram):
        Resize and normalize the spectrogram image.
    process_train(group_index_files, files, df):
        Process a group of audio files and save the spectrograms and labels as .npy files.
    process_inference(group_index_files, files):
        Process a group of audio files 
                
        
    """
    def __init__(self, audio_dir, output_base_dir, frame_size=1028, hop_size=128,window="hamming",lowcut=5000,highcut=100000,order_bandpass=6,size_image = (150, 150),cmap="magma",grey_option=False):
        self.audio_dir = audio_dir
        self.output_base_dir = output_base_dir
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.window=window
        self.lowcut = lowcut
        self.highcut = highcut
        self.order_bandpass = order_bandpass
        self.size_image = size_image
        self.cmap = cmap
        self.grey_option = grey_option

    def generate_signals(self, audio_path):
        """
        Load the audio file and return the audio signal and its sampling rate.

        Parameters
        ----------
        audio_path : str
            The path of the audio file.

        Returns
        -------
        sound_info : np.array
            The audio signal.
        sampling_rate : int
            The sampling rate of the audio signal.
        """
        sound_info, sampling_rate = librosa.load(audio_path, sr=None)
        return sound_info, sampling_rate

    def butter_bandpass_filter(self, data, sr):
        """
        Apply a bandpass filter to the audio signal.

        Parameters
        ----------
        data : np.array
            The audio signal.
        sr : int
            The sampling rate of the audio signal.

        Returns
        -------
        y : np.array
            The filtered audio signal.
        """
        sos = butter(self.order_bandpass, [self.lowcut, self.highcut], btype='band', fs=sr, output='sos')
        y = sosfiltfilt(sos, data)
        return y
    
    def reduce_noise(self, signal, sr):
        """
        Reduce the noise in the audio signal using noisereduce.

        Parameters
        ----------
        signal : np.array
            The audio signal.
        sr : int
            The sampling rate of the audio signal.

        Returns
        -------
        reduced_noise_signal : np.array
            The audio signal with reduced noise.
        """
        reduced_noise_signal = nr.reduce_noise(signal, sr=sr, n_std_thresh_stationary=1.5, stationary=True)
        return reduced_noise_signal
    

    def create_spectrogram(self, audio_path):
        """
        Create a spectrogram from the audio
        
        Parameters
        ----------
        audio_path : str
            The path of the audio file.

        Returns
        -------
        Y_log_scale : np.array
            The log-scaled spectrogram.
        """
    
        sig, sr = self.generate_signals(audio_path)
        y = self.butter_bandpass_filter(sig, sr)
        reduced_noise_y = self.reduce_noise(y, sr)
        #Use fft to more speeder algorithm
        S_scale = librosa.stft (reduced_noise_y, n_fft=self.frame_size, hop_length=self.hop_size,window=self.window)
        Y_log_scale = librosa.power_to_db(S_scale)
        return Y_log_scale


    def resize_and_normalize_spectrogram(self, spectrogram):
        """
        Resize and normalize the spectrogram image.

        The function resizes the input spectrogram to the desired dimensions and normalizes it by
        scaling its values to the range [0, 1]. Then, it converts the normalized spectrogram to
        an RGB image using the specified colormap or can also converted into grey scale.

        Parameters
        ----------
        spectrogram : np.array
            The input spectrogram.

        Returns
        -------
        rgb_image : np.array
            The resized and normalized RGB (or grey) image of the spectrogram.
        """
        resized_spectrogram = cv2.resize(spectrogram, self.size_image, interpolation=cv2.INTER_CUBIC)
        normalized_spectrogram = (resized_spectrogram - np.min(resized_spectrogram)) / (np.max(resized_spectrogram) - np.min(resized_spectrogram))  * 255.0
        rgb_image = plt.get_cmap(self.cmap)(normalized_spectrogram)
        rgb_image = (rgb_image[:, :, :3]).astype(np.uint8)
        if self.grey_option is True:
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        return rgb_image

    def process_train(self,group_index_files, files, labels_df):
        """
        Process a group of audio files and save the spectrograms and labels as .npy files.

        Parameters
        ----------
        group_index_files : str
            The name of the group of audio files.
        files : list
            A list of audio file names.
        df : DataFrame
            A DataFrame containing the labels for the audio files.

        Returns
        -------
        None
        """
        
        print(f"PROCESSING {group_index_files}")
        output_dir = os.path.join(self.output_base_dir, group_index_files)

        spectrograms_file = os.path.join(output_dir, "spectrograms.npy")
        labels_file = os.path.join(output_dir, "labels.npy")

        if os.path.exists(output_dir) and os.path.isfile(spectrograms_file) and os.path.isfile(labels_file):
            print(f"Skipping {group_index_files} as files already exist.")
            return

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        spectrograms = []
        labels = []

        for audio_file in files:
            if audio_file.endswith(".wav"):
                audio_path = os.path.join(self.audio_dir, audio_file)
                spectrogram = self.create_spectrogram(audio_path)
                spectrograms.append(spectrogram)
                label = labels_df["pos_label"][labels_df["id"] == audio_file].values[0]
                labels.append(label)

        resized_spectrograms = []
        for spectrogram in spectrograms:
            resized_spectrogram = self.resize_and_normalize_spectrogram(spectrogram)
            resized_spectrograms.append(resized_spectrogram)

        resized_spectrograms_array = np.stack(resized_spectrograms)

        np.save(spectrograms_file, resized_spectrograms_array)
        np.save(labels_file,labels)


    def process_inference(self, group_index_files, files):
        """
        Process a group of test audio files and save the spectrograms as .npy files.

        Parameters
        ----------
        group_index_files : str
            The name of the group of test audio files.
        files : list
            A list of test audio file names.

        Returns
        -------
        None
        """
        
        print(f"PROCESSING {group_index_files}")
        output_dir = os.path.join(self.output_base_dir, group_index_files)

        spectrograms_file = os.path.join(output_dir, "spectograms.npy")

        if os.path.exists(output_dir) and os.path.isfile(spectrograms_file):
            print(f"Skipping {group_index_files} as files already exist.")
            return

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        spectrograms = []

        for audio_file in files:
            if audio_file.endswith(".wav"):
                audio_path = os.path.join(self.audio_dir, audio_file)
                spectrogram = self.create_spectrogram(audio_path)
                spectrograms.append(spectrogram)

        resized_spectrograms = []
        for spectrogram in spectrograms:
            resized_spectrogram = self.resize_and_normalize_spectrogram(spectrogram)
            resized_spectrograms.append(resized_spectrogram)

        resized_spectrograms_array = np.stack(resized_spectrograms)

        np.save(spectrograms_file, resized_spectrograms_array)
        
        
        
def median_noise_reduction(audio, sr, hop_length=128, n_fft=1024):
    # Calculer le spectrogramme du signal d'entrée
    input_spectrogram = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))

    # Calculer le spectrogramme médian du bruit de fond
    noise_audio = audio
    noise_spectrogram = np.abs(librosa.stft(noise_audio, n_fft=n_fft, hop_length=hop_length))
    median_noise_spectrogram = np.median(noise_spectrogram, axis=1, keepdims=True)

    # Soustraire le spectrogramme médian du bruit de fond du spectrogramme du signal d'entrée
    reduced_spectrogram = input_spectrogram - median_noise_spectrogram
    reduced_spectrogram = np.maximum(reduced_spectrogram, 0)  # Enlever les valeurs négatives

    # Reconstruire le signal débruité à partir du spectrogramme réduit
    phase = np.angle(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    reduced_signal = librosa.istft(reduced_spectrogram * np.exp(1j * phase), hop_length=hop_length)

    return reduced_signal