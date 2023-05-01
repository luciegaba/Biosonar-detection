import plotly.express as px
import numpy as np
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft


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
    print(audio_ids)
    audio_signals = {}
    for audio in audio_ids:
        print(audio)
        audio_signals[audio] = load_audio_in_dict(audio, audio_path)
    return audio_signals

def plot_site_count_by_pos_label(df):
    """
    Plot the count of IDs by site and positive label.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing the "id" and "pos_label_" columns.
    
    Returns:
    plotly.graph_objs._figure.Figure: Plotly figure object representing the bar plot.
    """
    
    def get_upcase_letters(string):
        """
        Extract uppercase letters from a string.
        
        Parameters:
        string (str): Input string.
        
        Returns:
        str: String containing only uppercase letters from the input string.
        """
        upcase_letters = ""
        for char in string:
            if char.isupper():
                upcase_letters += char
        return upcase_letters

    df["site"] = df["id"].apply(lambda x: get_upcase_letters(x))

    grouped_df = df.groupby(['site', 'pos_label'], as_index=False).count()
    grouped_df = grouped_df.rename(columns={'id': 'count'})
    fig = px.bar(grouped_df, x="site", y="count", color="pos_label", barmode="group")
    return fig


def plot_freq(sig, sig_rate, lw=0.1, fmax=100):
    """
    Plots the time domain and frequency domain representations of a given signal.
    
    Parameters:
        sig (array-like): The input signal.
        sig_rate (float): The sampling rate of the signal in Hz.
        lw (float, optional): The line width for the time domain plot. Default is 0.1.
        fmax (float, optional): The maximum frequency to display in the frequency domain plot. Default is 100.
        
    Returns:
        None
    """
    N = len(sig)
    delta_t = 1 / sig_rate
    times = np.arange(0, N) / sig_rate
    signalf = fft(sig)
    freqs = np.linspace(0.0, 1.0 / (2.0 * delta_t), N // 2)

    fig, axs = plt.subplots(1, 2, figsize=(20, 4))
    
    axs[0].plot(times, sig, linewidth=lw)
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Amplitude')
    axs[0].set_title('Time Domain Representation')

    axs[1].plot(freqs, 2.0 / N * np.abs(signalf[0:N // 2]), linewidth=0.4)
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Amplitude')
    axs[1].set_title('Frequency Domain Representation')
    axs[1].set_xlim([0, fmax])

    plt.show()


def plot_spectrograms(filtered_signal, sr, nperseg_values=[64, 256, 1024, 2048, 4096], hop_size_values = [32, 128, 512, 2048], wins = ['boxcar', 'hamming', 'hann', 'blackman', 'mel'],cmap="magma"):
    # Boucle sur les différentes valeurs de nperseg et hop_size
    for n_idx, nperseg in enumerate(nperseg_values):
        for h_idx, hop_size in enumerate(hop_size_values):
            # Créer une nouvelle figure pour chaque combinaison de nperseg et hop_size
            fig, axs = plt.subplots(1, len(wins), figsize=(5*len(wins), 5), sharey=False)
            fig.tight_layout(pad=3.0)
            fig.suptitle(f'Nperseg={nperseg}, Hop={hop_size}', fontsize=16, y=1.05)

            # Boucle sur les différentes fenêtres
            for w_idx, win in enumerate(wins):
                ax = axs[w_idx]
                if win != "mel":
                    if isinstance(win, str):
                        win_name = win
                    if w_idx == 0:
                        ax.set_ylabel('Frequency (Hz)')

                    D = librosa.amplitude_to_db(np.abs(librosa.stft(filtered_signal, hop_length=hop_size, n_fft=nperseg, window=win)),ref=np.max)
                    ax.pcolormesh(D, cmap=cmap)
                    ax.set_xlabel('Time(s)')
                    ax.set_title(f'STFT with {win_name.capitalize()} Window')
                elif win == "mel":
                    # Compute mel spectrogram
                    mel_spectrogram = librosa.feature.melspectrogram(y=filtered_signal,sr=sr, hop_length=hop_size, n_fft=nperseg)
                    # Convert amplitude to decibels
                    mel_spectrogram_db = librosa.amplitude_to_db(mel_spectrogram,ref=np.max)

                    # Create a new axis sharing the same x-axis
                    ax2 = ax.twinx()
                    # Plot mel spectrogram
                    ax2.pcolormesh(mel_spectrogram_db, cmap=cmap)
                    ax2.set_xlabel('Time(s)')
                    ax2.set_ylabel('Mel')
                    ax.set_title('Mel Spectrogram')
                    
            # Afficher la figure
            plt.show()


def visualize_spectrogram(signal,hop_length=128,n_fft=1024,window="hamming"):
    spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(signal, hop_length=hop_length, n_fft=n_fft, window=window)),ref=np.max)
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='magma')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar()
    plt.show()