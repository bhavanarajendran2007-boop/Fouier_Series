import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.fft as fft
from scipy.stats import gmean
import librosa
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import io
from scipy.io.wavfile import write

# ---------------------------------------------------------
# PAGE SETTINGS & UI
# ---------------------------------------------------------
st.set_page_config(
    page_title="SoundSense Pro",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ SoundSense – Advanced Fourier STEM Analyzer")
st.markdown("### Investigating Sound through Mathematics & Signal Processing")

# ---------------------------------------------------------
# MATHEMATICAL FUNCTIONS (Fourier & DSP)
# ---------------------------------------------------------
def apply_window(signal, type="Blackman-Harris"):
    """Applies a window function to reduce spectral leakage in the FFT."""
    N = len(signal)
    if type == "Blackman-Harris":
        return signal * np.blackman(N)
    elif type == "Hann":
        return signal * np.hanning(N)
    return signal

def compute_fft(audio, sr):
    """Computes the Fast Fourier Transform (FFT) of the signal."""
    # Apply windowing for STEM accuracy
    windowed_audio = apply_window(audio)
    N = len(windowed_audio)
    freqs = fft.rfftfreq(N, 1/sr)
    spectrum = np.abs(fft.rfft(windowed_audio))
    return freqs, spectrum

def extract_stem_features(audio, sr):
    """Extracts scientific parameters for sound analysis."""
    # RMS and dB Calculation
    rms = np.sqrt(np.mean(audio**2))
    ref = 0.00002
    db = 20 * np.log10(rms/ref + 1e-9)

    # Spectral Centroid (Center of mass of the sound)
    centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))

    # Zero Crossing Rate (Measure of noisiness)
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))

    # Spectral Flatness (Mathematical 'pureness' of a tone)
    # 1.0 = White Noise, 0.0 = Pure Tone
    _, spectrum = compute_fft(audio, sr)
    flatness = gmean(spectrum + 1e-9) / (np.mean(spectrum) + 1e-9)

    return db, centroid, zcr, flatness

# ---------------------------------------------------------
# SIDEBAR SETTINGS
# ---------------------------------------------------------
st.sidebar.header("STEM Parameters")
st.sidebar.info("The Fourier Transform decomposes a complex signal into its constituent sine waves.")
duration = st.sidebar.slider("Analysis Clip Length (s)", 2, 10, 5)

# ---------------------------------------------------------
# AUDIO CAPTURE & PROCESSING
#