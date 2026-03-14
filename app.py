"""
Smart Noise Pollution Awareness & Sound Analyzer
A professional Streamlit application for environmental sound analysis using FFT.
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import scipy.signal as signal
import scipy.fft as fft
import io
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Page config – must be first Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SoundSense – Noise Pollution Analyzer",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# Global styling
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
  --bg:       #0a0d14;
  --surface:  #111520;
  --border:   #1e2535;
  --accent:   #00e5ff;
  --accent2:  #ff3d71;
  --accent3:  #a259ff;
  --warn:     #ffaa00;
  --ok:       #00e096;
  --text:     #e2e8f0;
  --muted:    #6b7a99;
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] { background: transparent !important; }

h1,h2,h3 { font-family: 'Space Mono', monospace !important; }

/* Sidebar */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}

/* Metric cards */
div[data-testid="metric-container"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  padding: 1rem !important;
}

/* Buttons */
.stButton > button {
  background: linear-gradient(135deg, var(--accent), var(--accent3)) !important;
  color: #000 !important;
  font-family: 'Space Mono', monospace !important;
  font-weight: 700 !important;
  border: none !important;
  border-radius: 8px !important;
  padding: .6rem 1.4rem !important;
  transition: opacity .2s !important;
}
.stButton > button:hover { opacity: .85 !important; }

/* Sliders */
.stSlider > div > div > div { background: var(--accent) !important; }

/* Progress bar */
.stProgress > div > div > div > div {
  background: linear-gradient(90deg, var(--ok), var(--accent), var(--warn), var(--accent2)) !important;
}

/* Expanders */
.streamlit-expanderHeader {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  font-family: 'Space Mono', monospace !important;
  color: var(--accent) !important;
}
.streamlit-expanderContent {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-top: none !important;
  border-radius: 0 0 8px 8px !important;
}

/* Code blocks */
code { color: var(--accent) !important; }

/* Divider */
hr { border-color: var(--border) !important; }

/* Alert banners */
div[data-baseweb="notification"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
}

/* Select box */
.stSelectbox > div > div {
  background: var(--surface) !important;
  border-color: var(--border) !important;
  color: var(--text) !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Matplotlib dark theme
# ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#111520",
    "axes.facecolor":    "#0d1018",
    "axes.edgecolor":    "#1e2535",
    "axes.labelcolor":   "#e2e8f0",
    "axes.titlecolor":   "#e2e8f0",
    "xtick.color":       "#6b7a99",
    "ytick.color":       "#6b7a99",
    "grid.color":        "#1e2535",
    "grid.linestyle":    "--",
    "text.color":        "#e2e8f0",
    "font.family":       "monospace",
})

# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────

def record_audio(duration: int, sample_rate: int = 44100, gain: float = 1.0) -> np.ndarray:
    """Record audio from microphone via sounddevice."""
    import sounddevice as sd
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate,
                   channels=1, dtype="float32")
    sd.wait()
    audio = audio.flatten() * gain
    # Normalise to [-1, 1]
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio


def compute_fft(audio: np.ndarray, sr: int):
    freqs = fft.rfftfreq(len(audio), 1 / sr)
    spectrum = np.abs(fft.rfft(audio))
    return freqs, spectrum


def compute_rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio ** 2)))


def rms_to_db(rms: float) -> float:
    if rms <= 0:
        return -120.0
    return float(20 * np.log10(rms + 1e-9))


def dominant_frequency(freqs, spectrum) -> float:
    idx = np.argmax(spectrum)
    return float(freqs[idx])


def spectral_centroid(freqs, spectrum) -> float:
    total = np.sum(spectrum)
    if total == 0:
        return 0.0
    return float(np.sum(freqs * spectrum) / total)


def zero_crossing_rate(audio: np.ndarray, sr: int) -> float:
    zcr = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))
    return float(zcr * sr)


def classify_sound(dom_freq: float, zcr: float, rms: float, centroid: float) -> tuple[str, str]:
    """Rule-based sound classifier. Returns (label, emoji)."""
    if zcr > 3000 and rms > 0.15 and dom_freq > 1000:
        return "Clap / Percussion", "👏"
    if 80 <= dom_freq <= 300 and 50 <= zcr <= 2500:
        return "Human Voice", "🗣️"
    if dom_freq < 250 and rms > 0.12 and zcr < 800:
        return "Vehicle / Traffic Noise", "🚗"
    if 150 <= centroid <= 4000 and zcr > 1000 and rms > 0.08:
        return "Music / Song", "🎵"
    return "Ambient Environmental Sound", "🌿"


def noise_level_label(db: float) -> tuple[str, str]:
    if db < -40:
        return "Quiet", "#00e096"
    elif db < -20:
        return "Moderate", "#00e5ff"
    elif db < -10:
        return "Loud", "#ffaa00"
    else:
        return "Dangerous", "#ff3d71"


# ─────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────

def plot_waveform(audio: np.ndarray, sr: int) -> plt.Figure:
    t = np.linspace(0, len(audio) / sr, len(audio))
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.plot(t, audio, color="#00e5ff", linewidth=0.6, alpha=0.9)
    ax.fill_between(t, audio, alpha=0.18, color="#00e5ff")
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Amplitude", fontsize=9)
    ax.set_title("Waveform – Time Domain", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.4)
    ax.set_xlim([0, t[-1]])
    fig.tight_layout()
    return fig


def plot_spectrum(freqs, spectrum) -> plt.Figure:
    mask = freqs <= 8000
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.semilogx(freqs[mask], 20 * np.log10(spectrum[mask] + 1e-9),
                color="#a259ff", linewidth=1.2)
    ax.fill_between(freqs[mask], 20 * np.log10(spectrum[mask] + 1e-9),
                    alpha=0.2, color="#a259ff")
    ax.set_xlabel("Frequency (Hz) – log scale", fontsize=9)
    ax.set_ylabel("Magnitude (dB)", fontsize=9)
    ax.set_title("Frequency Spectrum – FFT", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    return fig


def plot_spectrogram(audio: np.ndarray, sr: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 2.5))
    cmap = LinearSegmentedColormap.from_list(
        "ss", ["#0a0d14", "#0d1f4f", "#a259ff", "#00e5ff", "#ffffff"])
    Pxx, freqs_s, bins, _ = ax.specgram(
        audio, NFFT=1024, Fs=sr, noverlap=512, cmap=cmap)
    ax.set_ylim(0, 8000)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Frequency (Hz)", fontsize=9)
    ax.set_title("Spectrogram – Time vs Frequency", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_fourier_demo() -> plt.Figure:
    """Illustrate Fourier series by building a square-ish wave from harmonics."""
    t = np.linspace(0, 1, 2000)
    wave = np.zeros_like(t)
    colors = ["#00e5ff", "#a259ff", "#ff3d71", "#ffaa00", "#00e096"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4.5))

    for i, k in enumerate([1, 3, 5, 7, 9]):
        component = (1 / k) * np.sin(2 * np.pi * k * t)
        wave += component
        ax1.plot(t, component, color=colors[i % len(colors)],
                 linewidth=1.2, label=f"{k}f₀", alpha=0.85)

    ax1.set_title("Individual Sine Wave Components", fontsize=10, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8, ncol=5,
               facecolor="#111520", edgecolor="#1e2535", labelcolor="#e2e8f0")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("Time (s)", fontsize=8)

    ax2.plot(t, wave, color="#00e5ff", linewidth=1.5, label="Combined Signal")
    ax2.fill_between(t, wave, alpha=0.15, color="#00e5ff")
    ax2.set_title("Fourier Sum → Complex Waveform", fontsize=10, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=8,
               facecolor="#111520", edgecolor="#1e2535", labelcolor="#e2e8f0")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("Time (s)", fontsize=8)

    fig.suptitle("Fourier Series Demonstration", fontsize=12, fontweight="bold",
                 color="#a259ff")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
# UI Sections
# ─────────────────────────────────────────────

def section_title():
    st.markdown("""
    <div style="text-align:center; padding: 2.5rem 0 1rem;">
        <div style="font-family:'Space Mono',monospace; font-size:2.4rem; font-weight:700;
                    background:linear-gradient(90deg,#00e5ff,#a259ff,#ff3d71);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    line-height:1.2;">
            🎙️ SoundSense
        </div>
        <div style="font-family:'DM Sans',sans-serif; font-size:1.05rem; color:#6b7a99;
                    margin-top:.4rem; letter-spacing:.05em;">
            Smart Noise Pollution Awareness &amp; Sound Analyzer
        </div>
        <div style="margin-top:.8rem; height:2px;
                    background:linear-gradient(90deg,transparent,#00e5ff,#a259ff,#ff3d71,transparent);
                    border-radius:2px;"/>
    </div>
    """, unsafe_allow_html=True)


def section_awareness():
    with st.expander("📚 Noise Pollution Awareness", expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.markdown("""
**What is Noise Pollution?**

Noise pollution is unwanted or harmful sound that disrupts the natural environment.
It comes from traffic, industry, construction, aircraft, and urban living.
Unlike visible pollution, it's invisible — but its damage is very real.
""")
        col2.markdown("""
**Sound Level Reference Chart**

| Level | Source |
|-------|--------|
| 30 dB | Whisper |
| 60 dB | Normal conversation |
| 70 dB | Vacuum cleaner |
| 85 dB | Heavy traffic ⚠️ |
| 100 dB | Concerts 🔴 |
| 120 dB | Jet engine 💀 |

*Safe limit: below 70 dB continuous.*
""")
        col3.markdown("""
**Health Effects**

- 🧠 **Stress & Anxiety** — chronic noise elevates cortisol
- 👂 **Hearing Loss** — irreversible above 85 dB
- 😴 **Sleep Disruption** — even 40 dB at night is harmful
- ❤️ **Cardiovascular Risk** — linked to hypertension
- 🎓 **Cognitive Impairment** — reduces focus and memory

*Source: WHO Environmental Noise Guidelines*
""")


def section_fourier_demo():
    with st.expander("📐 Fourier Series Concept Demo", expanded=False):
        st.markdown("""
**How FFT Decomposes Sound**

Every complex sound wave is a *sum of simple sine waves* at different frequencies and amplitudes.
The **Fast Fourier Transform (FFT)** reverses this — it breaks any recorded signal into its
frequency components, revealing which pitches are present and how loud each one is.
""")
        fig = plot_fourier_demo()
        st.pyplot(fig)
        plt.close(fig)
        st.markdown("""
> **Formula:** `f(t) = Σ (aₙ cos(nω₀t) + bₙ sin(nω₀t))`  
> The FFT computes coefficients *aₙ* and *bₙ* efficiently in **O(n log n)** time.
""")


def section_recording_controls() -> tuple:
    st.markdown("---")
    st.markdown("### 🎤 Recording Controls")

    col1, col2 = st.columns(2)
    with col1:
        duration = st.slider(
            "Recording Duration (seconds)",
            min_value=2, max_value=120, value=10, step=1,
            help="Choose how long to record — up to 2 minutes."
        )
        # Friendly time display
        mins, secs = divmod(duration, 60)
        label = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
        st.caption(f"⏱ Duration: **{label}**  •  Est. file size: ~{duration * 44100 * 2 / 1024:.0f} KB")

    with col2:
        gain = st.slider(
            "Microphone Sensitivity / Gain",
            min_value=0.5, max_value=10.0, value=1.0, step=0.5,
            help="Boost weak microphone signals. Use 1.0 for normal sensitivity; go higher for quiet environments."
        )
        # Sensitivity tier label
        if gain <= 1.0:
            sens_label, sens_color = "Normal", "#00e096"
        elif gain <= 3.0:
            sens_label, sens_color = "Boosted", "#00e5ff"
        elif gain <= 6.0:
            sens_label, sens_color = "High", "#ffaa00"
        else:
            sens_label, sens_color = "Maximum", "#ff3d71"
        st.markdown(
            f'<span style="color:{sens_color}; font-family:\'Space Mono\',monospace; '
            f'font-size:.85rem;">🔊 Sensitivity: <b>{sens_label}</b> (×{gain})</span>',
            unsafe_allow_html=True
        )

    st.markdown("<br/>", unsafe_allow_html=True)
    record_btn = st.button("⏺  Start Recording", use_container_width=True)
    return duration, gain, record_btn


def section_graphs(audio: np.ndarray, sr: int):
    st.markdown("### 📊 Signal Visualisation")
    tab1, tab2, tab3 = st.tabs(["🌊 Waveform", "📈 Frequency Spectrum", "🌈 Spectrogram"])

    with tab1:
        fig = plot_waveform(audio, sr)
        st.pyplot(fig); plt.close(fig)

    with tab2:
        freqs, spectrum = compute_fft(audio, sr)
        fig = plot_spectrum(freqs, spectrum)
        st.pyplot(fig); plt.close(fig)

    with tab3:
        fig = plot_spectrogram(audio, sr)
        st.pyplot(fig); plt.close(fig)


def section_detection(audio: np.ndarray, sr: int):
    freqs, spectrum = compute_fft(audio, sr)
    rms = compute_rms(audio)
    dom_freq = dominant_frequency(freqs, spectrum)
    centroid = spectral_centroid(freqs, spectrum)
    zcr = zero_crossing_rate(audio, sr)
    db = rms_to_db(rms)
    level_label, level_color = noise_level_label(db)
    sound_label, sound_emoji = classify_sound(dom_freq, zcr, rms, centroid)

    st.markdown("---")
    st.markdown("### 🔍 Sound Detection Result")
    c1, c2 = st.columns(2)
    c1.markdown(f"""
<div style="background:#111520; border:1px solid #1e2535; border-radius:12px;
            padding:1.2rem; text-align:center;">
  <div style="font-size:2.8rem;">{sound_emoji}</div>
  <div style="font-family:'Space Mono',monospace; font-size:1.1rem;
              color:#00e5ff; margin-top:.4rem;">{sound_label}</div>
  <div style="color:#6b7a99; font-size:.85rem; margin-top:.3rem;">Detected sound type</div>
</div>
""", unsafe_allow_html=True)

    c2.markdown(f"""
<div style="background:#111520; border:1px solid #1e2535; border-radius:12px;
            padding:1.2rem; text-align:center;">
  <div style="font-size:2.8rem;">📶</div>
  <div style="font-family:'Space Mono',monospace; font-size:1.1rem;
              color:{level_color}; margin-top:.4rem;">{level_label}</div>
  <div style="color:#6b7a99; font-size:.85rem; margin-top:.3rem;">{db:.1f} dB estimated</div>
</div>
""", unsafe_allow_html=True)

    return rms, dom_freq, centroid, zcr, db, level_label, level_color


def section_noise_meter(db: float, level_label: str, level_color: str):
    st.markdown("---")
    st.markdown("### 📡 Noise Level Meter")
    # Map db range roughly -60 to 0 → 0 to 1
    progress = float(np.clip((db + 60) / 60, 0, 1))
    st.progress(progress)
    st.markdown(f"""
<div style="font-family:'Space Mono',monospace; font-size:1.3rem;
            color:{level_color}; text-align:center; margin-top:.3rem;">
  {level_label.upper()}  •  {db:.1f} dB
</div>
""", unsafe_allow_html=True)
    cols = st.columns(4)
    badges = [("Quiet", "#00e096", "< −40 dB"),
              ("Moderate", "#00e5ff", "−40 – −20 dB"),
              ("Loud", "#ffaa00", "−20 – −10 dB"),
              ("Dangerous", "#ff3d71", "> −10 dB")]
    for col, (name, color, rng) in zip(cols, badges):
        active = name == level_label
        border = f"2px solid {color}" if active else "1px solid #1e2535"
        col.markdown(f"""
<div style="background:#111520; border:{border}; border-radius:10px;
            padding:.7rem; text-align:center;">
  <div style="color:{color}; font-family:'Space Mono',monospace;
              font-size:.85rem; font-weight:700;">{name}</div>
  <div style="color:#6b7a99; font-size:.75rem;">{rng}</div>
</div>""", unsafe_allow_html=True)


def section_health_recommendations(level_label: str):
    st.markdown("---")
    st.markdown("### 💊 Health Recommendations")
    recommendations = {
        "Quiet": {
            "icon": "✅",
            "color": "#00e096",
            "msg": "Environment is safe. No hearing protection needed.",
            "tips": [
                "Maintain this comfortable sound environment.",
                "Use this space for rest, reading, or focused work.",
                "Ideal for sleep and recovery."
            ]
        },
        "Moderate": {
            "icon": "ℹ️",
            "color": "#00e5ff",
            "msg": "Acceptable levels for short-term exposure.",
            "tips": [
                "Prolonged exposure (8+ hours) may cause fatigue.",
                "Take regular quiet breaks every 1–2 hours.",
                "Consider noise-reducing headphones for focus work."
            ]
        },
        "Loud": {
            "icon": "⚠️",
            "color": "#ffaa00",
            "msg": "Elevated noise — reduce exposure time.",
            "tips": [
                "Wear ear protection if exposure exceeds 30 minutes.",
                "Move away from the noise source if possible.",
                "Avoid listening to music at high volume in this environment.",
                "Report persistent loud noise to local authorities."
            ]
        },
        "Dangerous": {
            "icon": "🚨",
            "color": "#ff3d71",
            "msg": "DANGEROUS noise levels detected! Immediate action required.",
            "tips": [
                "Leave the area or use proper hearing protection immediately.",
                "Exposure over 2 minutes can cause irreversible hearing damage.",
                "Use industrial-grade earplugs or earmuffs (NRR 25+).",
                "Consult an audiologist if you experience ringing in ears.",
                "Document and report the source — this may violate regulations."
            ]
        }
    }
    rec = recommendations.get(level_label, recommendations["Quiet"])
    st.markdown(f"""
<div style="background:#111520; border:1.5px solid {rec['color']};
            border-radius:12px; padding:1.4rem; margin-bottom:1rem;">
  <div style="font-size:1.6rem; display:inline;">{rec['icon']}</div>
  <span style="font-family:'Space Mono',monospace; color:{rec['color']};
               font-size:1rem; margin-left:.6rem; font-weight:700;">{rec['msg']}</span>
  <ul style="margin-top:.8rem; color:#e2e8f0; font-size:.92rem; line-height:1.8;">
    {''.join(f"<li>{t}</li>" for t in rec['tips'])}
  </ul>
</div>
""", unsafe_allow_html=True)


def section_statistics(rms, dom_freq, centroid, zcr, db):
    st.markdown("---")
    st.markdown("### 📐 Signal Statistics")
    cols = st.columns(5)
    metrics = [
        ("RMS Amplitude", f"{rms:.4f}", "Signal energy"),
        ("Dominant Freq", f"{dom_freq:.1f} Hz", "Strongest frequency"),
        ("Spectral Centroid", f"{centroid:.1f} Hz", "Spectral centre of mass"),
        ("Zero-Crossing Rate", f"{zcr:.0f} /s", "Sign changes per second"),
        ("Estimated Level", f"{db:.1f} dB", "Relative to full scale"),
    ]
    for col, (label, value, help_text) in zip(cols, metrics):
        col.metric(label=label, value=value, help=help_text)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    section_title()
    section_awareness()
    section_fourier_demo()

    duration, gain, record_btn = section_recording_controls()

    if record_btn:
        try:
            import sounddevice as sd
            import time as _time
            sr = 44100

            # Live countdown bar
            mins, secs = divmod(duration, 60)
            dur_label = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
            st.markdown(f"#### 🔴 Recording in progress — {dur_label}")
            progress_bar = st.progress(0)
            status_text  = st.empty()

            audio_buf = sd.rec(int(duration * sr), samplerate=sr,
                               channels=1, dtype="float32")

            for i in range(duration):
                _time.sleep(1)
                pct = (i + 1) / duration
                progress_bar.progress(pct)
                remaining = duration - i - 1
                if remaining:
                    status_text.markdown(
                        f'<span style="color:#6b7a99; font-size:.88rem;">⏳ {remaining}s remaining…</span>',
                        unsafe_allow_html=True)
                else:
                    status_text.markdown(
                        '<span style="color:#00e096; font-size:.88rem;">✅ Done! Processing…</span>',
                        unsafe_allow_html=True)

            sd.wait()
            progress_bar.empty()
            status_text.empty()

            # Apply gain & normalise
            audio = audio_buf.flatten() * gain
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak

            st.success(f"✅ Recorded {duration}s of audio — {len(audio):,} samples @ {sr} Hz")

            # Playback
            st.markdown("---")
            st.markdown("### ▶️ Audio Playback")
            import scipy.io.wavfile as wavfile
            buf = io.BytesIO()
            wavfile.write(buf, sr, (audio * 32767).astype(np.int16))
            buf.seek(0)
            st.audio(buf, format="audio/wav")

            section_graphs(audio, sr)
            rms, dom_freq, centroid, zcr, db, level_label, level_color = \
                section_detection(audio, sr)
            section_noise_meter(db, level_label, level_color)
            section_health_recommendations(level_label)
            section_statistics(rms, dom_freq, centroid, zcr, db)

        except Exception as e:
            st.error(f"❌ Recording failed: {e}")
            st.info("Make sure `sounddevice` is installed and a microphone is available.\n\n"
                    "`pip install sounddevice`")

    else:
        # Demo mode with synthetic signal
        st.markdown("---")
        st.markdown("""
<div style="background:#111520; border:1px dashed #1e2535; border-radius:12px;
            padding:2rem; text-align:center; color:#6b7a99;">
  <div style="font-size:2rem;">🎙️</div>
  <div style="font-family:'Space Mono',monospace; margin-top:.6rem; color:#e2e8f0;">
    Press <span style="color:#00e5ff;">⏺ Record</span> to analyse live sound
  </div>
  <div style="font-size:.88rem; margin-top:.4rem;">
    Adjust duration &amp; sensitivity above, then hit the button
  </div>
</div>
""", unsafe_allow_html=True)

        with st.expander("🔬 Preview with Synthetic Demo Signal", expanded=False):
            st.caption("This is a synthesised signal for demonstration. Press Record for real audio.")
            sr_demo = 44100
            t_demo = np.linspace(0, 2, sr_demo * 2)
            demo_audio = (
                0.5 * np.sin(2 * np.pi * 440 * t_demo) +
                0.3 * np.sin(2 * np.pi * 880 * t_demo) +
                0.15 * np.sin(2 * np.pi * 1320 * t_demo) +
                0.05 * np.random.randn(len(t_demo))
            )
            demo_audio /= np.max(np.abs(demo_audio))
            section_graphs(demo_audio, sr_demo)
            rms, dom_freq, centroid, zcr, db, level_label, level_color = \
                section_detection(demo_audio, sr_demo)
            section_noise_meter(db, level_label, level_color)
            section_health_recommendations(level_label)
            section_statistics(rms, dom_freq, centroid, zcr, db)

    # Footer
    st.markdown("---")
    st.markdown("""
<div style="text-align:center; color:#6b7a99; font-size:.8rem; padding:1rem 0;">
  SoundSense · Built with Streamlit + NumPy + SciPy + Matplotlib ·
  <span style="color:#a259ff;">FFT-powered</span> noise pollution awareness
</div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()