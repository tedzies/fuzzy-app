import librosa
import numpy as np
SR = 22050  # Sampling rate

def extract_features(path):
    y, sr = librosa.load(path, sr=SR, mono=True)

    # --- TIMBRE ---
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    # --- PITCH ---
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    pitch_contour = pitches[pitches > 0].mean() if np.any(pitches > 0) else 0

    harmonic = librosa.effects.harmonic(y)
    noise = y - harmonic
    hnr = np.mean(librosa.feature.rms(y=harmonic)) / (np.mean(librosa.feature.rms(y=noise)) + 1e-6)

    # --- RHYTHM ---
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # --- INTENSITY ---
    rmse = librosa.feature.rms(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)

    raw = {
        "mfcc_mean": mfcc.mean(),
        "mfcc_std": mfcc.std(),
        "spec_centroid_mean": spec_centroid.mean(),
        "spec_contrast_mean": spec_contrast.mean(),
        "spec_rolloff_mean": spec_rolloff.mean(),
        "chroma_mean": chroma.mean(),
        "pitch_contour": pitch_contour,
        "hnr": hnr,
        "tempo": tempo,
        "rmse_mean": rmse.mean(),
        "zcr_mean": zcr.mean()
    }

    timbre = np.mean([
        raw["mfcc_mean"],
        raw["spec_centroid_mean"],
        raw["spec_contrast_mean"],
        raw["spec_rolloff_mean"]
    ])

    pitch = np.mean([
        raw["chroma_mean"],
        raw["pitch_contour"],
        raw["hnr"]
    ])

    rhythm = raw["tempo"]

    intensity = np.mean([
        raw["rmse_mean"],
        raw["zcr_mean"]
    ])

    aggregated = {
        "timbre": timbre,
        "pitch": pitch,
        "rhythm": rhythm,
        "intensity": intensity
    }

    return raw, aggregated
