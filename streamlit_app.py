import streamlit as st
import pickle
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Import the extraction function from your separate file
from feature_extract import extract_features

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Music Mood Classifier", page_icon="🎵")

st.title("🎵 AI Music Mood Classifier")
st.markdown("""
Klasifikasi mood musik menggunakan logika **Fuzzy Mamdani** dengan optimasi **K-Medoids** dan **Feature Weight Analysis**.
""")

# --- 1. LOAD MODEL ---
@st.cache_resource
def load_model():
    with open('fuzzy_music_model.pkl', 'rb') as f:
        return pickle.load(f)

data = load_model()
params = data['params']
k_medoid_rules = data['k_medoid_rules']
f_weights = data['f_weights']
features = data['features']
inv_mood_map = data['inv_mood_map']

# --- 2. FUZZY LOGIC FUNCTIONS ---
def get_membership(x, p_feat, term):
    a, b, c = p_feat['p15'], p_feat['p50'], p_feat['p95']
    if term == 'Low':
        return max(0, min(1, (b-x)/(b-a))) if x < b else 0.0
    if term == 'Medium':
        if x <= a or x >= c: return 0.0
        return (x-a)/(b-a) if x <= b else (c-x)/(c-b)
    if term == 'High':
        return max(0, min(1, (x-b)/(c-b))) if x > b else 0.0

def fuzzy_inference_cog(feat_values, rules):
    mems = {f: {t: get_membership(feat_values[f], params[f], t) for t in ['Low', 'Medium', 'High']} for f in features}
    mood_activation = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
    
    for rule in rules:
        # Strength calculation with Feature Weights
        strength = min(mems[f][rule['if'][f]] * f_weights[f] for f in features)
        if strength > 0:
            # Weight by Certainty Factor (Purity)
            mood_activation[rule['then']] = max(mood_activation[rule['then']], strength * rule['cf'])

    num = sum(mood_activation[m] * m for m in [1, 2, 3, 4])
    den = sum(mood_activation.values())
    
    res_idx = int(round(num / den)) if den > 0 else 2
    return inv_mood_map[res_idx], mood_activation

# --- 3. UI INTERACTION ---
uploaded_file = st.file_uploader("Unggah file MP3", type=["mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    
    with st.spinner("Sedang mengekstrak fitur audio..."):
        try:
            # Save temp file for librosa processing
            with open("temp.mp3", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Extract features (using the function from feature_extractor.py)
            features_raw = extract_features("temp.mp3")
            
            # Handle tuple output if necessary
            feat = features_raw[1] if isinstance(features_raw, tuple) else features_raw
            
            st.subheader("Hasil Analisis Fitur")
            st.write(pd.DataFrame([feat]))

            # Inference
            mood_result, activation = fuzzy_inference_cog(feat, k_medoid_rules)
            
            st.success(f"### Prediksi Mood: {mood_result.upper()}")

            # Display Probabilities/Activation
            st.write("**Distribusi Aktivasi Mood:**")
            total_act = sum(activation.values())
            if total_act > 0:
                cols = st.columns(4)
                for i, m_id in enumerate([1, 2, 3, 4]):
                    pct = (activation[m_id] / total_act) * 100
                    cols[i].metric(inv_mood_map[m_id].capitalize(), f"{pct:.1f}%")
            
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")