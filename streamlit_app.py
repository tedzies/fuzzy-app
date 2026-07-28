import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
# Import the extraction function from your separate file
from feature_extract import extract_features

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Music Mood Classifier", page_icon="🎵")

st.title("🎵 Fuzzy Music Mood Classifier")
st.markdown("""
Klasifikasi mood musik menggunakan logika **Fuzzy Mamdani** dengan rule-base yang dibangung dengan **K-Medoids**.
""")
st.markdown("""
Akurasi sistem hanya **44%** 
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
    val = float(x.item()) if hasattr(x, 'item') else float(x)
    a, b, c = p_feat['p15'], p_feat['p50'], p_feat['p95']
    if term == 'Low':
        return max(0, min(1, (b-val)/(b-a))) if val < b else 0.0
    if term == 'Medium':
        if val <= a or val >= c: return 0.0
        return (val-a)/(b-a) if val <= b else (c-val)/(c-b)
    if term == 'High':
        return max(0, min(1, (val-b)/(c-b))) if val > b else 0.0

def fuzzy_inference_cog(feat_values, rules):
    mems = {f: {t: get_membership(feat_values[f], params[f], t) for t in ['Low', 'Medium', 'High']} for f in features}
    mood_activation = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
    activated_rules_details = []

    for i, rule in enumerate(rules):
        strengths = {f: mems[f][rule['if'][f]] * f_weights[f] for f in features}
        raw_strength = min(strengths.values())

        if raw_strength > 0:
            weighted_strength = raw_strength * rule['cf']
            mood_activation[rule['then']] = max(mood_activation[rule['then']], weighted_strength)
             # Clean formatting for the rules table
            conditions_str = " AND ".join([f"{k} is {v}" for k, v in rule['if'].items()])
            activated_rules_details.append({
                'Rule ID': i + 1,
                'IF (Conditions)': conditions_str,
                'THEN (Mood)': inv_mood_map[rule['then']].capitalize(),
                'CF': rule['cf'],
                'Final Strength': round(weighted_strength, 4)
            })

    num = sum(mood_activation[m] * m for m in [1, 2, 3, 4])
    den = sum(mood_activation.values())

    res_idx = int(round(num / den)) if den > 0 else 2
    return inv_mood_map[res_idx], mood_activation, activated_rules_details

# --- 3. UI INTERACTION ---
uploaded_file = st.file_uploader("Unggah file MP3", type=["mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    with st.spinner("Sedang mengekstrak fitur audio..."):
        try:
            with open("temp.mp3", "wb") as f:
                f.write(uploaded_file.getbuffer())

            features_raw = extract_features("temp.mp3")
            feat = features_raw[1] if isinstance(features_raw, (tuple, list)) else features_raw

            st.subheader("Hasil Analisis Fitur")
            st.write(pd.DataFrame([feat]))

            # Inference
            mood_result, activation, active_rules = fuzzy_inference_cog(feat, k_medoid_rules)

            st.success(f"### Prediksi Mood: {mood_result.upper()}")

            st.write("**Distribusi Aktivasi Mood:**")
            total_act = sum(activation.values())
            if total_act > 0:
                cols = st.columns(4)
                for i, m_id in enumerate([1, 2, 3, 4]):
                    pct = (activation[m_id] / total_act) * 100
                    cols[i].metric(inv_mood_map[m_id].capitalize(), f'{pct:.1f}%')

            if active_rules:
                st.subheader("Aturan yang Aktif")
                st.table(pd.DataFrame(active_rules))
            else:
                st.info("Tidak ada aturan yang aktif untuk track ini.")

        except Exception as e:
            st.error(f'Terjadi kesalahan: {e}')



st.markdown("""
Berdasarkan Penelitian:
[Colab Link](https://colab.research.google.com/drive/1QMOL83xfB_pfQ8nIvvT2u_m3vYPQualm?usp=sharing)
""")