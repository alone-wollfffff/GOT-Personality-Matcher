import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import base64
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="GOT Personality Matcher",
    page_icon="got.png",
    layout="wide"
)

# ---------------- MEDIA HANDLING ----------------
def get_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background_and_audio(video_path):
    if os.path.exists(video_path):
        video_b64 = get_base64(video_path)
        
        design_html = f"""
        <style>
        #bg-video {{
            position: fixed;
            top: 0;
            left: 0;
            min-width: 98%; 
            min-height: 98%;
            z-index: -1;
            object-fit: cover;
            filter: brightness(60%);
        }}
        .stApp {{
            background: transparent;
            color: white;
        }}
        h1, h2, h3, p {{
            text-shadow: 2px 2px 8px #000000;
        }}
        </style>

        <video autoplay loop muted playsinline id="bg-video">
            <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
        </video>
        """
        st.markdown(design_html, unsafe_allow_html=True)
        st.sidebar.header("🔊 Valar Morghulis")
        st.sidebar.audio(video_path, loop=True)
    else:
        st.error("Video file not found.")

set_background_and_audio("got.mp4")

# ---------------- DATA & LOGIC ----------------
@st.cache_resource
def load_data():
    df = pd.read_csv("characters.csv")
    with open("embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)
    return df, embeddings

@st.cache_resource
def get_api_characters():
    url = "https://thronesapi.com/api/v2/Characters"
    try:
        data = requests.get(url).json()
        # Normalize API names
        return {char['fullName'].strip().lower(): char['imageUrl'] for char in data}
    except:
        return {}

df, embeddings = load_data()
api_images = get_api_characters()

# ---------------- UI ----------------
st.title("🐉 GOT Personality Matcher")

character = st.selectbox("Who are you in Westeros?", df['char'].values)

if st.button("Reveal My Match"):
    # Find match using cosine similarity
    idx = df[df['char'] == character].index[0]
    sims = cosine_similarity(embeddings[idx].reshape(1, -1), embeddings)[0]
    sims[idx] = -1  # Exclude self
    match_idx = np.argmax(sims)
    match_name = df.iloc[match_idx]['char']
    
    col1, col2 = st.columns(2)
    
    # Floating image CSS
    st.markdown("""
    <style>
    .floating-img {
        width: 100px;
        height: 150px;
        object-fit: contain;
        filter: drop-shadow(0px 0px 15px rgba(255, 215, 0, 0.5));
        border-radius: 10px;
        margin: 20px auto;
        display: block;
        transition: transform 0.3s;
    }
    .floating-img:hover {
        transform: scale(1.05);
    }
    </style>
    """, unsafe_allow_html=True)

    # Helper function to get image
    def get_image(name):
        img_url = api_images.get(name.strip().lower(), None)
        if img_url:
            return img_url
        else:
            # fallback local image if API fails
            fallback_path = f"images/{name.replace(' ', '_')}.jpg"
            if os.path.exists(fallback_path):
                return fallback_path
        return None

    # Left column - User
    with col1:
        st.markdown(f"## {character}")
        img1 = get_image(character)
        if img1:
            st.image(img1, caption=character, use_container_width=True, clamp=True, output_format="auto")
        else:
            st.warning("Image not available")

    # Right column - Match
    with col2:
        st.markdown(f"## Match: {match_name}")
        img2 = get_image(match_name)
        if img2:
            st.image(img2, caption=match_name, use_container_width=True, clamp=True, output_format="auto")
        else:
            st.warning("Image not available")