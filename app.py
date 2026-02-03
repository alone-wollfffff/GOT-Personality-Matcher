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


def set_bg_and_audio(video_path, audio_path):
    if os.path.exists(video_path) and os.path.exists(audio_path):
        with open(video_path, "rb") as f:
            v_b64 = base64.b64encode(f.read()).decode()

        st.markdown(f"""
            <style>
                /* Import the Trajan-style font */
                @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&display=swap');

                #bg-video {{
                    position: fixed;
                    top: 0; left: 0;
                    width: 100vw; height: 100vh;
                    z-index: -1;
                    object-fit: cover;
                    filter: brightness(50%);
                }}

                /* Targeting ALL headers and specific text to match GoT style */
                h1, h2, h3, .stMarkdown p {{
                    font-family: 'Cinzel', serif !important;
                    
                    letter-spacing: 5px !important;
                }}

                /* Metallic Gold Gradient for the Main Titles */
                h1, h2 {{
                    background: linear-gradient(to bottom, #f3e5ab 0%, #8a7345 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    filter: drop-shadow(3px 3px 5px rgba(0,0,0,0.8));
                    font-weight: 700 !important;
                }}

                /* Styling for the instruction text */
                .stMarkdown p {{
                    color: #c0c0c0 !important; /* Silver/Stone color */
                    letter-spacing: 2px !important;
                    font-size: 1.1rem;
                }}

                /* Transparent Streamlit UI */
                .stApp {{ background: transparent; }}
                
                
                /* Make the dropdown labels match the theme */
                label {{
                    font-family: 'Cinzel', serif !important;
                    color: #e2d1a6 !important;
                    text-transform: uppercase;
                    letter-spacing: 2px;
                }}
            </style>
            
            <video id="bg-video" loop muted playsinline>
                <source src="data:video/mp4;base64,{v_b64}" type="video/mp4">
            </video>
        """, unsafe_allow_html=True)

        # 2. Page Content using the new styles
        st.title("𐌕𝔥𝔢 𝔰𝔬𝔫𝔤 𝔬𝔣 𝔦𝔠𝔢 𝔞𝔫𝔡 𝔣𝔦𝔯𝔢")
        st.write("Click anywhere on the screen to Dracarys...")

        # 3. Audio Widget
        st.sidebar.audio(audio_path, format="audio/mp4", loop=True)

        # 4. Synchronized Play Script
        st.components.v1.html("""
            <script>
                const startExperience = () => {
                    const doc = window.parent.document;
                    const video = doc.getElementById('bg-video');
                    const audios = doc.querySelectorAll('audio');
                    if (video) video.play();
                    audios.forEach(a => { a.muted = false; a.play(); });
                };
                window.parent.document.addEventListener('click', startExperience, { once: true });
            </script>
        """, height=0)

    else:
        st.error("Check filenames! Ensure 'got.mp4' and 'got.m4a' are in the root folder.")

# Call the function
set_bg_and_audio("got.mp4", "got.m4a")

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
st.title("ｖ𝚊𝚕𝚊𝚛 ｍ𝚘𝚛𝚐𝚑𝚞𝚕𝚒𝚜...")
character = st.selectbox("Select Character in Westeros ?", df['char'].values)

if st.button("Reveal Match.."):
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
        st.markdown(f"## Match : {match_name}")
        img2 = get_image(match_name)
        if img2:
            st.image(img2, caption=match_name, use_container_width=True, clamp=True, output_format="auto")
        else:
            st.warning("Image not available")