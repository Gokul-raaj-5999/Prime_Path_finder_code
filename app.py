import streamlit as st
import pandas as pd
import random
import torch
import base64
from app_rnn_model import recomendation_rnn_model
from app_movie_image import get_poster_url_from_title

def get_top10_recommendations(movie_id):
    top_k = 10
    movie_id = min(movie_id, 1554)  # keep inside [0, 1554]
    x = torch.tensor([[movie_id]], dtype=torch.long)
    with torch.no_grad():
        logits = ppf_model(x)
        probs = torch.softmax(logits, dim=1).numpy()[0]
    topk_preds = probs.argsort()[-top_k:][::-1]
    return topk_preds.tolist()

def load_data():
    df = pd.read_csv("final_combined.csv")
    df = df.drop_duplicates(subset=["movieId", "title"])
    return df

def get_image_as_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ---------- SESSION STATE ----------
if "selected_movies" not in st.session_state:
    st.session_state.selected_movies = []

if "random_movies" not in st.session_state:
    st.session_state.random_movies = []

st.set_page_config(page_title="🎬 Prime Path Finder", layout="wide")
df = load_data()

c1, c2 = st.columns([1,0.2])
with c1:
    st.subheader("🎥 Prime Path finder : A Movie Recomendation System")
with c2:
    if st.button("🔄 Refresh"):
        st.session_state.random_movies = df.sample(10, random_state=random.randint(0, 10000))[["movieId", "title"]].values.tolist()
        st.rerun()

num_movies = df["movieId"].nunique() + 1
ppf_model = recomendation_rnn_model(num_movies=1555)
ppf_model.load_state_dict(torch.load("model_256_0_001.pth", map_location="cpu"))
ppf_model.eval()

# Only generate random movies once per load 
if not st.session_state.random_movies:
    st.session_state.random_movies = df.sample(10, random_state=random.randint(0, 10000))[["movieId", "title"]].values.tolist()
random_movies = st.session_state.random_movies

cols = st.columns(5)

for i, (movie_id, movie_title) in enumerate(random_movies):
    col = cols[i % 5]
    with col:
        # Fetch poster for each movie title
        print(movie_title)
        poster_url = get_poster_url_from_title(movie_title)
        # Display poster image
        # st.image(poster_url,width=10, use_container_width=True)
        if not poster_url:
            image_base64 = get_image_as_base64("Image-not-found.png")
            poster_url = f"data:image/png;base64,{image_base64}"
        st.markdown(
            f"""
            <div style="text-align:center;">
                <img src="{poster_url}" style="width:150px; height:150px; object-fit:cover; border-radius:8px;">
                <p style="font-size:13px;">{movie_title}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Movie button
        if st.button(f"🎬 {movie_id} : {movie_title}", key=f"btn_{movie_id}"):
            movie_entry = {"movieId": movie_id, "title": movie_title}
            if movie_entry not in st.session_state.selected_movies:
                st.session_state.selected_movies.append(movie_entry)
                st.success("✅ Added!")
                
                # Get top 10 recommendations and update random_movies
                top10_ids = get_top10_recommendations(movie_id)
                recommended_movies = df[df["movieId"].isin(top10_ids)][["movieId", "title"]].values.tolist()
                st.session_state.random_movies = recommended_movies
                st.rerun()


    if (i + 1) % 5 == 0 and i + 1 < len(random_movies):
        cols = st.columns(5)

# ---------- Display Selected Movies ----------
if st.session_state.selected_movies:
    st.subheader("🎯 Selected Movies")
    selected_df = pd.DataFrame(st.session_state.selected_movies)
    st.dataframe(selected_df, use_container_width=True)
else:
    st.info("No movies selected yet.")
