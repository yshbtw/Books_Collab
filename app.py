import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Book Recommender", layout="wide")

st.title("Book Recommendation System")

@st.cache_data
def load_data():
    model = pickle.load(open('model.pkl', 'rb'))
    pt = pickle.load(open('pt1.pkl', 'rb'))
    
    return model, pt

model, pt = load_data()
books_names = sorted(pt.index.tolist())

selected_book = st.selectbox(
    'Choose a book you like:',
    books_names
)

if st.button('Get Recommendations'):
    index = np.where(pt.index == selected_book)[0][0]

    distances, indices = model.kneighbors(pt.iloc[index,:].values.reshape(1, -1), n_neighbors=6)
    
    st.subheader("Here are 5 books you might like:")
    
    cols = st.columns(5)
    

    for idx, col in enumerate(cols):
        col.write(f"**{pt.index[indices[0][idx+1]]}**")
