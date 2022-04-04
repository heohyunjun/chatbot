import streamlit as st
from streamlit_chat import message
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

'''
모델 로드를 사용할떄마다 계속 부르게 되면 latency가 매우 길어짐
따라서 한번 부르고 cache 형태로 불러옴
'''
@st.cache(allow_output_mutation = True)
def cached_model():
    model = SentenceTransformer('/model')
    return model