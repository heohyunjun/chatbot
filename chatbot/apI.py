import streamlit as st
from streamlit_chat import message
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

'''
모델 로드를 사용할떄마다 계속 부르게 되면 latency가 매우 길어짐
따라서 한번 부르고 cache 형태로 불러옴

모델과 동일하게 데이터셋도 캐싱함
'''
@st.cache(allow_output_mutation = True)
def cached_model():
    model = SentenceTransformer('/model')
    return model

@st.cache(allow_output_mutation = True)
def get_dataset():
    df = pd.read_csv('chat_text.csv')
    df['embedding'] = df['embedding'].apply(json.loads)
    return df

model = cached_model()
df = get_dataset()