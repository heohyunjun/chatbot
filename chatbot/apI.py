import streamlit as st
from streamlit_chat import message
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

'''
python 3.9 에서 streamlit 에러가 나지 않음

모델 로드를 사용할떄마다 계속 부르게 되면 latency가 매우 길어짐
따라서 한번 부르고 cache 형태로 불러옴

모델과 동일하게 데이터셋도 캐싱함

with st.form('form', clear_on_submit=True):
- 사용자 질문 입력할 form 생성


if submitted and user_input:
- 만약 유저가 잘입력했으면 ,
1. 유저가 입력한 텍스트를 임베딩
2. 발화 라벨 전부와 코사인 유사도 계산해서 'similarity' 컬럼에 저장
3. 유사도가 가장 높은 행 추출
'''
@st.cache(allow_output_mutation = True)
def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model

@st.cache(allow_output_mutation = True)
def get_dataset():
    df = pd.read_csv('chat_text.csv')
    df['embedding'] = df['embedding'].apply(json.loads)
    return df

model = cached_model()
df = get_dataset()

st.header('chat bot')

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []


# form 작성
# clear_on_submit = True :  텍스트 박스에 입력하고 전송 누르면, 텍스트 박스가 clear
with st.form('form', clear_on_submit=True):
    # 입력 텍스트 박스
    user_input = st.text_input('당신 :', "")

    # 전송 버튼
    submitted = st.form_submit_button('전송')

if submitted and user_input:
    embedding = model.encode(user_input)

    df['similarity'] = df['embedding'].map(lambda x : cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['similarity'].idxmax()]

    # 유저 질문 저장
    st.session_state.past.append(user_input)

    # 챗봇 답변 저장
    st.session_state.generated.append(answer['chatbot'])

for i in range(len(st.session_state['past'])):
    message(st.session_state['past'][i], is_user = True, key = str(i) + '_user')
    if len(st.session_state['generated']) > i :
        message(st.session_state['generated'][i], key= str(i) + '_bot')
