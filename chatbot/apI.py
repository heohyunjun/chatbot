import streamlit as st
from streamlit_chat import message
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import datetime
from confirm_button_hack import cache_on_button_press
'''

No streamlit error in python 3.9

If I keep calling every time you use the model load, the latency is very long.
Therefore, it is called once and loaded in the form of cache.

모델과 동일하게 데이터셋도 캐싱함

with st.form('form', clear_on_submit=True):
- 사용자 질문 입력할 form 생성


if submitted and user_input:
- 만약 유저가 잘입력했으면 ,
1. 유저가 입력한 텍스트를 임베딩
2. 발화 라벨 전부와 코사인 유사도 계산해서 'similarity' 컬럼에 저장
3. 유사도가 가장 높은 행 추출
'''

# 불용어 제거


st.title("chatbot")
tfidf = TfidfVectorizer(stop_words='english')
def main():

    @st.cache(allow_output_mutation=True)
    def cached_model():
        model = SentenceTransformer('jhgan/ko-sroberta-multitask')

        return model


    @st.cache(allow_output_mutation=True)
    def get_dataset():
        df = pd.read_csv('./chat_text.csv')
        df['embedding'] = df['embedding'].apply(json.loads)

        return df


    @st.cache(allow_output_mutation=True)
    def get_netflix():
        df2 = pd.read_csv('./net_rec.csv')

        return df2


    @st.cache(allow_output_mutation=True)
    def cosine_netflix(x: pd.DataFrame):
        # 영화설명 컬럼 학습
        tfidf_matrix = tfidf.fit_transform(x['description'])
        # linear_kernel : 사이킷런에서 제공하는 문서 유사도(코사인 유사도)를 계산 할 수 있음
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

        return cosine_sim


    # model_load
    model = cached_model()

    # dataset load
    df = get_dataset()

    # movie_dataset load
    df2 = get_netflix()

    # cosine_similarity load
    cosine_sim = cosine_netflix(df2)

    # d-day for switch jobs
    today = datetime.date.today()
    target_date = datetime.date(2024, 4, 7)
    d_day = target_date - today




    '''
        clear_on_submit = True
            - if type in the TextBox and submit, eht TextBox will clear
    '''

    st.header('chat bot')

    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []


    # st.dataframe(df2[['title']].T)


    # st.write('You selected', option)
    with st.form('form', clear_on_submit=True):
        # text_box for input

        option = st.selectbox(
            '추천가능한 영화 리스트',
            df2['title']

        )

        # TODO
        user_input = st.text_input("당신", "")

        col1, col2 = st.columns(2)

        # submit_buttion
        with col1:
            submitted = st.form_submit_button('전송')
        with col2:
            # 영화 추천 버튼
            netflix_submitted = st.form_submit_button('영화 추천 받기')

    if netflix_submitted:
        st.session_state.past.append(option)
        idx = df2[df2['title'] == option].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        movie_idx = sim_scores[1][0]
        title = df2['title'].iloc[movie_idx]

        answer = title + "를 추천합니다"
        st.session_state.generated.append(answer)

    if user_input == "":
        if submitted:
            answer = '궁금한것을 말씀해주세요'
            st.session_state.past.append(user_input)
            st.session_state.generated.append(answer)
    else:
        # save user_chatting
        st.session_state.past.append(user_input)
        print(user_input)
        if submitted:
            if user_input == "카카오 이직까지":
                m_answer = f"{user_input} {d_day.days}일 남았습니다"
                st.session_state.generated.append(m_answer)
            embedding = model.encode(user_input)
            df['similarity'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
            answer = df.loc[df['similarity'].idxmax()]

            # 챗봇 답변 저장
            st.session_state.generated.append(answer['챗봇'])
        else:
            pass


    for i in range(len(st.session_state['past'])):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        if len(st.session_state['generated']) > i:
            message(st.session_state['generated'][i], key=str(i) + '_bot')

root_password = 'hyunjun'

@cache_on_button_press('Authenticate')
def authenticate(password):
    return password == root_password

password = st.text_input('password', type="password")

if authenticate(password):
    st.success('You are authenticated!')
    main()
else:
    # st.header("you are not authenticated")
    st.error('The password is invalid.')