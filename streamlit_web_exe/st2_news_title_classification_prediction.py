# 使用者在網頁介面上與之互動後，結果會顯示在網頁上。 
# 執行指令：python -m streamlit run streamlit_web_exe/st2_news_title_classification_prediction.py
import streamlit as st
import numpy as np
import jieba
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import joblib
import pickle

MAX_LEN = 30
NUM_WORDS = 10000
# 1. load models
@st.cache_resource
def load_lstm_model():
    model = load_model("model_files/lstm_news_model.h5")
    
    # 載入 tokenizer
    with open("model_files/tokenizer_lstm.pickle", "rb") as f:
        tokenizer = pickle.load(f)
    
    # 載入 LabelEncoder
    with open("model_files/label_encoder_lstm.pickle", "rb") as f:
        label_encoder = pickle.load(f)
    
    return model, tokenizer, label_encoder
@st.cache_resource
def load_lr_model():
    model = joblib.load("model_files/LogisticRegression.joblib")
    vectorizer = joblib.load("model_files/tfidf_vectorizer.joblib")
    return model, vectorizer
@st.cache_resource
def load_dt_model():
    model = joblib.load("model_files/DecisionTree1.joblib")
    vectorizer = joblib.load("model_files/tfidf_vectorizer.joblib")
    return model, vectorizer

# 2. predict
def predict_lstm(title, model, tokenizer, label_encoder, max_len=30, return_top_3=True):
    # 1️⃣ jieba 斷詞
    cut_title = " ".join(jieba.cut(title))
    
    # 2️⃣ 轉成序列
    seq = tokenizer.texts_to_sequences([cut_title])
    padded = pad_sequences(seq, maxlen=max_len, padding="post")
    
    # 3️⃣ 模型預測
    pred_probs = model.predict(padded)[0]
    idx = np.argmax(pred_probs)
    
    top_label = label_encoder.inverse_transform([idx])[0]
    top_probability = float(pred_probs[idx])
    
    if return_top_3:
        label_probability_pairs = [
            {'標籤': label_encoder.inverse_transform([i])[0], '機率': float(f"{prob:.4f}")}
            for i, prob in enumerate(pred_probs)
        ]
        sorted_pairs = sorted(label_probability_pairs, key=lambda x: x['機率'], reverse=True)
        top_3_pairs = sorted_pairs[:3]
        return top_label, top_probability, top_3_pairs
    else:
        return top_label, top_probability


def predict_tfidf(title, model, vectorizer, label_map, return_top_3=True):
    cut_text = " ".join(jieba.cut(title))
    vec = vectorizer.transform([cut_text])
    probs = model.predict_proba(vec)[0]
    idx = np.argmax(probs)
    if idx >= len(label_map):
        idx = len(label_map) - 1
    top_probability = float(probs[idx])
    top_label = label_map[idx]
    if return_top_3:
        label_probability_pairs = []
        for i, prob in enumerate(probs):
            label_index = min(i, len(label_map) - 1)
            prob_formatted = f"{prob:.4f}"
            label_probability_pairs.append({'標籤': label_map[label_index], '機率': float(prob_formatted)})
        sorted_pairs = sorted(label_probability_pairs, key=lambda item: item['機率'], reverse=True)
        top_3_pairs = sorted_pairs[:3]
        return top_label, top_probability, top_3_pairs
    else:
        return top_label, top_probability

# 3. input
st.set_page_config("新聞分類預測器", layout="centered")
st.title(" 新聞標題自動分類推薦系統")
model_type = st.radio("選擇模型：", ["Decision Tree", "Logistic Regression", "LSTM"])
title_input = st.text_area("請輸入新聞標題：", height=80)
label_map = {0: "國際", 1: "政治", 2: "焦點", 3: "生活", 4: "社會",5:"蒐奇",6:"財經",7:"財經週報",8:"軍武"}  
if st.button("開始預測"):
    if title_input.strip() == "":
        st.warning("請輸入一段新聞標題文字。")
    else:
        if model_type == "LSTM":
            model, lstm_tokenizer, label_encoder = load_lstm_model()
            label, prob, res = predict_lstm(title_input, model, lstm_tokenizer, label_encoder)
        elif model_type == "Decision Tree":
            model, tfidf_vectorizer = load_dt_model()
            label, prob, res = predict_tfidf(title_input, model, tfidf_vectorizer, label_map)
        else:
            model, tfidf_vectorizer = load_lr_model()
            label, prob, res = predict_tfidf(title_input, model, tfidf_vectorizer, label_map)
        st.markdown(f"### 預測分類：**{label}**")
        st.markdown(f"預測機率：`{prob*100:.2f}%`")
        st.markdown(f"前三高的預測標籤與機率：`{res}`")
