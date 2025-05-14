# 使用者在網頁介面上與之互動後，結果會顯示在網頁上。 # 執行指令：python -m streamlit run st2.py
import streamlit as st
import numpy as np
import jieba
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import joblib

MAX_LEN = 30
NUM_WORDS = 10000
# 1. load models
@st.cache_resource
def load_lstm_model():
    #model = load_model("model_files/lstm_news_model.h5")
    model = joblib.load("model_files/nb_model.joblib")
    with open("model_files/tokenizer_lstm.pickle", "rb") as f:
        import pickle
        tokenizer = pickle.load(f)  # tokenizer 將文字轉換成機器可讀的數值序列
    return model, tokenizer
@st.cache_resource
def load_nb_model():
    model = joblib.load("model_files/nb_model.joblib")
    vectorizer = joblib.load("model_files/tfidf_vectorizer.joblib")
    return model, vectorizer
@st.cache_resource
def load_dt_model():
    model = joblib.load("model_files/DecisionTree1.joblib")
    vectorizer = joblib.load("model_files/tfidf_vectorizer.joblib")
    return model, vectorizer

# 2. predict
def predict_lstm(title, model, tokenizer, label_map, return_top_3=True):
    seq = tokenizer.texts_to_sequences([" ".join(jieba.cut(title))])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    pred_probs = model.predict(padded)[0]
    idx = np.argmax(pred_probs)
    if idx >= len(label_map):
        idx = len(label_map) - 1
    top_probability = float(pred_probs[idx])
    top_label = label_map[idx]
    if return_top_3:
        label_probability_pairs = []
        for i, prob in enumerate(pred_probs):
            label_index = min(i, len(label_map) - 1)
            prob_formatted = f"{prob:.4f}"
            label_probability_pairs.append({'標籤': label_map[label_index], '機率': float(prob_formatted)})
        sorted_pairs = sorted(label_probability_pairs, key=lambda item: item['機率'], reverse=True)
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
model_type = st.radio("選擇模型：", ["Decision Tree", "Naive Bayes", "LSTM"])
title_input = st.text_area("請輸入新聞標題：", height=80)
label_map = {0: "國際", 1: "政治", 2: "焦點", 3: "生活", 4: "社會",5:"蒐奇",6:"財經",7:"財經週報",8:"其他"}  
if st.button("開始預測"):
    if title_input.strip() == "":
        st.warning("請輸入一段新聞標題文字。")
    else:
        if model_type == "LSTM":
            model, lstm_tokenizer = load_lstm_model()
            label, prob, res = predict_lstm(title_input, model, lstm_tokenizer, label_map)
            #res = predict_lstm_top_3_probs(title_input, model, lstm_tokenizer, label_map)
        elif model_type == "Decision Tree":
            model, tfidf_vectorizer = load_dt_model()
            label, prob, res = predict_tfidf(title_input, model, tfidf_vectorizer, label_map)
        else:
            model, tfidf_vectorizer = load_nb_model()
            label, prob, res = predict_tfidf(title_input, model, tfidf_vectorizer, label_map)
        st.markdown(f"### 預測分類：**{label}**")
        st.markdown(f"預測機率：`{prob*100:.2f}%`")
        st.markdown(f"前三高的預測標籤與機率：`{res}`")
