# 執行指令：python -m streamlit run st1.py
import streamlit as st
import pandas as pd
import requests
import time
from bs4 import BeautifulSoup
from transformers import pipeline
import altair as alt

st.set_page_config("新聞網爬蟲 + 情緒分析", layout="centered")
@st.cache_resource
def load_bert_pipeline():
    return pipeline("text-classification", model="bert-base-chinese")
bert_classifier = load_bert_pipeline()
st.title("新聞網標題爬蟲 + 情緒分析儀表板")
start_date = st.text_input("起始日期（YYYYMMDD）", value="20250101")
end_date = st.text_input("結束日期（YYYYMMDD）", value="20250105")
keyword = st.text_input("關鍵字", value="AI")
if st.button("開始爬蟲 + 情緒分析"):
    with st.spinner("爬蟲進行中，請稍候..."):
        df = pd.DataFrame(columns=["title", "class", "time", "link"])
        page = 1
        while True:
            # 抓取資料
            time.sleep(1.5)
            url = f"https://search.ltn.com.tw/list?keyword={keyword}&start_time={start_date}&end_time={end_date}&page={page}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
            }
            response = requests.get(url, headers=headers)
            response.encoding = 'utf-8'

            if response.status_code != 200 or "沒有找到資料" in response.text:
                break
            soup = BeautifulSoup(response.text, 'html.parser')
            objTag = soup.find_all('div', class_='cont')
            if not objTag:
                break
            for data in objTag:
                title = data.find(class_="tit").get_text()
                try:
                    class1 = data.find(class_="immtag chan").get_text()
                except:
                    class1 = ""
                try:
                    class1 = data.find(class_="immtag chan11").get_text()
                except:
                    pass
                time1 = data.find('span', class_="time").get_text()
                link = data.find('a', class_="tit").get('href')
                df.loc[len(df)] = [title, class1, time1, link]
            page += 1

        st.success(f"共抓取 {len(df)} 筆新聞標題")
    # 進行情緒分析
    with st.spinner("進行情緒分析..."):
        def get_sentiment(text):
            try:
                result = bert_classifier(str(text)[:512])[0]
                return result['label']
            except:
                return "無法判斷"
        df["sentiment"] = df["title"].apply(get_sentiment)
        st.subheader("各新聞分類的情緒分布")
        st.text("LABLE_0: 負面情緒\nLABLE_1: 正面情緒")
        bias_df = df.groupby(["class", "sentiment"]).size().reset_index(name="count")
        bar_chart = alt.Chart(bias_df).mark_bar().encode(
            x="class:N", y="count:Q", color="sentiment:N"
        ).properties(width=800)
        st.altair_chart(bar_chart, use_container_width=True)
        st.subheader("分析結果預覽 & 匯出")
        st.dataframe(df.head(20))
        csv = df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button("下載結果 CSV", data=csv, file_name="ltn_sentiment_result.csv", mime="text/csv")
