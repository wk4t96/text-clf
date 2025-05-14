# 專案摘要

本專案分成兩個部分，第一部分對一個新聞網站進行爬蟲、提取新聞的標題、進行標題的情感分析並統計正向與負向情緒；第二部分提取大量的新聞標題及其分類，經過多種模型的訓練與測試後，將其應用於預測一段新的新聞標題所屬的分類。

這兩個部分的streamlit執行檔分別為存放在子目錄streamlit_web_exe中的st1_sentiment_analysis.py和st2_news_title_classification_prediction.py。並已佈署在Streamlit Community Cloud上。可直接點開以下網址來執行互動式網頁：
1. https://text-clf-st1.streamlit.app/
2. https://text-clf-1248.streamlit.app/

對標題分類模型的訓練與測試發現，多種模型如K-近鄰(KNN)、決策樹、支持向量分類器(SVC)、高斯單純貝葉斯(GaussianNB)、邏輯迴歸、多項式單純貝葉斯(MultinomialNB)，以及長短期記憶(LSTM)模型都具有相當良好的準確率(98%以上)。

