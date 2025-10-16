# 專案摘要

本專案分成兩個部分，第一部分利用公開的「財經新聞情緒分類數據集」來微調 ”IDEA-CCNL/Erlangshen-RoBERTa-110M-Sentiment” 模型，接著將此訓練後的模型應用於 Streamlit 平台上對一個新聞網站進行爬蟲、提取新聞的標題、進行標題的情緒分析並統計正向與負向情緒；第二部分提取大量的新聞標題及其分類，經過多種機器學習模型：長短期記憶(LSTM)、邏輯迴歸、多項式單純貝葉斯(MultinomialNB)，以及高斯單純貝葉斯(GaussianNB)模型的訓練與測試後，將其應用於預測一段新的新聞標題所屬的分類。

本專案的子目錄及其內容如下描述：
* model_files：存放訓練後的「新聞標題分類模型」檔；
* sentiment_fine_tuned_model：存放訓練後的「情緒分析模型」檔；
* streamlit_web_exe：存放 Streamlit 執行檔 -- st1_sentiment_analysis.py 和 st2_news_title_classification_prediction.py；
* train_model：記錄新聞標題分類模型以及情緒分析模型的訓練過程及模型評估結果。

本專案中的兩個部分已佈署在Streamlit Community Cloud上，可直接點開以下網址來執行互動式網頁：
1. https://text-clf-st1.streamlit.app/
2. https://text-clf-1248.streamlit.app/