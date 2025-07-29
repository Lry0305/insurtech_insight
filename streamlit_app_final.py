
import streamlit as st
import pandas as pd
import json
import plotly.express as px
from matplotlib import rcParams
import matplotlib.pyplot as plt
from collections import Counter
import os
from wordcloud import WordCloud
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import re

from agent.agent_api import query_industry_agent

rcParams['font.sans-serif'] = ['SimHei']

# 页面设置
st.set_page_config(page_title="保险科技智能分析平台", layout="wide")
st.title("📊 保险科技行业智能观点分析平台")

# 加载数据
@st.cache_data
def load_data():
    return pd.read_csv("insurtech_results.csv")

df = load_data()
st.markdown(f"共加载 **{len(df)}** 条新闻数据")

# ========== JSON 解析 ==========
def extract_json_fields(df):
    sentiments, opinions, keywords_all, subjects = [], [], [], []
    for raw in df["原始输出"]:
        try:
            content = json.loads(raw.replace("```json", "").replace("```", ""))
            sentiments.append(content.get("情绪", "未提取"))
            opinions.append(content.get("观点", ""))
            kws = content.get("关键词", [])
            ents = content.get("主体", [])
            if isinstance(kws, list): keywords_all.extend(kws)
            if isinstance(ents, list): subjects.append(ents)
        except:
            sentiments.append("解析失败")
            opinions.append("")
            subjects.append([])
    return sentiments, opinions, keywords_all, subjects

df["情绪"], df["观点"], all_keywords, df["主体"] = extract_json_fields(df)

# ========== 情绪 & 关键词分析 ==========
st.subheader("📊 情绪 & 关键词趋势总览")
sentiment_counts = df["情绪"].value_counts().reset_index()
sentiment_counts.columns = ["情绪", "数量"]

kw_freq = Counter(all_keywords)
top_kw = kw_freq.most_common(20)
top_df = pd.DataFrame(top_kw, columns=["关键词", "出现次数"])

col1, col2 = st.columns(2)
with col1:
    fig_sentiment = px.pie(sentiment_counts, names="情绪", values="数量", title="情绪占比")
    st.plotly_chart(fig_sentiment, use_container_width=True)
with col2:
    fig_bar = px.bar(top_df, x="关键词", y="出现次数", title="关键词出现频率", text="出现次数")
    st.plotly_chart(fig_bar, use_container_width=True)

# ========== 词云 ==========
st.subheader("☁️ 关键词词云图")
def get_chinese_font():
    possible_paths = [
        "C:/Windows/Fonts/simhei.ttf",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/usr/share/fonts/truetype/arphic/ukai.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

font_path = get_chinese_font()
if font_path:
    wordcloud = WordCloud(font_path=font_path, background_color="white", width=800, height=400)
    wordcloud.generate_from_frequencies(kw_freq)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
else:
    st.warning("⚠️ 未找到可用中文字体，词云将无法显示中文。")

# ========== 聚类分析 ==========
st.subheader("🧠 观点聚类分析")
vectorizer = TfidfVectorizer(stop_words="english", max_features=100)
X = vectorizer.fit_transform(df["观点"].fillna(""))
k = 4
model = KMeans(n_clusters=k, random_state=42)
df["聚类标签"] = model.fit_predict(X)
cluster_counts = df["聚类标签"].value_counts().sort_index()
fig_cluster = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                     labels={"x": "聚类标签", "y": "样本数"}, title="观点聚类分布")
st.plotly_chart(fig_cluster, use_container_width=True)

st.markdown("#### 每类代表性观点（每类前3条）")
for i in range(k):
    st.markdown(f"**🌀 聚类 {i}：**")
    sample = df[df["聚类标签"] == i]["观点"].dropna().head(3).tolist()
    for text in sample:
        st.markdown(f"- {text}")

# ========== 时间趋势分析 ==========
st.subheader("📈 保险科技报道随时间变化")
def extract_date(source_info):
    match = re.search(r"(\d{4}-\d{2}-\d{2})", str(source_info))
    return match.group(1) if match else None

df["日期"] = df["来源信息"].apply(extract_date)
timeline_df = df.dropna(subset=["日期"])
trend_data = timeline_df.groupby(["日期", "情绪"]).size().reset_index(name="数量")
fig_time = px.line(trend_data, x="日期", y="数量", color="情绪", markers=True,
                   title="不同情绪报道随时间趋势")
st.plotly_chart(fig_time, use_container_width=True)

# ========== 主体公司情绪分析 ==========
st.subheader("🏢 主体公司情绪雷达图")
def extract_entities(df):
    entities = []
    for raw in df["原始输出"]:
        try:
            content = json.loads(raw.replace("```json", "").replace("```", ""))
            ents = content.get("主体", [])
            if isinstance(ents, list):
                for e in ents:
                    entities.append((e, content.get("情绪", "未提取")))
        except:
            continue
    return pd.DataFrame(entities, columns=["主体", "情绪"])

ent_df = extract_entities(df)
if not ent_df.empty:
    pivot = ent_df.groupby(["主体", "情绪"]).size().unstack(fill_value=0)
    top_entities = pivot.sum(axis=1).sort_values(ascending=False).head(6).index
    radar = pivot.loc[top_entities]
    fig_radar = px.line_polar(radar.reset_index(), r=radar.sum(axis=1), theta=radar.index,
                              line_close=True, title="高频公司/机构情绪关注强度")
    st.plotly_chart(fig_radar, use_container_width=True)
else:
    st.info("未成功提取公司主体")

# ========== 智能体提示 ========== 
st.subheader("🤖 行业智能体建议（实验功能）")
st.markdown("你可以将聚类结果或关键词作为输入提示词，用于 AI 分析行业态势：")
st.code("请分析当前保险科技行业的关键词：智能核保、数字风控、客户体验")

# ========== 导出报告 ==========
st.subheader("📤 一键导出报告")
csv = df.to_csv(index=False).encode("utf-8-sig")
st.download_button("📥 下载分析结果（CSV）", csv, file_name="insurtech_analysis.csv", mime="text/csv")

# ========== 原始数据 ==========
st.subheader("📰 原始新闻数据")
st.dataframe(df[["标题", "情绪", "链接"]], use_container_width=True)

with st.chat_message("user"):
    user_question = st.text_input("💬 提问保险科技相关问题", key="chat")

if user_question:
    with st.spinner("正在分析，请稍候..."):
        answer = query_industry_agent(user_question, api_key="sk-7b432ed6557d42939e8c52ae59a442c1")
        st.chat_message("ai").markdown(answer)
