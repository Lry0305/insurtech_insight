# 重构美化后的最终 streamlit_app.py 内容，带分页、筛选器和错误修复
final_beautified_code = '''
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

rcParams['font.sans-serif'] = ['SimHei']
st.set_page_config(page_title="保险科技智能分析平台", layout="wide")

# 读取数据
@st.cache_data
def load_data():
    return pd.read_csv("insurtech_results.csv")

df = load_data()

# 解析 JSON 字段
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

# 从正文中提取时间（简单规则）
def extract_date(text):
    match = re.search(r"(\\d{4}-\\d{2}-\\d{2})", str(text))
    return match.group(1) if match else None

df["日期"] = df["正文"].apply(extract_date)

# ==== 侧边栏筛选器 ====
with st.sidebar:
    st.header("🔍 数据筛选器")
    sentiment_options = ["全部"] + sorted(df["情绪"].dropna().unique().tolist())
    selected_sentiment = st.selectbox("按情绪筛选", sentiment_options)
    if selected_sentiment != "全部":
        df = df[df["情绪"] == selected_sentiment]

# ==== 页面结构划分 ====
st.title("📊 保险科技行业智能观点分析平台")
st.markdown("🚀 当前共加载新闻：**{}** 条".format(len(df)))
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["📊 情绪与关键词", "🧠 聚类分析", "🏢 公司与时间", "📄 数据总览"])

with tab1:
    st.subheader("📊 情绪分布")
    sentiment_counts = df["情绪"].value_counts().reset_index()
    sentiment_counts.columns = ["情绪", "数量"]
    fig_sentiment = px.pie(sentiment_counts, names="情绪", values="数量", title="情绪占比")
    st.plotly_chart(fig_sentiment, use_container_width=True)

    st.subheader("🔥 高频关键词 Top20")
    kw_freq = Counter(all_keywords)
    top_kw = kw_freq.most_common(20)
    top_df = pd.DataFrame(top_kw, columns=["关键词", "出现次数"])
    fig_bar = px.bar(top_df, x="关键词", y="出现次数", text="出现次数")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("☁️ 关键词词云图")
    def get_chinese_font():
        paths = [
            "C:/Windows/Fonts/simhei.ttf",
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/usr/share/fonts/truetype/arphic/ukai.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ]
        for path in paths:
            if os.path.exists(path): return path
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

with tab2:
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

    st.markdown("#### 每类代表性观点（前3条）")
    for i in range(k):
        st.markdown(f"**🌀 聚类 {i}：**")
        sample = df[df["聚类标签"] == i]["观点"].dropna().head(3).tolist()
        for text in sample:
            st.markdown(f"- {text}")

with tab3:
    st.subheader("📈 情绪随时间变化趋势")
    timeline_df = df.dropna(subset=["日期"])
    trend_data = timeline_df.groupby(["日期", "情绪"]).size().reset_index(name="数量")
    fig_time = px.line(trend_data, x="日期", y="数量", color="情绪", markers=True,
                       title="不同情绪报道随时间趋势")
    st.plotly_chart(fig_time, use_container_width=True)

    st.subheader("🏢 主体公司情绪分析")
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

    st.subheader("🤖 行业智能体建议")
    st.markdown("提示词示例：")
    st.code("请分析当前保险科技行业的关键词：智能核保、数字风控、客户体验")

with tab4:
    st.subheader("📄 原始新闻数据总览")
    st.dataframe(df[["标题", "情绪", "链接"]], use_container_width=True)

    st.subheader("📥 一键导出报告")
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 下载 CSV", csv, file_name="insurtech_analysis.csv", mime="text/csv")
'''

# 保存为最终优化版文件
with open("D:/比赛/insurtech/streamlit_app_final_beautified.py", "w", encoding="utf-8") as f:
    f.write(final_beautified_code)
