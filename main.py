from crawler.news_crawler import crawl_insurtech_news
from analysis.deepseek_analysis import extract_insight_from_article
from utils.save_results import save_results_to_json, save_results_to_csv
import json

# 1. 爬取新闻
df = crawl_insurtech_news(["保险科技", "数字保险"], max_articles_per_keyword=10)

# 2. 调用大模型分析
results = []
for i, row in df.iterrows():
    print(f"\n🎯 正在分析文章：{row['标题']}")
    output = extract_insight_from_article(row['正文'])
    try:
        parsed = json.loads(output) if isinstance(output, str) and output.startswith("{") else {"原始输出": output}
    except Exception:
        parsed = {"原始输出": output}

    parsed["标题"] = row["标题"]
    parsed["正文"] = row["正文"]
    parsed["链接"] = row["链接"]
    results.append(parsed)

# 3. 保存结果
save_results_to_json(results, "insurtech_results.json")
save_results_to_csv(results, "insurtech_results.csv")