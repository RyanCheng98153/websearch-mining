import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import ndcg_score

# 讀取 qrel 資訊 (相關性標記)
def load_qrel(filepath):
    qrel = pd.read_csv(filepath, sep='\s+', header=None, names=["qid", "Q0", "doc_id", "relevance"])
    qrel = qrel.drop(columns=["Q0"])  # 不需要的欄位
    return qrel

# 讀取 rankings 資訊
def load_rankings(filepath):
    rankings = pd.read_csv(filepath, sep='\s+', header=None, names=["qid", "Q0", "doc_id", "rank", "score", "method"])
    return rankings

# 整合特徵
def merge_features_and_labels(rankings_files, qrel_file):
    qrel = load_qrel(qrel_file)
    features = qrel[["qid", "doc_id"]]
    for method, file in rankings_files.items():
        rankings = load_rankings(file)
        features = features.merge(rankings[["qid", "doc_id", "score"]], on=["qid", "doc_id"], how="left")
        features.rename(columns={"score": f"{method}_score"}, inplace=True)
    features = features.merge(qrel, on=["qid", "doc_id"], how="left")
    return features.fillna(0)

# 模型訓練與測試
def train_and_evaluate(features, test_features):
    X = features.drop(columns=["qid", "doc_id", "relevance"])
    y = features["relevance"]
    X_test = test_features.drop(columns=["qid", "doc_id", "relevance"])
    y_test = test_features["relevance"]

    model = GradientBoostingRegressor()
    model.fit(X, y)

    y_pred = model.predict(X_test)
    ndcg = ndcg_score([y_test], [y_pred], k=10)
    print(f"NDCG@10: {ndcg}")

    return model

def main():
    # 路徑設定
    qrel_40_path = "query/qrels_40.txt"
    qrel_10_path = "query/qrels_10.txt"
    rankings_40_files = {
        "bm25": "rankings/qrels_40.txt/bm25/none.runs",
        "lmJM": "rankings/qrels_40.txt/lm_JM_similar/none.runs",
        "lmMLE": "rankings/qrels_40.txt/lm_MLE_smooth/none.runs",
        # 可添加其他方法...
    }

    # 整合數據
    train_features = merge_features_and_labels(rankings_40_files, qrel_40_path)
    test_features = merge_features_and_labels(rankings_40_files, qrel_10_path)

    # 模型訓練與評估
    model = train_and_evaluate(train_features, test_features)

if __name__ == "__main__":
    main()