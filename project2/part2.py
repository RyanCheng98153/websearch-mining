import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import os
import argparse

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
    features.fillna(0, inplace=True)
    
    # 特徵標準化 (使用 MinMaxScaler)
    scaler = MinMaxScaler()
    feature_cols = [col for col in features.columns if "_score" in col]
    features[feature_cols] = scaler.fit_transform(features[feature_cols])
    
    return features
def train_and_evaluate(features, test_features, args):
    X = features.drop(columns=["qid", "doc_id", "relevance"])
    y = features["relevance"]
    X_test = test_features.drop(columns=["qid", "doc_id", "relevance"])
    
    # 模型初始化
    model = RandomForestClassifier()
    model.fit(X, y)
    
    # 預測
    predictions = model.predict_proba(X_test)
    test_features["score"] = [p[1] for p in predictions]  # 提取正類機率
    
    # 排序與輸出結果
    test_features = test_features.sort_values(by=["qid", "score"], ascending=[True, False])
    test_features = test_features[test_features["score"] > 0.0]
    test_features["rank"] = test_features.groupby("qid").cumcount() + 1
    test_features = test_features.groupby("qid").head(1000)  # 保留每個查詢的前 1000 筆
    
    # 儲存結果到文件
    if not os.path.exists("./rankings/qrels_10.txt/_EXP"):
        os.makedirs("./rankings/qrels_10.txt/_EXP")
    output_file = f"./rankings/qrels_10.txt/_EXP/{args.index}.runs"
    with open(output_file, "w") as f:
        for _, row in test_features.iterrows():
            f.write(
                f"{row['qid']} Q0 {row['doc_id']} {row['rank']} {row['score']:.5f} Exp\n"
            )
    
    # print(test_features.head())
    print(f"result has written in {output_file}")
    return test_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--index", type=str, default="porter", help="{porter, krovetz, none}" )
    parser.add_argument( "--eval", type=int, help="40(qrels 401-440) or 10(qrels 441-450)", default=10 )
    args = parser.parse_args()

    # dataset path
    trainset_path = "query/qrels_40.txt"
    testset_path = "./query/qrels_40.txt" if args.eval == 40 else "./query/qrels_10.txt"
    
    rankings_40_files = {
        "bm25": f"rankings/qrels_40.txt/bm25/{args.index}.runs",
        # "lmJM": f"rankings/qrels_40.txt/lm_JM_similar/{args.index}.runs",
        # "lmMLE": f"rankings/qrels_40.txt/lm_MLE_smooth/{args.index}.runs",
        # "axiomatic_f1_exp": "rankings/qrels_40.txt/lm_MLE_smooth/{args.index}.runs",
        # "axiomatic_f1_log": "rankings/qrels_40.txt/axiomatic_f1_log/{args.index}.runs",
        # "axiomatic_f2_exp": "rankings/qrels_40.txt/axiomatic_f2_exp/{args.index}.runs",
        # "axiomatic_f2_log:": "rankings/qrels_40.txt/axiomatic_f2_log/{args.index}.runs"
    }

    # 整合數據
    train_features = merge_features_and_labels(rankings_40_files, trainset_path)
    test_features = merge_features_and_labels(rankings_40_files, testset_path)

    # 模型訓練與評估並輸出結果
    train_and_evaluate(train_features, test_features, args)

if __name__ == "__main__":
    main()