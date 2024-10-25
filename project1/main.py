from __future__ import division, unicode_literals
import sys
import src.util as utils
import src.VectorSpace as vsm
import fire
from src.Parser import Parser

# documents = utils.read_files(sys.argv[1])
# documents = [ documents[k] for k in documents.keys()]
# filelist = [ f"News{k}.txt" for k in test ] [:]
# test_query = ["Typhoon", "Taiwan", "war"]
# test = [12780, 10184, 12428, 13724, 10152, 10355, 12944, 10460, 6715, 6825]

def prob1(Eng_query: str = "",
          use_tqdm: bool = False
          ):
    
    documents = utils.read_files("./EnglishNews/")
    vsEng = vsm.VectorSpace(documents=documents, use_tqdm=use_tqdm)
    # vsEng.build_query(query=Eng_query)
    
    # 使用 TF 向量和 Cosine Similarity
    tf_Cosine_ratings = vsEng.search( query=Eng_query, method="cosine", use_tfidf=False, topN_results=10)
    print("TF Cosine")
    printRatings(tf_Cosine_ratings)

    # 使用 TF-IDF 向量和 Cosine Similarity
    tfidf_Cosine_ratings = vsEng.search( query=Eng_query, method="cosine", use_tfidf=True, topN_results=10)
    print("TF-IDF Cosine")
    printRatings(tfidf_Cosine_ratings)
    
    # 使用 TF 向量和 Euclidean Distance
    tf_Euclidean_ratings = vsEng.search( query=Eng_query, method="euclidean", use_tfidf=False, topN_results=10)
    print("TF Euclidean")
    printRatings(tf_Euclidean_ratings)
    
    # 使用 TF-IDF 向量和 Euclidean Distance
    tfidf_Euclidean_ratings = vsEng.search( query=Eng_query, method="euclidean", use_tfidf=True, topN_results=10)
    print("TF-IDF Euclidean")
    printRatings(tfidf_Euclidean_ratings)

def prob2(Eng_query: str = "",
          use_tqdm: bool = False
          ):
    
    documents = utils.read_files("./EnglishNews/")
    vsEng = vsm.VectorSpace(documents=documents, use_tqdm=use_tqdm)
    # vsEng.build_query(query=Eng_query)
    
    tfidf_Cosine_ratings = vsEng.search( query=Eng_query, method="cosine", use_tfidf=True, topN_results=10)
    
    bestQ_fpath, score = tfidf_Cosine_ratings[0]
    # bestQ_fpath = 'News12428.txt'
    
    best_query = utils.read_file("./EnglishNews/" + bestQ_fpath)
    best_query = utils.getVerbNoun(best_query)
    
    # vsEng.build_query(query=Eng_query, query_support=best_query)
    
    # 使用 TF 向量和 Cosine Similarity
    tf_Cosine_ratings = vsEng.search( query=Eng_query, query_support=best_query, method="cosine", use_tfidf=False, topN_results=10)
    print("TF Cosine")
    printRatings(tf_Cosine_ratings)

    # 使用 TF-IDF 向量和 Cosine Similarity
    tfidf_Cosine_ratings = vsEng.search( query=Eng_query, query_support=best_query, method="cosine", use_tfidf=True, topN_results=10)
    print("TF-IDF Cosine")
    printRatings(tfidf_Cosine_ratings)
    
    # 使用 TF 向量和 Euclidean Distance
    tf_Euclidean_ratings = vsEng.search( query=Eng_query, query_support=best_query, method="euclidean", use_tfidf=False, topN_results=10)
    print("TF Euclidean")
    printRatings(tf_Euclidean_ratings)
    
    # 使用 TF-IDF 向量和 Euclidean Distance
    tfidf_Euclidean_ratings = vsEng.search( query=Eng_query, query_support=best_query, method="euclidean", use_tfidf=True, topN_results=10)
    print("TF-IDF Euclidean")
    printRatings(tfidf_Euclidean_ratings)

from src.Parser import JiebaParser
    
def prob3(Chi_query: str = "",
          use_tqdm: bool = False
          ):    
    # Chinese News
    documents = utils.read_files("./ChineseNews/")
    vsChi = vsm.VectorSpace(documents=documents, use_tqdm=use_tqdm, parser=JiebaParser())
    # vsChi.build_query(query=Chi_query)
    
    # 使用 TF 向量和 Cosine Similarity
    tf_Cosine_ratings = vsChi.search( query=Chi_query, method="cosine", use_tfidf=False, topN_results=10)
    print("TF Cosine")
    printRatings(tf_Cosine_ratings)

    # 使用 TF-IDF 向量和 Cosine Similarity
    tfidf_Cosine_ratings = vsChi.search( query=Chi_query, method="cosine", use_tfidf=True, topN_results=10)
    print("TF-IDF Cosine")
    printRatings(tfidf_Cosine_ratings)
    
    # 使用 TF 向量和 Euclidean Distance
    tf_Euclidean_ratings = vsChi.search( query=Chi_query, method="euclidean", use_tfidf=False, topN_results=10)
    print("TF Euclidean")
    printRatings(tf_Euclidean_ratings)
    
    # 使用 TF-IDF 向量和 Euclidean Distance
    tfidf_Euclidean_ratings = vsChi.search( query=Chi_query, method="euclidean", use_tfidf=True, topN_results=10)
    print("TF-IDF Euclidean")
    printRatings(tfidf_Euclidean_ratings)

import csv
from itertools import islice

def take(n, iterable):
    """Return the first n items of the iterable as a list."""
    return list(islice(iterable, n))

def prob4(use_tqdm: bool = False):
    
    documents = take(100, utils.read_files("./smaller_dataset/collections/").items())
    queries = utils.read_files("./smaller_dataset/queries/")
    # print(queries.keys())
    
    relevances = dict()
    with open("./smaller_dataset/rel.tsv", 'r', newline='') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            relevances[row[0]] = row[1].strip('[]').split(', ')
    
    vsEng = vsm.VectorSpace(documents=dict(documents), use_tqdm=use_tqdm)
    
    # print(documents)
    # print(queries)
    
    from tqdm import tqdm
    
    all_results: dict[str, list[str]] = dict()
    for k in queries.keys():
        result = vsEng.search(query=queries[k], method="cosine", use_tfidf=True, topN_results=20) 
        result = [doc for doc, score in result] 
        all_results[k] = [(item.split('.')[0][1:]) for item in result if item.startswith("d")] [:10]
    
    MRR_count = 0
    
    for query, prediction in all_results.items():
        # Calculate MRR
        for i, doc in enumerate(prediction):
            if doc in relevances[query.split(".")[0]]:
                MRR_count += 1 / (i + 1)
                break

    # Calculate MAP
    MAP_count = 0
    
    for query, prediction in all_results.items():
        avg_pre = 0
        correct = 0
        for i, doc in enumerate(prediction):    
            if doc in relevances[query.split(".")[0]]:
                correct += 1
                avg_pre += correct / (i + 1)  # Precision at i + 1
        if correct == 0:
            continue
        avg_pre /= correct  # Average precision for this query
        MAP_count += avg_pre

    # Calculate Recall
    
    recall_count = 0
    
    for query, prediction in all_results.items():
        correct = 0
        for i, doc in enumerate(prediction):
            if doc in relevances[query.split(".")[0]]:
                correct += 1
        recall_count += correct / 10  # Assuming a fixed recall denominator of 10
    
    print("----------------------")
    print("TF-IDF Cosine")
    print("MRR@10:    ", round(MRR_count / len(all_results), 6))
    print("MAP@10:    ", round(MAP_count / len(all_results), 6))
    print("RECALL@10: ", round(recall_count / len(all_results), 6))
    print("----------------------")

def printRatings(ratings: list):
    print("NewsID" + " "*9 + "Score")
    for id, rating in ratings:
        print(f"{id: <15}{round(rating, 7)}")
    print("------------------------\n")

def main(# filepath: str = "./EnglishNews/",
         Eng_query: str = "",
         Chi_query: str = "",
         use_tqdm: bool = False):
    
    
    print("problem 1")
    prob1(Eng_query=Eng_query, use_tqdm=use_tqdm)
    print("problem 2")
    prob2(Eng_query=Eng_query, use_tqdm=use_tqdm)
    print("problem 3")
    prob3(Chi_query=Chi_query, use_tqdm=use_tqdm)
    print("problem 4")
    prob4(use_tqdm=use_tqdm)
    
if __name__ == "__main__":
    fire.Fire(main)

"""
for i, blob in enumerate(bloblist):
    # print(blob)
    blob = tfidf.tb(blob)
    if len( blob.split() ) < 2:
         continue
    
    wordbag = utils.removeDuplicates(target+blob)
    print(target)
    print(blob)
    print(wordbag)
    
    tf_vector =     [ tfidf.tf(word, blob) for word in wordbag]
    idf_vector =    [ tfidf.idf(word, bloblist) for word in wordbag]
    tfidf_vector =  [ tfidf.tfidf(word, blob, bloblist) for word in wordbag]
    
    print(f"Document: {filelist[i]}")
    print(f"TF Vector: {tf_vector}")
    print(f"IDF Vector: {idf_vector}")
    print(f"TF-IDF Vector: {tfidf_vector}")
    print()

"""
########################################################33
