from __future__ import division, unicode_literals
import sys
import src.util as utils
import src.tfidf as tfidf
import src.VectorSpace as vsm
import fire

# documents = utils.read_files(sys.argv[1])
# documents = [ documents[k] for k in documents.keys()]
# filelist = [ f"News{k}.txt" for k in test ] [:]
# test_query = ["Typhoon", "Taiwan", "war"]
# test = [12780, 10184, 12428, 13724, 10152, 10355, 12944, 10460, 6715, 6825]

def printRatings(ratings: list):
    print("NewsID" + " "*9 + "Score")
    for id, rating in ratings:
        print(f"{id: <15}{round(rating, 7)}")
    print("------------------------\n")

def main(# filepath: str = "./EnglishNews/",
         query: str = "",
         use_tqdm: bool = False):
    filepath: str = "./EnglishNews/"
         
    documents = utils.read_files(filepath, )
    
    # print(filepath)
    query = query.split()
    # query = "".join(sys.argv[2:])
    # query = test_query
    vectorSpace = vsm.VectorSpace(documents=documents, query=query, use_tqdm=use_tqdm)
    
    # 使用 TF 向量和 Cosine Similarity
    # tf_Cosine_ratings = vectorSpace.search( method="cosine", use_tfidf=False, topN_results=10)
    # print("TF Cosine")
    # printRatings(tf_Cosine_ratings)

    # 使用 TF-IDF 向量和 Cosine Similarity
    tfidf_Cosine_ratings = vectorSpace.search( method="cosine", use_tfidf=True, topN_results=10)
    print("TF-IDF Cosine")
    printRatings(tfidf_Cosine_ratings)
    
    # 使用 TF 向量和 Euclidean Distance
    # tf_Euclidean_ratings = vectorSpace.search( method="euclidean", use_tfidf=False, topN_results=10)
    # print("TF Euclidean")
    # printRatings(tf_Euclidean_ratings)
    
    # 使用 TF-IDF 向量和 Euclidean Distance
    tfidf_Euclidean_ratings = vectorSpace.search( method="euclidean", use_tfidf=True, topN_results=10)
    print("TF-IDF Euclidean")
    printRatings(tfidf_Euclidean_ratings)
    
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
