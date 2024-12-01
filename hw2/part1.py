from pyserini.search.lucene import LuceneSearcher
from pyserini.pyclass import autoclass
import argparse
import os
import re

def parse(path: str):
    parsed = []
    
    with open(path, 'r') as f:
        data = f.read()
    
        matches = re.findall(pattern=r"<num> Number: (\d+)\s+<title> (.+)", string=data)
        for match in matches:
            parsed.append({
                'number': match[0],
                'title': match[1].split()
            })
    return parsed, len(parsed)

# Load Lucene Similarity classes
LMJelinekMercerSimilarity = autoclass("org.apache.lucene.search.similarities.LMJelinekMercerSimilarity")
IndexSearcher = autoclass("org.apache.lucene.search.IndexSearcher")

# Custom Searcher Class for LM Jelinek-Mercer Similarity
class LMJelinekMercerSearcher(LuceneSearcher):
    def set_qld(self, mu=1000.0):
        """Sets the Jelinek-Mercer smoothing parameter for the searcher."""
        self.object.similarity = LMJelinekMercerSimilarity(mu)
        self.object.searcher = IndexSearcher(self.object.reader)
        self.object.searcher.setSimilarity(self.object.similarity)


# Import required Lucene classes
AxiomaticF1EXP = autoclass("org.apache.lucene.search.similarities.AxiomaticF1EXP")
AxiomaticF1LOG = autoclass("org.apache.lucene.search.similarities.AxiomaticF1LOG")
AxiomaticF2EXP = autoclass("org.apache.lucene.search.similarities.AxiomaticF2EXP")
AxiomaticF2LOG = autoclass("org.apache.lucene.search.similarities.AxiomaticF2LOG")

class AxiomaticSearcher(LuceneSearcher):
    """
    Custom searcher that uses Axiomatic Similarity for scoring documents.
    """

    def set_axiomatic_similarity(self, axiomatic_model, s=0.5):
        """
        Configures the IndexSearcher to use a specified Axiomatic model.

        :param axiomatic_model: An instance of Axiomatic similarity (e.g., AxiomaticF1EXP).
        :param s: Smoothing factor. A small positive real number, default is 0.5.
        """
        self.object.similarity = axiomatic_model(s)
        self.object.searcher = IndexSearcher(self.object.reader)
        self.object.searcher.setSimilarity(self.object.similarity)



# if __name__ == "__main__":
#     parsed, num = parse("./query/trec40.txt")
#     for p in parsed:
#         print(f"{p['number']}. {p['title']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( "--index", type=str, default="porter", help="{porter, krovetz, none}" )
    parser.add_argument( "--searcher", type=str, default="bm25", help="{bm25, lm_MLE_smooth, lm_JM_similar}" )
    parser.add_argument( "--eval", type=int, help="40(qrels 401-440) or 10(qrels 441-450)", default=40 )
    args = parser.parse_args()

    eval = "./qrels_40.txt" if args.eval == 40 else "./qrels_10.txt"
    parse_file = "./query/trec40.txt" if args.eval == 40 else "./query/trec10.txt"
    output_file = f"./rankings/{eval}/{args.searcher}/{args.index}.runs"
    os.makedirs(f"./rankings/{eval}/{args.searcher}", exist_ok=True)

    index = f"indexes/{args.index}"
    searcher = None
    
    if args.searcher == "bm25":
        searcher = LuceneSearcher(index)
        searcher.set_bm25(k1=1.2, b=0.75)
    # Language Modeling with Laplace Smoothing
    elif args.searcher == "lm_MLE_smooth":
        searcher = LuceneSearcher(index)
        searcher.set_qld()
    # Language Modeling Jelinek-Mercer Similarity
    elif args.searcher == "lm_JM_similar":
        searcher = LMJelinekMercerSearcher(index)
        searcher.set_qld(0.8)
    elif args.searcher == "axiomatic_f1_exp":
        searcher = AxiomaticSearcher(index)
        searcher.set_axiomatic_similarity(AxiomaticF1EXP, s=0.3)
    elif args.searcher ==  "axiomatic_f1_log":
        searcher = AxiomaticSearcher(index)
        searcher.set_axiomatic_similarity(AxiomaticF1LOG, s=0.5)
    elif args.searcher ==  "axiomatic_f2_exp":
        searcher = AxiomaticSearcher(index)
        searcher.set_axiomatic_similarity(AxiomaticF2EXP, s=0.7)
    elif args.searcher ==  "axiomatic_f2_log":
        searcher = AxiomaticSearcher(index)
        searcher.set_axiomatic_similarity(AxiomaticF2LOG, s=0.9)
    else:
        print("Invalid searcher")
        exit(1)

    parsed, num = parse(parse_file)
    with open(file=output_file, mode="w") as f:
        for p in parsed:
            hits = searcher.search(q=f"{p['number']}. {p['title']}", k=1000)
            for i in range(len(hits)):
                if hits[i].score <= 0.0:
                    break
                f.write( f"{p['number']} Q0 {hits[i].docid} {i+1} {hits[i].score:.5f} {args.searcher}\n " )
