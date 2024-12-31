# perl script to evaluate the search results
# perl .\src\trec_eval.pl .\query\qrels_40.txt .\rankings\qrels_40.txt\bm25\none.runs

for searcher in bm25 lm_MLE_smooth lm_JM_similar; do
    for index in porter krovetz none; do
        echo "Evaluate:  Index: $index, Searcher: $searcher, Running..."
        if [ ! -d ./results/qrels_40.txt/$searcher/$index ]; then
            mkdir -p ./results/qrels_40.txt/$searcher/
        fi
        perl ./src/trec_eval.pl -q ./query/qrels_40.txt ./rankings/qrels_40.txt/$searcher/$index.runs > ./results/qrels_40.txt/$searcher/$index.result
    done
done

for searcher in axiomatic_f1_exp axiomatic_f1_log axiomatic_f2_exp axiomatic_f2_log; do
    for index in porter krovetz none; do
        if [ ! -d ./results/qrels_40.txt/$searcher/$index ]; then
            mkdir -p ./results/qrels_40.txt/$searcher/
        fi
        echo "Evaluate:  Index: $index, Searcher: $searcher, Running..."
        perl ./src/trec_eval.pl -q ./query/qrels_40.txt ./rankings/qrels_40.txt/$searcher/$index.runs > ./results/qrels_40.txt/$searcher/$index.result
    done
done

echo "Evaluation done"