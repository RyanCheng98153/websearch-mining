# perl script to evaluate the search results
# perl .\src\trec_eval.pl .\query\qrels_10.txt .\rankings\qrels_10.txt\bm25\none.runs

for index in porter krovetz none; do
    echo "Evaluate:  Index: $index, Searcher: _EXP, Running..."
    if [ ! -d ./results/qrels_10.txt/_EXP/$index ]; then
        mkdir -p ./results/qrels_10.txt/_EXP/
    fi
    perl ./src/trec_eval.pl -q ./query/qrels_10.txt ./rankings/qrels_10.txt/_EXP/$index.runs > ./results/qrels_10.txt/_EXP/$index.result
done

for searcher in bm25 lm_MLE_smooth lm_JM_similar; do
    for index in porter krovetz none; do
        echo "Evaluate:  Index: $index, Searcher: $searcher, Running..."
        if [ ! -d ./results/qrels_10.txt/$searcher/$index ]; then
            mkdir -p ./results/qrels_10.txt/$searcher/
        fi
        perl ./src/trec_eval.pl -q ./query/qrels_10.txt ./rankings/qrels_10.txt/$searcher/$index.runs > ./results/qrels_10.txt/$searcher/$index.result
    done
done

for searcher in axiomatic_f1_exp axiomatic_f1_log axiomatic_f2_exp axiomatic_f2_log; do
    for index in porter krovetz none; do
        echo "Evaluate:  Index: $index, Searcher: $searcher, Running..."
        if [ ! -d ./results/qrels_10.txt/$searcher/$index ]; then
            mkdir -p ./results/qrels_10.txt/$searcher/
        fi
        perl ./src/trec_eval.pl -q ./query/qrels_10.txt ./rankings/qrels_10.txt/$searcher/$index.runs > ./results/qrels_10.txt/$searcher/$index.result
    done
done

echo "Evaluation done"