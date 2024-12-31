
for index in porter krovetz none; do
    echo "Index: $index, Searcher: _EXP, Running..."
    python ./part2.py --index $index --eval 10 
done


for index in porter krovetz none; do
    echo "Evaluate:  Index: $index, Searcher: _EXP, Running..."
    if [ ! -d ./results/qrels_10.txt/_EXP/$index ]; then
        mkdir -p ./results/qrels_10.txt/_EXP/
    fi
    perl ./src/trec_eval.pl -q ./query/qrels_10.txt ./rankings/qrels_10.txt/_EXP/$index.runs > ./results/qrels_10.txt/_EXP/$index.result
done
