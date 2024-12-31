# # convert the WT2G dataset to jsonl format
# if [ "$1" == "test" ]; then
#     python src/convert_wt2g_to_jsonl.py test
# else
#     python src/convert_wt2g_to_jsonl.py
# fi

# # index the WT2G dataset
# sh ./index.sh
# echo "Indexing done"

# # # run the queries
# echo "Running queries"
# for searcher in bm25 lm_MLE_smooth lm_JM_similar; do
#     for index in porter krovetz none; do
#         echo "Index: $index, Searcher: $searcher, Query: $eval, Running..."
#         python ./part1.py --index $index --searcher $searcher --eval 40
#     done
# done

for searcher in bm25 lm_MLE_smooth lm_JM_similar; do
    for index in porter krovetz none; do
        echo "Index: $index, Searcher: $searcher, Running..."
        success=false
        attempt=1
        max_attempts=3
        
        # Retry loop
        until $success || [ $attempt -gt $max_attempts ]; do
            python ./part1.py --index $index --searcher $searcher --eval 10
            if [ $? -eq 0 ]; then
                success=true
                echo "Index: $index, Searcher: $searcher, Completed successfully."
                echo ""
            else
                echo "Attempt $attempt for Index: $index, Searcher: $searcher failed. Retrying..."
                echo ""
                ((attempt++))
                sleep 2  # Optional: Wait for a few seconds before retrying
            fi
        done

        if ! $success; then
            echo "Index: $index, Searcher: $searcher failed after $max_attempts attempts."
        fi
    done
done


# Index: porter, Searcher: lm_JM_similar, Query: , Running...

# echo "Queries done"

# run the more queries
# echo "Running queries"
# for searcher in axiomatic_f1_exp axiomatic_f1_log axiomatic_f2_exp axiomatic_f2_log; do
#     for index in porter krovetz none; do
#         echo ""
#         echo "Index: $index, Searcher: $searcher, Query: $eval, Running..."
#         python ./part1.py --index $index --searcher $searcher --eval 40
#     done
# done

for searcher in axiomatic_f1_exp axiomatic_f1_log axiomatic_f2_exp axiomatic_f2_log; do
    for index in porter krovetz none; do
        echo "Index: $index, Searcher: $searcher, Running..."
        success=false
        attempt=1
        max_attempts=3
        
        # Retry loop
        until $success || [ $attempt -gt $max_attempts ]; do
            python ./part1.py --index $index --searcher $searcher --eval 10
            if [ $? -eq 0 ]; then
                success=true
                echo "Index: $index, Searcher: $searcher, Completed successfully."
                echo ""
            else
                echo "Attempt $attempt for Index: $index, Searcher: $searcher failed. Retrying..."
                echo ""
                ((attempt++))
                sleep 2  # Optional: Wait for a few seconds before retrying
            fi
        done

        if ! $success; then
            echo "Index: $index, Searcher: $searcher failed after $max_attempts attempts."
        fi
    done
done

for index in porter krovetz none; do
    echo "Index: $index, Searcher: _EXP, Running..."
    python ./part2.py --index $index --eval 10 
done

echo "Queries done"

echo "Cleaning up searcher logs..."
rm ./hs_err_pid*.log
echo "Part 2 Done"