# convert the WT2G dataset to jsonl format
echo "Converting WT2G dataset to jsonl format"
python src/convert_wt2g_to_jsonl.py
echo "Conversion done"

# Description: This script is used to build index for the given collection
echo "Building index"
sh src/build_index.sh --nostem
sh src/build_index.sh --porter
sh src/build_index.sh --kstem
echo "Indexing done"

# Description: This script is used to build index for the given collection
# sh build_trecweb_index.sh --nostem
# sh build_trecweb_index.sh --porter
# sh build_trecweb_index.sh --kstem