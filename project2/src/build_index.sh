if [ "$1" == "--porter" ]; then
    # Command for Porter Stemmer
    python -m pyserini.index.lucene \
      --collection JsonCollection \
      --input ./data/collection \
      --index ./indexes/porter \
      --stemmer porter \
      --generator DefaultLuceneDocumentGenerator \
      --threads 1 \
      --storePositions --storeDocvectors --storeRaw
elif [ "$1" == "--kstem" ]; then
    # Command for Krovetz Stemmer
    python -m pyserini.index.lucene \
      --collection JsonCollection \
      --input ./data/collection \
      --index ./indexes/krovetz \
      --stemmer krovetz \
      --generator DefaultLuceneDocumentGenerator \
      --threads 1 \
      --storePositions --storeDocvectors --storeRaw
elif [ "$1" == "--nostem" ]; then
    # Command without stemming
    python -m pyserini.index.lucene \
      --collection JsonCollection \
      --input ./data/collection \
      --index ./indexes/none \
      --stemmer none \
      --generator DefaultLuceneDocumentGenerator \
      --threads 1 \
      --storePositions --storeDocvectors --storeRaw
else
    echo "Usage: $0 [--porter | --kstem | --nostem]"
    exit 1
fi
