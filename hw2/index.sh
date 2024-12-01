# Description: This script is used to build index for the given collection
sh src/build_index.sh --nostem
sh src/build_index.sh --porter
sh src/build_index.sh --kstem

# Description: This script is used to build index for the given collection
# sh build_trecweb_index.sh --nostem
# sh build_trecweb_index.sh --porter
# sh build_trecweb_index.sh --kstem