# script to be sourced in travis yml
# setup all enviroment variables
export LIBRARY_PATH=$LIBRARY_PATH:${PWD}/travis/mxnet/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PWD}/travis/mxnet/lib/

