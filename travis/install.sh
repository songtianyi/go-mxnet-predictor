#!/bin/bash
MY_PATH="`dirname \"$0\"`"  # relative
MY_PATH="`( cd \"$MY_PATH\" && pwd )`" # absolutized and normalized

cd $MY_PATH/travis && git clone https://github.com/dmlc/mxnet.git --recursive
cd $MY_PATH/travis/mxnet && make -j4
ln -s $MY_PATH/travis/lib/libmxnet.so /usr/lib/libmxnet.so
go get github.com/anthonynsimon/bild
sed -i "s/MXNET_SRC_DIR/$MY_PATH\/travis\/mxnet/g" $MY_PATH/tools/mxnet.pc
exit 0
