#!/bin/bash
set -x
MY_PATH="`dirname \"$0\"`"  # relative
MY_PATH="`( cd \"$MY_PATH\" && pwd )`" # absolutized and normalized

checkRet(ret) {
	if [[ $ret -eq 0 ]];then
		return
	else
		exit 1
	fi	
}

cd $MY_PATH/travis && git clone https://github.com/dmlc/mxnet.git --recursive
cd $MY_PATH/travis/mxnet && make -j4
checkRet $0

go get github.com/anthonynsimon/bild

sed -i "s/MXNET_SRC_DIR/$MY_PATH\/travis\/mxnet/g" $MY_PATH/tools/mxnet.pc
pkg-config --libs mxnet
checkRet $0

exit 0
