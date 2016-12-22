#!/bin/bash
set -x
MY_PATH="`dirname \"$0\"`"  # relative
MY_PATH="`( cd \"$MY_PATH\" && pwd )`" # absolutized and normalized

checkRet() {
	ret=$1
	if [[ $ret -eq 0 ]];then
		return
	else
		exit 1
	fi	
}

cd $MY_PATH/travis && git clone https://github.com/dmlc/mxnet.git --recursive
cd $MY_PATH/travis/mxnet && make -j4
checkRet $?

go get github.com/anthonynsimon/bild

sed -i "s/MXNET_SRC_DIR/$MY_PATH\/travis\/mxnet/g" $MY_PATH/tools/mxnet.pc
sudo cp $MY_PATH/tools/mxnet.pc /usr/local/lib/pkgconfig/
pkg-config --libs mxnet
checkRet $?

exit 0
