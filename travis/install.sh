#!/bin/bash
MPWD=`pwd`

checkRet() {
	ret=$1
	if [[ $ret -eq 0 ]];then
		return
	else
		exit 1
	fi	
}

cd $MPWD/travis && git clone https://github.com/dmlc/mxnet.git --recursive
cd $MPWD/travis/mxnet && make -j4
checkRet $?

go get github.com/anthonynsimon/bild

sed -i "s/MXNET_SRC_DIR/$MPWD\/travis\/mxnet/g" $MPWD/tools/mxnet.pc
sudo cp $MPWD/tools/mxnet.pc /usr/lib/pkgconfig/
pkg-config --libs mxnet
checkRet $?

exit 0
