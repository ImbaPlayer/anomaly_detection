#!/bin/sh

init(){
    a=200
    b=0.1
    python svm_train.py $a $b >> result/result.log &
}


main(){
    init
    exit 0;
}


main