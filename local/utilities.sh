#!/bin/bash

write_vars () {
    echo "--norm-means=false --norm-vars=false" > $1/cmvn_opts
    echo "--delta-order=0" > $1/delta_opts
    echo "--left-context=0 --right-context=0" > $1/splice_opts
    echo "$left_context $right_context" > $1/context_opts
    echo 1 > $1/frame_subsampling_factor
}

prepare_training_data () {
    data=$1
    num_splits=$2
    echo "Warning: Very few dev examples"
    mkdir $data/keras_train_split

    sort -R $data/feats.scp > $data/keras_train_split/all_feats.scp
    tail -n +5 $data/keras_train_split/all_feats.scp > $data/keras_train_split/feats.scp
    split --additional-suffix .scp --numeric-suffixes -n l/$num_splits -a 4 $data/keras_train_split/feats.scp $data/keras_train_split/feats_

    mkdir $data/keras_val_split
    head -n 5 $data/keras_train_split/all_feats.scp > $data/keras_val_split/feats.scp
    split --additional-suffix .scp --numeric-suffixes -n l/10 -a 4 $data/keras_val_split/feats.scp $data/keras_val_split/feats_

    rm $data/keras_train_split/all_feats.scp
}

split_data_opt () {
    # doesn't generate dev data
    data=$1
    train_split=$2
    num_splits=$3

    mkdir $train_split

    sort -R $data/feats.scp > $train_split/all_feats.scp

    copy-feats scp:$train_split/all_feats.scp ark,scp:$train_split/feats.ark,$train_split/feats.scp
    split --additional-suffix .scp --numeric-suffixes -n l/$num_splits -a 4 $train_split/feats.scp $train_split/feats_
}

split_data () {
    # doesn't generate dev data
    data=$1
    train_split=$2
    num_splits=$3

    mkdir $train_split

    sort -R $data/feats.scp > $train_split/feats.scp

    split --additional-suffix .scp --numeric-suffixes -n l/$num_splits -a 4 $train_split/feats.scp $train_split/feats_
}

prepare_training_data_opt () {
    data=$1
    num_splits=$2
    
    train_split=$data/keras_train_split_opt
    val_split=$data/keras_val_split_opt
    mkdir $train_split $val_split

    sort -R $data/feats.scp > $train_split/all_feats.scp

    tail -n +301 $train_split/all_feats.scp | sort -R \
        | copy-feats scp:- ark,scp:$train_split/feats.ark,$train_split/feats.scp
    split --additional-suffix .scp --numeric-suffixes -n l/$num_splits -a 4 $train_split/feats.scp $train_split/feats_

    head -n 300 $train_split/all_feats.scp | sort -R \
        | copy-feats scp:- ark,scp:$val_split/feats.ark,$val_split/feats.scp
    split --additional-suffix .scp --numeric-suffixes -n l/10 -a 4 $val_split/feats.scp $val_split/feats_
}
