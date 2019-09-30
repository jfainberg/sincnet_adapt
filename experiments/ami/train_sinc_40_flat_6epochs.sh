#!/bin/bash

# Trains a model from scratch

. path.sh
. cmd.sh
. local/utilities.sh

export CUDA_VISIBLE_DEVICES=1
export TF_CPP_MIN_LOG_LEVEL=2

cmd=run.pl
ali="exp/ihm/tri3_cleaned_ali_train_cleaned"

data="data/ihm/train_cleaned_hires/"
utt2spk=$data/utt2spk
pdfs="ark:ali-to-pdf $ali/final.mdl ark:'gunzip -c $ali/ali.*.gz |' ark,t:- |"
left_context=0
right_context=0
output="exp/ami_sinc_40_flat_6epochs/"
mkdir -p $output

graph_dir="exp/ihm/tri3_cleaned/graph_ami_fsh.o3g.kn.pr1-7"
write_vars $output # cmvn_opts, etc.
ln -sf `pwd`/$ali/final.mdl $output/final.mdl

# Prepare raw data
nj=30
sdata=$data/split${nj}utt;
utils/split_data.sh --per-utt $data $nj
$cmd JOB=1:$nj --max-jobs-run 30 $output/log/extract_segments.JOB.log \
    extract-segments scp:$sdata/JOB/wav.scp $sdata/JOB/segments ark,scp:$sdata/JOB/wavseg.ark,$sdata/JOB/wavseg.scp
for i in `seq 1 $nj`; do
    mkdir -p $sdata/$i/wavs
    cat $sdata/$i/wavseg.scp | xargs -P 30 -I % sh -c 'utt=`echo % | cut -d" " -f1`; rspec=`echo % | cut -d" " -f2-`; wav-copy $rspec '"$sdata/$i"'/wavs/$utt.wav'
    cat $sdata/$i/wavseg.scp | xargs -P 30 -I % sh -c 'utt=`echo % | cut -d" " -f1`; echo $utt '"$sdata/$i/wavs/"'$utt.wav >> '"$data/wav.lst"
done

# TODO: avoid writing out wavs in an intermediate step between extract-segments and compute-raw-feats.py

scps=
for j in `seq 300`; do
    scps="$scps wav$j.scp"
done
mkdir $data/split300
split_scp.pl $data/wav.lst $data/split300/$scps

for j in `seq 1 300`; do
    python3 local/compute-raw-feats.py scp:wav$j.scp ark,scp:data/ihm/raw_AMI_200ms/new_train/$j.ark,data/ihm/raw_AMI_200ms/new_train/$j.scp
done

cat data/ihm/raw_AMI_200ms/new_train/*.scp  > data/ihm/raw_AMI_200ms/new_train/feats.scp

rawdata=data/ihm/raw_AMI_200ms/new_train
num_splits=10000
if [ ! -d $rawdata/keras_train_split_opt ]; then
    prepare_training_data_opt $rawdata $num_splits
fi

echo $output
if [ ! -f $output/model.best.h5 ]; then
    mkdir -p $output
    python2.7 experiments/ami/train_sinc_40_flat_6epochs.py $rawdata/keras_train_split_opt $rawdata/keras_val_split_opt $utt2spk "$pdfs" $left_context $right_context $output $rawdata/keras_test_split
fi

cp pdf_counts $output/
for test_set in dev eval; do
    echo "Decoding..."
    if [ ! -f $output/decode_${test_set}/lat.1.gz ]; then
        local/decode_raw.sh --nj 4 data/ihm/raw_AMI_200ms/${test_set} $output $graph_dir $output/decode_$test_set
    fi
    best_lmwt=`grep "Percent Total Error" $output/decode_$test_set/ascore_*/*.dtl \
                   | sed 's/\s\s*/ /g;' | sort -nk 5 | head -n 1 | grep -Po 'score_[0-9]*' | grep -Po '[0-9]*'`
    grep -A 6 "Percent Total Error" $output/decode_$test_set/ascore_$best_lmwt/*.dtl
done
