#!/bin/bash

. cmd.sh
. path.sh

cmd=run.pl
max_active=7000 # max-active
beam=15.0
latbeam=7.0
acwt=0.1
model="model.best.h5"
nj=32
skip_scoring=false

. parse_options.sh || exit 1;

data=$1
model_dir=$2
graph_dir=$3
decode_dir=$4

for f in $graph_dir/HCLG.fst $data/feats.scp; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

cmvn_opts=`cat $model_dir/cmvn_opts`
delta_opts=`cat $model_dir/delta_opts`
splice_opts=`cat $model_dir/splice_opts`
context_opts=`cat $model_dir/context_opts`
frame_subsampling_factor=`cat $model_dir/frame_subsampling_factor`

# Split data by speaker
# nj=`cat $data/spk2utt | wc -l`
utils/split_data.sh $data $nj
sdata=$data/split${nj}

 # $cmd JOB=1:$nj --max-jobs-run 30 $decode_dir/log/extract_segments.JOB.log \
#     extract-segments scp:$sdata/JOB/wav.scp $sdata/JOB/segments ark,scp:$sdata/JOB/wavseg.ark,$sdata/JOB/wavseg.scp
# for i in `seq 1 $nj`; do
#     mkdir -p $sdata/$i/wavs
#     cat $sdata/$i/wavseg.scp | xargs -P 30 -I % sh -c 'utt=`echo % | cut -d" " -f1`; rspec=`echo % | cut -d" " -f2-`; wav-copy $rspec '"$sdata/$i"'/wavs/$utt.wav'
#     cat $sdata/$i/wavseg.scp | xargs -P 30 -I % sh -c 'utt=`echo % | cut -d" " -f1`; echo $utt '"$sdata/$i/wavs/"'$utt.wav >> '"$data/wav.lst"
# done
# $cmd JOB=1:$nj --max-jobs-run 4 $decode_dir/log/compute_raw_feats.JOB.log \
#     python compute-raw-feats.py scp:$sdata/JOB/wav.scp ark,scp:$sdata/JOB/raw.ark,$sdata/JOB/raw.scp
# exit 0
# Prepare raw features by running ../pytorch-kaldi/save_raw_fea.py on $data
# Make sure that my new feats are the same as these, then continue
# mkdir data/ihm/raw_AMI_200ms/new_train

# Write predictions

adaptation_pdfs="ark:$decode_dir/pdfs"

feats="ark,s,cs:copy-feats scp:$sdata/JOB/feats.scp ark:- |"
feats="$feats python2.7 local/nnet3_forward.py $model_dir/$model $model_dir/pdf_counts $frame_subsampling_factor $context_opts JOB |"
# feats="$feats grep -v 'import dot_parser' |"
decode_opts="--max-active=$max_active --beam=$beam --lattice-beam=$latbeam --acoustic-scale=$acwt --allow-partial=true"
lat_wspecifier="ark:| gzip -c > $decode_dir/lat.JOB.gz"

mkdir -p $decode_dir
echo "Decoding to $decode_dir"
$cmd JOB=1:$nj --max-jobs-run 4 $decode_dir/log/decode.JOB.log \
  latgen-faster-mapped-parallel --num-threads=4 $decode_opts --word-symbol-table=$graph_dir/words.txt $model_dir/final.mdl $graph_dir/HCLG.fst "$feats" "$lat_wspecifier"

echo $nj > $decode_dir/num_jobs

if ! $skip_scoring; then
    bash steps/score_kaldi.sh --min-lmwt 5 $data $graph_dir $decode_dir
fi
