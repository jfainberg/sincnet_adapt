#!/bin/bash

# Adapt to speaker on eval/adapt and test on same speaker on eval/test

. path.sh
. cmd.sh
. local/utilities.sh

export CUDA_VISIBLE_DEVICES=0
export TF_CPP_MIN_LOG_LEVEL=2

cmd=run.pl
ali="exp/ihm/tri3_cleaned_pfstar_train"

data="data/pfstar/train"
utt2spk=$data/utt2spk
pdfs="ark:ali-to-pdf $ali/final.mdl ark:'gunzip -c $ali/ali.*.gz |' ark,t:- |"
left_context=0
right_context=0

graph_dir="exp/ihm/tri3_cleaned/graph_ami+pfstar"

# Prepare raw data
nj=30
sdata=$data/split${nj}utt;


for spk in `cut -d" " -f1 data/pfstar/eval/adapt/spk2utt`; do
    output="exp/ami_sinc_40_flat_6epochs/adapt_pfstar_nobn_lhuc0sinc_lr15e-4_8epochs_spk$spk"
    # Prepare for total WER per epoch across speakers
    total_output="exp/ami_sinc_40_flat_6epochs/adapt_pfstar_nobn_lhuc0sinc_lr15e-4_8epochs_spkcomb"
    mkdir -p $output
    write_vars $output # cmvn_opts, etc.
    ln -sf `pwd`/$ali/final.mdl $output/final.mdl

    if [ ! -f $output/feats_lores.scp ]; then
        cat data/raw_PFSTAR_200ms/eval/adapt/feats.scp > $output/feats_all_raw.scp
        cat data/pfstar/eval/adapt/feats.scp > $output/feats_all.scp
        grep -h "$spk$" data/pfstar/eval/adapt/utt2spk > $output/uttlist
        utils/filter_scp.pl -f 1 $output/uttlist $output/feats_all_raw.scp > $output/feats.scp
        utils/filter_scp.pl -f 1 $output/uttlist $output/feats_all.scp > $output/feats_lores.scp

        utils/subset_data_dir.sh --utt-list $output/uttlist data/pfstar/eval/adapt data/pfstar/eval/adapt_spk$spk
    fi

    ali="exp/ihm/tri3_cleaned_pfstar_spk$spk"
    if [ ! -f $ali/ali.1.gz ]; then
        steps_kaldi/align_fmllr.sh --nj 1 data/pfstar/eval/adapt_spk$spk data/lang exp/ihm/tri3_cleaned $ali || exit 1
    fi
    pdfs="ark:ali-to-pdf $ali/final.mdl ark:'gunzip -c $ali/ali.*.gz |' ark,t:- |"

    num_utts=`wc -l < $output/feats.scp`
    echo "$0: Adapting to $num_utts for spk $spk"

    num_frames=`feat-to-len scp:$output/feats_lores.scp ark,t:- | awk '{sum+=$2} END {print sum}'`
    echo "$0: Adapting to $num_frames"

    data=$output
    num_splits=1
    if [ ! -d $data/keras_train_split ]; then
        split_data $data $data/keras_train_split $num_splits
    fi
    
    echo "WARN: val same as train"
    if [ ! -f $output/model.best.h5 ]; then
        python2.7 experiments/ami/adapt_pfstar_40_flat_speaker.py $data/keras_train_split $data/keras_train_split $utt2spk "$pdfs" $left_context $right_context $output "LHUC0+SINC" $num_frames || exit 1
    fi

    # Prepare age split test set
    if [ ! -d data/raw_PFSTAR_200ms/eval/test_spk$spk ]; then
        grep -h "$spk$" data/pfstar/eval/test/utt2spk > $output/test_uttlist
        utils/subset_data_dir.sh --utt-list $output/test_uttlist data/raw_PFSTAR_200ms/eval/test data/raw_PFSTAR_200ms/eval/test_spk$spk
    fi


    for e in `seq 1 8`; do
        if [ ! -f $output/decode_pfstar_test_spk${spk}_epoch${e}/lat.1.gz ]; then
            cp pdf_counts $output/
            steps/nnet3/decode_raw.sh --model "model.0${e}.h5" --skip-scoring true --nj 1 data/raw_PFSTAR_200ms/eval/test_spk$spk $output $graph_dir $output/decode_pfstar_test_spk${spk}_epoch$e
            echo "Done decoding adapted..."
            steps_kaldi/score_kaldi.sh data/raw_PFSTAR_200ms/eval/test_spk$spk $graph_dir $output/decode_pfstar_test_spk${spk}_epoch$e
        fi

        mkdir -p $total_output/epoch$e
        ln -sf `pwd`/$output/decode_pfstar_test_spk${spk}_epoch$e/lat.1.gz $total_output/epoch$e/lat.${spk}.gz
    done


    # Get score for baseline on subsetted test set
    # output_base="exp/ami_sinc_40_flat_6epochs/decode_pfstar_test"
    # if [ ! -f $output_base/subset_age$age/lat.1.gz ]; then
    #     mkdir -p $output_base/subset_age$age
    #     lattice-copy --include=data/raw_PFSTAR_200ms/test_age$age/utt2spk ark:"gunzip -c $output_base/lat.*.gz |" "ark:| gzip -c > $output_base/subset_age$age/lat.1.gz"
    #     steps_kaldi/score_kaldi.sh data/raw_PFSTAR_200ms/test_age$age $graph_dir $output_base/subset_age$age
    # fi

    # echo "Unadapted age $age:"
    # more $output_base/subset_age$age/scoring_kaldi/best_wer

    # echo "Adapted age $age:"
    # more $output/decode_pfstar_test_age$age/scoring_kaldi/best_wer
done

# Get total WER per epoch across speakers
for e in `seq 1 8`; do
    steps_kaldi/score_kaldi.sh data/raw_PFSTAR_200ms/eval/test $graph_dir $total_output/epoch$e
done
