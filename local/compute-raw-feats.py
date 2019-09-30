#!/usr/bin/env python

from __future__ import division, print_function

import sys
import numpy as np

from kaldi.feat.mfcc import Mfcc, MfccOptions
from kaldi.matrix import Vector
from kaldi.util.options import ParseOptions
from kaldi.util.table import (MatrixWriter, RandomAccessFloatReaderMapped,
                              SequentialWaveReader)

def extract_windows(signal, window_length, label_length, label_shift):
    def rolling_window(arr, window, stride):
        shape = arr.shape[:-1] + (int((arr.shape[-1] - window)/stride + 1), window)
        strides = (stride*arr.strides[-1],) + (arr.strides[-1],)
        return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

    # Apply padding relative to centre labels
    label_center = label_length // 2 - 1

    start_idx = label_center - (window_length // 2)
    pad_left = abs(start_idx) if start_idx < 0 else 0

    remaining_frames = (signal.shape[0] + pad_left) % label_shift
    pad_right = (window_length // 2) - remaining_frames - label_shift // 2
    pad_right = pad_right if pad_right > 0 else 0

    signal = np.pad(signal, [pad_left, pad_right], mode='constant')

    return rolling_window(signal, window_length, label_shift)



def compute_mfcc_feats(wav_rspecifier, feats_wspecifier, opts, mfcc_opts):
    mfcc = Mfcc(mfcc_opts)

    # Shift by label window length so that feats align
    lab_window_len_sample = int((opts.sampling_rate * opts.label_window_length) / 1000)
    lab_window_shift_sample = int((opts.sampling_rate * opts.label_window_shift) / 1000)
    sig_window_len_sample = int((opts.sampling_rate * opts.signal_window_length) / 1000)

    num_utts, num_success = 0, 0
    with SequentialWaveReader(wav_rspecifier) as reader, \
         MatrixWriter(feats_wspecifier) as writer:
        for num_utts, (key, wave) in enumerate(reader, 1):
            if wave.duration < opts.min_duration:
                print("File: {} is too short ({} sec): producing no output."
                      .format(key, wave.duration), file=sys.stderr)
                continue

            num_chan = wave.data().num_rows
            if opts.channel >= num_chan:
                print("File with id {} has {} channels but you specified "
                      "channel {}, producing no output.", file=sys.stderr)
                continue
            channel = 0 if opts.channel == -1 else opts.channel

            try:
                # Move signal from integers to floats
                signal = wave.data()[channel].numpy()
                signal = signal.astype(float) / 2**15 # 32768  # int to float
                signal /= np.max(np.abs(signal)) # normalise

                # Extract windows
                feats = extract_windows(signal, sig_window_len_sample,
                                        lab_window_len_sample, lab_window_shift_sample)
            except:
                print("Failed to compute features for utterance", key,
                      file=sys.stderr)
                continue

            if opts.subtract_mean:
                mean = Vector(feats.num_cols)
                mean.add_row_sum_mat_(1.0, feats)
                mean.scale_(1.0 / feats.num_rows)
                for i in range(feats.num_rows):
                    feats[i].add_vec_(-1.0, mean)

            writer[key] = feats
            num_success += 1

            if num_utts % 10 == 0:
                print("Processed {} utterances".format(num_utts),
                      file=sys.stderr)

    print("Done {} out of {} utterances".format(num_success, num_utts),
          file=sys.stderr)

    return num_success != 0


if __name__ == '__main__':
    usage = """Create MFCC feature files.

    Usage:  compute-mfcc-feats [options...] <wav-rspecifier> <feats-wspecifier>
    """
    po = ParseOptions(usage)

    mfcc_opts = MfccOptions()
    mfcc_opts.register(po)

    po.register_int("sampling-rate", 16000, "Sampling rate of waveforms and labels.")
    po.register_int("signal-window-length", 200, "Window length in ms (what will be presented to the network).")
    po.register_int("label-window-length", 25, "Window length of alignments / labels in ms.")
    po.register_int("label-window-shift", 10, "Window shift of alignments / labels in ms.")
    po.register_bool("subtract-mean", False, "Subtract mean of each feature"
                     "file [CMS]; not recommended to do it this way.")
    po.register_int("channel", -1, "Channel to extract (-1 -> expect mono, "
                    "0 -> left, 1 -> right)")
    po.register_float("min-duration", 0.0, "Minimum duration of segments "
                      "to process (in seconds).")

    opts = po.parse_args()

    if (po.num_args() != 2):
      po.print_usage()
      sys.exit()

    wav_rspecifier = po.get_arg(1)
    feats_wspecifier = po.get_arg(2)

    compute_mfcc_feats(wav_rspecifier, feats_wspecifier, opts, mfcc_opts)
