import math
import numpy as np
import random
import collections
import tensorflow as tf
import h5py
try:
    import kaldi_io
except:
    print("WARNING: Failed to import kaldi_io, assuming this is intentional (e.g. no Kaldi)!")

SILENCE_PDFS = set([0, 41, 43, 60, 118])

def load_dataset(feats_dir, utt_to_spk, utt_to_pdfs, chunk_size, subsampling_factor=1, left_context=0, right_context=0):
    if subsampling_factor != 1:
        raise ValueError('Data generator works only with subsampling_factor=1')

    def _map_fn(path):
        if path.endswith("hdf5"):
            f = h5py.File(path, 'r')
            feats_reader = [(utt, f[utt].value) for utt in f]
        else:
            feats_reader = kaldi_io.SequentialBaseFloatMatrixReader("scp:%s" % path)

        feats = []
        spks = []
        pdfs = []
        for (utt, utt_feats) in feats_reader:
            if utt not in utt_to_pdfs:
                continue

            # spk = utt_to_spk[utt] if random.random() < 0.5 else 0
            spk = 0  # HACK
            utt_pdfs = utt_to_pdfs[utt]

            utt_subsampled_length = utt_feats.shape[0] / subsampling_factor
            if abs(utt_subsampled_length - utt_pdfs.shape[0]) > 1:
                continue

            utt_feats = utt_feats[:utt_subsampled_length * subsampling_factor]
            utt_pdfs = utt_pdfs[:utt_subsampled_length]
            chunks = create_chunks(utt_feats, utt_pdfs, utt_pdfs, chunk_size, left_context, right_context, subsampling_factor)

            feats.extend([chunk[0] for chunk in chunks])
            spks.extend([spk for chunk in chunks])
            pdfs.extend([chunk[1] for chunk in chunks])

        return np.array(feats, dtype=np.float32), np.array(spks, dtype=np.int32), np.array(pdfs, dtype=np.int32)

    def _reshape_fn(x, y, z):
        return (
            tf.reshape(x, [-1, 3200, chunk_size - left_context + right_context]),
            tf.reshape(y, [-1, 1]),
            tf.reshape(z, [-1, chunk_size, 1])
        )

    dataset = tf.data.Dataset.list_files("%s/feats_*" % feats_dir, seed=0)
    dataset = dataset.map(lambda path: tf.py_func(_map_fn, [path], [tf.float32, tf.int32, tf.int32]))
    dataset = dataset.map(_reshape_fn)
    dataset = dataset.apply(tf.contrib.data.unbatch())
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(10000, seed=0))

    return dataset

def load_utt_to_pdfs(pdfs):
    utt_to_pdfs = {}
    if pdfs.endswith("hdf5"):
        f = h5py.File(pdfs, 'r')
        reader = [(utt, f[utt].value) for utt in f]
    else:
        reader = kaldi_io.SequentialInt32VectorReader(pdfs)

    for utt, utt_pdfs in reader:
        utt_to_pdfs[utt] = utt_pdfs.reshape((-1, 1))

    return utt_to_pdfs

def load_utt_to_spk(utt2spk):
    spks = {'unk': 0}
    utt_to_spk = {}
    with open(utt2spk, 'r') as f:
        for line in f:
            (utt, spk) = line.split()

            if spk not in spks:
                spks[spk] = len(spks)

            utt_to_spk[utt] = spks[spk]

    return utt_to_spk

def create_chunks(feats, adapt_pdfs, test_pdfs, chunk_size, left_context, right_context, subsampling_factor, trim_silence=False):
    chunks = []
    feats = pad_feats(feats, left_context, right_context)

    for start, end in get_segments(test_pdfs, chunk_size, trim_silence):
        if end - start < chunk_size:
            continue

        for offset in get_offsets(start, end, chunk_size):
            chunk_feats = feats[offset:offset + chunk_size + right_context - left_context]
            chunk_adapt_pdfs = adapt_pdfs[offset:offset + chunk_size]
            chunk_test_pdfs = test_pdfs[offset:offset + chunk_size]
            chunks.append((chunk_feats, chunk_adapt_pdfs, chunk_test_pdfs))

    return chunks


def get_segments(pdfs, chunk_size, trim_silence):
    if not trim_silence:
        return [(0, pdfs.shape[0])]

    last_end = 0
    is_silence = False
    silences = [(0, 0)]
    pdfs = pdfs.flatten()
    for i, pdf in enumerate(pdfs):
        if is_silence:
            if pdf not in SILENCE_PDFS:
                silences.append((i - silence_length, i))
                is_silence = False
                silence_length = 0
            else:
                silence_length += 1
        else:
            if pdf in SILENCE_PDFS:
                is_silence = True
                silence_length = 1

    if is_silence:
        silences.append((pdfs.shape[0] - silence_length, pdfs.shape[0]))
    else:
        silences.append((pdfs.shape[0], pdfs.shape[0]))

    return [(e1, s2) for ((s1, e1), (s2, e2)) in zip(silences, silences[1:])]

def pad_feats(feats, left_context, right_context):
    if left_context == 0 and right_context == 0:
        return feats

    padded_feats = np.zeros((feats.shape[0] - left_context + right_context, feats.shape[1]))
    padded_feats[:-left_context,:] = feats[0]
    padded_feats[-right_context:,:] = feats[-1]
    padded_feats[-left_context:-right_context,:] = feats

    return padded_feats

def get_offsets(start, end, window):
    length = end - start
    num_chunks = (length - window) / window

    if num_chunks == 0:
        return [start]
    else:
        # Distribute chunks uniformly on the segment, there might be gaps between segments.
        shift = float(length - window) / num_chunks
        return [start + int(shift * i) for i in range(num_chunks)] + [end - window]
