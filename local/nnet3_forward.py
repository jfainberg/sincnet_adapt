import itertools
import json
import math
from signal import signal, SIGPIPE, SIG_DFL
import os
import sys


if __name__ == '__main__':
    model = sys.argv[1]
    counts = sys.argv[2]
    frame_subsampling_factor = int(sys.argv[3])
    left_context = int(sys.argv[4])
    right_context = int(sys.argv[5])

    if len(sys.argv) > 6:
        jobid = int(sys.argv[6])
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = str(jobid - 1)

    import numpy as np
    import keras
    import kaldi_io
    import tensorflow as tf

    from learning_to_adapt.model import FeatureTransform, LHUC, Renorm, UttBatchNormalization

    from layers import SincConv
    from optimizers import MultiAdam

    config = tf.ConfigProto()
    config.intra_op_parallelism_threads=1
    config.inter_op_parallelism_threads=1
    keras.backend.tensorflow_backend.set_session(tf.Session(config=config))


    # if len(sys.argv) > 6:
    #     apply_exp = bool(sys.argv[6])
    # else:
    apply_exp = False

    if not model.endswith('.h5'):
        raise TypeError ('Unsupported model type. Please use h5 format. Update Keras if needed')

    ## Load model
    m = keras.models.load_model(model, custom_objects={
        'FeatureTransform': FeatureTransform,
        'LHUC': LHUC,
        'Renorm': Renorm,
        'UttBatchNormalization': UttBatchNormalization,
        'SincConv': SincConv,
        'MultiAdam': MultiAdam})

    if os.path.isfile(counts):
        with open(counts, 'r') as f:
            counts = np.fromstring(f.read().strip(" []"), dtype='float32', sep=' ')
        priors = counts / np.sum(counts)
        priors[priors==0] = 1e-5 # floor zero counts
    else:
        priors = 1

    with kaldi_io.SequentialBaseFloatMatrixReader("ark:-") as arkIn, \
            kaldi_io.BaseFloatMatrixWriter("ark,t:-") as arkOut:
        signal(SIGPIPE, SIG_DFL)

        for utt, feats in arkIn:
            feats = np.expand_dims(feats, 2)

            logProbMat = np.log(m.predict(feats).squeeze() / priors)
            logProbMat[logProbMat == -np.inf] = -100

            if apply_exp:
                arkOut.write(utt, np.exp(logProbMat))
            else:
                arkOut.write(utt, logProbMat)
