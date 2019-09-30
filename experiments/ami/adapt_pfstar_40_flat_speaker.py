import sys
import numpy as np
import os

from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
from keras.models import Model
from keras.layers import Input, Activation, Conv1D, BatchNormalization, Dense, MaxPooling1D
from keras.optimizers import Adam

from learning_to_adapt.model import LHUC, Renorm
from learning_to_adapt.utils import load_dataset, load_utt_to_spk, load_utt_to_pdfs

from layers import SincConv
from optimizers import MultiAdam

import keras
import tensorflow as tf

config = tf.ConfigProto()
config.intra_op_parallelism_threads=8
config.inter_op_parallelism_threads=8
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

def setup_model_adapt(model, method):

    if method == "SINC":
        model.trainable = True
        for l in model.layers:
            l.trainable = l.name.startswith('sinc')

        learning_rate = 0.0015

    if method == "ALL-SINC":
        model.trainable = True
        for l in model.layers:
            l.trainable = not l.name.startswith('sinc')

        learning_rate = 0.00015

    if method == "LHUC1+SINC":
        model.trainable = True
        model = insert_lhuc_layers(model, 1)
        for l in model.layers:
            l.trainable = l.name.startswith('lhuc') or l.name.startswith('sinc')

        learning_rate = 0.0015

    elif method == "LHUC0+SINC":
        model.trainable = True
        model = insert_lhuc_layer_after_sinc(model)
        for l in model.layers:
            l.trainable = l.name.startswith('lhuc') or l.name.startswith('sinc')

        learning_rate = 0.0015

    elif method.startswith("LHUC"):
        model.trainable = True
        if method == "LHUC1":
            model = insert_lhuc_layers(model, 1)
        elif method == "LHUC0":
            model = insert_lhuc_layer_after_sinc(model)
        else:
            model = insert_lhuc_layers(model)
        for l in model.layers:
            l.trainable = l.name.startswith('lhuc')

        learning_rate = 0.8


    return learning_rate, set_test_mode_for_batchnorm(model)

def insert_lhuc_layer_after_sinc(model):
    layer_list = []
    lhuc_layers = 0
    for i, layer in enumerate(model.layers):
        layer.trainable = False
        layer_list.append(layer)
        if layer.name.startswith("sinc"):
            layer_list.append(LHUC(name="lhuc.%s" % i, trainable=True))
            lhuc_layers += 1

    x = layer_list[0].output
    for layer in layer_list[1:]:
        x = layer(x)

    return Model(input=model.input, output=x)

def insert_lhuc_layers(model, num_layers=100):
    layer_list = []
    lhuc_layers = 0
    for i, layer in enumerate(model.layers):
        layer.trainable = False
        layer_list.append(layer)
        if layer.name.endswith("affine") and lhuc_layers < num_layers:
            layer_list.append(LHUC(name="lhuc.%s" % i, trainable=True))
            lhuc_layers += 1

    x = layer_list[0].output
    for layer in layer_list[1:]:
        x = layer(x)

    return Model(input=model.input, output=x)


def set_test_mode_for_batchnorm(m):
    input_length = 16000 * 200 / 1000 # 3200

    x = y = keras.layers.Input(shape=(input_length, 1))
    for l in m.layers:
        if l.name.startswith('input'):
            continue

        if l.name.endswith('batchnorm') and isinstance(l, keras.layers.BatchNormalization):
            y = l(y, training=False)
        else:
            y = l(y)

    m = keras.models.Model(inputs=x, outputs=y)
    # m.compile(loss='sparse_categorical_crossentropy', optimizer='sgd')

    return m


if __name__ == '__main__':
    train_data = sys.argv[1]
    val_data = sys.argv[2]
    utt2spk = sys.argv[3]
    pdfs = sys.argv[4]
    left_context = int(sys.argv[5])
    right_context = int(sys.argv[6])
    output_path = sys.argv[7]
    method = sys.argv[8]
    num_frames = int(sys.argv[9])

    # method = 'SINC'

    model = os.path.dirname(output_path) + '/model.best.h5'  # Assume model dir one up from adapt dir
    if not model.endswith('.h5'):
        raise TypeError ('Unsupported model type. Please use h5 format. Update Keras if needed')

    ## Load model
    model = keras.models.load_model(model, custom_objects={'SincConv':SincConv})

    epochs = 8
    num_samples = 40000
    if num_frames < num_samples:
        num_samples = num_frames
    num_iterations = (num_frames // num_samples) * epochs
    batch_size = 256
    # learning_rate = 0.0015

    utt_to_spk = load_utt_to_spk(utt2spk)
    utt_to_pdfs = load_utt_to_pdfs(pdfs)

    train_dataset = load_dataset(train_data, utt_to_spk, utt_to_pdfs, chunk_size=1, subsampling_factor=1, left_context=left_context, right_context=right_context)
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    x, _, y = train_dataset.make_one_shot_iterator().get_next()

    val_dataset = load_dataset(val_data, utt_to_spk, utt_to_pdfs, chunk_size=1, subsampling_factor=1, left_context=left_context, right_context=right_context)
    val_dataset = val_dataset.batch(batch_size, drop_remainder=True)
    val_dataset = val_dataset.take(128).cache().repeat()
    val_x, _, val_y = val_dataset.make_one_shot_iterator().get_next()

    learning_rate, model = setup_model_adapt(model, method)

    multiply_vars = list()
    for l in model.layers:
        if l.name.startswith('lhuc'):
            multiply_vars.extend(l.weights)

    # multiplier applies to lr on lhuc layers (multiply_vars)
    model.compile(
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        optimizer=MultiAdam(lr=learning_rate, amsgrad=True, clipvalue=1., multiplier=500, multiply_vars=multiply_vars)
    )

    callbacks = [
        CSVLogger(output_path + "/model.csv"),
        ModelCheckpoint(filepath=output_path + "/model.{epoch:02d}.h5", save_best_only=False, period=1),
        ModelCheckpoint(filepath=output_path + "/model.best.h5", save_best_only=True),
        LearningRateScheduler(lambda epoch, lr: learning_rate - epoch * (learning_rate - learning_rate / 10) / num_iterations, verbose=1)
    ]

    print(model.summary())
    model.fit(x, y,
        steps_per_epoch=num_samples//batch_size,
        epochs=num_iterations,
        validation_data=(val_x, val_y),
        validation_steps=(num_samples//batch_size),
        callbacks=callbacks
    )
