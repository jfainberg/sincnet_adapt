import sys
import numpy as np

from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
from keras.models import Model
from keras.layers import Input, Conv1D, BatchNormalization, MaxPooling1D
from keras.optimizers import Adam

from learning_to_adapt.utils import load_dataset, load_utt_to_spk, load_utt_to_pdfs

from layers import SincConv

import keras
import tensorflow as tf

config = tf.ConfigProto()
config.intra_op_parallelism_threads=8
config.inter_op_parallelism_threads=8
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

def create_model(hidden_dim=850, num_pdfs=4208):
    input_length = 16000 * 200 / 1000 # 3200
    feats = Input(shape=(input_length, 1))
    x = SincConv(filters=40, kernel_size=129, initializer='flat')(feats)
    x = MaxPooling1D(3)(x)

    x = Conv1D(hidden_dim, kernel_size=2, strides=1, activation="relu", name="1.affine" )(x)
    x = BatchNormalization(name="1.renorm")(x)
    x = MaxPooling1D(3)(x)

    x = Conv1D(hidden_dim, kernel_size=2, dilation_rate=3, activation="relu", name="2.affine")(x)
    x = BatchNormalization(name="2.renorm")(x)
    x = MaxPooling1D(3)(x)

    x = Conv1D(hidden_dim, kernel_size=2, dilation_rate=6, activation="relu", name="3.affine")(x)
    x = BatchNormalization(name="3.renorm")(x)
    x = MaxPooling1D(3)(x)

    x = Conv1D(hidden_dim, kernel_size=2, dilation_rate=9, activation="relu", name="4.affine")(x)
    x = BatchNormalization(name="4.renorm")(x)
    x = MaxPooling1D(3)(x)

    x = Conv1D(hidden_dim, kernel_size=2, dilation_rate=6, activation="relu", name="5.affine")(x)
    x = BatchNormalization(name="5.renorm")(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(hidden_dim, kernel_size=1, dilation_rate=1, activation="relu", name="6.affine")(x)

    y = Conv1D(num_pdfs, kernel_size=1, activation="softmax", name="output.affine")(x)

    return Model(inputs=[feats], outputs=[y])


if __name__ == '__main__':
    train_data = sys.argv[1]
    val_data = sys.argv[2]
    utt2spk = sys.argv[3]
    pdfs = sys.argv[4]
    left_context = int(sys.argv[5])
    right_context = int(sys.argv[6])
    output_path = sys.argv[7]

    # 1 epoch is roughly 25270800*0.97 = 24512676 frames 
    # 1 epoch is roughly 61 iterations of 400000 frames each
    num_iterations = 61*6
    batch_size = 256
    learning_rate = 0.0015

    utt_to_spk = load_utt_to_spk(utt2spk)
    utt_to_pdfs = load_utt_to_pdfs(pdfs)

    train_dataset = load_dataset(train_data, utt_to_spk, utt_to_pdfs, chunk_size=1, subsampling_factor=1, left_context=left_context, right_context=right_context)
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    train_dataset = train_dataset.prefetch(512)
    x, _, y = train_dataset.make_one_shot_iterator().get_next()

    val_dataset = load_dataset(val_data, utt_to_spk, utt_to_pdfs, chunk_size=1, subsampling_factor=1, left_context=left_context, right_context=right_context)
    val_dataset = val_dataset.batch(batch_size, drop_remainder=True)
    val_dataset = val_dataset.take(256).cache().repeat()
    val_x, _, val_y = val_dataset.make_one_shot_iterator().get_next()

    model = create_model(800, 3976)
    model.compile(
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        optimizer=Adam(lr=learning_rate, amsgrad=True, clipvalue=1.)
    )

    callbacks = [
        CSVLogger(output_path + "model.csv"),
        ModelCheckpoint(filepath=output_path + "model.{epoch:02d}.h5", save_best_only=False, period=10),
        ModelCheckpoint(filepath=output_path + "model.best.h5", save_best_only=True),
        LearningRateScheduler(lambda epoch, lr: learning_rate - epoch * (learning_rate - learning_rate / 10) / num_iterations, verbose=0)
    ]

    print(model.summary())
    model.fit(x, y,
        steps_per_epoch=400000//batch_size,
        epochs=num_iterations,
        validation_data=(val_x, val_y),
        validation_steps=(400000//batch_size // 10),
        callbacks=callbacks
    )
