# -*- coding: future_fstrings -*-

import os
from time import time

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import MobileNet
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, \
    GlobalAveragePooling2D, GlobalMaxPooling2D, AvgPool2D, Lambda, \
    Dropout, GlobalAveragePooling2D, multiply, LocallyConnected2D, \
    BatchNormalization
from keras.models import Sequential, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, \
    LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

import numpy as np

import data_preparation
import params


def create_data_generator(dataset,
                          labels,
                          batch_size,
                          color_mode="rgb",
                          target_size=params.IMG_SIZE):
    '''
    Creates a Keras DataGenerator for the input dataset

    Args:
      dataset: The images subset to use
      labels: The labels to use
      batch_size: The batch_size of the generator
      color_mode: one of "grayscale", "rgb". Default: "rgb". 
      target_size: The (x, y) image size to scale the images

    Returns:
      The created ImageDataGenerator.
    '''

    dataset['newLabel'] = dataset.apply(
        lambda x: x['Finding Labels'].split('|'), axis=1)

    image_generator = ImageDataGenerator(samplewise_center=True,
                                         samplewise_std_normalization=False,
                                         horizontal_flip=True,
                                         vertical_flip=False,
                                         height_shift_range=0.1,
                                         width_shift_range=0.1,
                                         brightness_range=[0.7, 1.5],
                                         rotation_range=5,
                                         shear_range=0.01,
                                         fill_mode='nearest',
                                         zoom_range=0.125,
                                         preprocessing_function=preprocess_input)

    dataset_generator = image_generator.flow_from_dataframe(dataframe=dataset,
                                                            directory=None,
                                                            x_col='path',
                                                            y_col='newLabel',
                                                            class_mode='categorical',
                                                            classes=labels,
                                                            target_size=target_size,
                                                            color_mode=color_mode,
                                                            batch_size=batch_size)

    return dataset_generator


def _create_attention_model(frozen_model, labels):
    '''
      Creates an attention model to train on the frozen VGG19 
      output features

      Args:
        frozen_model: The VGG19 frozen network
        labels: The labels to use

      Returns:
        The created Model.
      '''

    frozen_features = Input(frozen_model.get_output_shape_at(0)[
        1:], name='feature_input')
    frozen_depth = frozen_model.get_output_shape_at(0)[-1]
    new_features = BatchNormalization()(frozen_features)

    # here we do an attention mechanism to turn pixels in the GAP on an off
    attention_layer = Conv2D(128, kernel_size=(1, 1), padding='same',
                             activation='elu')(new_features)
    attention_layer = Conv2D(32, kernel_size=(1, 1), padding='same',
                             activation='elu')(attention_layer)
    attention_layer = Conv2D(16, kernel_size=(1, 1), padding='same',
                             activation='elu')(attention_layer)
    attention_layer = AvgPool2D((2, 2), strides=(1, 1), padding='same')(
        attention_layer)  # smooth results
    attention_layer = Conv2D(1,
                             kernel_size=(1, 1),
                             padding='valid',
                             activation='sigmoid')(attention_layer)

    # fan it out to all of the channels
    up_c2_w = np.ones((1, 1, 1, frozen_depth))
    up_c2 = Conv2D(frozen_depth, kernel_size=(1, 1), padding='same',
                   activation='linear', use_bias=False, weights=[up_c2_w])
    up_c2.trainable = False
    attention_layer = up_c2(attention_layer)

    mask_features = multiply([attention_layer, new_features])
    gap_features = GlobalAveragePooling2D()(mask_features)
    gap_mask = GlobalAveragePooling2D()(attention_layer)

    # to account for missing values from the attention model
    gap = Lambda(lambda x: x[0]/x[1],
                 name='RescaleGAP')([gap_features, gap_mask])
    gap_dr = Dropout(0.5)(gap)
    dr_steps = Dropout(0.5)(Dense(128, activation='elu')(gap_dr))
    out_layer = Dense(len(labels), activation='sigmoid')(dr_steps)

    # creating the final model
    attention_model = Model(inputs=[frozen_features], outputs=[
        out_layer], name='attention_model')

    attention_model.compile(optimizer='adam', loss='binary_crossentropy',
                            metrics=['binary_accuracy'])

    return attention_model


def _create_VGG19_model(labels, input_shape):
    '''
    Creates a VGG19 based model for transfer learning

    Args:
      labels: The labels to use
      input_shape: The shape of the Network input

    Returns:
      The created Model.

    '''
    frozen_model = VGG19(weights="imagenet",
                                      include_top=False,
                                      input_shape=input_shape)
    frozen_model.trainable = False

    return frozen_model


def create_VGG19_attention_model(labels, input_shape):
    '''
    Creates a VGG19 attention model for transfer learning

    Args:
      labels: The labels to use
      input_shape: The shape of the Network input

    Returns:
      The created Model.

    '''

    frozen_model = _create_VGG19_model(labels, input_shape)

    print(f'{frozen_model.summary()}')

    attention_model = _create_attention_model(frozen_model, labels)
    print(f'{attention_model.summary()}')

    model = Sequential(name='combined_model')
    model.add(frozen_model)
    model.add(attention_model)
    model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy',
                  metrics=['binary_accuracy'])

    print(f'{model.summary()}')

    return model


def create_simple_model(labels, input_shape):
    '''
    Creates a simple model based on MobileNet


    Args:
      labels: The labels to use
      input_shape: The shape of the Network input

    Returns:
      The created Model.

    '''

    base_mobilenet_model = MobileNet(input_shape=input_shape,
                                     include_top=False,
                                     weights=None)
    model = Sequential()
    model.add(base_mobilenet_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(len(labels), activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy', 'mae'])
    print(f'{model.summary()}')
    return model


def fit_model(model, train, valid):
    '''
    Fits the model.

    Args:
      model: The model to train
      train: The training data generator
      valid: The validation data generator

    Returns:
      The created Model.
    '''

    weight_path = "{}_weights.best.hdf5".format('xray_class')

    checkpoint = ModelCheckpoint(weight_path,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min',
                                 save_weights_only=True)

    early = EarlyStopping(monitor="val_loss",
                          mode="min",
                          patience=5)

    tensorboard = TensorBoard(log_dir=params.RESULTS_FOLDER)

    callbacks_list = [tensorboard, checkpoint, early]

    model.fit_generator(train,
                        validation_data=valid,
                        validation_steps=valid.samples//valid.batch_size,
                        steps_per_epoch=params.STEPS_PER_EPOCH,
                        epochs=params.EPOCHS,
                        callbacks=callbacks_list,
                        use_multiprocessing=True,
                        workers=params.WORKERS)


def train():
    '''
    Trains a CNN.
    '''

    metadata = data_preparation.load_metadata()
    metadata, labels = data_preparation.preprocess_metadata(metadata)
    train, valid = data_preparation.stratify_train_test_split(metadata)

    train_generator = create_data_generator(train, labels, params.BATCH_SIZE)
    validation_generator = create_data_generator(
        valid, labels, params.VALIDATION_BATCH_SIZE)

    sample_X, sample_Y = next(create_data_generator(
        train, labels, params.BATCH_SIZE))

    #model = create_simple_model(labels, sample_X.shape[1:])
    model = create_VGG19_attention_model(labels, sample_X.shape[1:])

    fit_model(model, train_generator, validation_generator)


if __name__ == '__main__':
    train()
