# -*- coding: future_fstrings -*-

import os
from time import time
import datetime

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input as MobileNet_preprocess_input
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as VGG19_preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input as InceptionResNetV2_preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as InceptionV3_preprocess_input
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input as MobileNetV2_preprocess_input
from keras.applications.nasnet import NASNetLarge
from keras.applications.nasnet import preprocess_input as NASNetLarge_preprocess_input

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
import reset
import gradient_accumulation
from utils import plot_train_metrics, save_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

RUN_TIMESTAMP = datetime.datetime.now().isoformat('-')


def create_data_generator(dataset,
                          labels,
                          batch_size,
                          preprocessing_function,
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
                                         samplewise_std_normalization=True,
                                         horizontal_flip=True,
                                         vertical_flip=False,
                                         height_shift_range=0.05,
                                         width_shift_range=0.1,
                                         brightness_range=[0.7, 1.5],
                                         rotation_range=5,
                                         shear_range=0.1,
                                         fill_mode='reflect',
                                         zoom_range=0.15,
                                         preprocessing_function=preprocessing_function)

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


def _create_attention_model(frozen_model, labels, optimizer='adam'):
    '''
      Creates an attention model to train on a pre-trained model
      output features

      Args:
        frozen_model: The VGG19 frozen network
        labels: The labels to use
        optimizer: The optimizer to use

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

    attention_model.compile(optimizer=optimizer, loss='binary_crossentropy',
                            metrics=['binary_accuracy'])

    return attention_model


def _create_base_model(Model, labels, input_shape, trainable=True, weights="imagenet"):
    '''
        Creates a Keras base model for transfer learning

        Args:
        Model: The Keras class to initialize
        labels: The labels to use
        input_shape: The shape of the Network input
        trainable: Is the model be able to be trained
        weights: Which pre-trained weights to use if any

        Returns:
        The created Model.

        '''

    base_model = Model(weights=weights,
                       include_top=False,
                       input_shape=input_shape)
    base_model.trainable = trainable

    return base_model


def create_simple_model(base_model, labels, optimizer='adam'):
    '''
    Creates a simple model by adding dropout, pooling, and dense layer to a pretrained model


    Args:
      base_model: The Keras base model
      labels: The labels to use
      optimizer: The optimizer to use

    Returns:
      The created Model.

    '''

    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(len(labels), activation='sigmoid'))
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy', 'mae'])
    print(f'{model.summary()}')
    return model


def create_attention_model(base_model, labels, optimizer='adam'):
    '''
    Creates an attention model by adding attention layers to base_model


    Args:
      base_model: The Keras Base Model to start with
      labels: The labels to use
      optimizer: The optimizer to use

    Returns:
      The created attention Model.

    '''

    attention_model = _create_attention_model(
        base_model, labels, optimizer=optimizer)

    model = Sequential(name='combined_model')
    model.add(base_model)
    model.add(attention_model)
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                  metrics=['binary_accuracy'])

    print(f'{model.summary()}')

    return model


def fit_model(model, model_name, train, valid):
    '''
    Fits the model.

    Args:
      model: The model to train
      train: The training data generator
      valid: The validation data generator
    '''
    results_folder = os.path.join(params.RESULTS_FOLDER, model_name)
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    weight_path = os.path.join(results_folder, params.WEIGHT_FILE_NAME)

    checkpoint = ModelCheckpoint(weight_path,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min',
                                 save_weights_only=True)

    early = EarlyStopping(monitor="val_loss",
                          mode="min",
                          patience=params.EARLY_STOPPING_PATIENCE)

    tensorboard = TensorBoard(log_dir=os.path.join(
        params.RESULTS_FOLDER, params.TENSORBOARD_BASE_FOLDER, model_name))

    #dynamicLR = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
    #                              patience=2, min_lr=params.LEARNING_RATE/1000)

    #callbacks_list = [tensorboard, checkpoint, dynamicLR, early]
    callbacks_list = [tensorboard, checkpoint, early]

    history = model.fit_generator(train,
                                  validation_data=valid,
                                  validation_steps=valid.samples//valid.batch_size,
                                  steps_per_epoch=train.samples//train.batch_size,
                                  epochs=params.EPOCHS,
                                  callbacks=callbacks_list)

    # save loss and accuracy plots to disk
    loss_fig_path, acc_fig_path = plot_train_metrics(
        history, model_name, results_folder,  RUN_TIMESTAMP)
    print(f'Saved loss plot -> {loss_fig_path}')
    print(f'Saved accuracy plot -> {acc_fig_path}')

    # save json model config file and trained weights to disk
    json_path, weights_path = save_model(
        model, history, model_name, results_folder, RUN_TIMESTAMP)
    print(f'Saved json config -> {json_path}')
    print(f'Saved weights -> {weights_path}')


def train_model(_Model, input_shape, preprocessing_function,
                train, valid, labels,
                extend_model_callback, optimizer, name_prefix):
    '''
    Trains a model based on the give Keras pre-trained class.
    '''

    train_generator = create_data_generator(
        train, labels, params.BATCH_SIZE, preprocessing_function, target_size=input_shape)
    validation_generator = create_data_generator(
        valid, labels, params.VALIDATION_BATCH_SIZE, preprocessing_function, target_size=input_shape)

    sample_X, sample_Y = next(create_data_generator(
        train, labels, params.BATCH_SIZE, preprocessing_function, target_size=input_shape))

    baseModel = _create_base_model(_Model,
                                   labels,
                                   sample_X.shape[1:],
                                   trainable=True,
                                   weights=None)

    model = extend_model_callback(baseModel, labels, optimizer)

    fit_model(model, f'{name_prefix}_{baseModel.name}',
              train_generator, validation_generator)

def train_multiple_networks():
    '''
    Trains list of CNNs.
    '''

    # for these image sizes, we don't need gradient_accumulation to achieve BATCH_SIZE = 256
    optimizer = 'adam'
    if params.BATCH_SIZE < 256:
        optimizer = gradient_accumulation.AdamAccumulate(
            lr=params.LEARNING_RATE, accum_iters=params.ACCUMULATION_STEPS)

    base_models = [
        #[MobileNet, params.IMG_SIZE, MobileNet_preprocess_input],
        [InceptionResNetV2, params.IMG_SIZE,
            InceptionResNetV2_preprocess_input],
        [VGG19, params.IMG_SIZE, VGG19_preprocess_input],
        [InceptionV3, params.IMG_SIZE, InceptionV3_preprocess_input],
        [MobileNetV2, params.IMG_SIZE, MobileNetV2_preprocess_input],
        [NASNetLarge, params.IMG_SIZE, NASNetLarge_preprocess_input],
    ]

    #Generate Train and Validation Data
    metadata = data_preparation.load_metadata()
    metadata, labels = data_preparation.preprocess_metadata(metadata)
    train, valid = data_preparation.stratify_train_test_split(metadata)

    for [_Model, input_shape, preprocess_input] in base_models:
        #Train Model
        train_model(_Model, input_shape, preprocess_input,
                    train, valid, labels,
                    create_simple_model, optimizer, 'simple')

    for [_Model, input_shape, preprocess_input] in base_models:
        #Train Model
        train_model(_Model, input_shape, preprocess_input,
                    train, valid, labels,
                    create_attention_model, optimizer, 'attention')

if __name__ == '__main__':
    reset.reset_keras()
    train_multiple_networks()
