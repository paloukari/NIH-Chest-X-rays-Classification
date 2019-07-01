# -*- coding: future_fstrings -*-

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import MobileNet
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, \
    GlobalAveragePooling2D, GlobalMaxPooling2D, \
    Dropout
from keras.models import Sequential, Model
from keras import optimizers, callbacks, regularizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

import data_preparation
import params


def create_data_generator(dataset,
                          labels,
                          batch_size,
                          target_size=params.IMG_SIZE):
    '''
    Creates a Keras DataGenerator for the input dataset

    Args:
      dataset: The images subset to use
      labels: The labels to use
      target_size: The (x, y) image size to scale the images
      batch_size: The batch_size of the generator

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
                                         rotation_range=5,
                                         shear_range=0.1,
                                         fill_mode='reflect',
                                         zoom_range=0.15)

    dataset_generator = image_generator.flow_from_dataframe(dataframe=dataset,
                                                            directory=None,
                                                            x_col='path',
                                                            y_col='newLabel',
                                                            class_mode='categorical',
                                                            classes=labels,
                                                            target_size=target_size,
                                                            color_mode='grayscale',
                                                            batch_size=batch_size)

    return dataset_generator


def create_VGG_model(labels, input_shape):
    '''
    '''
    model = applications.VGG19(weights="imagenet",
                               include_top=False,
                               input_shape=input_shape)

    model = MaxPooling2D((2, 2), strides=(
        2, 2), padding='same', name='pool4')(model)
    model = Flatten(name='flatten_')(model)
    model = Dense(4096, activation='relu', name='vgg_fc1/fc1_1')(model)
    model = Dense(4096, activation='relu', name='vgg_fc1/fc1_2')(model)
    model = Dense(labels, activation='relu', name='vgg_fc2')(model)
    predictions = Dense(16, activation="softmax")(model)

    # creating the final model
    model = Model(input=model.input, output=predictions)
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
    callbacks_list = [checkpoint, early]

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
        valid, labels, params.BATCH_SIZE)

    sample_X, sample_Y = next(create_data_generator(
        train, labels, params.BATCH_SIZE))

    model = create_simple_model(labels, sample_X.shape[1:])
    #model = create_VGG_model(labels, sample_X.shape[1:])

    fit_model(model, train_generator, validation_generator)


if __name__ == '__main__':
    train()
