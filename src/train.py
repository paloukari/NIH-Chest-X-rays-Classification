# -*- coding: future_fstrings -*-

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import MobileNet
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from keras.models import Sequential
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

    # TODO: replace the next() with the proper Keras param
    valid_X, valid_Y = next(valid)
    model.fit_generator(train,
                        validation_data=(valid_X, valid_Y),
                        steps_per_epoch=1000,
                        epochs=1,
                        callbacks=callbacks_list)


def train():
    '''
    Trains a CNN.
    '''

    metadata = data_preparation.load_metadata()
    metadata, labels = data_preparation.preprocess_metadata(metadata)
    train, valid = data_preparation.stratify_train_test_split(metadata)

    train_generator = create_data_generator(train, labels, 32)
    validation_generator = create_data_generator(valid, labels, 1024)

    sample_X, sample_Y = next(create_data_generator(train, labels, 32))

    model = create_simple_model(labels, sample_X.shape[1:])

    fit_model(model, train_generator, validation_generator)


if __name__ == '__main__':
    train()
