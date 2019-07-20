#!/usr/bin/env python
# coding: utf-8

import numpy as np 
import pandas as pd
import os
import time
import matplotlib.pyplot as plt

from glob import glob

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from itertools import chain
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import MobileNet
from keras.applications.vgg19 import VGG19
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from keras.models import Sequential
from keras import optimizers, callbacks, regularizers

def main():
    all_xray_df = pd.read_csv('../data/Data_Entry_2017.csv')
    all_image_paths = {os.path.basename(x): x for x in 
                   glob(os.path.join('..', 'data', 'images*', '*.png'))}
    print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])
    all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)
    all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)

    label_counts = all_xray_df['Finding Labels'].value_counts()[:15]

    all_xray_df['Finding Labels'] = all_xray_df['Finding Labels'].map(lambda x: x.replace('No Finding', ''))

    all_labels = np.unique(list(chain(*all_xray_df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
    all_labels = [x for x in all_labels if len(x)>0]
    print('All Labels ({}): {}'.format(len(all_labels), all_labels))
    for c_label in all_labels:
        if len(c_label)>1: # leave out empty labels
            all_xray_df[c_label] = all_xray_df['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)

    # keep at least 1000 cases
    MIN_CASES = 1000
    all_labels = [c_label for c_label in all_labels if all_xray_df[c_label].sum()>MIN_CASES]
    print('Clean Labels ({})'.format(len(all_labels)), 
      [(c_label,int(all_xray_df[c_label].sum())) for c_label in all_labels])

    # since the dataset is very unbiased, we can resample it to be a more reasonable collection

    # weight is 0.1 + number of findings
    sample_weights = all_xray_df['Finding Labels'].map(lambda x: len(x.split('|')) if len(x)>0 else 0).values + 4e-2
    sample_weights /= sample_weights.sum()
    all_xray_df = all_xray_df.sample(40000, weights=sample_weights)

    label_counts = all_xray_df['Finding Labels'].value_counts()[:15]

    label_counts = 100*np.mean(all_xray_df[all_labels].values,0)

    # # Prepare Training Data
    # Here we split the data into training and validation sets and create a single vector (disease_vec) with the 0/1 outputs for the disease status
    all_xray_df['disease_vec'] = all_xray_df.apply(lambda x: [x[all_labels].values], 1).map(lambda x: x[0])

    train_df, valid_df = train_test_split(all_xray_df, 
                                   test_size = 0.25, 
                                   random_state = 2018,
                                   stratify = all_xray_df['Finding Labels'].map(lambda x: x[:4]))
    print('train', train_df.shape[0], 'validation', valid_df.shape[0])

    # # Create Data Generators
    # Here we make the data generators for loading and randomly transforming images

    valid_df['newLabel'] = valid_df.apply(lambda x: x['Finding Labels'].split('|'), axis=1)
    train_df['newLabel'] = train_df.apply(lambda x: x['Finding Labels'].split('|'), axis=1)

    core_idg = ImageDataGenerator(samplewise_center=True, 
                              samplewise_std_normalization=True, 
                              horizontal_flip = True, 
                              vertical_flip = False, 
                              height_shift_range= 0.05, 
                              width_shift_range=0.1, 
                              rotation_range=5, 
                              shear_range = 0.1,
                              fill_mode = 'reflect',
                              zoom_range=0.15)

    def MakeMobileNetModel(IMG_SIZE, bs, channels=1):
        model = Sequential()
        base_model = MobileNet(input_shape =  (*IMG_SIZE, channels), #Need to define the shape here from IMG_SIZE
                                 include_top = False, weights = None)
        model.add(base_model)
        model.add(GlobalAveragePooling2D())
        model.add(Dropout(0.5))
        model.add(Dense(512))
        model.add(Dropout(0.5))
        model.add(Dense(len(all_labels), activation = 'sigmoid'))
        return model

    # # Create Models and Load Weights

    # ## Define Parameters

    imageSize = (512, 512)
    colorMode = 'rgb'
    channels = 3
    batchSize = 16

    valid_gen = core_idg.flow_from_dataframe(dataframe=valid_df, 
                         directory=None,
                         x_col = 'path',
                        y_col = 'newLabel', 
                         class_mode = 'categorical',
                        classes = all_labels,
                        target_size = imageSize,
                         color_mode = colorMode,
                        batch_size = batchSize,
                        seed=1234) 

    def createROC(all_labels, test_Y, pred_Y):
        aucScores = []
     
        for (idx, c_label) in enumerate(all_labels):
            fpr, tpr, thresholds = roc_curve(test_Y[:,idx].astype(int), pred_Y[:,idx])
            aucScores.append(auc(fpr,tpr))
        return aucScores

    # ## MobileNet

    mobilenet_model = MakeMobileNetModel(imageSize, batchSize, channels)
    mobilenet_model.load_weights('xray_class_weights.best.hdf5')

    start = time.time()
    predictions = []
    y_labels = []

    print("Total Batches to predict: {}".format(valid_gen.__len__()))
    for i in range(valid_gen.__len__()):
        if(i%10==0):
            print("Predicting batch:", i)
        x, y = next(valid_gen)
        predictions.append(mobilenet_model.predict(x))
        y_labels.append(y)

    end = time.time()
    duration = end - start
    print("Prediction Took {:.2} seconds".format(duration))

    test_Y = np.concatenate(y_labels)
    mobilenet_preds = np.concatenate(predictions)

    mobilenetAUC = createROC(all_labels, test_Y, mobilenet_preds)

    summaryDF = pd.DataFrame(
        {'Class':all_labels,
        'Mobilenet': mobilenetAUC,
        })

    print("Summary ROC Scores of Mobile Net Inference")
    print(summaryDF.round(3))

    print("Results Saved to CSV")
    summaryDF.to_csv("./cl_mobilenetInferenceDF.csv", index=False)


if __name__ == "__main__":
    main()




