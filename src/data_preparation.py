# -*- coding: future_fstrings -*-


from itertools import chain
import numpy as np
import pandas as pd
import os
from glob import glob
import params
from sklearn.model_selection import train_test_split



def load_metadata(data_folder=params.DATA_FOLDER,
                  metadata_file=params.INDICES_FILE):
    '''
    Loads the metadata from the indices_file csv file and scans the file system to map
    the png files to the metadata records.

    Args:
      data_folder: The path to the data folder
      metadata_file: The filename of the metadata csv file

    Returns:
      The metadata DataFrame with a file system mapping of the data (images)
    '''

    metadata = pd.read_csv(os.path.join(data_folder, metadata_file))
    file_system_scan = {os.path.basename(x): x for x in
                        glob(os.path.join(data_folder, 'images', '*.png'))}
    if len(file_system_scan) != metadata.shape[0]:
        raise Exception(
            'ERROR: Different number metadata records and png files.'.format())

    metadata['path'] = metadata['Image Index'].map(file_system_scan.get)
    print('Total x-ray records:{}.'.format((metadata.shape[0])))

    return metadata


def preprocess_metadata(metadata, minimum_cases=params.MIN_CASES):
    '''
    Preprocessing of the metadata df. We remove the 'No Finding' records
    and all labels with less than minimum_cases records. 

    Args:
      metadata: The metadata DataFrame

    Returns:
      metadata, labels : The preprocessed metadata DataFrame and the 
      valid labels left in the metadata.
    '''

    metadata['Finding Labels'] = metadata['Finding Labels'].map(
        lambda x: x.replace('No Finding', ''))

    labels = np.unique(
        list(chain(*metadata['Finding Labels'].map(lambda x: x.split('|')).tolist())))
    labels = [x for x in labels if len(x) > 0]

    for c_label in labels:
        if len(c_label) > 1:  # leave out empty labels
            metadata[c_label] = metadata['Finding Labels'].map(
                lambda finding: 1.0 if c_label in finding else 0)

    labels = [c_label for c_label in labels if metadata[c_label].sum()
              > minimum_cases]

    sample_weights = metadata['Finding Labels'].map(
        lambda x: len(x.split('|')) if len(x) > 0 else 0).values + 4e-2
    sample_weights /= sample_weights.sum()
    metadata = metadata.sample(40000, weights=sample_weights)

    labels_count = [(c_label, int(metadata[c_label].sum()))
                    for c_label in labels]

    print('Labels ({}:{})'.format((len(labels)), (labels_count)))
    print('Total x-ray records:{}.'.format((metadata.shape[0])))

    return metadata, labels


def stratify_train_test_split(metadata):
    '''
    Creates a train/test stratification of the dataset

    Args:
      metadata: The metadata DataFrame

    Returns:
      train, valid: The stratified train/test DataFrames
    '''
    stratify = metadata['Finding Labels'].map(lambda x: x[:4])
    train, valid = train_test_split(metadata,
                                    test_size=0.25,
                                    random_state=2018,
                                    stratify=stratify)
    return train, valid



if __name__ == '__main__':
    metadata = load_metadata()
    metadata, labels = preprocess_metadata(metadata)
    train, valid = stratify_train_test_split(metadata)
