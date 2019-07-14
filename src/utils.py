# -*- coding: future_fstrings -*-

"""
Adapted from:https://github.com/paloukari/OrcaDetector
"""

# import matplotlib this way to run without a display
import params
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('Agg')

from sklearn.metrics import classification_report, confusion_matrix

def calculate_accuracies(results, labels=None, run_timestamp='unspecified'):
    """
        Calculate, displays, and savees various accuracy metrics after a test run.  This method is
        only for post-processing; Keras reports its built-in accuracy calculations during
        training/validation runs.
        Args:
            results: list, with each list item being a dict of name/value pairs
            labels: list of labels
        Returns:
            no return value
    """

    if labels is None:
        return

    print(classification_report(labels, results))

    # also save classification report and confusion matrix to disk
    clf_report = classification_report(labels, results, output_dict=True)
    clf_filename = f'classification_report_{run_timestamp}.json'
    clf_path = os.path.join(params.RESULTS_FOLDER, clf_filename)
    df = pd.DataFrame(clf_report)
    df.to_json(clf_path, orient='columns')
    print(f'Classification report saved to {clf_path}')

    conf_matrix = confusion_matrix(labels, results)
    conf_matrix_filename = f'confusion_matrix_{run_timestamp}.csv'
    conf_matrix_path = os.path.join(params.RESULTS_FOLDER,
                                    conf_matrix_filename)
    df = pd.DataFrame(conf_matrix)
    df.to_csv(conf_matrix_path, index=True)
    print(f'Confusion matrix saved to {conf_matrix_path}')

    
def plot_train_metrics(model_history, model_name, run_timestamp='unspecified'):
    """
        Generate and save figure with plot of train and validation losses.
        Currently, plot_type='epochs' is the only option supported.
        Args:
            model_history: Keras History instance; History.history dict contains a list of metrics per epoch
            run_timestamp: time of the run, to be used in naming saved artifacts from the same run
        Returns:
            loss_fig_path: string representing path to the saved loss plot for the run
            acc_fig_path: string represeenting path to the saved accuracy plot for the run
    """

    # extract data from history dict
    train_losses = model_history.history['loss']
    val_losses = model_history.history['val_loss']
    final_val_loss = val_losses[-1]

    train_acc = model_history.history['binary_accuracy']
    val_acc = model_history.history['val_binary_accuracy']
    final_val_acc = val_acc[-1]

    # define filenames
    loss_filename = f'{model_name}_loss_plot_val_loss_{final_val_loss:.4f}_val_acc_{final_val_acc:.4f}_{run_timestamp}.png'
    loss_fig_path = os.path.join(params.RESULTS_FOLDER, loss_filename)
    acc_filename = f'{model_name}_accuracy_plot_val_loss_{final_val_loss:.4f}_val_acc_{final_val_acc:.4f}_{run_timestamp}.png'
    acc_fig_path = os.path.join(params.RESULTS_FOLDER, acc_filename)

    # generate and save loss plot
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.xticks(np.arange(0, len(train_losses), step=1))
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Model\nRun time: {run_timestamp}')
    plt.legend(('Training', 'Validation'))
    plt.savefig(loss_fig_path)

    # clear axes and figure to reset for next plot
    plt.cla()
    plt.clf()

    # generate and save accuracy plot
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.xticks(np.arange(0, len(train_acc), step=1))
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} Model\nRun time: {run_timestamp}')
    plt.legend(('Training', 'Validation'))
    plt.savefig(acc_fig_path)

    return loss_fig_path, acc_fig_path


def save_model(model, model_history, model_name, run_timestamp='unspecified'):
    """
        Generate and save figure with plot of train and validation losses.
        Currently, plot_type='epochs' is the only option supported.
        Args:
            model: trained Keras Model
            model_history: Keras History object, used to extract loss for file names.
            run_timestamp: time of the run, to be used in naming saved artifacts from the same run
        Returns:
            json_path: string representing path to the saved model config json file
            weights_path: string representing path to the saved model weights
    """

    # extract data from history dict
    final_val_loss = model_history.history['val_loss'][-1]
    final_val_acc = model_history.history['val_binary_accuracy'][-1]

    # save model config
    json_filename = f'{model_name}_config_val_loss_{final_val_loss:.4f}_val_acc_{final_val_acc:.4f}_{run_timestamp}.json'
    output_json = os.path.join(
        params.RESULTS_FOLDER, json_filename)
    with open(output_json, 'w') as json_file:
        json_file.write(model.to_json())

    # save trained model weights
    weights_filename = f'{model_name}_weights_val_loss_{final_val_loss:.4f}_val_acc_{final_val_acc:.4f}_{run_timestamp}.hdf5'
    output_weights = os.path.join(params.RESULTS_FOLDER, weights_filename)
    model.save_weights(output_weights)

    # Create symbolic link to the most recent weights (to use for testing)
    symlink_path = os.path.join(
        params.RESULTS_FOLDER, f'{model_name}_weights_latest.hdf5')
    try:
        os.symlink(output_weights, symlink_path)
    except FileExistsError:
        # If the symlink already exist, delete and create again
        os.remove(symlink_path)
        os.symlink(output_weights, symlink_path)
    print(f'Created symbolic link to final weights -> {symlink_path}')

    return output_json, output_weights
