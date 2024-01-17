import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
font = {'fontname': 'Helvetica'}

import seaborn as sb

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



def data_split_metric_values(Y_true, Y_pred):
    """
    args:
        Y_true - a vector of the real Y values of a data split e.g. the 
        training set, validation set, test

        Y_pred - a vector of the predicted Y values of an ML model given 
        a data split e.g. a training set, validation set, test set

        unique_labels - the unique values of the target/real Y output
        values. Note that it is not a good idea to pass the unique labels
        of one data split since it may not contain all unique labels

        given these arguments it creates a bar graph of all the relevant
        metrics in evaluating an ML model e.g. accuracy, precision,
        recall, and f1-score.
    """

    unique_labels = np.unique(Y_true)

    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(Y_true, Y_pred)
    print('Accuracy: {:.2%}'.format(accuracy))

    # precision tp / (tp + fp)
    precision = precision_score(Y_true, Y_pred, labels=unique_labels, average='weighted')
    print('Precision: {:.2%}'.format(precision))

    # recall: tp / (tp + fn)
    recall = recall_score(Y_true, Y_pred, labels=unique_labels,average='weighted')
    print('Recall: {:.2%}'.format(recall))

    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(Y_true, Y_pred, labels=unique_labels,average='weighted')
    print('F1 score: {:.2%}\n'.format(f1))

    return accuracy, precision, recall, f1

def multi_class_heatmap(conf_matrix, img_title: str="untitled", cmap: str='YlGnBu', save_img: bool=True):
    """
    takes in the confusion matrix returned by the confusion_matrix()
    function from sklearn e.g. conf_matrix_train = confusion_matrix(
        Y_true_train, Y_pred_train, labels=np.unique(Y_true_train)
    )

    other args:
        cmap - the color map you want the confusion matrix chart to have.
        Other values can be 'flare'
    """
    axis = sb.heatmap(conf_matrix, cmap=cmap, annot=True, fmt='g')
    axis.set_title(img_title, )

    if save_img:
        plt.savefig(f'./figures & images/{img_title}.png')
        plt.show()

def view_metric_values(metrics_df, img_title: str="untitled", save_img: bool=True, colors: list=['#2ac5b9', '#1ca3b6', '#0a557a', '#01363e',]):
    """
    given a each list of the training, validation, and testing set
    groups accuracy, precision, recall, and f1-score, plot a bar
    graph that separates these three groups metric values

    calculate accuracy, precision, recall, and f1-score for every 
    data split using the defined data_split_metric_values() function 
    above:

    train_acc, train_prec, train_rec, train_f1 = data_split_metric_values(Y_true_train, Y_pred_train)
    val_acc, val_prec, val_rec, val_f1 = data_split_metric_values(Y_true_val, Y_pred_val)
    test_acc, test_prec, test_rec, test_f1 = data_split_metric_values(Y_true_test, Y_pred_test)

    metrics_df = pd.DataFrame({
        'data_split': ['training', 'validation', 'testing'],
        'accuracy': [train_acc, val_acc, test_acc], 
        'precision': [train_prec, val_prec, test_prec], 
        'recall': [train_rec, val_rec, test_rec], 
        'f1-score': [train_f1, val_f1, test_f1]
    })
    """

    fig = plt.figure(figsize=(15, 10))
    axis = fig.add_subplot()

    # uses the given array of the colors you want to use
    sb.set_palette(sb.color_palette(colors))

    # create accuracy, precision, recall, f1-score of training group
    # create accuracy, precision, recall, f1-score of validation group
    # create accuracy, precision, recall, f1-score of testing group
    df_exp = metrics_df.melt(id_vars='data_split', var_name='metric', value_name='score')
    
    axis = sb.barplot(data=df_exp, x='data_split', y='score', hue='metric', ax=axis)
    axis.set_title(img_title, )
    axis.set_yscale('log')
    axis.legend()

    if save_img:
        plt.savefig(f'./figures & images/{img_title}.png')
        plt.show()

def view_classified_labels(df, img_title: str="untitled", save_img: bool=True, colors: list=['#db7f8e', '#b27392']):
    """
    given a each list of the training, validation, and testing set
    groups accuracy, precision, recall, and f1-score, plot a bar
    graph that separates these three groups metric values

    calculates all misclassified vs classified labels for training,
    validation, and testing sets by taking in a dataframe called
    classified_df created with the following code:

    num_right_cm_train = conf_matrix_train.trace()
    num_right_cm_val = conf_matrix_val.trace()
    num_right_cm_test = conf_matrix_test.trace()

    num_wrong_cm_train = train_labels.shape[0] - num_right_cm_train
    num_wrong_cm_val = val_labels.shape[0] - num_right_cm_val
    num_wrong_cm_test = test_labels.shape[0] - num_right_cm_test

    classified_df = pd.DataFrame({
        'data_split': ['training', 'validation', 'testing'],
        'classified': [num_right_cm_train, num_right_cm_val, num_right_cm_test], 
        'misclassified': [num_wrong_cm_train, num_wrong_cm_val, num_wrong_cm_test]}, 
        index=["training set", "validation set", "testing set"])
    """

    fig = plt.figure(figsize=(15, 10))
    axis = fig.add_subplot()

    # uses the given array of the colors you want to use
    sb.set_palette(sb.color_palette(colors))

    # create classified and misclassified of training group
    # create classified and misclassified of validation group
    # create classified and misclassified of testing group
    df_exp = df.melt(id_vars='data_split', var_name='status', value_name='score')
    
    axis = sb.barplot(data=df_exp, x='data_split', y='score', hue='status', ax=axis)
    axis.set_title(img_title, )
    axis.legend()

    if save_img:
        plt.savefig(f'./figures & images/{img_title}.png')
        plt.show()

def view_label_freq(label_freq, img_title: str="untitled", save_img: bool=True, labels: list | pd.Series | np.ndarray=["DER", "NDG", "OFF", "HOM"], horizontal: bool=True):
    """
    suitable for all discrete input

    main args:
        label_freq - is actually a the returned value of the method
        of a pandas series, e.g.
            label_freq = df['label'].value_counts()
            label_freq

        labels - a list of all the labels we want to use in the 
        vertical bar graph
    """

    # plots the unique labels against the count of these unique labels

    axis = sb.barplot(x=label_freq.values, y=labels, palette="flare") \
        if horizontal == True else sb.barplot(x=labels, y=label_freq.values, palette="flare")
    x_label = "frequency" if horizontal == True else "value"
    y_label = "value" if horizontal == True else "frequency"
    axis.set_xlabel(x_label, )
    axis.set_ylabel(y_label, )
    axis.set_title(img_title, )

    if save_img:
        plt.savefig(f'./figures & images/{img_title}.png')
        plt.show()

class ModelResults:
    def __init__(self, history, epochs):
        """
        args:
            history - the history dictionary attribute extracted 
            from the history object returned by the self.fit() 
            method of the tensorflow Model object 

            epochs - the epoch list attribute extracted from the history
            object returned by the self.fit() method of the tensorflow
            Model object
        """
        self.history = history
        self.epochs = epochs

    def _build_results(self, metrics_to_use: list):
        """
        builds the dictionary of results based on history object of 
        a tensorflow model

        returns the results dictionary with the format {'loss': 
        [24.1234, 12.1234, ..., 0.2134], 'val_loss': 
        [41.123, 21.4324, ..., 0.912]} and the number of epochs 
        extracted from the attribute epoch of the history object from
        tensorflow model.fit() method

        args:
            metrics_to_use - a list of strings of all the metrics to extract 
            and place in the dictionary
        """

        # extract the epoch attribute from the history object
        epochs = self.epochs
        results = {}
        for metric in metrics_to_use:
            if metric not in results:
                # extract the history attribute from the history object
                # which is a dictionary containing the metrics as keys, and
                # the the metric values over time at each epoch as the values
                results[metric] = self.history[metric]

        return results, epochs
    
    def export_results(self, dataset_id: str="untitled", metrics_to_use: list=['loss', 
                                            'val_loss', 
                                            'binary_crossentropy', 
                                            'val_binary_crossentropy', 
                                            'binary_accuracy', 
                                            'val_binary_accuracy', 
                                            'precision', 
                                            'val_precision', 
                                            'recall', 
                                            'val_recall', 
                                            'f1_m', 
                                            'val_f1_m', 
                                            'auc', 
                                            'val_auc',
                                            'categorical_crossentropy',
                                            'val_categorical_crossentropy'], save_img: bool=True):
        """
        args:
            metrics_to_use - a list of strings of all the metrics to extract 
            and place in the dictionary, must always be of even length
        """

        # extracts the dictionary of results and the number of epochs
        results, epochs = self._build_results(metrics_to_use)
        results_items = list(results.items())

        # we want to leave the user with the option to 
        for index in range(0, len(metrics_to_use) - 1, 2):
            # say 6 was the length of metrics to use
            # >>> list(range(0, 6 - 1, 2))
            # [0, 2, 4]
            metrics_indeces = (index, index + 1)
            curr_metric, curr_metric_perf = results_items[metrics_indeces[0]]
            curr_val_metric, curr_val_metric_perf = results_items[metrics_indeces[1]]
            print(curr_metric)
            print(curr_val_metric)
            curr_result = {
                curr_metric: curr_metric_perf,
                curr_val_metric: curr_val_metric_perf
            }
            print(curr_result)

            self.view_train_cross_results(
                results=curr_result,
                epochs=epochs, 
                curr_metrics_indeces=metrics_indeces,
                save_img=save_img,
                img_title="model performance using {} dataset for {} metric".format(dataset_id, curr_metric)
            )

    def view_train_cross_results(self, results: dict, epochs: list, curr_metrics_indeces: tuple, save_img: bool, img_title: str="untitled"):
        """
        plots the number of epochs against the cost given cost values 
        across these epochs.
        
        main args:
            results - is a dictionary created by the utility preprocessor
            function build_results()
        """

        figure = plt.figure(figsize=(15, 10))
        axis = figure.add_subplot()

        styles = [
            ('p:', '#f54949'), 
            ('h-', '#f59a45'), 
            ('o--', '#afb809'), 
            ('x:','#51ad00'), 
            ('+:', '#03a65d'), 
            ('8-', '#035aa6'), 
            ('.--', '#03078a'), 
            ('>:', '#6902e6'),
            ('p-', '#c005e6'),
            ('h--', '#fa69a3'),
            ('o:', '#240511'),
            ('x-', '#052224'),
            ('+--', '#402708'),
            ('8:', '#000000')]

        for index, (key, value) in enumerate(results.items()):
            # value contains the array of metric values over epochs
            # e.g. [213.1234, 123.43, 43.431, ..., 0.1234]

            if key == "loss" or key == "val_loss":
                # e.g. loss, val_loss has indeces 0 and 1
                # binary_cross_entropy, val_binary_cross_entropy 
                # has indeces 2 and 3
                axis.plot(
                    np.arange(len(epochs)), 
                    value, 
                    styles[curr_metrics_indeces[index]][0], 
                    color=styles[curr_metrics_indeces[index]][1], 
                    alpha=0.5, 
                    label=key, 
                    markersize=10, 
                    linewidth=3)
            else:
                # here if the metric value is not hte loss or 
                # validation loss each element is rounded by 2 
                # digits and converted to a percentage value 
                # which is why it is multiplied by 100 in order
                # to get much accurate depiction of metric value
                # that is not in decimal format
                metric_perc = [round(val * 100, 2) for val in value]
                axis.plot(
                    np.arange(len(epochs)), 
                    metric_perc, 
                    styles[curr_metrics_indeces[index]][0], 
                    color=styles[curr_metrics_indeces[index]][1], 
                    alpha=0.5, 
                    label=key, 
                    markersize=10, 
                    linewidth=3)

        # annotate end of lines
        for index, (key, value) in enumerate(results.items()):        
            if key == "loss" or key == "val_loss":
                last_loss_rounded = round(value[-1], 2)
                axis.annotate(last_loss_rounded, xy=(epochs[-1], value[-1]), color='black', alpha=1)
            else: 
                last_metric_perc = round(value[-1] * 100, 2)
                axis.annotate(last_metric_perc, xy=(epochs[-1], value[-1] * 100), color='black', alpha=1)

        axis.set_ylabel('metric value', )
        axis.set_xlabel('epochs', )
        axis.set_title(img_title, )
        axis.legend()

        if save_img == True:
            plt.savefig(f'./figures & images/{img_title}.png')
            plt.show()

        # delete figure
        del figure

def view_images(data_gen, grid_dims: tuple=(2, 6), size: tuple=(25, 10), model=None, img_title="untitled", save_img: bool=True):
    """
    views images created by the ImageGenerator() class 
    from tensorflow.keras.preprocessing.image

    args: 
        data_gen - the data generator created from the ImageGenerator()
        method self.flow_from_directory()
    """
    class_names = list(data_gen.class_indices.keys())
    
    # The plotting configurations
    n_rows, n_cols = grid_dims
    n_images = n_rows * n_cols
    plt.figure(figsize=size)
    
    # gets a batch of the image data for visualization
    # This process can take a little time because of 
    # the large batch size
    images, labels = next(data_gen) 

    # sample n_images of indeces from array of indeces 
    # of length len(images) without replacement
    sampled_indeces = np.random.choice([num for num in range(len(images))], n_images, replace=False)

    # Iterate through the subplots.
    for i, id in enumerate(sampled_indeces, start=1):
        # use the randomly sampled id as index to access 
        # an image and its respective label
        image, label = images[id], class_names[np.argmax(labels[id], axis=0)]
        
        # Plot the sub plot
        plt.subplot(n_rows, n_cols, i)
        plt.imshow(image)
        plt.axis('off')
        
        # If model is available make predictions
        if model is not None:
            pred = class_names[np.argmax(model.predict(image[np.newaxis,...]))]
            title = f"Class : {label}\nPred : {pred}"
        else:
            title = f"Class : {label}"
        
        plt.title(title, )
    
    if save_img == True:
        plt.savefig(f'./figures & images/{img_title}.png')
        plt.show()


def view_all_splits_results(history_dict: dict, save_img: bool=True, img_title: str="untitled"):
    """
    
    """
    history_df = pd.DataFrame(history_dict)
    print(history_df)

    palettes = np.array(['#f54949', '#f59a45', '#afb809', '#51ad00', '#03a65d', '#035aa6', '#03078a', '#6902e6', '#c005e6', '#fa69a3', '#240511', '#052224', '#402708', '#000000'])
    markers = np.array(['o', 'v', '^', '8', '*', 'p', 'h', ])#'x', '+', '>', 'd', 'H', '3', '4'])

    sampled_indeces = np.random.choice(list(range(len(markers))), size=history_df.shape[1], replace=False)

    print(palettes[sampled_indeces])
    print(markers[sampled_indeces])

    figure = plt.figure(figsize=(15, 10))
    axis = sb.lineplot(data=history_df, 
        palette=palettes[sampled_indeces].tolist(),
        markers=markers[sampled_indeces].tolist(), 
        linewidth=3.0,
        markersize=9,
        alpha=0.75)
    
    axis.set_ylabel('metric value', )
    axis.set_xlabel('epochs', )
    axis.set_title(img_title, )
    axis.legend()

    if save_img == True:
        print(save_img)
        plt.savefig(f'./figures & images/{img_title}.png')
        plt.show()

def show_image(img):
    plt.imshow(img)