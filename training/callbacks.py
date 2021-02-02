import io
import itertools
import logging
import logging.handlers  # noqa
import operator
import os
import warnings

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import Callback
from tensorflow.python.keras.utils import tf_utils


class ModelCheckpointLogged(tf.keras.callbacks.ModelCheckpoint):
    """
    This class add a log file with the saved model. It is helpful to see the metrics about the model when it was saved.
    """
    def __init__(self, *args, **kwargs):
        """
        See more information in tensorflow.keras.callbacks.ModelCheckPoint

        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        log_level = logging.INFO
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        file_handler = logging.handlers.RotatingFileHandler(f"{os.path.splitext(self.filepath)[0]}.log")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)

        self._logger = logging.getLogger(f"ModelCheckpointLogged_{self.monitor}")
        self._logger.addHandler(file_handler)
        self._logger.setLevel(log_level)

    def on_epoch_end(self, batch, logs=None):
        if isinstance(self.save_freq, int) or self.epochs_since_last_save + 1 >= self.period:
            logs = logs or {}
            logs = tf_utils.to_numpy_or_python_type(logs)
            current = logs.get(self.monitor)
            if self.monitor_op(current, self.best):
                log_message = f"last_best: {self.best}"
                for key in logs:
                    log_message = ", ".join([log_message, f"{key}: {logs[key]}"])
                self._logger.info(log_message)
        super().on_epoch_end(batch, logs)


class ConfusionMatrixTB(Callback):
    """
    Callback that generate a confusion matrix and plot this image into the tensorboard.
    """
    def __init__(self, data_val, class_names, logdir, period=1, monitor=None, mode=None):
        """

        :param data_val: [tuple[numpy.ndarray, [numpy.ndarray]]] A tuple with tha index 0 as the input data and index 1
            as output data.
        :param class_names:[list[str]] A list with all labels.
        :param logdir: [str] Path where the tensorboard log file will be saved.
        :param period: [int] The period tha this callback will be executed. If the monitor arg was set, this parameter
            will be ignored.
        :param monitor: [str] Set some metric in the training to execute this callback whenever this metric improves.
        :param mode: [str] "max" or "min" Set the direction of the monitor.
        """
        super().__init__()
        self._logdir = logdir
        self._class_names = class_names
        self._data_val = data_val
        self._file_writer_cm = tf.summary.create_file_writer(self._logdir + '/cm')
        self._period = period
        self._monitor = monitor

        if self._monitor is not None and mode is None:
            raise ValueError("Please set 'mode' with parameters 'max' or 'min'")
        if mode not in ["max", "min"]:
            raise ValueError("The argument 'mode' must be 'max' or 'min'")
        self._mode = mode

        self._operator = operator.gt if mode == "max" else operator.lt
        self._best = np.inf if self._mode is not None and self._mode == "min" else -np.inf

    @staticmethod
    def _plot_confusion_matrix(cm, class_names):
        """
          Returns a matplotlib figure containing the plotted confusion matrix.

          Args:
            cm (array, shape = [n, n]): a confusion matrix of integer classes
            class_names (array, shape = [n]): String names of the integer classes
          """
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Compute the labels from the normalized confusion matrix.
        labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return figure

    @staticmethod
    def _plot_to_image(figure, path=None):
        """
        Converts the matplotlib plot specified by 'figure' to a PNG image and
          returns it. The supplied figure is closed and inaccessible after this call.
          """
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image

    def log_confusion_matrix(self, epoch, logs):
        # Use the model to predict the values from the test_images.

        true_labels = []
        pred_labels = []

        pred_raw = self.model.predict(self._data_val[0])
        if len(self._class_names) > 2:
            true_labels += list(self._data_val[1])
            pred_labels += list(pred_raw.argmax(axis=1))
        else:
            true_labels += list(self._data_val[1])
            pred_labels += list(np.round(pred_raw.flatten()))
        # Calculate the confusion matrix using sklearn.metrics
        cm = confusion_matrix(true_labels, pred_labels)

        figure = self._plot_confusion_matrix(cm, class_names=self._class_names)
        cm_image = self._plot_to_image(figure)

        with self._file_writer_cm.as_default():
            tf.summary.image(f"Confusion Matrix ({self._monitor})", cm_image, step=epoch)

    def on_epoch_end(self, epoch, logs=None):
        if self._monitor is not None:
            logs = tf_utils.to_numpy_or_python_type(logs)
            if self._monitor not in logs:
                warnings.warn(f"'{self._monitor}' does not exist, ignoring confusion matrix for this epoch.")
                return
            value = logs[self._monitor]
            if self._operator(value, self._best):
                self._best = value
                self.log_confusion_matrix(epoch, logs)
        elif epoch % self._period == 0:
            self.log_confusion_matrix(epoch, logs)
