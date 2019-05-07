
import numpy as np
from sklearn.model_selection import ParameterGrid
import tensorflow as tf
from tensorflow import keras


def main():
    # load dataset
    fashion_dataset = keras.datasets.fashion_mnist
    (train_dataset, train_label), (test_dataset, test_label) = fashion_dataset.load_data()

    train_dataset = train_dataset / 255.0
    test_dataset = test_dataset / 255.0

    param_grid = {'n_layer': [8, 16, 32, 64, 128, 256, 512], 'n_epoch': [3, 5]}
    grid = ParameterGrid(param_grid)
    best_accuracy = None
    best_param = None
    best_model = None

    for param in grid:
        # define model
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=train_dataset.shape[1:]),
            keras.layers.Dense(param['n_layer'], activation=tf.nn.relu),
            keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
        model.compile(
            optimizer=tf.train.AdamOptimizer(),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        model.fit(train_dataset, train_label, epochs=param['n_epoch'])

        # evaluate model
        test_loss, test_accuracy = model.evaluate(test_dataset, test_label)

        if best_accuracy is None or test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model = model
            best_param = param

    print('best accuracy', best_accuracy)
    print('best param', best_param)


if __name__ == '__main__':
    main()
