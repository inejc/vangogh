import argparse
import os
import sys

import keras.backend as K
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img

from common import dir_type, file_type

BATCH_SIZE = 512


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        '-p', '--preprocess_stats', type=file_type, required=True,
        help='path to the file with preprocessing statistics')

    parser.add_argument(
        '-m', '--model', type=file_type, required=True,
        help='path to the keras model')

    parser.add_argument(
        '-f', '--features_layer', type=int, required=True,
        help='index of model\'s features layer')

    parser.add_argument(
        '-l', '--list', type=file_type, required=True,
        help='file containing list of images to process')

    parser.add_argument(
        '-i', '--input', type=dir_type, required=True,
        help='input images directory')

    parser.add_argument(
        '-o', '--output', type=dir_type, required=True,
        help='output features directory')

    args = parser.parse_args(args=argv)
    return args


def _load_trained_cnn_layer(model_path, layer_index):
    model = load_model(model_path)
    dense_output = K.function(
        [model.layers[0].input, K.learning_phase()],
        [model.layers[layer_index].output])
    # output in test mode = 0
    return lambda X: dense_output([X, 0])[0]


def main(argv):
    args = parse_args(argv)

    with open(args.list) as f:
        allnames = f.read().splitlines()
        allnames = [x.split(' ')[-1].strip() for x in allnames[1:]]

    prep_stats = np.load(args.preprocess_stats)
    model = _load_trained_cnn_layer(args.model, args.features_layer)

    for sub in xrange(0, len(allnames), BATCH_SIZE):
        fnames = allnames[sub:sub + BATCH_SIZE]

        # preprocess batch
        batch = np.zeros((BATCH_SIZE, 3, 256, 256), dtype=K.floatx())
        for idx, fname in enumerate(fnames):
            fpath = os.path.join(args.input, fname)
            image = load_img(fpath)

            X = img_to_array(image, dim_ordering='th')
            X -= prep_stats['mean']
            X /= (prep_stats['std'] + 1e-7)
            batch[idx] = X

        features = model(batch)
        print features.shape

        # save features
        for idx, fname in enumerate(fnames):
            path = os.path.join(args.output, os.path.dirname(fname))
            if not os.path.exists(path):
                os.makedirs(path)
            fpath = os.path.join(args.output, fname + ".feat")

            feature = features[idx]
            if len(feature.shape) > 1:
                feature = feature.flatten()

            np.savetxt(fpath, feature)


if __name__ == "__main__":
    main(sys.argv[1:])
