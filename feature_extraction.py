import pickle
import tensorflow as tf
# TODO: import Keras layers you need here

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('dataset', '', "Use the bottleneck features from this dataset, one of 'cifar10', or 'traffic'")
flags.DEFINE_string('network', '', "Use the bottleneck features from this network, one of 'vgg', 'inception', or 'resnet'")


def load_bottleneck_data(network, dataset):
    """
    Utility function to load bottleneck features. Assumes bottleneck
    feature files are in the same directory.

    Arguments:
        network - String, one of 'resnet', 'vgg', 'inception'
        dataset - String, one of 'cifar10', 'traffic'
    """
    train_file = 'bottlenecks/{}_{}_bottleneck_features_train.p'.format(network, dataset)
    validation_file = 'bottlenecks/{}_{}_bottleneck_features_validation.p'.format(network, dataset)

    print("Training file", train_file)
    print("Validation file", validation_file)

    with open(train_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val

def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.network, FLAGS.dataset)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    # TODO: define your model and hyperparams here
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic


    # TODO: train your model here


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
