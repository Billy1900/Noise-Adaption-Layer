import os
import os.path
import copy
import hashlib
import errno
import numpy as np
from numpy.testing import assert_array_almost_equal
from parse_config import args
from data.noise import build_for_cifar100


def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def download_url(url, root, filename, md5):
    from six.moves import urllib

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath)
        except:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath)


def list_dir(root, prefix=False):
    """List all directories at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(
        filter(
            lambda p: os.path.isdir(os.path.join(root, p)),
            os.listdir(root)
        )
    )

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]

    return directories


def list_files(root, suffix, prefix=False):
    """List all files ending with a suffix at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files


# basic function
def multiclass_noisify(y, P, random_state):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    # print(np.max(y), P.shape[0])
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    # print(m)
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


# noisify_pairflip call the function "multiclass_noisify"
def noisify_pairflip(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    print('----------Pair noise----------')
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        if args.dataset == 'mnist':
            """mistakes:
            1 <- 7
            2 -> 7
            3 -> 8
            5 <-> 6
            """
            # 1 <- 7
            P[7, 7], P[7, 1] = 1. - n, n
            # 2 -> 7
            P[2, 2], P[2, 7] = 1. - n, n
            # 5 <-> 6
            P[5, 5], P[5, 6] = 1. - n, n
            P[6, 6], P[6, 5] = 1. - n, n
            # 3 -> 8
            P[3, 3], P[3, 8] = 1. - n, n
            y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
            actual_noise = (y_train_noisy != y_train).mean()
            assert actual_noise > 0.0
            print('Actual noise %.2f' % actual_noise)
            y_train = y_train_noisy
            return y_train, actual_noise, P
        elif args.dataset == 'cifar10':
            """mistakes:
            automobile <- truck
            bird -> airplane
            cat <-> dog
            deer -> horse
            """
            # automobile <- truck
            P[9, 9], P[9, 1] = 1. - n, n
            # bird -> airplane
            P[2, 2], P[2, 0] = 1. - n, n
            # cat <-> dog
            P[3, 3], P[3, 5] = 1. - n, n
            P[5, 5], P[5, 3] = 1. - n, n
            # automobile -> truck
            P[4, 4], P[4, 7] = 1. - n, n
            y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
            actual_noise = (y_train_noisy != y_train).mean()
            assert actual_noise > 0.0
            print('Actual noise %.2f' % actual_noise)
            y_train = y_train_noisy
            return y_train, actual_noise, P
        elif args.dataset == 'cifar100':
            nb_superclasses = 20
            nb_subclasses = 5
            for i in np.arange(nb_superclasses):
                init, end = i * nb_subclasses, (i+1) * nb_subclasses
                P[init:end, init:end] = build_for_cifar100(nb_subclasses, n)

            y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
            actual_noise = (y_train_noisy != y_train).mean()
            assert actual_noise > 0.0
            print('Actual noise %.2f' % actual_noise)
            y_train = y_train_noisy
            return y_train, actual_noise, P
        else: # binary classes
            """mistakes:
            1 -> 0: n
            0 -> 1: .05
            """
            P[1, 1], P[1, 0] = 1.0 - n, n
            P[0, 0], P[0, 1] = 0.95, 0.05
            y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
            actual_noise = (y_train_noisy != y_train).mean()
            assert actual_noise > 0.0
            print('Actual noise %.2f' % actual_noise)
            y_train = y_train_noisy
            return y_train, actual_noise, P
    else:
        print('Actual noise %.2f, not a right one, right range is (0.0, 1.0)' % noise)
        return y_train, noise, P


def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    print('----------Symmetric noise----------')
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes - 1):
            P[i, i] = 1. - n
        P[nb_classes - 1, nb_classes - 1] = 1. - n

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
        # print(P)
        return y_train, actual_noise, P
    else:
        print('Actual noise %.2f' % noise)
        return y_train, noise, P


def noisify(dataset='mnist', nb_classes=10, train_labels=None, noise_type=None, noise_rate=0, random_state=1):
    global train_noisy_labels, actual_noise_rate
    if noise_type == 'pairflip':
        train_noisy_labels, actual_noise_rate, _ = noisify_pairflip(train_labels, noise_rate, random_state=1,
                                                                    nb_classes=nb_classes)
    if noise_type == 'symmetric':
        train_noisy_labels, actual_noise_rate, _ = noisify_multiclass_symmetric(train_labels, noise_rate,
                                                                                random_state=1,
                                                                                nb_classes=nb_classes)
    return train_noisy_labels, actual_noise_rate


def norm(T):
    row_sum = np.sum(T, 1)
    T_norm = T / row_sum
    return T_norm


def error(T, T_true):
    error = np.sum(np.abs(T - T_true)) / np.sum(np.abs(T_true))
    return error


def transition_matrix_generate(noise_rate=0.5, num_classes=10):
    P = np.ones((num_classes, num_classes))
    n = noise_rate
    P = (n / (num_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, num_classes - 1):
            P[i, i] = 1. - n
        P[num_classes - 1, num_classes - 1] = 1. - n
    return P


def fit(X, num_classes, filter_outlier=False):
    # number of classes
    c = num_classes
    T = np.empty((c, c))
    eta_corr = X
    for i in np.arange(c):
        if not filter_outlier:
            idx_best = np.argmax(eta_corr[:, i])
        else:
            eta_thresh = np.percentile(eta_corr[:, i], 97, interpolation='higher')
            robust_eta = eta_corr[:, i]
            robust_eta[robust_eta >= eta_thresh] = 0.0
            idx_best = np.argmax(robust_eta)
        for j in np.arange(c):
            T[i, j] = eta_corr[idx_best, j]
    return T


# flip clean labels to noisy labels
# train set and val set split
def dataset_split(train_images, train_labels, noise_rate=0.0, split_per=0.9, random_seed=1, num_classes=10):
    clean_train_labels = train_labels[:, np.newaxis]
    noisy_labels = clean_train_labels
    if args.noise_type == 'symmetric' and noise_rate > 0.0:
        noisy_labels, real_noise_rate, transition_matrix = noisify_multiclass_symmetric(clean_train_labels,
                                                                                        noise=noise_rate,
                                                                                        random_state=random_seed,
                                                                                        nb_classes=num_classes)
    elif args.noise_type == 'pairflip' and noise_rate > 0.0:
        noisy_labels, real_noise_rate, _ = noisify_pairflip(clean_train_labels, noise=noise_rate, random_state=random_seed,
                                                            nb_classes=num_classes)
    noisy_labels = noisy_labels.squeeze()
    #    print(noisy_labels)
    num_samples = int(noisy_labels.shape[0])
    np.random.seed(random_seed)
    train_set_index = np.random.choice(num_samples, int(num_samples * split_per), replace=False)
    index = np.arange(train_images.shape[0])
    val_set_index = np.delete(index, train_set_index)

    train_set, val_set = train_images[train_set_index, :], train_images[val_set_index, :]
    train_labels, val_labels = noisy_labels[train_set_index], noisy_labels[val_set_index]

    return train_set, val_set, train_labels, val_labels
