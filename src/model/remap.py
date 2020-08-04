import copy
import logging
import numpy as np
import os
import sys
import time
import warnings
from scipy.sparse import lil_matrix
from scipy.special import expit, softmax
from sklearn import preprocessing
from sklearn.utils._joblib import Parallel, delayed
from utility.access_file import save_data, load_data

logger = logging.getLogger(__name__)
EPSILON = np.finfo(np.float).eps
UPPER_BOUND = np.log(sys.float_info.max) * 10
LOWER_BOUND = np.log(sys.float_info.min) * 10
np.random.seed(12345)
np.seterr(divide='ignore', invalid='ignore')


class reMap:
    def __init__(self, alpha=16, binarize_input_feature=True, fit_intercept=True, decision_threshold=0.5,
                 learning_type="optimal", lr=0.0001, lr0=0.0, forgetting_rate=0.9, delay_factor=1.0, max_sampling=3,
                 subsample_input_size=0.3, subsample_labels_size=50, cost_subsample_size=100, min_bags=10, max_bags=50,
                 score_strategy=True, loss_threshold=0.05, early_stop=False, pi=0.4, calc_bag_cost=True,
                 calc_label_cost=True, calc_total_cost=False, varomega=0.3, varrho=0.7, min_negatives_ratio=0.3,
                 lambdas=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                 label_bag_sim=True, label_closeness_sim=True, corr_bag_sim=True, corr_label_sim=True,
                 corr_input_sim=True, batch=10, num_epochs=5, num_jobs=1, display_interval=2, shuffle=True,
                 random_state=12345, log_path='../../log'):
        logging.basicConfig(filename=os.path.join(log_path, 'reMap_events'), level=logging.DEBUG)
        self.binarize_input_feature = binarize_input_feature
        self.fit_intercept = fit_intercept
        self.lambdas = lambdas
        self.decision_threshold = decision_threshold
        self.alpha = alpha
        self.learning_type = learning_type
        self.lr = lr
        self.lr0 = lr0
        self.forgetting_rate = forgetting_rate
        self.delay_factor = delay_factor
        self.max_sampling = max_sampling
        self.subsample_input_size = subsample_input_size
        self.subsample_labels_size = subsample_labels_size
        self.cost_subsample_size = cost_subsample_size
        self.min_bags = min_bags
        self.max_bags = max_bags
        score_strategy = score_strategy
        if score_strategy and corr_bag_sim:
            self.check_loss_diff = False
        else:
            self.check_loss_diff = True
        self.loss_threshold = loss_threshold
        self.early_stop = early_stop
        self.calc_bag_cost = calc_bag_cost
        self.calc_label_cost = calc_label_cost
        self.calc_total_cost = calc_total_cost
        self.pi = pi
        self.varomega = varomega
        self.varrho = varrho
        self.min_negatives_ratio = min_negatives_ratio
        self.label_bag_sim = label_bag_sim
        self.label_closeness_sim = label_closeness_sim
        self.corr_bag_sim = corr_bag_sim
        self.corr_label_sim = corr_label_sim
        self.corr_input_sim = corr_input_sim
        self.num_epochs = num_epochs
        self.batch = batch
        self.display_interval = display_interval
        self.shuffle = shuffle
        self.random_state = random_state
        self.num_jobs = num_jobs
        self.verbose = 0
        self.log_path = log_path
        warnings.filterwarnings("ignore", category=Warning)

    def __print_arguments(self, **kwargs):
        argdict = dict()
        argdict.update({'num_bags': 'Number of bags: {0}'.format(self.num_bags)})
        argdict.update({'num_labels': 'Number of labels: {0}'.format(self.num_labels)})
        argdict.update({'alpha': 'A hyper-parameter for controlling bags centroids: {0}'.format(self.alpha)})
        argdict.update({'binarize': 'Binarize data? {0}'.format(self.binarize_input_feature)})
        argdict.update({'fit_intercept': 'Whether the intercept should be estimated '
                                         'or not? {0}'.format(self.fit_intercept)})
        argdict.update({'decision_threshold': 'The decision cutoff threshold: {0}'.format(self.decision_threshold)})
        argdict.update({'learning_type': 'The learning rate schedule: {0}'.format(self.learning_type)})
        if self.learning_type == "optimal":
            argdict.update({'lr': 'The learning rate: {0}'.format(self.lr)})
            argdict.update({'lr0': 'The initial learning rate: {0}'.format(self.lr0)})
        else:
            argdict.update({'forgetting_rate': 'Forgetting rate to control how quickly old '
                                               'information is forgotten: {0}'.format(self.forgetting_rate)})
            argdict.update(
                {'delay_factor': 'Delay factor down weights early iterations: {0}'.format(self.delay_factor)})
        argdict.update({'max_sampling': 'Maximum number of random samplings: {0}'.format(self.max_sampling)})
        argdict.update({'subsample_labels_size': 'Subsampling labels: {0}'.format(self.subsample_labels_size)})
        argdict.update({'subsample_labels_size': 'Subsampling labels: {0}'.format(self.subsample_labels_size)})
        argdict.update(
            {'cost_subsample_size': 'Subsampling size for computing a cost: {0}'.format(self.cost_subsample_size)})
        argdict.update({'label_bag_sim': 'Whether to enforce labels to a bag '
                                         'similarity constraint? {0}'.format(self.label_bag_sim)})
        argdict.update({'label_closeness_sim': 'Whether to enforce labels similarity '
                                               'constraint? {0}'.format(self.label_closeness_sim)})
        argdict.update({'corr_bag_sim': 'Whether to enforce bags correlation '
                                        'constraint from dataset? {0}'.format(self.corr_bag_sim)})
        argdict.update({'corr_label_sim': 'Whether to enforce labels correlation '
                                          'constraint from dataset? {0}'.format(self.corr_label_sim)})
        argdict.update({'corr_input_sim': 'Whether to enforce instances correlation '
                                          'constraint from a dataset? {0}'.format(self.corr_input_sim)})
        argdict.update({'calc_label_cost': 'Whether to include labels cost? {0}'.format(self.calc_label_cost)})
        argdict.update({'calc_bag_cost': 'Whether to include bags cost? {0}'.format(self.calc_bag_cost)})
        argdict.update({'calc_total_cost': 'Whether to compute total cost? {0}'.format(self.calc_total_cost)})
        argdict.update({'min_bags': 'Minimum number of bags for each sample: {0}'.format(self.min_bags)})
        argdict.update({'max_bags': 'Maximum number of bags for each sample: {0}'.format(self.max_bags)})
        score_strategy = False if self.check_loss_diff else True
        argdict.update({'score_strategy': 'Whether to update bags based on score '
                                          'threshold strategy? {0}'.format(score_strategy)})
        argdict.update({'loss_threshold': 'A cutoff threshold of the differences of loss between two consecutive '
                                          'rounds: {0}'.format(self.loss_threshold)})
        argdict.update({'pi': 'A prior parameter for positive bags: {0}'.format(self.pi)})
        argdict.update({'varomega': 'A prior parameter for biased negative bags: {0}'.format(self.varomega)})
        argdict.update({'varrho': 'A hyper-parameter for positive bags: {0}'.format(self.varrho)})
        argdict.update({'min_negatives_ratio': 'A hyper-parameter for creating a balanced '
                                               'positive/negative bags: {0}'.format(self.min_negatives_ratio)})
        argdict.update({'lambdas': 'Six hyper-parameters for constraints: {0}'.format(self.lambdas)})
        argdict.update({'batch': 'Number of examples to use in each iteration: {0}'.format(self.batch)})
        argdict.update({'num_epochs': 'Maximum number of times the model loops over '
                                      'training set: {0}'.format(self.num_epochs)})
        argdict.update({'shuffle': 'Shuffle the datset? {0}'.format(self.shuffle)})
        argdict.update({'log_path': 'The location of the log information: {0}'.format(self.log_path)})
        argdict.update({'display_interval': 'How often to evaluate? {0}'.format(self.display_interval)})
        argdict.update({'num_jobs': 'Number of parallel workers: {0}'.format(self.num_jobs)})
        argdict.update({'random_state': 'The random number generator: {0}'.format(self.random_state)})

        for key, value in kwargs.items():
            argdict.update({key: value})
        args = list()
        for key, value in argdict.items():
            args.append(value)
        args = [str(item[0] + 1) + '. ' + item[1] for item in zip(list(range(len(args))), args)]
        args = '\n\t\t'.join(args)
        print('\t>> The following arguments are applied:\n\t\t{0}'.format(args), file=sys.stderr)
        logger.info('\t>> The following arguments are applied:\n\t\t{0}'.format(args))

    def __shffule(self, num_samples):
        if self.shuffle:
            idx = np.arange(num_samples)
            np.random.shuffle(idx)
            return idx

    def __check_bounds(self, X):
        X = np.clip(X, LOWER_BOUND, UPPER_BOUND)
        if len(X.shape) > 1:
            if X.shape[0] == X.shape[1]:
                min_x = np.min(X) + EPSILON
                max_x = np.max(X) + EPSILON
                X = X - min_x
                X = X / (max_x - min_x)
                X = 2 * X - 1
        return X

    def __init_variables(self, num_samples):
        """Initialize latent variables.
        :param num_samples:
        """

        # initialize parameters for bags_labels
        self.coef_bag = np.random.uniform(low=-1 / np.sqrt(self.bag_feature_size),
                                          high=1 / np.sqrt(self.bag_feature_size),
                                          size=(self.num_bags, self.bag_feature_size))
        self.intercept_bag = np.zeros(shape=(self.num_bags, 1))

        # initialize parameters for pathways
        self.coef_label = np.random.uniform(low=-1 / np.sqrt(self.input_feature_size),
                                            high=1 / np.sqrt(self.input_feature_size),
                                            size=(self.num_labels, self.input_feature_size))
        self.intercept_label = np.zeros(shape=(self.num_labels, 1))

        # initialize a linear transformation matrix
        if self.label_bag_sim:
            self.U = np.random.uniform(low=-1 / np.sqrt(self.input_feature_size),
                                       high=1 / np.sqrt(self.bag_feature_size),
                                       size=(self.input_feature_size, self.bag_feature_size))

        self.W = np.random.uniform(low=-1 / np.sqrt(self.num_bags),
                                   high=1 / np.sqrt(self.num_labels),
                                   size=(self.num_bags, self.num_labels))

        # initialize a similarity matrix
        init_gamma = 100.
        init_var = 1. / init_gamma
        self.S = np.random.gamma(shape=init_gamma, scale=init_var, size=(num_samples, num_samples))
        np.fill_diagonal(self.S, 0)
        self.S = self.S / np.sum(self.S, axis=0)[:, np.newaxis]
        i_lower = np.tril_indices(num_samples, -1)
        self.S[i_lower] = self.S.T[i_lower]
        self.S = lil_matrix(self.S)

    def __optimal_learning_rate(self, alpha):
        def _loss(p, y):
            z = p * y
            # approximately equal and saves the computation of the log
            if z > 18:
                return np.exp(-z)
            if z < -18:
                return -z
            return np.log(1.0 + np.exp(-z))

        typw = np.sqrt(1.0 / np.sqrt(alpha))
        # computing lr0, the initial learning rate
        initial_eta0 = typw / max(1.0, _loss(-typw, 1.0))
        # initialize t such that lr at first sample equals lr0
        optimal_init = 1.0 / (initial_eta0 * alpha)
        return optimal_init

    def __sigmoid(self, X):
        return expit(X)

    def __softmax(self, X, axis=None):
        return softmax(X, axis=axis)

    def __log_logistic(self, X, negative=True):
        param = 1
        if negative:
            param = -1
        X = np.clip(X, EPSILON, 1 - EPSILON)
        X = param * np.log(1 + np.exp(X))
        return X

    def __norm_l21(self, M):
        if M.size == 0:
            return 0.0
        if len(M.shape) == 2:
            ret = np.sum(np.power(M, 2), axis=1)
        else:
            ret = np.power(M, 2)
        ret = np.sum(np.sqrt(ret))
        return ret

    def __grad_l21_norm(self, M):
        if len(M.shape) == 2:
            D = 1 / (2 * np.linalg.norm(M, axis=1))
            ret = np.dot(np.diag(D), M)
        else:
            D = (2 * np.linalg.norm(M) + EPSILON)
            ret = M / D
        return ret

    def __threshold_closeness(self, H):
        average_prob_labels = np.mean(H, axis=0)
        G = 1 - average_prob_labels
        prob_bags2sample = np.multiply(average_prob_labels, G) + 0.0001
        prob_bags2sample = prob_bags2sample / np.sum(prob_bags2sample)
        return prob_bags2sample

    def __extract_centroids(self, y, labels, bags):
        num_samples = y.shape[0]
        norm_label_features = self.label_features / np.linalg.norm(self.label_features, axis=1)[:, np.newaxis]
        if len(bags) > 1:
            labels = labels[0]
            c_hat = np.array([np.multiply(y[n], self.bags_labels) for n in np.arange(num_samples)])
            c_hat = c_hat[:, bags, :]
            c_hat = np.dot(c_hat, norm_label_features)
            c_hat = self.alpha * (c_hat / np.sum(self.bags_labels[bags], axis=1)[:, np.newaxis])
        else:
            bags = bags[0]
            c_hat = np.array([np.multiply(y[n], self.bags_labels[bags]) for n in np.arange(num_samples)])
            c_hat = np.dot(c_hat, norm_label_features)
            c_hat = self.alpha * (c_hat / np.sum(self.bags_labels[bags]))
        c_bar = np.abs(c_hat - self.centroids[bags])
        return c_bar

    def __scale_diagonal(self, D):
        assert D.shape[0] == D.shape[1]
        with np.errstate(divide='ignore'):
            D = 1.0 / np.sqrt(D)
        D[np.isinf(D)] = 0
        return D

    def __normalize_laplacian(self, A):
        A.setdiag(values=0)
        A = A.toarray()
        D = A.sum(axis=1)
        D = np.diag(D)
        L = D - A
        D = self.__scale_diagonal(D=D)
        return D.dot(L.dot(D))

    def __feed_forward(self, X, y, y_Bag=None, current_batch=-1, total_batches=-1, transform=False,
                       snapshot_history=False, epoch=-1, file_name='reMap', rspath='.'):
        X = X.toarray()
        y = y.toarray()
        num_samples = X.shape[0]

        if self.binarize_input_feature:
            preprocessing.binarize(X, copy=False)
        if self.fit_intercept:
            X = np.concatenate((np.ones((num_samples, 1)), X), axis=1)

        if self.subsample_labels_size != self.num_labels:
            num_labels_example = np.sum(y, axis=0)
            weight_labels = 1 / num_labels_example
            weight_labels[weight_labels == np.inf] = 0.0
            weight_labels = weight_labels / np.sum(weight_labels)
            labels = np.unique(np.where(y == 1)[1])
            if labels.shape[0] > self.subsample_labels_size:
                labels = np.random.choice(labels, self.subsample_labels_size, replace=False,
                                          p=weight_labels[labels])
            labels = np.sort(labels)
        else:
            labels = np.arange(self.num_labels)

        ##TODO: delete below and uncomment the remaining
        # prob_bag = np.random.uniform(0, 1, size=(num_samples, self.num_bags)) + EPSILON
        prob_bag = np.zeros((num_samples, self.num_bags)) + EPSILON
        coef_intercept_label = self.coef_label[labels]
        if self.fit_intercept:
            coef_intercept_label = np.hstack((self.intercept_label[labels], coef_intercept_label))
        prob_label = np.mean(self.__sigmoid(np.dot(X, coef_intercept_label.T)), axis=0)
        bags = np.unique(np.nonzero(self.bags_labels[:, labels])[0])
        coef_intercept_bag = self.coef_bag[bags]
        if self.fit_intercept:
            coef_intercept_bag = np.hstack((self.intercept_bag[bags], coef_intercept_bag))
        for label_idx, label in enumerate(labels):
            c_bar = self.__extract_centroids(y=y, labels=[label], bags=bags)
            if self.fit_intercept:
                c_bar = np.array([np.concatenate((np.ones((c_bar.shape[1], 1)), c_bar[n, :]), axis=1) for n in
                                  np.arange(num_samples)])
            coef = np.array([np.diagonal(np.dot(c_bar[idx], coef_intercept_bag.T)).T for idx in np.arange(num_samples)])
            tmp = np.array([self.__sigmoid(coef[n]) for n in np.arange(num_samples)])
            del coef, c_bar
            bags_idx = np.nonzero(self.bags_labels[:, label])[0]
            bags_label_idx = np.array([int(np.where(bags == b)[0]) for b in bags_idx])
            if len(bags_idx) > 0:
                prob_bag[:, bags_idx] += np.multiply(tmp[:, bags_label_idx], prob_label[label_idx])
        prob_bag = prob_bag / self.num_bags

        if y_Bag is not None:
            y_Bag = y_Bag.toarray()
        else:
            y_Bag = np.zeros((num_samples, self.num_bags))

        prob_bags2sample = np.zeros((num_samples, self.max_bags))
        bags2sample_idx = np.zeros((num_samples, self.max_bags), dtype=np.int)
        # history of probabilities
        H = np.zeros((self.max_sampling, num_samples, self.num_bags)) + EPSILON
        prob_prev_label = 0.0
        prob_curr_label = 0.0
        # any probabilities below some value will be truncated
        eta = (1 - self.pi - self.varomega)
        for idx in np.arange(num_samples):
            for i in np.arange(self.max_sampling):
                # perform sequential additive and calculate predictions
                # while selecting a sample do enumeration of bags in the
                # sequential treatment
                bags = np.random.permutation(self.num_bags)
                if bags.shape[0] > self.subsample_labels_size:
                    bags = np.random.choice(bags, self.subsample_labels_size, replace=False)
                bag_pairs = [(bags[idx], bags[idx + 1]) for idx, bag in enumerate(bags) if idx < len(bags) - 1]
                prev_labels = list()
                for prev_label, curr_label in bag_pairs:
                    z = self.bags_correlation[prev_label, curr_label]
                    if z < 0.0001:
                        continue
                    prev_labels.append(prev_label)
                    prob_prev_label = prob_bag[idx, prev_label]
                    prob_curr_label = prob_bag[idx, curr_label]
                    H[i, idx, curr_label] = z * self.__sigmoid(np.multiply(prob_prev_label, prob_curr_label))
            # compute threshold_closeness
            tmp_prob_bags = self.__threshold_closeness(H=H[:, idx, :])
            eta = np.percentile(tmp_prob_bags, eta * 100)
            tmp_prob_bags[tmp_prob_bags < eta] = EPSILON
            tmp_bags2sample_idx = np.argsort(-tmp_prob_bags)
            tmp_prob_bags = -np.sort(-tmp_prob_bags)
            tmp_bags2sample_idx = tmp_bags2sample_idx[:self.max_bags]
            tmp_prob_bags = tmp_prob_bags[:self.max_bags]
            tmp_prob_bags = tmp_prob_bags / np.sum(tmp_prob_bags)
            y_Bag[idx, tmp_bags2sample_idx] += 1
            y_Bag[y_Bag == 2] = +1
            total_bags = np.sum(np.multiply(y[idx], self.bags_labels), axis=1)[tmp_bags2sample_idx]
            arg_bags = np.argsort(total_bags)[::-1]
            not_optimized = self.max_bags - 1
            while self.min_bags < not_optimized:
                p = tmp_prob_bags[not_optimized]
                if total_bags[arg_bags[not_optimized]] == np.count_nonzero(y[idx]):
                    y_Bag[idx, tmp_bags2sample_idx[not_optimized]] = np.random.binomial(1, p=p)
                if np.sum(y_Bag[:, tmp_bags2sample_idx[not_optimized]]) > int(num_samples * self.min_negatives_ratio):
                    y_Bag[idx, tmp_bags2sample_idx[not_optimized]] = np.random.binomial(1, p=p)
                else:
                    y_Bag[idx, tmp_bags2sample_idx[not_optimized]] = 1
                not_optimized = not_optimized - 1
            prob_bags2sample[idx] = tmp_prob_bags
            bags2sample_idx[idx] = tmp_bags2sample_idx

        if snapshot_history and epoch > -1:
            save_data(data=H, file_name=file_name + '_' + str(current_batch) + '_' + str(epoch) + '.pkl',
                      save_path=rspath, mode="wb", print_tag=False)
        # delete unnecessary variables
        del H, prob_bag, prob_prev_label, prob_curr_label

        if transform:
            y_Bag[y_Bag == 0] = -1
        desc = '\t\t\t--> Computed {0:.4f}%...'.format((((current_batch + 1) / total_batches) * 100))
        print(desc, end="\r")
        return y_Bag, prob_bags2sample, bags2sample_idx

    def __batch_forward(self, X, y, y_Bag=None, transform=False, snapshot_history=False,
                        epoch=-1, file_name='reMap', rspath='.'):
        print('  \t\t>>>------------>>>------------>>>')
        print('  \t\t>> Forward step...')
        logger.info('\t\t>> Forward step...')
        list_batches = np.arange(start=0, stop=X.shape[0], step=self.batch)
        parallel = Parallel(n_jobs=self.num_jobs, verbose=max(0, self.verbose - 1))
        if transform:
            results = parallel(delayed(self.__feed_forward)(X[batch:batch + self.batch],
                                                            y[batch:batch + self.batch], None,
                                                            idx, len(list_batches),
                                                            transform, snapshot_history,
                                                            epoch, file_name, rspath)
                               for idx, batch in enumerate(list_batches))
        else:
            results = parallel(delayed(self.__feed_forward)(X[batch:batch + self.batch],
                                                            y[batch:batch + self.batch],
                                                            y_Bag[batch:batch + self.batch],
                                                            idx, len(list_batches),
                                                            transform, snapshot_history, epoch,
                                                            file_name, rspath)
                               for idx, batch in enumerate(list_batches))
        desc = '\t\t\t--> Computed {0:.4f}%...'.format(((len(list_batches) / len(list_batches)) * 100))
        logger.info(desc)
        print(desc)

        # merge result
        y_Bag, prob_bags2sample, bags2sample_idx = zip(*results)
        y_Bag = np.vstack(y_Bag)
        prob_bags2sample = np.vstack(prob_bags2sample)
        bags2sample_idx = np.vstack(bags2sample_idx)

        # store sufficient_stats in a dictionary
        sufficient_stats = {"y_Bag": lil_matrix(y_Bag), "prob_bags2sample": lil_matrix(prob_bags2sample),
                            "bags2sample_idx": lil_matrix(bags2sample_idx)}

        return sufficient_stats

    def __optimize_w(self, y, y_Bag, learning_rate):
        desc = '\t\t\t--> Optimizing {0}: {1:.2f}%...'.format("W", 100)
        logger.info(desc)
        print(desc, end="\r")

        lam_3 = self.lambdas[2]
        y = y.toarray()
        y_Bag = y_Bag.toarray()
        num_samples = y.shape[0]

        gradient = np.dot(np.dot(y_Bag.T, y_Bag), self.W)
        gradient -= np.dot(y_Bag.T, y)
        gradient = gradient * 2
        gradient = gradient / (self.num_bags * num_samples)

        # compute the R lam3 * D_W * W
        R = lam_3 * self.__grad_l21_norm(M=self.W)

        # average by the number of bags
        gradient = gradient + R

        # gradient of W = W_old - learning_type * gradient value of W
        tmp = self.W - learning_rate * gradient
        self.W = self.__check_bounds(tmp)

    def __optimize_u(self, learning_rate):
        lam_3 = self.lambdas[2]
        gradient = 0.0

        # compute Theta^path.T * Theta^path * U
        label_label_U = np.dot(self.coef_label.T, self.coef_label)
        label_label_U = np.dot(label_label_U, self.U)

        # compute theta^path * theta^bag.T * U
        for bag_idx in np.arange(self.num_bags):
            desc = '\t\t\t--> Optimizing {0}: {1:.2f}%...'.format("U", ((bag_idx + 1) / self.num_bags) * 100)
            print(desc, end="\r")
            labels = np.nonzero(self.bags_labels[bag_idx])[0]
            P = self.coef_label[np.array(labels)][np.newaxis, :, :]
            B = np.tile(self.coef_bag[bag_idx][np.newaxis, :], (len(labels), 1))
            B = B[:, np.newaxis, :]
            gradient += np.tensordot(P, B, axes=[[1, 0], [0, 1]])

        gradient = (2 * gradient) / np.count_nonzero(self.bags_labels)
        gradient = label_label_U - gradient

        # compute the R lam3 * D_U * U
        R = lam_3 * self.__grad_l21_norm(M=self.U)

        # average by the number of bags
        gradient = gradient / self.num_bags + R

        # gradient of U = U_old - learning_type * gradient value of U
        tmp = self.U - learning_rate * gradient
        self.U = self.__check_bounds(tmp)

    def __optimize_theta_bag(self, y, y_Bag, prob_bags2sample, learning_rate, batch_idx, total_progress):
        lam_2 = self.lambdas[1]
        num_samples = y.shape[0]
        y = y.toarray()
        y_Bag = y_Bag.toarray()
        count = batch_idx + 1

        # max v
        v = np.array([np.max(prob_bags2sample[i].toarray()) for i in np.arange(num_samples)])

        # compute log-loss
        R_1 = np.zeros((self.num_bags, self.bag_feature_size))
        coef_bag = np.zeros((self.num_bags, self.bag_feature_size))
        if self.fit_intercept:
            coef_bag = np.zeros((self.num_bags, self.bag_feature_size + 1))
        for bag_idx in np.arange(self.num_bags):
            count += 1
            desc = '\t\t\t--> Optimizing {0}: {1:.2f}%...'.format("Bag's parameters",
                                                                  (count / total_progress) * 100)
            print(desc, end="\r")

            labels = np.nonzero(self.bags_labels[bag_idx])[0]
            c_bar = self.__extract_centroids(y=y, labels=labels, bags=[bag_idx])
            coef_intercept_bag = self.coef_bag[bag_idx]
            if self.fit_intercept:
                coef_intercept_bag = np.hstack((self.intercept_bag[bag_idx], coef_intercept_bag))
                c_bar = np.concatenate((np.ones((num_samples, 1)), c_bar), axis=1)
            positive_indices = np.where(y_Bag[:, bag_idx] == 1)[0]
            negative_indices = np.where(y_Bag[:, bag_idx] == -1)[0]
            total_samples = np.sort(np.append(positive_indices, negative_indices))
            y_Bag_hat = y_Bag[total_samples, :]
            y_Bag_hat[y_Bag_hat == -1] = 0
            c_bar = c_bar[total_samples]
            cond = -(2 * y_Bag_hat[:, bag_idx] - 1)
            coef = np.dot(c_bar, coef_intercept_bag)
            coef = np.multiply(coef, cond)
            logit = 1 / (np.exp(-coef) + 1)
            coef = np.multiply(c_bar, cond[:, np.newaxis])
            coef = np.multiply(coef, logit[:, np.newaxis])
            coef = np.multiply(coef, v[total_samples][:, np.newaxis])
            coef_bag[bag_idx] = np.mean(coef, axis=0)
            del coef, logit, c_bar, coef_intercept_bag, cond

            # compute 2 * (- U^T * Theta^path + Theta^bag)
            if self.label_bag_sim:
                u_theta = -np.dot(self.coef_label[np.array(labels)], self.U)
                R_1[bag_idx] = np.sum(2 * (u_theta + self.coef_bag[bag_idx]), axis=0)
                R_1[bag_idx] = R_1[bag_idx] / len(labels)
                del u_theta

        # compute the constraint lam2 * D_Theta^bag * Theta^bag
        R_2 = lam_2 * self.__grad_l21_norm(M=self.coef_bag)

        # gradient of Theta^bag = Theta^bag_old + learning_type * gradient value of Theta^bag
        if self.fit_intercept:
            gradient = coef_bag[:, 1:] + R_1 + R_2
            self.intercept_bag = coef_bag[:, 0][:, np.newaxis]
        else:
            gradient = coef_bag + R_1 + R_2

        tmp = self.coef_bag - learning_rate * gradient
        self.coef_bag = self.__check_bounds(tmp)

    def __optimize_theta_label(self, X, y, prob_bags2sample, S, learning_rate, batch_idx, total_progress):
        lam_5 = self.lambdas[5]
        X = X.toarray()
        y = y.toarray()
        S = S.toarray()
        num_samples = X.shape[0]
        count = batch_idx + 1

        if self.binarize_input_feature:
            preprocessing.binarize(X, copy=False)

        # max v
        v = np.array([np.max(prob_bags2sample[i].toarray()) for i in np.arange(num_samples)])

        # compute log-loss
        R = np.zeros((self.num_labels, self.input_feature_size))
        coef_label = np.zeros((self.num_labels, self.input_feature_size))
        if self.fit_intercept:
            coef_label = np.zeros((self.num_labels, self.input_feature_size + 1))
            X = np.concatenate((np.ones((num_samples, 1)), X), axis=1)

        for label_idx in np.arange(self.num_labels):
            count += 1
            desc = '\t\t\t--> Optimizing {0}: {1:.2f}%...'.format("Label's parameters",
                                                                  (count / total_progress) * 100)
            print(desc, end="\r")
            bags = np.nonzero(self.bags_labels[:, label_idx])[0]
            labels = np.unique([l for bag_idx in bags for l in np.nonzero(self.bags_labels[bag_idx])[0]])
            if self.label_bag_sim:
                # compute the constraint -2 * U * Theta^bag
                R[label_idx] = -2 * np.dot(self.U, np.mean(self.coef_bag[bags], axis=0))
            # compute the constraint 1/bags (2/|B| * (Theta^path - Theta^path))
            R[label_idx] += np.mean(self.coef_label[labels] - self.coef_label[label_idx], axis=0) / len(bags)

            coef_intercept_label = self.coef_label[label_idx]
            if self.fit_intercept:
                coef_intercept_label = np.hstack((self.intercept_label[label_idx], coef_intercept_label))
            cond = -(2 * y[:, label_idx] - 1)
            coef = np.dot(X, coef_intercept_label)
            coef = np.multiply(coef, cond)
            logit = 1 / (np.exp(-coef) + 1)
            coef = np.multiply(X, cond[:, np.newaxis])
            coef = np.multiply(coef, logit[:, np.newaxis])
            coef = np.multiply(coef, v[:, np.newaxis])
            coef_label[label_idx] = np.mean(coef, axis=0)

        # compute the constraint 2 * U * U^T * Theta^path
        if self.label_bag_sim:
            R += 2 * np.dot(np.dot(self.U, self.U.T), self.coef_label.T).T

        # compute the constraint X^T * L * X * Theta^path
        if self.corr_input_sim:
            L = self.__normalize_laplacian(lil_matrix(S))
            R += np.dot(np.dot(np.dot(X[:, 1:].T, L), X[:, 1:]), self.coef_label.T).T
            del L

        # compute the constraint lam6 * D_Theta^path * Theta^path
        R += lam_5 * self.__grad_l21_norm(M=self.coef_label)

        # gradient of Theta^path = Theta^path_old + learning_type * gradient value of Theta^path
        if self.fit_intercept:
            gradient = coef_label[:, 1:] + R
            self.intercept_label = coef_label[:, 0][:, np.newaxis]
        else:
            gradient = coef_label + R

        tmp = self.coef_label - learning_rate * gradient
        self.coef_label = self.__check_bounds(tmp)

    def __optimize_s(self, X, y, y_Bag, S, samples_idx, learning_rate, batch_idx, batch, total_progress):
        desc = '\t\t\t--> Optimizing {0}: {1:.2f}%...'.format("S", ((batch_idx + 1) / total_progress) * 100)
        if (batch_idx + 1) != total_progress:
            print(desc, end="\r")
        if (batch_idx + 1) == total_progress:
            print(desc)
            logger.info(desc)

        def __func_jac_s(lam_1, kappa, lam_4):
            gradient = lam_1 * np.dot(y_Bag, y_Bag.T)
            gradient += lam_4 * np.dot(y, y.T)
            gradient += np.dot(np.dot(np.dot(X, self.coef_label.T), self.coef_label), X.T)
            gradient += kappa * 2 * (S - 1)
            return gradient

        lam_1 = self.lambdas[0]
        kappa = self.lambdas[3]
        lam_4 = self.lambdas[4]
        X = X.toarray()
        y = y.toarray()
        y_Bag = y_Bag.toarray()
        S = S.toarray()
        num_samples = X.shape[0]

        gradient = __func_jac_s(lam_1, kappa, lam_4)
        S = S - learning_rate * gradient
        S = S / np.sum(S, axis=1)
        S[S < 0] = 0
        np.fill_diagonal(S, 0)
        i_lower = np.tril_indices(num_samples, -1)
        S[i_lower] = S.T[i_lower]
        self.S[samples_idx[:, None], samples_idx] = lil_matrix(S)

    def __optimize_d(self, y, y_Bag, prob_bags2sample, S, old_cost, new_cost, old_loss_bag, new_loss_bag, epoch,
                     learning_rate, batch_idx, total_progress):
        desc = '\t\t\t--> Optimizing {0}: {1:.2f}%...'.format("D", ((batch_idx + 1) / total_progress) * 100)
        if (batch_idx + 1) != total_progress:
            print(desc, end="\r")
        if (batch_idx + 1) == total_progress:
            print(desc)
            logger.info(desc)

        y = y.toarray()
        y_Bag = y_Bag.toarray()
        relative_change = np.abs((new_cost - old_cost) / old_cost)

        if not self.check_loss_diff:
            lam_1 = self.lambdas[0]
            lam_3 = self.lambdas[2]
            num_samples = y.shape[0]
            L = self.__normalize_laplacian(lil_matrix(S))

            for bag_idx in np.arange(self.num_bags):
                positive_indices = np.where(y_Bag[:, bag_idx] == +1)[0]
                negative_indices = np.where(y_Bag[:, bag_idx] == -1)[0]
                total_samples = np.sort(np.append(positive_indices, negative_indices))
                if len(total_samples) == 0:
                    continue
                labels = np.nonzero(self.bags_labels[bag_idx])[0]
                c_bar = self.__extract_centroids(y=y, labels=labels, bags=[bag_idx])

                coef_intercept_bag = self.coef_bag[bag_idx]
                if self.fit_intercept:
                    coef_intercept_bag = np.hstack((self.intercept_bag[bag_idx], coef_intercept_bag))
                    c_bar = np.concatenate((np.ones((num_samples, 1)), c_bar), axis=1)
                y_Bag_hat = y_Bag[total_samples, :]
                y_Bag_hat[y_Bag_hat == -1] = 0
                c_bar = c_bar[total_samples]
                cond = (2 * y_Bag_hat[:, bag_idx] - 1)
                c_bar_coef = np.dot(c_bar, coef_intercept_bag)
                coef = np.multiply(c_bar_coef, cond)
                logit = 1 / (np.exp(-coef) + 1)

                # max v
                v = np.array([np.max(prob_bags2sample[i].toarray()) for i in np.arange(num_samples)])
                coef = np.multiply(-c_bar_coef, logit[:, np.newaxis])
                coef = np.multiply(coef, v[total_samples][:, np.newaxis])
                tmp = learning_rate * np.mean(coef, axis=0)
                y_Bag[total_samples, bag_idx] = y_Bag[total_samples, bag_idx] - tmp
                del coef, logit, c_bar, coef_intercept_bag

            gradient = lam_1 * np.dot(L, y_Bag)
            gradient += 2 * np.dot(np.dot(y_Bag, self.W), self.W.T)
            gradient -= 2 * lam_3 * np.dot(y, self.W.T)
            gradient = gradient / (num_samples * self.num_bags)
            y_Bag = y_Bag - learning_rate * gradient
            y_Bag[y_Bag >= 0.9] = +1
            y_Bag[y_Bag <= 0] = -1
            if relative_change < self.loss_threshold or epoch == self.num_epochs:
                avg = np.mean(y_Bag[np.abs(y_Bag) < 1])
                np.nan_to_num(avg, copy=False)
                y_Bag[y_Bag >= avg] = +1
                y_Bag[y_Bag < avg] = -1
            else:
                y_Bag[np.abs(y_Bag) < 0.9] = 0
        else:
            eta = (1 - self.pi - self.varomega) * self.varrho
            loss_diff = new_loss_bag - old_loss_bag
            rows, cols = np.where(np.abs(loss_diff) >= self.loss_threshold)
            y_Bag[rows, cols] = y_Bag[rows, cols] * -1
            rows, cols = np.where(y_Bag == 0)
            tmp = loss_diff[rows, cols]
            if tmp.size > 0:
                perc = np.percentile(tmp, eta)
                if relative_change < self.loss_threshold or epoch == self.num_epochs:
                    tmp[tmp < perc] = +1
                    tmp[tmp >= perc] = -1
                else:
                    num_zeros = np.sum(tmp >= perc)
                    tmp[tmp >= perc] = np.random.choice(a=[+1, -1], size=num_zeros)
                    tmp[tmp < perc] = 0
            y_Bag[rows, cols] = tmp
        return y_Bag

    def __batch_optimize_d(self, y, y_Bag, prob_bags2sample, S, old_cost, new_cost, old_loss_bag, new_loss_bag, epoch,
                           learning_rate):
        desc = '  \t\t>> Optimizing bags choice...'
        print(desc)
        logger.info(desc)
        list_batches = np.arange(start=0, stop=y.shape[0], step=self.batch)
        parallel = Parallel(n_jobs=self.num_jobs, verbose=max(0, self.verbose - 1))
        results = parallel(delayed(self.__optimize_d)(y[batch:batch + self.batch],
                                                      y_Bag[batch:batch + self.batch],
                                                      prob_bags2sample[batch:batch + self.batch],
                                                      S[batch:batch + self.batch, batch:batch + self.batch],
                                                      old_cost, new_cost,
                                                      old_loss_bag[batch:batch + self.batch],
                                                      new_loss_bag[batch:batch + self.batch],
                                                      epoch, learning_rate, batch_idx, len(list_batches))
                           for batch_idx, batch in enumerate(list_batches))
        y_Bag = np.vstack(results)
        return lil_matrix(y_Bag)

    def __batch_backward(self, X, y, y_Bag, prob_bags2sample, samples_idx, learning_rate):
        print('  \t\t<<<------------<<<------------<<<')
        print('  \t\t>> Backward step...')
        logger.info('\t\t>> Backward step...')

        X = X[samples_idx, :]
        y = y[samples_idx, :]
        S = self.S[samples_idx[:, None], samples_idx]
        parallel = Parallel(n_jobs=self.num_jobs, verbose=max(0, self.verbose - 1))
        list_batches = np.arange(start=0, stop=X.shape[0], step=self.batch)

        # optimize W
        self.__optimize_w(y=y, y_Bag=y_Bag, learning_rate=learning_rate)

        # optimize U
        if self.label_bag_sim:
            self.__optimize_u(learning_rate=learning_rate)

        # optimize Theta^bag
        total_progress = len(list_batches) * self.num_bags
        parallel(delayed(self.__optimize_theta_bag)(y[batch:batch + self.batch],
                                                    y_Bag[batch:batch + self.batch],
                                                    prob_bags2sample[batch:batch + self.batch],
                                                    learning_rate, batch_idx, total_progress)
                 for batch_idx, batch in enumerate(list_batches))

        # optimize Theta^path
        total_progress = len(list_batches) * self.num_labels
        parallel(delayed(self.__optimize_theta_label)(X[batch:batch + self.batch],
                                                      y[batch:batch + self.batch],
                                                      prob_bags2sample[batch:batch + self.batch],
                                                      S[batch:batch + self.batch, batch:batch + self.batch],
                                                      learning_rate, batch_idx, total_progress)
                 for batch_idx, batch in enumerate(list_batches))

        # optimize S
        parallel(delayed(self.__optimize_s)(X[batch:batch + self.batch],
                                            y[batch:batch + self.batch],
                                            y_Bag[batch:batch + self.batch],
                                            S[batch:batch + self.batch, batch:batch + self.batch],
                                            samples_idx[batch:batch + self.batch],
                                            learning_rate, batch_idx, batch, len(list_batches))
                 for batch_idx, batch in enumerate(list_batches))

    def __cost_rec_error(self, y, y_Bag, bag_idx):
        desc = '\t\t\t--> Calculate reconstruction error cost: {0:.2f}%...'.format(
            ((bag_idx + 1) / self.num_bags) * 100)
        if (bag_idx + 1) != self.num_bags:
            print(desc, end="\r")
        if (bag_idx + 1) == self.num_bags:
            print(desc)
            logger.info(desc)

        lam_3 = self.lambdas[2]
        num_samples = y.shape[0]
        B = np.copy(y_Bag[:, bag_idx])
        B[B < 1] = 0

        # ||y - BW^bag||_2^2
        tmp = np.multiply(y, self.bags_labels[bag_idx])
        rec_error = np.dot(B[:, np.newaxis], self.W[bag_idx][np.newaxis])
        rec_error = np.linalg.norm(tmp - rec_error) ** 2
        rec_error = rec_error / num_samples
        # ||W^bag||_{2,1}
        rec_error += lam_3 * self.__norm_l21(M=self.W)

        return rec_error

    def __cost_bag(self, y, y_Bag, bag_idx, v, loss_bag):
        desc = '\t\t\t--> Calculate bag-label cost: {0:.2f}%...'.format(((bag_idx + 1) / self.num_bags) * 100)
        if (bag_idx + 1) != self.num_bags:
            print(desc, end="\r")
        if (bag_idx + 1) == self.num_bags:
            print(desc)
            logger.info(desc)

        lam_2 = self.lambdas[1]
        num_samples = y.shape[0]
        labels = np.nonzero(self.bags_labels[bag_idx])[0]
        c_bar = self.__extract_centroids(y=y, labels=labels, bags=[bag_idx])

        coef_intercept_bag = self.coef_bag[bag_idx]
        if self.fit_intercept:
            coef_intercept_bag = np.hstack((self.intercept_bag[bag_idx], coef_intercept_bag))
            c_bar = np.concatenate((np.ones((num_samples, 1)), c_bar), axis=1)
        coef = np.dot(c_bar, coef_intercept_bag)
        positive_indices = np.where(y_Bag[:, bag_idx] == 1)[0]
        negative_indices = np.where(y_Bag[:, bag_idx] == -1)[0]
        unlabeled_indices = np.where(y_Bag[:, bag_idx] == 0)[0]
        eta = (1 - self.pi - self.varomega) * self.varrho
        for idx in [0, +1, -1]:
            if idx == 1:
                if positive_indices.shape[0] != 0:
                    p_loss_1 = self.pi * -np.mean(self.__log_logistic(-coef[positive_indices]))
                    p_loss_2 = self.pi * -np.array(self.__log_logistic(coef[positive_indices]))
                    prob_p = np.array(self.__sigmoid(coef[positive_indices]))
                    prob_p[prob_p > eta] = 1
                    p_loss_2 = np.multiply(p_loss_2, (1 - prob_p) / prob_p)
                    p_loss_2 = np.mean(p_loss_2)
                    p_loss = p_loss_1 + p_loss_2
                    loss_bag[positive_indices, bag_idx] = p_loss
                    del prob_p
            elif idx == -1:
                if negative_indices.shape[0] != 0:
                    n_loss_1 = self.varomega * -np.mean(self.__log_logistic(coef[negative_indices]))
                    prob_n = np.array(self.__sigmoid(coef[negative_indices]))
                    prob_n[prob_n > eta] = 1
                    n_loss_2 = np.multiply(n_loss_1, (1 - prob_n) / prob_n)
                    n_loss_2 = np.mean(n_loss_2)
                    n_loss = n_loss_1 + n_loss_2
                    loss_bag[negative_indices, bag_idx] = n_loss
                    del prob_n
            else:
                if unlabeled_indices.shape[0] != 0:
                    u_loss = -np.array(self.__log_logistic(coef[unlabeled_indices]))
                    prob_u = np.array(self.__sigmoid(coef[unlabeled_indices]))
                    prob_u[prob_u <= eta] = 1
                    u_loss = np.multiply(u_loss, 1 - prob_u)
                    u_loss = np.mean(u_loss)
                    loss_bag[unlabeled_indices, bag_idx] = u_loss
                    del prob_u

        # cost log-bag
        cost_bag_path = np.mean(np.multiply(loss_bag[:, bag_idx], v))

        if self.calc_total_cost:
            # ||U^T * Theta^path - Theta^bag||_2^2
            if self.label_bag_sim:
                tmp = np.dot(self.coef_label[np.array(labels)], self.U)
                tmp = np.linalg.norm(tmp - coef_intercept_bag[1:]) ** 2
                cost_bag_path += tmp

            # ||Theta^bag||_2^2
            cost_bag_path += lam_2 * self.__norm_l21(M=self.coef_bag[bag_idx])

            # cost ||Theta^path_q - Theta^path_k||_2^2
            if self.label_closeness_sim:
                cost_bag_path += np.trace(np.dot(self.coef_label[labels], self.coef_label[labels].T))
        return cost_bag_path

    def __cost_label(self, X, y, s_cost_x, label_idx, v):
        desc = '\t\t\t--> Calculate label cost: {0:.2f}%...'.format(((label_idx + 1) / self.num_labels) * 100)
        if (label_idx + 1) != self.num_labels:
            print(desc, end="\r")
        if (label_idx + 1) == self.num_labels:
            print(desc)
            logger.info(desc)

        lam_5 = self.lambdas[5]
        coef_intercept_label = self.coef_label[label_idx]
        if self.fit_intercept:
            coef_intercept_label = np.hstack((self.intercept_label[label_idx], coef_intercept_label))
        cond = -(2 * y[:, label_idx] - 1)
        coef = np.dot(X, coef_intercept_label)
        coef = np.multiply(coef, cond)
        nll = -np.mean(self.__log_logistic(coef))

        # cost log-path
        cost_label = nll * v

        if self.calc_total_cost:
            # cost 1/2 * S_q,k ||Theta^path X_q - Theta^path X_k||_2^2
            if self.corr_input_sim:
                cost_label += s_cost_x[label_idx]
            # ||Theta^path||_2^2
            cost_label += lam_5 * self.__norm_l21(M=self.coef_label[label_idx])
        return cost_label

    def __total_cost(self, X, y, y_Bag, S, prob_bags2sample):
        print('  \t\t>> Compute cost...')
        logger.info('\t\t>> Compute cost...')

        # hyper-parameters
        lam_1 = self.lambdas[0]
        lam_3 = self.lambdas[2]
        kappa = self.lambdas[3]
        lam_4 = self.lambdas[4]
        s_cost = 0.0
        s_cost_x = 0.0
        s_cost_y = 0.0
        s_cost_bag = 0.0
        u_cost = 0.0
        w_cost = 0.0
        cost_label = 0.0
        cost_bag = 0.0

        # properties of dataset
        num_samples = X.shape[0]
        X = X.toarray()
        y = y.toarray()
        y_Bag = y_Bag.toarray()
        parallel = Parallel(n_jobs=self.num_jobs, verbose=max(0, self.verbose - 1))
        loss_bag = np.zeros((num_samples, self.num_bags))

        if self.fit_intercept:
            X = np.concatenate((np.ones((num_samples, 1)), X), axis=1)

        # mean v
        v = np.array([np.max(prob_bags2sample[i].toarray()) for i in np.arange(num_samples)])

        if self.corr_bag_sim or self.corr_label_sim or self.corr_input_sim:
            L = self.__normalize_laplacian(S)

        if self.calc_total_cost:
            # cost S
            if self.corr_bag_sim or self.corr_label_sim or self.corr_input_sim:
                S = S.toarray()
                s_cost = np.dot(S, np.ones((S.shape[0], 1)))
                s_cost = np.sum(s_cost - 1, axis=1)
                s_cost = kappa * np.linalg.norm(s_cost)

            # cost (lambda_1 / 2) * S_q,k ||B_q - B_k||_2^2
            if self.corr_bag_sim:
                s_cost_bag += lam_1 * np.trace(np.dot(np.dot(y_Bag.T, L), y_Bag))

        if self.calc_bag_cost:
            results = parallel(delayed(self.__cost_bag)(y, y_Bag, bag_idx, v, loss_bag)
                               for bag_idx in np.arange(self.num_bags))
            cost_bag += np.mean(results)
            del results
            # cost U
            if self.calc_total_cost and self.label_bag_sim:
                u_cost = lam_3 * self.__norm_l21(M=self.U)

        if self.calc_label_cost:
            if self.calc_total_cost:
                # cost (lambda_4 / 2) * S_q,k ||y_q - y_k||_2^2
                if self.corr_label_sim:
                    s_cost_y = lam_4 * np.trace(np.dot(np.dot(y.T, L), y))

                # cost 1/2 * S_q,k ||Theta^path X_q - Theta^path X_k||_2^2
                if self.corr_input_sim:
                    if self.fit_intercept:
                        s_cost_x = np.dot(X[:, 1:], self.coef_label.T)
                    else:
                        s_cost_x = np.dot(X, self.coef_label.T)
                    s_cost_x = np.diag(np.dot(np.dot(s_cost_x.T, L), s_cost_x))
            results = parallel(delayed(self.__cost_label)(X, y, s_cost_x, label_idx, v)
                               for label_idx in np.arange(self.num_labels))
            cost_label += np.mean(results)
            del results

        if self.calc_total_cost:
            results = parallel(delayed(self.__cost_rec_error)(y, y_Bag, bag_idx)
                               for bag_idx in np.arange(self.num_bags))
            w_cost = np.mean(results)
            del results

        total_cost_ = cost_bag + cost_label + s_cost_bag + u_cost + s_cost + s_cost_y + w_cost + EPSILON
        return total_cost_, loss_bag

    def fit(self, X, y, y_Bag, bags_labels, bags_correlation, label_features, centroids, model_name='reMap',
            model_path="../../model", result_path=".", snapshot_history=False, display_params: bool = True):

        if X is None:
            raise Exception("Please provide a dataset.")
        if y is None:
            raise Exception("Please provide labels for the dataset.")
        if y_Bag is None:
            raise Exception("Please provide bags_labels for the dataset.")
        if bags_labels is None:
            raise Exception("Labels for each bag must be included.")
        if bags_correlation is None:
            raise Exception("Correlation among bags must be included.")
        if label_features is None:
            raise Exception("Features for each label must be included.")
        if centroids is None:
            raise Exception("Bags' centroids must be included.")

        assert X.shape[0] == y.shape[0]
        assert X.shape[0] == y_Bag.shape[0]
        assert y.shape[0] == y_Bag.shape[0]

        # collect properties from data
        num_samples = X.shape[0]
        self.num_labels = bags_labels.shape[1]
        self.num_bags = y_Bag.shape[1]
        self.input_feature_size = X.shape[1]
        self.bag_feature_size = centroids.shape[1]
        self.bags_labels = bags_labels
        self.bags_correlation = bags_correlation
        self.label_features = label_features
        self.centroids = centroids

        if display_params:
            self.__print_arguments()
            time.sleep(2)

        cost_file_name = model_name + "_cost.txt"
        save_data('', file_name=cost_file_name, save_path=result_path, mode='w', w_string=True, print_tag=False)

        self.__init_variables(num_samples=num_samples)

        print('\t>> Training by reMap model...')
        logger.info('\t>> Training by reMap model...')
        n_epochs = self.num_epochs + 1

        if self.learning_type == "optimal":
            optimal_init = self.__optimal_learning_rate(alpha=self.lr)

        optimum_cost = np.inf
        timeref = time.time()

        for epoch in np.arange(start=1, stop=n_epochs):
            desc = '\t   {0:d})- Epoch count ({0:d}/{1:d})...'.format(epoch, n_epochs - 1)
            print(desc)
            logger.info(desc)

            if self.learning_type == "optimal":
                # usual optimization technique
                learning_rate = 1.0 / (self.lr * (optimal_init + epoch - 1))
            else:
                # using variational inference sgd
                learning_rate = np.power((epoch + self.delay_factor), -self.forgetting_rate)

            # shuffle dataset
            idx = self.__shffule(num_samples=num_samples)
            X = X[idx, :]
            y = y[idx, :]
            y_Bag = y_Bag[idx, :]

            # set epoch time
            start_epoch = time.time()

            # forward pass
            size_x = int(np.ceil(X.shape[0] * self.subsample_input_size))
            samples_idx = np.random.choice(np.arange(num_samples), size_x, replace=False)
            sstats = self.__batch_forward(X=X[samples_idx, :], y=y[samples_idx, :],
                                          y_Bag=y_Bag[samples_idx, :],
                                          snapshot_history=snapshot_history,
                                          epoch=epoch,
                                          file_name=model_name,
                                          rspath=result_path)

            # pick a subsample to compute loss
            ss_cost = samples_idx
            tmp = np.arange(len(ss_cost))
            if self.cost_subsample_size < len(ss_cost):
                tmp = np.random.choice(tmp, self.cost_subsample_size, replace=False)
                ss_cost = ss_cost[tmp]
            prob_bags2sample = sstats["prob_bags2sample"][tmp, :]
            sstats_y_Bag = sstats["y_Bag"][tmp, :]

            S = None
            if self.corr_bag_sim or self.corr_label_sim or self.corr_input_sim:
                S = self.S[ss_cost[:, None], ss_cost]

            # compute loss
            if epoch == 1:
                old_cost, old_loss_bag = self.__total_cost(X=X[ss_cost, :], y=y[ss_cost, :], y_Bag=sstats_y_Bag,
                                                           S=S, prob_bags2sample=prob_bags2sample)

            # backward pass
            self.__batch_backward(X=X, y=y, y_Bag=sstats_y_Bag, prob_bags2sample=prob_bags2sample,
                                  samples_idx=ss_cost, learning_rate=learning_rate)

            # optimize D
            new_cost, new_loss_bag = self.__total_cost(X=X[ss_cost, :], y=y[ss_cost, :], y_Bag=sstats_y_Bag, S=S,
                                                       prob_bags2sample=prob_bags2sample)
            tmp = self.__batch_optimize_d(y=y[ss_cost, :], y_Bag=y_Bag[ss_cost, :], prob_bags2sample=prob_bags2sample,
                                          S=S, old_cost=old_cost, new_cost=new_cost, old_loss_bag=old_loss_bag,
                                          new_loss_bag=new_loss_bag, epoch=epoch, learning_rate=learning_rate)
            print('\t\t  --> New cost: {0:.4f}; Old cost: {1:.4f}'.format(new_cost, old_cost))
            logger.info('\t\t  --> New cost: {0:.4f}; Old cost: {1:.4f}'.format(new_cost, old_cost))
            y_Bag[ss_cost, :] = tmp
            old_loss_bag = new_loss_bag
            old_cost = new_cost
            end_epoch = time.time()
            self.is_fit = True

            print('\t\t  ## Epoch {0} took {1} seconds...'.format(epoch, round(end_epoch - start_epoch, 3)))
            logger.info('\t\t  ## Epoch {0} took {1} seconds...'.format(epoch, round(end_epoch - start_epoch, 3)))
            data = str(epoch) + '\t' + str(round(end_epoch - start_epoch, 3)) + '\t' + str(new_cost) + '\n'
            save_data(data=data, file_name=cost_file_name, save_path=result_path, mode='a', w_string=True,
                      print_tag=False)
            # Save models parameters based on test frequencies
            if (epoch % self.display_interval) == 0 or epoch == 1 or epoch == n_epochs - 1:
                if optimum_cost >= new_cost or epoch == n_epochs - 1:
                    optimum_cost = new_cost
                    W_name = model_name + '_W.pkl'
                    U_name = model_name + '_U.pkl'
                    S_name = model_name + '_S.pkl'
                    model_file_name = model_name + '.pkl'
                    if epoch == n_epochs - 1:
                        W_name = model_name + '_W_final.pkl'
                        U_name = model_name + '_U_final.pkl'
                        S_name = model_name + '_S_final.pkl'
                        model_file_name = model_name + '_final.pkl'

                    print('\t\t  --> Storing the reMap\'s W parameters to: {0:s}'.format(W_name))
                    logger.info('\t\t  --> Storing the reMap\'s W parameters to: {0:s}'.format(W_name))
                    save_data(data=lil_matrix(self.W), file_name=W_name, save_path=model_path, mode="wb",
                              print_tag=False)
                    self.W = None

                    if self.label_bag_sim:
                        print('\t\t  --> Storing the reMap\'s U parameters to: {0:s}'.format(U_name))
                        logger.info('\t\t  --> Storing the reMap\'s U parameters to: {0:s}'.format(U_name))
                        save_data(data=lil_matrix(self.U), file_name=U_name, save_path=model_path, mode="wb",
                                  print_tag=False)
                        self.U = None

                    if self.corr_bag_sim or self.corr_label_sim or self.corr_input_sim:
                        print('\t\t  --> Storing the reMap\'s S parameters to: {0:s}'.format(S_name))
                        logger.info('\t\t  --> Storing the reMap\'s S parameters to: {0:s}'.format(S_name))
                        save_data(data=lil_matrix(self.S), file_name=S_name, save_path=model_path, mode="wb",
                                  print_tag=False)
                        self.S = None

                    print('\t\t  --> Storing the reMap model to: {0:s}'.format(model_file_name))
                    logger.info('\t\t  --> Storing the reMap model to: {0:s}'.format(model_file_name))
                    self.bags_labels = None
                    self.bags_correlation = None
                    self.label_features = None
                    self.centroids = None
                    save_data(data=copy.copy(self), file_name=model_file_name, save_path=model_path, mode="wb",
                              print_tag=False)
                    self.bags_labels = bags_labels
                    self.bags_correlation = bags_correlation
                    self.label_features = label_features
                    self.centroids = centroids

                    if epoch != n_epochs - 1:
                        self.W = load_data(file_name=W_name, load_path=model_path, tag="reMap\'s W parameters")
                        self.W = self.W.toarray()
                        if self.label_bag_sim:
                            self.U = load_data(file_name=U_name, load_path=model_path, tag="reMap\'s U parameters")
                            self.U = self.U.toarray()
                        if self.corr_bag_sim or self.corr_label_sim or self.corr_input_sim:
                            self.S = load_data(file_name=S_name, load_path=model_path, tag="reMap\'s S parameters")

                    if self.early_stop:
                        relative_change = np.abs((new_cost - old_cost) / old_cost)
                        desc = '\t\t  --> There is a little improvement in the cost '
                        desc += '(< {0}) for epoch {1}, hence, training is terminated...'.format(self.loss_threshold,
                                                                                                 epoch)
                        if relative_change < self.loss_threshold:
                            print(desc)
                            logger.info(desc)
                            break
        print('\t  --> Training consumed %.2f mintues' % (round((time.time() - timeref) / 60., 3)))
        logger.info('\t  --> Training consumed %.2f mintues' % (round((time.time() - timeref) / 60., 3)))

    def __check_validity(self, X, bags_correlation, bags_labels, batch_size, centroids, decision_threshold,
                         label_features, max_sampling, num_jobs, subsample_labels_size, y):
        if X is None:
            raise Exception("Please provide a dataset.")
        if y is None:
            raise Exception("Please provide labels for the dataset.")
        if bags_labels is None:
            raise Exception("Labels for each bag must be included.")
        if bags_correlation is None:
            raise Exception("Correlation among bags must be included.")
        if label_features is None:
            raise Exception("Features for each label must be included.")
        if centroids is None:
            raise Exception("Bags' centroids must be included.")
        if not self.is_fit:
            raise Exception("This instance is not fitted yet. Call 'fit' with "
                            "appropriate arguments before using this method.")
        # validate inputs
        if decision_threshold <= 0:
            self.decision_threshold = 0.5
        else:
            self.decision_threshold = decision_threshold
        if batch_size <= 0:
            self.batch = 30
        else:
            self.batch = batch_size
        if num_jobs <= 0:
            self.num_jobs = 1
        else:
            self.num_jobs = num_jobs
        if subsample_labels_size <= 0:
            self.subsample_labels_size = self.num_labels
        else:
            self.subsample_labels_size = subsample_labels_size
        if max_sampling <= 0:
            self.max_sampling = 3
        else:
            self.max_sampling = max_sampling
        # extract various properties
        self.bags_labels = bags_labels
        self.bags_correlation = bags_correlation
        self.label_features = label_features
        self.centroids = centroids

    def transform(self, X, y, bags_labels, bags_correlation, label_features, centroids,
                  subsample_labels_size=50, max_sampling=3, snapshot_history=False,
                  decision_threshold=0.5, batch_size=30, num_jobs=1, file_name='reMap',
                  result_path="."):
        assert X.shape[0] == y.shape[0]
        self.__check_validity(X, bags_correlation, bags_labels, batch_size, centroids, decision_threshold,
                              label_features, max_sampling, num_jobs, subsample_labels_size, y)

        sstats = self.__batch_forward(X=X, y=y, y_Bag=None, transform=True,
                                      snapshot_history=snapshot_history,
                                      file_name=file_name, rspath=result_path)
        return sstats["y_Bag"]

    def predictive_distribution(self, X, y, bags_labels, bags_correlation, label_features, centroids,
                                subsample_labels_size=50, max_sampling=3, snapshot_history=False,
                                decision_threshold=0.5, batch_size=30, num_jobs=1, file_name='reMap',
                                result_path="."):
        assert X.shape[0] == y.shape[0]
        self.__check_validity(X, bags_correlation, bags_labels, batch_size, centroids, decision_threshold,
                              label_features, max_sampling, num_jobs, subsample_labels_size, y)

        sstats = self.__batch_forward(X=X, y=y, y_Bag=None, transform=True,
                                      snapshot_history=snapshot_history,
                                      file_name=file_name, rspath=result_path)
        prob_bags2sample = sstats["prob_bags2sample"]
        return prob_bags2sample.mean()

    def score_and_predictive_distribution(self, X, y, bags_labels, bags_correlation, label_features, centroids,
                                          subsample_labels_size=50, max_sampling=3, snapshot_history=False,
                                          decision_threshold=0.5, batch_size=30, num_jobs=1, file_name='reMap',
                                          result_path="."):
        assert X.shape[0] == y.shape[0]
        self.__check_validity(X, bags_correlation, bags_labels, batch_size, centroids, decision_threshold,
                              label_features, max_sampling, num_jobs, subsample_labels_size, y)

        ## batch forward
        sstats = self.__batch_forward(X=X, y=y, y_Bag=None, transform=True,
                                      snapshot_history=snapshot_history,
                                      file_name=file_name, rspath=result_path)
        prob_bags2sample = sstats["prob_bags2sample"]
        accuracy = 0.0
        for s_idx, item in enumerate(sstats["y_Bag"].toarray()):
            bags_idx = np.where(item == 1)[0]
            lst_labels = np.vstack([self.get_labels(bag) for bag in bags_idx])
            labels = np.nonzero(y[s_idx])[1]
            total_labels = [len([lbl for i in lst_labels if lbl in i]) for lbl in labels]
            accuracy += len(total_labels) / lst_labels.size
        accuracy /= y.shape[0]
        return prob_bags2sample.mean(), accuracy

    def get_labels(self, bag_idx):
        labels = np.nonzero(self.bags_labels[bag_idx])[0]
        return labels

    def get_bags(self, label_idx):
        if self.bags_labels is not None:
            bags = np.nonzero(self.bags_labels[:, label_idx])[0]
        else:
            bags = None
        return bags
