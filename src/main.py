__author__ = "Abdur Rahman M. A. Basher"
__date__ = '03/08/2020'
__copyright__ = "Copyright 2020, The Hallam Lab"
__license__ = "GPL v3"
__version__ = "1.0"
__maintainer__ = "Abdur Rahman M. A. Basher"
__email__ = "arbasher@alumni.ubc.ca"
__status__ = "Production"
__description__ = "This file is the main entry to perform relabeling a multi-label dataset using reMap."

import datetime
import json
import os
import textwrap
from argparse import ArgumentParser

import utility.file_path as fph
from train import train
from utility.arguments import Arguments


def __print_header():
    os.system('clear')
    print('# ' + '=' * 50)
    print('Author: ' + __author__)
    print('Copyright: ' + __copyright__)
    print('License: ' + __license__)
    print('Version: ' + __version__)
    print('Maintainer: ' + __maintainer__)
    print('Email: ' + __email__)
    print('Status: ' + __status__)
    print('Date: ' + datetime.datetime.strptime(__date__,
                                                "%d/%m/%Y").strftime("%d-%B-%Y"))
    print('Description: ' + textwrap.TextWrapper(width=45,
                                                 subsequent_indent='\t     ').fill(__description__))
    print('# ' + '=' * 50)


def save_args(args):
    os.makedirs(args.log_dir, exist_ok=args.exist_ok)
    with open(os.path.join(args.log_dir, 'config.json'), 'wt') as f:
        json.dump(vars(args), f, indent=2)


def __internal_args(parse_args):
    arg = Arguments()

    arg.display_interval = parse_args.display_interval
    if parse_args.display_interval < 0:
        arg.display_interval = 1
    arg.random_state = parse_args.random_state
    arg.num_jobs = parse_args.num_jobs

    ##########################################################################################################
    ##########                                  ARGUMENTS FOR PATHS                                 ##########
    ##########################################################################################################

    arg.dspath = parse_args.dspath
    arg.mdpath = parse_args.mdpath
    arg.rspath = parse_args.rspath
    arg.rsfolder = parse_args.rsfolder
    arg.logpath = parse_args.logpath

    ##########################################################################################################
    ##########                          ARGUMENTS FOR FILE NAMES AND MODELS                         ##########
    ##########################################################################################################

    arg.bag_sigma_name = parse_args.bag_sigma_name
    arg.bag_phi_name = parse_args.bag_phi_name
    arg.hin_name = parse_args.hin_name
    arg.features_name = parse_args.features_name
    arg.bag_centroid_name = parse_args.bag_centroid_name
    arg.rho_name = parse_args.rho_name
    arg.vocab_name = parse_args.vocab_name
    arg.X_name = parse_args.X_name
    arg.y_name = parse_args.y_name
    arg.yB_name = parse_args.yB_name
    arg.bags_labels = parse_args.bags_labels
    arg.file_name = parse_args.file_name
    arg.model_name = parse_args.model_name

    ##########################################################################################################
    ##########                            ARGUMENTS PREPROCESSING FILES                             ##########
    ##########################################################################################################

    arg.preprocess_dataset = parse_args.preprocess_dataset

    ##########################################################################################################
    ##########                              ARGUMENTS USED FOR TRAINING                             ##########
    ##########################################################################################################

    arg.train = parse_args.train
    arg.transform = parse_args.transform
    arg.top_k = parse_args.top_k
    arg.alpha = parse_args.alpha
    arg.v_cos = parse_args.v_cos
    arg.define_bags = parse_args.define_bags
    arg.recover_max_bags = parse_args.recover_max_bags
    arg.alpha = parse_args.alpha
    arg.binarize_input_feature = parse_args.binarize
    arg.fit_intercept = parse_args.fit_intercept
    arg.decision_threshold = parse_args.decision_threshold
    arg.snapshot_history = parse_args.snapshot_history
    arg.learning_type = parse_args.learning_type
    arg.lr = parse_args.lr
    arg.lr0 = parse_args.lr0
    arg.forgetting_rate = parse_args.fr
    arg.delay_factor = parse_args.delay
    arg.max_sampling = parse_args.max_sampling
    arg.ssample_label_size = parse_args.ssample_label_size
    arg.ssample_input_size = parse_args.ssample_input_size
    arg.calc_subsample_size = parse_args.calc_subsample_size
    arg.min_bags = parse_args.min_bags
    arg.max_bags = parse_args.max_bags
    arg.score_strategy = parse_args.score_strategy
    arg.loss_threshold = parse_args.loss_threshold
    arg.early_stop = parse_args.early_stop
    arg.calc_label_cost = parse_args.calc_label_cost
    arg.calc_bag_cost = parse_args.calc_bag_cost
    arg.calc_total_cost = parse_args.calc_total_cost
    arg.pi = parse_args.pi
    arg.varomega = parse_args.varomega
    arg.varrho = parse_args.varrho
    arg.random_allocation = parse_args.random_allocation
    arg.theta_bern = parse_args.theta_bern
    arg.min_negatives_ratio = parse_args.min_neg_ratio
    arg.lambdas = parse_args.lambdas
    arg.label_bag_sim = parse_args.label_bag_sim
    arg.label_closeness_sim = parse_args.label_closeness_sim
    arg.corr_bag_sim = parse_args.corr_bag_sim
    arg.corr_label_sim = parse_args.corr_label_sim
    arg.corr_input_sim = parse_args.corr_input_sim
    arg.batch = parse_args.batch
    arg.num_epochs = parse_args.num_epochs
    arg.shuffle = parse_args.shuffle

    return arg


def parse_command_line():
    __print_header()
    # Parses the arguments.
    parser = ArgumentParser(description="Run reMap.")

    parser.add_argument('--display-interval', default=-1, type=int,
                        help='display intervals. -1 means display per each iteration.')
    parser.add_argument('--random_state', default=12345,
                        type=int, help='Random seed. (default value: 12345).')
    parser.add_argument('--num-jobs', type=int, default=1,
                        help='Number of parallel workers. Default is 2.')
    parser.add_argument('--batch', type=int, default=30,
                        help='Batch size. (default value: 30).')
    parser.add_argument('--num-epochs', default=3, type=int,
                        help='Number of epochs over the training set. (default value: 3).')

    # Arguments for path
    parser.add_argument('--dspath', default=fph.DATASET_PATH, type=str,
                        help='The path to the dataset after the samples are processed. '
                             'The default is set to dataset folder outside the source code.')
    parser.add_argument('--mdpath', default=fph.MODEL_PATH, type=str,
                        help='The path to the output models. The default is set to '
                             'train folder outside the source code.')
    parser.add_argument('--rspath', default=fph.RESULT_PATH, type=str,
                        help='The path to the results. The default is set to result '
                             'folder outside the source code.')
    parser.add_argument('--rsfolder', default="Prediction_reMap", type=str,
                        help='The result folder name. The default is set to Prediction_reMap.')
    parser.add_argument('--logpath', default=fph.LOG_PATH, type=str,
                        help='The path to the log directory.')

    # Arguments for file names and models
    parser.add_argument('--file-name', type=str, default='biocyc',
                        help='The file name to save an object. (default value: "biocyc")')
    parser.add_argument('--hin-name', type=str, default='hin.pkl',
                        help='The name of the hin model file. (default value: "hin.pkl")')
    parser.add_argument('--features-name', type=str, default='features.npz',
                        help='The features file name. (default value: "features.npz")')
    parser.add_argument('--bag-centroid-name', type=str, default='bag_centroid.npz',
                        help='The bags centroids file name. (default value: "bag_centroid.npz")')
    parser.add_argument('--rho-name', type=str, default='rho.npz',
                        help='The rho file name. (default value: "rho.npz")')
    parser.add_argument('--bag-sigma-name', type=str, default='sigma.npz',
                        help='The file name for bags covariance. (default value: "sigma.npz")')
    parser.add_argument('--bag-phi-name', type=str, default='phi.npz',
                        help='The file name for labels distribution over bags. '
                             '(default value: "phi.npz")')
    parser.add_argument('--vocab-name', type=str, default='vocab.pkl',
                        help='The vocab file name. (default value: "vocab.pkl").')
    parser.add_argument('--X-name', type=str, default='biocyc_X.pkl',
                        help='The X file name. (default value: "biocyc_X.pkl")')
    parser.add_argument('--y-name', type=str, default='biocyc_y.pkl',
                        help='The y file name. (default value: "biocyc_y.pkl")')
    parser.add_argument('--yB-name', type=str, default='biocyc_B.pkl',
                        help='The bags file name. (default value: "biocyc_B.pkl")')
    parser.add_argument('--bags-labels', type=str, default='bag_pathway.pkl',
                        help='The file name for bags consisting of associated labels. '
                             '(default value: "bag_pathway.pkl")')
    parser.add_argument('--model-name', type=str, default='reMap',
                        help='The file name, excluding extension, to save '
                             'an object. (default value: "reMap")')

    # Arguments for preprocessing dataset
    parser.add_argument('--preprocess-dataset', action='store_true', default=False,
                        help='Preprocess biocyc collection by building bags_labels centroids and '
                             'define maximum expected number of bags. (default value: False).')
    parser.add_argument('--define-bags', action='store_true', default=False,
                        help='Whether to construct bags to labels centroids. (default value: False).')
    parser.add_argument('--recover-max-bags', action='store_true', default=False,
                        help='Whether to recover maximum number of bags. (default value: False).')
    parser.add_argument("--v-cos", type=float, default=0.2,
                        help="A cutoff threshold for consine similarity. (default value: 0.2).")

    # Arguments for training and evaluation
    parser.add_argument('--train', action='store_true', default=False,
                        help='Whether to train the reMap model. (default value: False).')
    parser.add_argument('--transform', action='store_true', default=False,
                        help='Whether to transform labels to bags from inputs using '
                             'a pretrained reMap model. (default value: False).')
    parser.add_argument('--top-k', type=int, default=250,
                        help='Top k labels to be considered for each bag. (default value: 250).')
    parser.add_argument("--alpha", type=float, default=16,
                        help="A hyper-parameter for controlling bags centroids. (default value: 16).")
    parser.add_argument('--binarize', action='store_false', default=True,
                        help='Whether binarize data (set feature values to 0 or 1). (default value: True).')
    parser.add_argument('--fit-intercept', action='store_false', default=True,
                        help='Whether the intercept should be estimated or not. (default value: True).')
    parser.add_argument("--decision-threshold", type=float, default=0.5,
                        help="The cutoff threshold for reMap. (default value: 0.5)")
    parser.add_argument('--snapshot-history', action='store_true', default=False,
                        help='Whether to have a snapshot of history probabilities. (default value: False).')
    parser.add_argument('--learning-type', default='optimal', type=str, choices=['optimal', 'sgd'],
                        help='The learning rate schedule. (default value: "optimal")')
    parser.add_argument('--lr', default=0.0001, type=float,
                        help='The learning rate. (default value: 0.0001).')
    parser.add_argument('--lr0', default=0.0, type=float,
                        help='The initial learning rate. (default value: 0.0).')
    parser.add_argument('--fr', type=float, default=0.9,
                        help='Forgetting rate to control how quickly old information is forgotten. The value should '
                             'be set between (0.5, 1.0] to guarantee asymptotic convergence. (default value: 0.7).')
    parser.add_argument('--delay', type=float, default=1.,
                        help='Delay factor down weights early iterations. (default value: 0.9).')
    parser.add_argument('--max-sampling', default=3, type=int,
                        help='Maximum number of random samplings. (default value: 3).')
    parser.add_argument('--ssample-input-size', default=0.05, type=float,
                        help='The size of input subsample. (default value: 0.05).')
    parser.add_argument('--ssample-label-size', default=50, type=int,
                        help='Maximum number of labels to be sampled. (default value: 50).')
    parser.add_argument('--calc-subsample-size', type=int, default=50,
                        help='Compute loss on selected samples. (default value: 50).')
    parser.add_argument('--min-bags', default=10, type=int,
                        help='Minimum number of bags for each sample. (default value: 10).')
    parser.add_argument('--max-bags', default=50, type=int,
                        help='Maximum number of bags for each sample. (default value: 50).')
    parser.add_argument('--score-strategy', action='store_false', default=True,
                        help='Whether to update bags based on score threshold strategy or loss estimator. '
                             '(default value: "score threshold strategy").')
    parser.add_argument("--loss-threshold", type=float, default=0.001,
                        help="A hyper-parameter for deciding the cutoff threshold of the differences "
                             "of loss between two consecutive rounds. (default value: 0.001).")
    parser.add_argument("--early-stop", action='store_true', default=False,
                        help="Whether to terminate training based on relative change "
                             "between two consecutive iterations. (default value: False).")
    parser.add_argument("--calc-label-cost", action='store_true', default=False,
                        help="Compute label cost, i.e., cost of labels. (default value: False).")
    parser.add_argument("--calc-bag-cost", action='store_false', default=True,
                        help="Compute bag cost, i.e., cost of bags. (default value: True).")
    parser.add_argument("--calc-total-cost", action='store_true', default=False,
                        help="Compute total cost, i.e., cost of bags plus cost of labels."
                             " (default value: False).")
    parser.add_argument("--pi", type=float, default=0.4,
                        help="A prior parameter for positive bags. (default value: 0.4).")
    parser.add_argument("--varomega", type=float, default=0.3,
                        help="A prior parameter for biased negative bags. (default value: 0.3).")
    parser.add_argument("--varrho", type=float, default=0.7,
                        help="A hyper-parameter for positive bags. (default value: 0.7).")
    parser.add_argument("--random-allocation", action='store_true', default=False,
                        help='Whether to apply randomized allocation for reMap. (default value: False).')
    parser.add_argument('--theta-bern', type=float, default=0.3,
                        help='The Bernoulli probability value for allocating bags randomly to either -1, or +1. (default value: 0.3).')
    parser.add_argument("--min-neg-ratio", type=float, default=0.3,
                        help="A hyper-parameter for creating a balanced positive/negative bags. (default value: 0.3).")
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                        help="Six hyper-parameters for constraints. (default value: [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]).")
    parser.add_argument('--label-bag-sim', action='store_false', default=True,
                        help='Whether to apply similarity constraint among labels within a bag. (default value: True).')
    parser.add_argument('--corr-bag-sim', action='store_false', default=True,
                        help='Whether to apply similarity constraint among bags. (default value: True).')
    parser.add_argument('--label-closeness-sim', action='store_false', default=True,
                        help='Whether to apply closeness constraint of a label to other labels of a bag. '
                             '(default value: True).')
    parser.add_argument('--corr-label-sim', action='store_false', default=True,
                        help='Whether to apply similarity constraint among labels. (default value: True).')
    parser.add_argument('--corr-input-sim', action='store_false', default=True,
                        help='Whether to apply similarity constraint among instances. (default value: True).')
    parser.add_argument('--shuffle', action='store_false', default=True,
                        help='Whether or not the training data should be shuffled after each epoch. '
                             '(default value: True).')

    parse_args = parser.parse_args()
    args = __internal_args(parse_args)

    train(arg=args)


if __name__ == "__main__":
    # app.run(parse_command_line)
    parse_command_line()
