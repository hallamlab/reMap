"""
This file is the main entry used to train the input dataset using reMap
"""

import numpy as np
import os
import sys
import time
import traceback
from model.remap import reMap
from scipy.sparse import lil_matrix
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_distances
from utility.access_file import load_data, save_data

EPSILON = np.finfo(np.float).eps


###***************************        Private Main Entry        ***************************###

def __train(arg):
    # Setup the number of operations to employ
    steps = 1
    # Whether to display parameters at every operation
    display_params = True

    ##########################################################################################################
    ######################                        PREPROCESSING                         ######################
    ##########################################################################################################

    if arg.define_bags:
        print("\n{0})- Construct bags_labels centroids...".format(steps))
        steps = steps + 1

        # load a hin file
        hin = load_data(file_name=arg.hin_name, load_path=arg.mdpath,
                        tag="heterogeneous information network")
        node2idx_path2vec = dict((node[0], node[1]["mapped_idx"]) for node in hin.nodes(data=True))
        # map pathways indices of vocab to path2vec pathways indices
        vocab = load_data(file_name=arg.vocab_name, load_path=arg.dspath, tag="vocabulary")
        idxvocab = np.array([idx for idx, v in vocab.items() if v in node2idx_path2vec])
        del hin

        # define pathways 2 bags_labels
        phi = np.load(file=os.path.join(arg.mdpath, arg.bag_phi_name))
        phi = phi[phi.files[0]]
        bags_labels = np.argsort(-phi)
        bags_labels = bags_labels[:, :arg.top_k]
        labels_distr_idx = np.array([[pathway for pathway in bag if pathway in idxvocab] for bag in bags_labels])
        bags_labels = preprocessing.MultiLabelBinarizer().fit_transform(labels_distr_idx)
        labels_distr_idx = [[list(idxvocab).index(label_idx) for label_idx in bag_idx] for bag_idx in
                            labels_distr_idx]

        # get trimmed phi distributions
        phi = -np.sort(-phi)
        phi = phi[:, :arg.top_k]

        # calculate correlation
        sigma = np.load(file=os.path.join(arg.mdpath, arg.bag_sigma_name))
        sigma = sigma[sigma.files[0]]
        sigma[sigma < 0] = EPSILON
        C = np.diag(np.sqrt(np.diag(sigma)))
        C_inv = np.linalg.inv(C)
        rho = np.dot(np.dot(C_inv, sigma), C_inv)
        min_rho = np.min(rho)
        max_rho = np.max(rho)
        rho = rho - min_rho
        rho = rho / (max_rho - min_rho)

        # extracting pathway features
        path2vec_features = np.load(file=os.path.join(arg.mdpath, arg.features_name))
        path2vec_features = path2vec_features[path2vec_features.files[0]]
        pathways_idx = np.array([node2idx_path2vec[v] for idx, v in vocab.items() if v in node2idx_path2vec])
        features = path2vec_features[pathways_idx, :]
        features = features / np.linalg.norm(features, axis=1)[:, np.newaxis]

        # get centroids of bags_labels
        C = np.dot(bags_labels, features) / np.sum(bags_labels, axis=1)[:, np.newaxis]
        C = arg.alpha * C

        # save files
        np.savez(os.path.join(arg.dspath, arg.file_name + "_exp_phi_trim.npz"), phi)
        np.savez(os.path.join(arg.dspath, arg.file_name + "_rho.npz"), rho)
        np.savez(os.path.join(arg.dspath, arg.file_name + "_features.npz"), features)
        np.savez(os.path.join(arg.dspath, arg.file_name + "_bag_centroid.npz"), C)
        save_data(data=bags_labels, file_name=arg.file_name + "_bag_pathway.pkl", save_path=arg.dspath,
                  tag="bags_labels with associated pathways", mode="wb")
        save_data(data=idxvocab, file_name=arg.file_name + "_idxvocab.pkl", save_path=arg.dspath,
                  tag="pathway ids to pathway features ids", mode="wb")
        save_data(data=labels_distr_idx, file_name=arg.file_name + "_labels_distr_idx.pkl", save_path=arg.dspath,
                  tag="bags labels batch_idx with associated pathways", mode="wb")
        print("\t>> Done...")

    if arg.recover_max_bags:
        print("\n{0})- Recover maximum expected bags_labels...".format(steps))
        steps = steps + 1

        # load files
        features = np.load(file=os.path.join(arg.dspath, arg.file_name + "_features.npz"))
        features = features[features.files[0]]
        C = np.load(file=os.path.join(arg.dspath, arg.file_name + "_bag_centroid.npz"))
        C = C[C.files[0]]
        bags_labels = load_data(file_name=arg.file_name + "_bag_pathway.pkl", load_path=arg.dspath,
                                tag="bags_labels with associated pathways")
        idxvocab = load_data(file_name=arg.file_name + "_idxvocab.pkl", load_path=arg.dspath,
                             tag="pathway ids to pathway features ids")
        y = load_data(file_name=arg.y_name, load_path=arg.dspath, tag="y")
        y_Bag = np.zeros((y.shape[0], C.shape[0]), dtype=np.int)

        for s_idx, sample in enumerate(y):
            desc = "\t>> Recovering maximum number of bags_labels: {0:.2f}%...".format(
                ((s_idx + 1) / y.shape[0]) * 100)
            if (s_idx + 1) != y.shape[0]:
                print(desc, end="\r")
            if (s_idx + 1) == y.shape[0]:
                print(desc)
            pathways = np.zeros((len(list(idxvocab), )), dtype=np.int)
            for ptwy_idx in sample.rows[0]:
                if ptwy_idx in idxvocab:
                    pathways[list(idxvocab).index(ptwy_idx)] = 1
            pathways = np.diag(pathways)
            features = pathways @ features
            sample_bag_features = np.dot(bags_labels, features) / np.sum(bags_labels, axis=1)[:, np.newaxis]
            sample_bag_features = arg.alpha * sample_bag_features
            np.nan_to_num(sample_bag_features, copy=False)
            cos = cosine_distances(C, sample_bag_features) / 2
            cos = np.diag(cos)
            B_idx = np.argwhere(cos > arg.v_cos)
            B_idx = B_idx.reshape((B_idx.shape[0],))
            y_Bag[s_idx, B_idx] = 1

        # save dataset with maximum bags_labels
        save_data(data=lil_matrix(y_Bag), file_name=arg.file_name + "_B.pkl", save_path=arg.dspath, mode="wb",
                  tag="bags to labels data")
        print("\t>> Done...")

    ##########################################################################################################
    ######################                            TRAIN                             ######################
    ##########################################################################################################

    if arg.train:
        print("\n{0})- Training {1} dataset using reMap model...".format(steps, arg.y_name))
        steps = steps + 1

        # load files
        print("\t>> Loading files...")
        y_Bag = load_data(file_name=arg.yB_name, load_path=arg.dspath, tag="B")

        # set randomly bags
        if arg.random_allocation:
            num_samples = y_Bag.shape[0]
            y_Bag = y_Bag.toarray()
            for bag_idx in np.arange(y_Bag.shape[1]):
                if np.sum(y_Bag[:, bag_idx]) == num_samples:
                    y_Bag[:, bag_idx] = np.random.binomial(1, arg.theta_bern, num_samples)
            y_Bag[y_Bag == 0] = -1
            y_Bag = lil_matrix(y_Bag)
            # save dataset with maximum bags_labels
            save_data(data=lil_matrix(y_Bag), file_name=arg.model_name + "_B.pkl", save_path=arg.dspath, mode="wb",
                      tag="bags to labels data")
        else:
            features = np.load(file=os.path.join(arg.dspath, arg.features_name))
            features = features[features.files[0]]
            C = np.load(file=os.path.join(arg.dspath, arg.bag_centroid_name))
            C = C[C.files[0]]
            rho = np.load(file=os.path.join(arg.dspath, arg.rho_name))
            rho = rho[rho.files[0]]
            bags_labels = load_data(file_name=arg.bags_labels, load_path=arg.dspath,
                                    tag="bags_labels with associated pathways")
            X = load_data(file_name=arg.X_name, load_path=arg.dspath, tag="X")
            y = load_data(file_name=arg.y_name, load_path=arg.dspath, tag="y")
            model = reMap(alpha=arg.alpha, binarize_input_feature=arg.binarize_input_feature,
                          fit_intercept=arg.fit_intercept, decision_threshold=arg.decision_threshold,
                          learning_type=arg.learning_type, lr=arg.lr, lr0=arg.lr0, forgetting_rate=arg.forgetting_rate,
                          delay_factor=arg.delay_factor, max_sampling=arg.max_sampling,
                          subsample_input_size=arg.ssample_input_size, subsample_labels_size=arg.ssample_label_size,
                          cost_subsample_size=arg.calc_subsample_size, min_bags=arg.min_bags, max_bags=arg.max_bags,
                          score_strategy=arg.score_strategy, loss_threshold=arg.loss_threshold,
                          early_stop=arg.early_stop, pi=arg.pi, calc_bag_cost=arg.calc_bag_cost,
                          calc_label_cost=arg.calc_label_cost, calc_total_cost=arg.calc_total_cost,
                          varomega=arg.varomega, varrho=arg.varrho, min_negatives_ratio=arg.min_negatives_ratio,
                          lambdas=arg.lambdas, label_bag_sim=arg.label_bag_sim,
                          label_closeness_sim=arg.label_closeness_sim, corr_bag_sim=arg.corr_bag_sim,
                          corr_label_sim=arg.corr_label_sim, corr_input_sim=arg.corr_input_sim, batch=arg.batch,
                          num_epochs=arg.num_epochs, num_jobs=arg.num_jobs, display_interval=arg.display_interval,
                          shuffle=arg.shuffle, random_state=arg.random_state, log_path=arg.logpath)
            model.fit(X=X, y=y, y_Bag=y_Bag, bags_labels=bags_labels, bags_correlation=rho, label_features=features,
                      centroids=C, model_name=arg.model_name, model_path=arg.mdpath, result_path=arg.rspath,
                      snapshot_history=arg.snapshot_history, display_params=display_params)


    ##########################################################################################################
    ######################                           TRANSFORM                          ######################
    ##########################################################################################################
    
    if arg.transform:
        print("\n{0})- Predicting dataset using a pre-trained reMap model...".format(steps))

        # load files
        print("\t>> Loading files...")
        features = np.load(file=os.path.join(arg.dspath, arg.features_name))
        features = features[features.files[0]]
        C = np.load(file=os.path.join(arg.dspath, arg.bag_centroid_name))
        C = C[C.files[0]]
        rho = np.load(file=os.path.join(arg.dspath, arg.rho_name))
        rho = rho[rho.files[0]]
        bags_labels = load_data(file_name=arg.bags_labels, load_path=arg.dspath,
                                tag="bags_labels with associated pathways")

        # load data
        X = load_data(file_name=arg.X_name, load_path=arg.dspath, tag="X")
        y = load_data(file_name=arg.y_name, load_path=arg.dspath, tag="y")
        model = load_data(file_name=arg.model_name + ".pkl", load_path=arg.mdpath, tag="reMap model")

        print("\t>> Predict bags...")
        y_Bag = model.transform(X=X, y=y, bags_labels=bags_labels, bags_correlation=rho, label_features=features,
                                centroids=C, subsample_labels_size=arg.ssample_label_size,
                                max_sampling=arg.max_sampling, snapshot_history=arg.snapshot_history, 
                                decision_threshold=arg.decision_threshold, batch_size=arg.batch, num_jobs=arg.num_jobs, 
                                file_name=arg.file_name, result_path=arg.rspath)
        # save dataset with maximum bags_labels
        save_data(data=lil_matrix(y_Bag), file_name=arg.file_name + "_B.pkl", save_path=arg.dspath, mode="wb",
                  tag="bags to labels data")


def train(arg):
    try:
        if arg.define_bags or arg.recover_max_bags or arg.train or arg.transform:
            actions = list()
            if arg.define_bags:
                actions += ["CONSTRUCT BAGs"]
            if arg.recover_max_bags:
                actions += ["RECOVER MAXIMUM NUMBER OF BAGs"]
            if arg.train:
                actions += ["TRAIN reMap"]
            if arg.transform:
                actions += ["TRANSFORM RESULTS USING a PRETRAINED MODEL"]
            desc = [str(item[0] + 1) + ". " + item[1] for item in zip(list(range(len(actions))), actions)]
            desc = " ".join(desc)
            print("\n*** APPLIED ACTIONS ARE: {0}".format(desc))
            timeref = time.time()
            __train(arg)
            print("\n*** The selected actions consumed {1:f} SECONDS\n".format("", round(time.time() - timeref, 3)),
                  file=sys.stderr)
        else:
            print("\n*** PLEASE SPECIFY AN ACTION...\n", file=sys.stderr)
    except Exception:
        print(traceback.print_exc())
        raise
