#!/bin/bash
#set -e
#set -x

#pip install -r requirements.txt

# Note -- these are not the recommended settings for this dataset.  This is just so the open-source tests will finish quickly.

############## Preprecess the dataset
# python main.py --preprocess-dataset --define-bags --recover-max-bags --v-cos 0.1 --hin-name "hin_cmt.pkl" --vocab-name "vocab_biocyc.pkl" --bag-phi-name "soap/soap_biocyc_26_exp_phi.npz" --bag-sigma-name "soap/soap_biocyc_26_sigma.npz" --features-name "path2vec/path2vec_cmt_final_tf_embeddings.npz" --top-k 350 --file-name "biocyc_50" --X-name "biocyc_X.pkl" --y-name "biocyc_y.pkl"

# python main.py --preprocess-dataset --define-bags --recover-max-bags --v-cos 0.1 --hin-name "hin_cmt.pkl" --vocab-name "vocab_biocyc.pkl" --bag-phi-name "soap/soap_biocyc_27_exp_phi.npz" --bag-sigma-name "soap/soap_biocyc_27_sigma.npz" --features-name "path2vec/path2vec_cmt_final_tf_embeddings.npz" --top-k 190 --file-name "biocyc_100" --X-name "biocyc_X.pkl" --y-name "biocyc_y.pkl"

# python main.py --preprocess-dataset --define-bags --recover-max-bags --v-cos 0.1 --hin-name "hin_cmt.pkl" --vocab-name "vocab_biocyc.pkl" --bag-phi-name "soap/soap_biocyc_28_exp_phi.npz" --bag-sigma-name "soap/soap_biocyc_28_sigma.npz" --features-name "path2vec/path2vec_cmt_final_tf_embeddings.npz" --top-k 120 --file-name "biocyc_150" --X-name "biocyc_X.pkl" --y-name "biocyc_y.pkl"

## apply this data to all experiments, where bags are set to 200
## TODO: this one for README.md
#python main.py --preprocess-dataset --define-bags --recover-max-bags --v-cos 0.1 --hin-name "hin_cmt.pkl" --vocab-name "vocab_biocyc.pkl" --bag-phi-name "soap/soap_biocyc_29_exp_phi.npz" --bag-sigma-name "soap/soap_biocyc_29_sigma.npz" --features-name "path2vec/path2vec_cmt_final_tf_embeddings.npz" --top-k 90 --file-name "biocyc" --X-name "biocyc_X.pkl" --y-name "biocyc_y.pkl"

# python main.py --preprocess-dataset --define-bags --recover-max-bags --v-cos 0.1 --hin-name "hin_cmt.pkl" --vocab-name "vocab_biocyc.pkl" --bag-phi-name "spreat_biocyc_29_exp_phi.npz" --bag-sigma-name "spreat_biocyc_29_sigma.npz" --features-name "path2vec_cmt_final_tf_embeddings.npz" --top-k 90 --file-name "biocyc_spreat_collapsed" --X-name "biocyc_X.pkl" --y-name "biocyc_y.pkl"

## add supplementary as well
# python main.py --preprocess-dataset --define-bags --recover-max-bags --v-cos 0.1 --hin-name "hin_cmt.pkl" --vocab-name "vocab_biocyc.pkl" --bag-phi-name "soap_biocyc_9_exp_phi.npz" --bag-sigma-name "soap_biocyc_9_sigma.npz" --features-name "path2vec_cmt_final_tf_embeddings.npz" --top-k 225 --file-name "biocyc_soap" --X-name "biocyc_X.pkl" --y-name "biocyc_y.pkl"
# python main.py --preprocess-dataset --define-bags --recover-max-bags --v-cos 0.1 --hin-name "hin_cmt.pkl" --vocab-name "vocab_biocyc.pkl" --bag-phi-name "spreat_biocyc_9_exp_phi.npz" --bag-sigma-name "spreat_biocyc_9_sigma.npz" --features-name "path2vec_cmt_final_tf_embeddings.npz" --top-k 260 --file-name "biocyc_spreat" --X-name "biocyc_X.pkl" --y-name "biocyc_y.pkl"
# python main.py --preprocess-dataset --define-bags --recover-max-bags --v-cos 0.1 --hin-name "hin_cmt.pkl" --vocab-name "vocab_biocyc.pkl" --bag-phi-name "ctm_biocyc_9_exp_omega.npz" --bag-sigma-name "ctm_biocyc_9_sigma.npz" --features-name "path2vec_cmt_final_tf_embeddings.npz" --top-k 150 --file-name "biocyc_ctm" --X-name "biocyc_X.pkl" --y-name "biocyc_y.pkl"

# python main.py --preprocess-dataset --define-bags --recover-max-bags --v-cos 0.1 --hin-name "hin_cmt.pkl" --vocab-name "vocab_biocyc.pkl" --bag-phi-name "soap/soap_biocyc_30_exp_phi.npz" --bag-sigma-name "soap/soap_biocyc_30_sigma.npz" --features-name "path2vec/path2vec_cmt_final_tf_embeddings.npz" --top-k 80 --file-name "biocyc_300" --X-name "biocyc_X.pkl" --y-name "biocyc_y.pkl"

#######################################################################################################
###################################            First test           ###################################
#######################################################################################################
## train and estimate cost
#python main.py --train --random-allocation --bags-labels "biocyc_bag_pathway.pkl" --X-name "biocyc_X.pkl" --y-name "biocyc_y.pkl" --yB-name "biocyc_B.pkl" --file-name "biocyc" --model-name "reMap_1" --num-epochs 3 --num-jobs 10

## TODO: this one for README.md
# python main.py --train --calc-label-cost --ssample-input-size 0.05 --ssample-label-size 50 --calc-subsample-size 50 --decision-threshold 0.5 --loss-threshold 0.001 --max-sampling 1 --min-bags 10 --max-bags 50 --bags-labels "biocyc_bag_pathway.pkl" --features-name "biocyc_features.npz" --bag-centroid-name "biocyc_bag_centroid.npz" --rho-name "biocyc_rho.npz" --X-name "biocyc_X.pkl" --y-name "biocyc_y.pkl" --yB-name "biocyc_B.pkl" --file-name "biocyc" --model-name "reMap_2" --num-epochs 3 --num-jobs 10

# python main.py --train --calc-label-cost --ssample-input-size 0.05 --ssample-label-size 50 --calc-subsample-size 50 --decision-threshold 0.5 --loss-threshold 0.001 --max-sampling 1 --min-bags 10 --max-bags 50 --bags-labels "biocyc_spreat_collapsed_bag_pathway.pkl" --features-name "biocyc_spreat_collapsed_features.npz" --bag-centroid-name "biocyc_spreat_collapsed_bag_centroid.npz" --rho-name "biocyc_spreat_collapsed_rho.npz" --X-name "biocyc_X.pkl" --y-name "biocyc_y.pkl" --yB-name "biocyc_spreat_collapsed_B.pkl" --file-name "biocyc_spreat_collapsed" --model-name "reMap_3" --num-epochs 3 --num-jobs 10
# python main.py --train --calc-label-cost --ssample-input-size 0.05 --ssample-label-size 50 --calc-subsample-size 50 --decision-threshold 0.5 --loss-threshold 0.001 --max-sampling 1 --min-bags 10 --max-bags 50 --bags-labels "biocyc_soap_bag_pathway.pkl" --features-name "biocyc_soap_features.npz" --bag-centroid-name "biocyc_soap_bag_centroid.npz" --rho-name "biocyc_soap_rho.npz" --X-name "biocyc_X.pkl" --y-name "biocyc_y.pkl" --yB-name "biocyc_spreat_B.pkl" --file-name "biocyc_soap" --model-name "reMap_4_1" --num-epochs 3 --num-jobs 10
# python main.py --train --calc-label-cost --ssample-input-size 0.05 --ssample-label-size 50 --calc-subsample-size 50 --decision-threshold 0.5 --loss-threshold 0.001 --max-sampling 1 --min-bags 10 --max-bags 50 --bags-labels "biocyc_spreat_bag_pathway.pkl" --features-name "biocyc_spreat_features.npz" --bag-centroid-name "biocyc_spreat_bag_centroid.npz" --rho-name "biocyc_spreat_rho.npz" --X-name "biocyc_X.pkl" --y-name "biocyc_y.pkl" --yB-name "biocyc_spreat_B.pkl" --file-name "biocyc_spreat" --model-name "reMap_4_2" --num-epochs 3 --num-jobs 10
# python main.py --train --calc-label-cost --ssample-input-size 0.5 --ssample-label-size 50 --calc-subsample-size 50 --decision-threshold 0.5 --loss-threshold 0.001 --max-sampling 1 --min-bags 10 --max-bags 50 --bags-labels "biocyc_ctm_bag_pathway.pkl" --features-name "biocyc_ctm_features.npz" --bag-centroid-name "biocyc_ctm_bag_centroid.npz" --rho-name "biocyc_ctm_rho.npz" --X-name "biocyc_X.pkl" --y-name "biocyc_y.pkl" --yB-name "biocyc_ctm_B.pkl" --file-name "biocyc_ctm" --model-name "reMap_4_3" --num-epochs 3 --num-jobs 10

## generate samples
## TODO: this one for README.md
# python main.py --transform --ssample-label-size 100 --decision-threshold 0.5 --max-sampling 3 --min-bags 10 --max-bags 50 --bags-labels "biocyc_bag_pathway.pkl" --features-name "biocyc_features.npz" --bag-centroid-name "biocyc_bag_centroid.npz" --rho-name "biocyc_rho.npz" --X-name "biocyc_X.pkl" --y-name "biocyc_y.pkl" --file-name "biocyc" --model-name "reMap_2" --num-epochs 3 --num-jobs 10

# python main.py --transform --ssample-label-size 100 --decision-threshold 0.5 --loss-threshold 0.001 --max-sampling 1 --min-bags 10 --max-bags 50 --bags-labels "biocyc_spreat_collapsed_bag_pathway.pkl" --features-name "biocyc_spreat_collapsed_features.npz" --bag-centroid-name "biocyc_spreat_collapsed_bag_centroid.npz" --rho-name "biocyc_spreat_collapsed_rho.npz" --X-name "biocyc_X.pkl" --y-name "biocyc_y.pkl" --file-name "reMap_3" --model-name "reMap_3" --num-epochs 3 --num-jobs 10
# python main.py --transform --ssample-label-size 100 --decision-threshold 0.5 --loss-threshold 0.001 --max-sampling 1 --min-bags 10 --max-bags 50 --bags-labels "biocyc_soap_bag_pathway.pkl" --features-name "biocyc_soap_features.npz" --bag-centroid-name "biocyc_soap_bag_centroid.npz" --rho-name "biocyc_spreat_rho.npz" --X-name "biocyc_X.pkl" --y-name "biocyc_y.pkl" --file-name "reMap_4_1" --model-name "reMap_4_1" --num-epochs 3 --num-jobs 10
# python main.py --transform --ssample-label-size 100 --decision-threshold 0.5 --loss-threshold 0.001 --max-sampling 1 --min-bags 10 --max-bags 50 --bags-labels "biocyc_spreat_bag_pathway.pkl" --features-name "biocyc_spreat_features.npz" --bag-centroid-name "biocyc_spreat_bag_centroid.npz" --rho-name "biocyc_spreat_rho.npz" --X-name "biocyc_X.pkl" --y-name "biocyc_y.pkl" --file-name "reMap_4_2" --model-name "reMap_4_2" --num-epochs 3 --num-jobs 10
# python main.py --transform --ssample-label-size 100 --decision-threshold 0.5 --loss-threshold 0.001 --max-sampling 1 --min-bags 10 --max-bags 50 --bags-labels "biocyc_ctm_bag_pathway.pkl" --features-name "biocyc_ctm_features.npz" --bag-centroid-name "biocyc_ctm_bag_centroid.npz" --rho-name "biocyc_ctm_rho.npz" --X-name "biocyc_X.pkl" --y-name "biocyc_y.pkl" --file-name "reMap_4_3" --model-name "reMap_4_3" --num-epochs 3 --num-jobs 10

#######################################################################################################
###################################           Second test           ###################################
#######################################################################################################
# then set bags gradually within range {50, 100, 150, 200, 300}
# remember that 200 bags already being trained on
# python main.py --train --calc-label-cost --ssample-input-size 0.5 --ssample-label-size 50 --calc-subsample-size 50 --decision-threshold 0.5 --loss-threshold 0.001 --max-sampling 1 --min-bags 10 --max-bags 50 --bags-labels "biocyc_50_bag_pathway.pkl" --features-name "biocyc_50_features.npz" --bag-centroid-name "biocyc_50_bag_centroid.npz" --rho-name "biocyc_50_rho.npz" --X-name "biocyc_X.pkl" --y-name "biocyc_y.pkl" --yB-name "biocyc_50_B.pkl" --file-name "biocyc_50" --model-name "reMap_5" --num-epochs 3 --num-jobs 10
# python main.py --transform --ssample-label-size 100 --decision-threshold 0.5 --max-sampling 3 --min-bags 10 --max-bags 50 --bags-labels "biocyc_50_bag_pathway.pkl" --features-name "biocyc_50_features.npz" --bag-centroid-name "biocyc_50_bag_centroid.npz" --rho-name "biocyc_50_rho.npz" --X-name "biocyc_X.pkl" --y-name "biocyc_y.pkl" --file-name "biocyc_50" --model-name "reMap_5" --num-epochs 3 --num-jobs 10
#
# python main.py --train --calc-label-cost --ssample-input-size 0.5 --ssample-label-size 50 --calc-subsample-size 50 --decision-threshold 0.5 --loss-threshold 0.001 --max-sampling 1 --min-bags 10 --max-bags 50 --bags-labels "biocyc_100_bag_pathway.pkl" --features-name "biocyc_100_features.npz" --bag-centroid-name "biocyc_100_bag_centroid.npz" --rho-name "biocyc_100_rho.npz" --X-name "biocyc_X.pkl" --y-name "biocyc_y.pkl" --yB-name "biocyc_100_B.pkl" --file-name "biocyc_100" --model-name "reMap_6" --num-epochs 3 --num-jobs 10
# python main.py --transform --ssample-label-size 100 --decision-threshold 0.5 --max-sampling 3 --min-bags 10 --max-bags 50 --bags-labels "biocyc_100_bag_pathway.pkl" --features-name "biocyc_100_features.npz" --bag-centroid-name "biocyc_100_bag_centroid.npz" --rho-name "biocyc_100_rho.npz" --X-name "biocyc_X.pkl" --y-name "biocyc_y.pkl" --file-name "biocyc_100" --model-name "reMap_6" --num-epochs 3 --num-jobs 10

# python main.py --train --calc-label-cost --ssample-input-size 0.5 --ssample-label-size 50 --calc-subsample-size 50 --decision-threshold 0.5 --loss-threshold 0.001 --max-sampling 1 --min-bags 10 --max-bags 50 --bags-labels "biocyc_150_bag_pathway.pkl" --features-name "biocyc_150_features.npz" --bag-centroid-name "biocyc_150_bag_centroid.npz" --rho-name "biocyc_150_rho.npz" --X-name "biocyc_X.pkl" --y-name "biocyc_y.pkl" --yB-name "biocyc_150_B.pkl" --file-name "biocyc_150" --model-name "reMap_7" --num-epochs 3 --num-jobs 10
# python main.py --transform --ssample-label-size 100 --decision-threshold 0.5 --max-sampling 3 --min-bags 10 --max-bags 50 --bags-labels "biocyc_150_bag_pathway.pkl" --features-name "biocyc_150_features.npz" --bag-centroid-name "biocyc_150_bag_centroid.npz" --rho-name "biocyc_150_rho.npz" --X-name "biocyc_X.pkl" --y-name "biocyc_y.pkl" --file-name "biocyc_150" --model-name "reMap_7" --num-epochs 3 --num-jobs 10

# python main.py --train --calc-label-cost --ssample-input-size 0.5 --ssample-label-size 50 --calc-subsample-size 50 --decision-threshold 0.5 --loss-threshold 0.001 --max-sampling 1 --min-bags 10 --max-bags 50 --bags-labels "biocyc_300_bag_pathway.pkl" --features-name "biocyc_300_features.npz" --bag-centroid-name "biocyc_300_bag_centroid.npz" --rho-name "biocyc_300_rho.npz" --X-name "biocyc_X.pkl" --y-name "biocyc_y.pkl" --yB-name "biocyc_300_B.pkl" --file-name "biocyc_300" --model-name "reMap_8" --num-epochs 3 --num-jobs 10
# python main.py --transform --ssample-label-size 100 --decision-threshold 0.5 --max-sampling 3 --min-bags 10 --max-bags 50 --bags-labels "biocyc_300_bag_pathway.pkl" --features-name "biocyc_300_features.npz" --bag-centroid-name "biocyc_300_bag_centroid.npz" --rho-name "biocyc_300_rho.npz" --X-name "biocyc_X.pkl" --y-name "biocyc_y.pkl" --file-name "biocyc_300" --model-name "reMap_8" --num-epochs 3 --num-jobs 10

#######################################################################################################
###################################            Third test           ###################################
#######################################################################################################
# choose the best model for predicting with snapshot-history is true and max-sampling is 10
# python main.py --transform --snapshot-history --ssample-label-size 100 --decision-threshold 0.5 --max-sampling 10 --min-bags 10 --max-bags 50 --bags-labels "biocyc_bag_pathway.pkl" --features-name "biocyc_features.npz" --bag-centroid-name "biocyc_bag_centroid.npz" --rho-name "biocyc_rho.npz" --X-name "golden_X.pkl" --y-name "golden_y.pkl" --file-name "biocyc_golden" --model-name "reMap_2_final" --num-epochs 3 --batch 65 --num-jobs 10
# python main.py --transform --snapshot-history --ssample-label-size 100 --decision-threshold 0.5 --max-sampling 10 --min-bags 10 --max-bags 50 --bags-labels "biocyc_bag_pathway.pkl" --features-name "biocyc_features.npz" --bag-centroid-name "biocyc_bag_centroid.npz" --rho-name "biocyc_rho.npz" --X-name "cami_X.pkl" --y-name "cami_y.pkl" --file-name "biocyc_cami" --model-name "reMap_2_final" --num-epochs 3 --batch 50 --num-jobs 10

# history
# python main.py --preprocess-dataset --define-bags --recover-max-bags --v-cos 0.1 --hin-name "hin_cmt.pkl" --vocab-name "vocab_biocyc.pkl" --bag-phi-name "soap/soap_biocyc_29_exp_phi.npz" --bag-sigma-name "soap/soap_biocyc_29_sigma.npz" --features-name "path2vec/path2vec_cmt_final_tf_embeddings.npz" --top-k 90 --file-name "golden" --X-name "golden_X.pkl" --y-name "golden_y.pkl"
# python main.py --train --snapshot-history --calc-label-cost --ssample-input-size 1 --ssample-label-size 100 --calc-subsample-size 50 --decision-threshold 0.5 --max-sampling 10 --min-bags 10 --max-bags 50 --bags-labels "biocyc_bag_pathway.pkl" --features-name "golden_features.npz" --bag-centroid-name "golden_bag_centroid.npz" --rho-name "golden_rho.npz" --X-name "golden_X.pkl" --y-name "golden_y.pkl" --yB-name "golden_B.pkl" --file-name "golden" --model-name "reMap_9" --num-epochs 10 --num-jobs 10

#######################################################################################################
###################################        Fifth test [FUTURE]      ###################################
#######################################################################################################
#  python main.py --train --score-strategy --calc-label-cost --ssample-input-size 0.05 --ssample-label-size 50 --calc-subsample-size 50 --decision-threshold 0.5 --loss-threshold 0.001 --max-sampling 1 --min-bags 10 --max-bags 50 --bags-labels "biocyc_bag_pathway.pkl" --X-name "biocyc_X.pkl" --y-name "biocyc_y.pkl" --yB-name "biocyc_B.pkl" --file-name "biocyc" --model-name "reMap_3" --num-epochs 3 --num-jobs 10

# python main.py --transform --ssample-label-size 100 --decision-threshold 0.5 --max-sampling 3 --min-bags 10 --max-bags 50 --bags-labels "biocyc_bag_pathway.pkl" --X-name "biocyc_X.pkl" --y-name "biocyc_y.pkl" --file-name "biocyc" --model-name "reMap_3" --num-epochs 3 --num-jobs 10

# below tests are only valid for loss based update
# first choose self.varrho within {0.3, 0.7, 0.9} where a higher self.varrho indicates
# more nU samples required; note 0.7 is defualt and we already run it using "reMap_3"
# so we just need to run two instances
# python main.py --train --score-strategy --calc-label-cost --ssample-input-size 0.5 --ssample-label-size 50 --calc-subsample-size 50 --decision-threshold 0.5 --loss-threshold 0.001 --max-sampling 1 --min-bags 10 --max-bags 50 --varrho 0.3 --bags-labels "biocyc_bag_pathway.pkl" --X-name "biocyc_X.pkl" --y-name "biocyc_y.pkl" --yB-name "biocyc_B.pkl" --file-name "biocyc" --model-name "reMap_9" --num-epochs 3 --num-jobs 10
# python main.py --transform --ssample-label-size 100 --decision-threshold 0.5 --max-sampling 3 --min-bags 10 --max-bags 50 --bags-labels "biocyc_bag_pathway.pkl" --X-name "biocyc_X.pkl" --y-name "biocyc_y.pkl" --file-name "biocyc" --model-name "reMap_9" --num-epochs 3 --num-jobs 10

# python main.py --train --score-strategy --calc-label-cost --ssample-input-size 0.5 --ssample-label-size 50 --calc-subsample-size 50 --decision-threshold 0.5 --loss-threshold 0.001 --max-sampling 1 --min-bags 10 --max-bags 50 --varrho 0.3 --varrho 0.9 --bags-labels "biocyc_bag_pathway.pkl" --X-name "biocyc_X.pkl" --y-name "biocyc_y.pkl" --yB-name "biocyc_B.pkl" --file-name "biocyc" --model-name "reMap_10" --num-epochs 3 --num-jobs 10
# python main.py --transform --ssample-label-size 100 --decision-threshold 0.5 --max-sampling 3 --min-bags 10 --max-bags 50 --bags-labels "biocyc_bag_pathway.pkl" --X-name "biocyc_X.pkl" --y-name "biocyc_y.pkl" --file-name "biocyc" --model-name "reMap_10" --num-epochs 3 --num-jobs 10
