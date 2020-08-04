![Workflow](flowchart.png)

## Basic Description
This repo contains an implementation of reMap (**re**labeling **M**ulti-label dataset based on b**a**g a**p**proach) a simple, and yet, generic framework, that performs relabeling examples to a different set of labels, characterized as bags, where a bag is comprised of correlated pathways. To obtain bags, any models from [CBT](https://github.com/hallamlab/cbt.git) package can be employed. After obtaining bags, reMap preforms relabeling by alternating between 1) assigning bags to each sample (or feed-forward) and 2) updating reMap's internal parameters (or feed-backward). reMap's effectiveness were evaluated on metabolic pathway prediction (using [leADS](https://github.com/hallamlab/leADS.git)) where resulting performance metrics equaled or exceeded other prediction methods on organismal genomes with improved sensitivity score.

## Dependencies
The codebase is tested to work under Python 3.7. To install the necessary requirements, run the following commands:

``pip install -r requirements.txt``

Basically, *reMap* requires following packages:
- [Anaconda](https://www.anaconda.com/)
- [NumPy](http://www.numpy.org/) (== 1.18.5)
- [scikit-learn](https://scikit-learn.org/stable/) (== 0.23)
- [scipy](https://www.scipy.org/index.html) (==1.5.0)


## Experimental Objects and Test Samples
Please download the following files from [Zenodo](https://zenodo.org/record/3711138#.Xn2fgXVKjeQ). 
- The link contains the following preprocessed files:
    - "biocyc.pkl": an object containing the preprocessed MetaCyc database.
    - "pathway2ec.pkl": a matrix representing Pathway-EC association of size (3650, 2526).
    - "pathway2ec_idx.pkl": the pathway2ec association indices.                        
    - "M.pkl": a sub matrix from pathway2ec.
    - "A.pkl": Pathway-Pathway interaction matrix of size (2526, 2526).
    - "B.pkl": EC-EC interaction matrix of size (3650, 3650).
    - "P.pkl": Pathway features matrix of size (2526, 128).
    - "E.pkl": EC features matrix of size (3650, 128).
    - "hin.pkl": a sample of heterogeneous information network. 
    - "pathway2vec_embeddings.npz": a sample of embeddings (nodes, dimension size). Based on your tests, you need to generate features using [pathway2vec](https://github.com/hallamlab/pathway2vec).
- We also provided pretrained models and samples for testing:
    - "golden_X.pkl": Golden dataset of size (63, 3650). First six examples correspond to: AraCyc, EcoCyc, HumanCyc, LeishCyc, TrypanoCyc, and YeastCyc.
    - "golden_Xe.pkl": Golden dataset of size (63, 3778. First six examples correspond to: AraCyc, EcoCyc, HumanCyc, LeishCyc, TrypanoCyc, and YeastCyc.
    - "golden_y.pkl": Golden dataset of size (63, 2526). First six examples correspond pathways to: AraCyc, EcoCyc, HumanCyc, LeishCyc, TrypanoCyc, and YeastCyc.
    - "symbionts_X.pkl": Symbiont dataset of size (3, 3650).
    - "symbionts_Xe.pkl": Symbiont dataset of size (3, 3778).
    - "symbionts_y.pkl": Symbiont dataset of size (3, 2526).
    - "cami_X.pkl": CAMI dataset of size (40, 3650).
    - "cami_Xe.pkl": CAMI dataset of size (40, 3778).
    - "cami_y.pkl": CAMI dataset of size (40, 2526).
    - "hots_4_X.pkl": HOT metagenomics dataset of size (6, 3650).
    - "hots_4_Xe.pkl": HOT metagenomics dataset of size (6, 3778).
    - "hots_4_y.pkl": HOT metagenomics dataset of size (6, 2526).
    - "biocyc205_tier23_9255_X.pkl": BioCyc (v20.5 tier 2 \& 3) dataset of size (9255, 3650).
    - "biocyc205_tier23_9255_Xe.pkl": BioCyc (v20.5 tier 2 \& 3) dataset of size (9255, 3778).
    - "biocyc205_tier23_9255_y.pkl": BioCyc (v20.5 tier 2 \& 3) dataset of size (9255, 2526).
    - "triUMPF_X.pkl": a pretrained model using "biocyc205_tier23_9255_X.pkl" and "biocyc205_tier23_9255_y.pkl".
    - "triUMPF_X_W.pkl": a pretrained latent factors for pathways of size (2526, 100).
    - "triUMPF_X_H.pkl": a pretrained basis matrix for ECs of size (3650, 100).
    - "triUMPF_X_U.pkl": an auxilary matrix of size (128, 100).
    - "triUMPF_X_V.pkl": an auxilary matrix of size (128, 100).
    - "triUMPF_X_T.pkl": a pathway community representation matrix of size (128, 90).
    - "triUMPF_X_C.pkl": a pathway community indicator matrix of size (2526, 90).
    - "triUMPF_X_R.pkl": a EC community representation matrix of size (128, 100).
    - "triUMPF_X_K.pkl": a EC community representation matrix of size (3650, 100).
    - "triUMPF_X_L.pkl": an auxilary matrix of size (9255, 128).
    - "triUMPF_X_Z.pkl": an auxilary matrix of size (9255, 3650).
    - "triUMPF_e.pkl": a pretrained model using "biocyc205_tier23_9255_Xe.pkl" and "biocyc205_tier23_9255_y.pkl".
    - "triUMPF_e\*.pkl": Descriptions about the remaining matrices are same as triUMPF_X\*.

## Installation and Basic Usage
Run the following commands to clone the repository to an appropriate location:

``git clone https://github.com/hallamlab/reMap.git``

For all experiments, navigate to ``src`` folder then run the commands of your choice. For example, to display *reMap*'s running options use: `python main.py --help`. It should be self-contained.

### Preprocessing
To preprocess data, we provide few examples.

#### Example 1
To preprocess datasets with **no noise** to the pathway2ec association matrix ("pathway2ec.pkl"), execute the following command:

``python main.py --preprocess-dataset --ssample-input-size 1 --object-name "biocyc.pkl" --pathway2ec-name "pathway2ec.pkl" --pathway2ec-pidx-name "pathway2ec_idx.pkl" --features-name "pathway2vec_embeddings.npz" --hin-name "hin.pkl" --mdpath [Location of the features] --ospath [Location to all objects except features]``

#### Example 2
To preprocess datasets with **20% noise** to the pathway2ec association matrix ("pathway2ec.pkl"), execute the following command:

``python main.py --preprocess-dataset --ssample-input-size 0.2 --object-name "biocyc.pkl" --pathway2ec-name "pathway2ec.pkl" --pathway2ec-pidx-name "pathway2ec_idx.pkl" --features-name "pathway2vec_embeddings.npz" --hin-name "hin.pkl" --mdpath [Location of the features] --ospath [Location to all objects except features]``

#### Example 3
To preprocess datasets with **20% noise** to the pathway2ec association (*pathway2ec.pkl*), the pathway to pathway association (*A*), and the EC to EC association (*B*) matrices, execute the following command:

``python main.py --preprocess-dataset --white-links --ssample-input-size 0.2 --object-name "biocyc.pkl" --pathway2ec-name "pathway2ec.pkl" --pathway2ec-pidx-name "pathway2ec_idx.pkl" --features-name "pathway2vec_embeddings.npz" --hin-name "hin.pkl" --mdpath [Location of the features] --ospath [Location to all objects except features]``

### Training
For trainning, we provide few examples. 

Description about arguments in all examples: *--cutting-point* is the cutting point after which binarize operation is halted in the input data, *--M-name* is the pathway2ec association matrix file name, *--W-name* is the W parameter, *--H-name* is the H parameter, and *--model-name* corresponds the name of the model excluding any *EXTENSION*. The model name will have *.pkl* extension. The arguments *--P-name* corresponds the pathway features file name, *--E-name* is the EC features file name, *--A-name* is the pathway to pathway association file name, *--B-name* corresponds the EC to EC association file name, *--X-name* is the input space of multi-label data, and *--y-name* is the pathway space of multi-label data. For the dataset, any multi-label dataset can be employed.

**Please** do not use "triUMPF_X.pkl" or "triUMPF_e.pkl" and all the associated models related files (e.g. triUMPF_X_C.pkl, triUMPF_X_H.pkl...etc) during this step, and change the name of the pretrained models or store them in a different folder to avoid conflict.

#### Example 1
To **decompose** *M* of 100 components, execute the following command:

``python main.py --train --num-components 100 --lambdas 0.01 0.01 0.01 0.01 0.001 10 --M-name "M.pkl" --model-name "[model name (without extension)]" --mdpath "[path to the model]" --logpath "[path to the log directory]" --batch 50 --max-inner-iter 5 --num-epochs 100 --num-jobs 2``

#### Example 2
To **decompose** *M* of 100 components by using **features**, execute the following command:

``python main.py --train --fit-features --num-components 100 --lambdas 0.01 0.01 0.01 0.01 0.001 10 --M-name "M.pkl" --P-name "P.pkl" --E-name "E.pkl" --model-name "[model name (without extension)]" --mdpath "[path to the model]" --logpath "[path to the log directory]" --batch 50 --max-inner-iter 5 --num-epochs 100 --num-jobs 2``

#### Example 3
If you wish to train multi-label dataset by **decomposing** *M* of 100 components while using **features** and **community**, execute the following command:

``python main.py --train --fit-features --fit-comm --binarize --use-external-features --cutting-point 3650 --num-components 100 --lambdas 0.01 0.01 0.01 0.01 0.001 10 --M-name "M.pkl" --P-name "P.pkl" --E-name "E.pkl"  --A-name "A.pkl" --B-name "B.pkl" --X-name "biocyc_Xe.pkl" --y-name "biocyc_y.pkl" --model-name "[model name (without extension)]" --mdpath "[path to the model]" --dspath "[path to the dataset]" --logpath "[path to the log directory]" --batch 50 --max-inner-iter 5 --num-epochs 100 --num-jobs 2``

#### Example 4
If you wish to use the previously **decomposed** *M* of 100 components to train multi-label dataset while using **features** and **community**, execute the following command:

``python main.py --train --no-decomposition --fit-features --fit-comm --binarize --use-external-features --cutting-point 3650 --num-components 100 --lambdas 0.01 0.01 0.01 0.01 0.001 10 --W-name "[Generated .pkl W file]" --H-name "[Generated .pkl H file]"  --P-name "P.pkl" --E-name "E.pkl"  --A-name "A.pkl" --B-name "B.pkl" --X-name "biocyc_Xe.pkl" --y-name "biocyc_y.pkl" --model-name "[model name (without extension)]" --mdpath "[path to the model]" --dspath "[path to the dataset]" --logpath "[path to the log directory]" --batch 50 --max-inner-iter 5 --num-epochs 100 --num-jobs 2``

### Predicting
For inference, we provide few examples. 

Description about arguments in all examples: *--binarize* is a boolean variable indicating whether to binarize data, *--cutting-point* is the cutting point after which binarize operation is halted in the input data,  *--decision-threshold* corresponds the cutoff threshold for prediction, *--object-name* is an object containing the preprocessed MetaCyc database, *--pathway2ec-name* is a matrix representing Pathway-EC association, *--pathway2ec-idx-name* corresponds the pathway2ec association indices, *--hin-name* is the heterogeneous information network, *--features-name* is features corresponding ECs and pathways, *--file-name* corresponds the name of several preprocessed files (without extension), *--batch* is batch size, *--num-jobs* corresponds the number of parallel workers, and *--model-name* corresponds the name of the model excluding any *EXTENSION*. The model name will have *.pkl* extension. The arguments *--X-name* is the input space of multi-label data. For the dataset, any multi-label dataset can be employed.


#### Example 1
To predict outputs from a dataset (e.g. "cami_Xe.pkl") using already trained model (e.g. "triUMPF_e.pkl") with decomposed *M* of 100 components while using **features** and **community**, execute the following command:

``python main.py --predict --binarize --cutting-point 3650 --decision-threshold 0.5 --X-name "cami_Xe.pkl" --file-name "triUMPF" --model-name "triUMPF_e.pkl" --dspath "[path to the dataset and to store predicted results]" --mdpath "[path to the model]" --logpath "[path to the log directory]" --batch 50 --num-jobs 2``

#### Example 2

To predict outputs and **compile pathway report** from a dataset (e.g. "symbionts_Xe.pkl"), generated by MetaPathways v2, using already trained model (e.g. "triUMPF_e.pkl") with decomposed *M* of 100 components while using **features** and **community**, execute the following command:

``python main.py --predict --binarize --pathway-report --cutting-point 3650 --decision-threshold 0.5 --object-name "biocyc.pkl" --pathway2ec-idx-name "pathway2ec_idx.pkl" --pathway2ec-name "pathway2ec.pkl" --hin-name "hin.pkl" --features-name "pathway2vec_embeddings.npz" --X-name "symbionts_Xe.pkl" --file-name "triUMPF" --model-name "triUMPF_e.pkl" --rsfolder "[name of the main folder]" --dspath "[path to the dataset and to store predicted results]" --mdpath "[path to the model]" --rspath "[path to storing results]" --logpath "[path to the log directory]" --batch 50 --num-jobs 2``

where *--pathway-report* enables to generate a detailed report for pathways for each instance.

## Citing
If you find *triUMPF* useful in your research, please consider citing the following paper:
- M. A. Basher, Abdur Rahman, McLaughlin, Ryan J., and Hallam, Steven J.. **["Metabolic pathway inference using non-negative matrix factorization with community detection."](https://doi.org/10.1101/2020.05.27.119826)**, bioRxiv (2020).

## Contact
For any inquiries, please contact: [arbasher@alumni.ubc.ca](mailto:arbasher@alumni.ubc.ca)
