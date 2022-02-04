# Reproducibility of Exacerbating Algorithmic Bias through Fairness Attacks
This repository contains code for reproducing the experiments in the [Exacerbating Algorithmic Bias through Fairness Attacks](https://arxiv.org/pdf/2012.08723.pdf) paper as part of the [Machine Learning Reproducibility Challenge](https://paperswithcode.com/rc2021).
Please cite the original paper if you find this useful:
```
@article{mehrabi2020exacerbating,
  title={Exacerbating Algorithmic Bias through Fairness Attacks},
  author={Mehrabi, Ninareh and Naveed, Muhammad and Morstatter, Fred and Galstyan, Aram},
  journal={arXiv preprint arXiv:2012.08723},
  year={2020}
}
```


# Requirements
The code was tested using the environment we provide in 'f_attack_env.yml'.
It is highly recommended to work in that environment by installing it using Anaconda.
To do this install Anaconda and simply run

```bash
conda env create -f f_attack_env.yml
```

```bash
conda activate f_attack
```

# Running Instructions

To replicate all our experimental results specified in the paper by evaluating the pre-trained models run the jupyter notebook 'Results_notebook.ipynb'.

To do this don't forget to first run ```python -m ipykernel install --user --name=f_attack``` to create the necessary kernel.

To retrain all these models from scratch simply run 'run_experiments.py'.

## Data preprocessing

The three datasets (german, compas, drug) used for the experiments are available in 'Custom_data_preprocessing' downloaded from the source.
Running the according data processing scripts creates a 'datasetname_data.npz' and a 'datasetname_group_label.npz' file in 'Fairness_attack/data'.

The 'datasetname_data.npz' file contains the full training and test data. <br/>
The 'datasetname_group_label.npz' file contains the labels of the sensitive feature (0,1) for all data points. In the case of our experiments Male=0, Female=1.

Running these scripts is not necessary since the processed data is already given in the repository.
The scripts are given to provide easy access to the preprocessing to be adapted to other datasets.

To use different datasets the files have to be placed in the same way in the 'Fairness_attack/data' directory.

## Run attacks

To get information on all arguments you can run ```python run_gradient_em_attack.py -h ```.

Run the following commands to use the different attacks on the 'german' dataset with a particular set of hyperparameters.

Note: The sensitive_feature_idx is set to be 0 by our preprocessing for all three datasets.

To run the influence attack on fairness (IAF):
```bash
python run_gradient_em_attack.py --total_grad_iter 10000 --dataset german --epsilon 0.5 --lamb 1 --method IAF --sensitive_feature_idx 0 --stopping_method Accuracy
```

To run the random anchoring attack (RAA):
```bash
python run_gradient_em_attack.py --total_grad_iter 10000 --dataset german --epsilon 0.5 --method RAA --sensitive_feature_idx 0 --stopping_method Accuracy
```

To run the non-random anchoring attack (NRAA):
```bash
python run_gradient_em_attack.py --total_grad_iter 10000 --dataset german --epsilon 0.5 --method NRAA --sensitive_feature_idx 0 --stopping_method Accuracy
```

To run the Koh baseline:
```bash
python run_gradient_em_attack.py --total_grad_iter 10000 --dataset german --epsilon 0.5 --method Koh --sensitive_feature_idx 0 --stopping_method Accuracy
```

To run the Solans baseline:
```bash
python run_gradient_em_attack.py --total_grad_iter 10000 --dataset german --epsilon 0.5 --method Solans --sensitive_feature_idx 0 --stopping_method Accuracy
```

## Eval mode

To evaluate a trained model simply pass ``` --eval_mode True ```.
One specifies which model to evaluate by the attack, dataset, and hyperparameters used for the model.

For example, the following evaluates the model which used the 'IAF' attack, was trained on the 'german' dataset with 'epsilon = 0.2' 'lamb = 1' and uses accuracy as the stopping method:

```bash
python run_gradient_em_attack.py --eval_mode True --dataset german --epsilon 0.2 --method IAF --sensitive_feature_idx 0 --lamb 1 --stopping_method Accuracy
```



# References

This code builds upon the [implemenation](https://github.com/Ninarehm/attack) developed by Mehrabi et al thus please cite:

```
@article{mehrabi2020exacerbating,
  title={Exacerbating Algorithmic Bias through Fairness Attacks},
  author={Mehrabi, Ninareh and Naveed, Muhammad and Morstatter, Fred and Galstyan, Aram},
  journal={arXiv preprint arXiv:2012.08723},
  year={2020}
}
```


Their code intern builds on the implementation of Pang Wei Koh and Percy Liang in 2017. <br/>
As Mehrabi et al we have left their LICENSE.md file to give due credit to these researchers, and to document that their license allows us to build upon their work. <br/>
Note: the 'Koh' baseline is also to be credited to these authors.

Please give them credit by citing:

 ```
@article{koh2018stronger,
  title={Stronger data poisoning attacks break data sanitization defenses},
  author={Koh, Pang Wei and Steinhardt, Jacob and Liang, Percy},
  journal={arXiv preprint arXiv:1811.00741},
  year={2018}
}
 ```
 ```
@inproceedings{koh2017understanding,
  title={Understanding black-box predictions via influence functions},
  author={Koh, Pang Wei and Liang, Percy},
  booktitle={Proceedings of the 34th International Conference on Machine Learning-Volume 70},
  pages={1885--1894},
  year={2017},
  organization={JMLR. org}
}
 ```

If you find the influence attack on fairness useful you may also cite:
 ```
@article{zafar2015learning,
  title={Learning fair classifiers},
  author={Zafar, Muhammad Bilal and Valera, Isabel and Rodriguez, Manuel Gomez and Gummadi, Krishna P},
  journal={stat},
  volume={1050},
  pages={29},
  year={2015}
}
 ```

For the Solans baseline attack please cite:
```
@misc{solans2020,
      title={Poisoning Attacks on Algorithmic Fairness},
      author={David Solans and Battista Biggio and Carlos Castillo},
      year={2020},
      eprint={2004.07401},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
 ```

The citations of the datasets are as follows:
  For German and Drug consumption datasets cite:
 ```
@misc{Dua:2019 ,
author = "Dua, Dheeru and Graff, Casey",
year = "2017",
title = "{UCI} Machine Learning Repository",
url = "http://archive.ics.uci.edu/ml",
institution = "University of California, Irvine, School of Information and Computer Sciences" }
 ```
 For the COMPAS dataset cite:
 ```
@article{larson2016compas,
  title={Compas analysis},
  author={Larson, J and Mattu, S and Kirchner, L and Angwin, J},
  journal={GitHub, available at: https://github. com/propublica/compas-analysis[Google Scholar]},
  year={2016}
}