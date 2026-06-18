# Homework 3 Answers and Code Explanation

## Exercise 1

The Gaussian cluster size and optimizer settings are not fully specified, so the notebook tests several sample sizes, learning rates, and epoch counts. It selects a sample size that gives clearly separated clusters and optimizer settings that give low BCE and high accuracy. Logistic regression has a linear score \(\theta^T\tilde x\), so its decision boundary \(\theta^T\tilde x=0\) is a line.

Code explanation: `n_candidates` tests possible cluster sizes. `cluster_scores` measures separation between empirical class centers. `sigmoid`, `bce`, `grad_bce`, and `accuracy` implement logistic regression from scratch. `train_full_gd` runs full gradient descent. `lr_candidates` and `epoch_candidates` select optimization settings. The selected model is plotted with its decision boundary and training curves.

## Exercise 2

The batch sizes are specified, but the learning rate is not. The notebook tests several SGD learning rates on batch size 10, selects the best one, and then compares batch sizes 1, 10, and full batch. Small batches produce noisier gradients and less smooth curves; large batches are smoother but each update is more expensive.

Code explanation: `train_logistic_sgd` performs mini-batch logistic-gradient updates. `lr_candidates_sgd` and `lr_scores_sgd` select the learning rate. `results` stores one training run for each required batch size. The plots show BCE and accuracy against epoch.

## Exercise 3

Lowering the threshold to 0.3 predicts more positives, which usually raises recall and lowers precision. Raising it to 0.7 predicts fewer positives, which usually raises precision and lowers recall. The best threshold depends on the practical cost of false positives and false negatives.

Code explanation: `best_theta` uses the batch-size-10 logistic model. `metrics` computes probabilities, thresholds them, counts TP, FP, FN, TN, and derives accuracy, precision, recall, and F1. The loop evaluates thresholds 0.3, 0.5, and 0.7.

## Exercise 4

The SGD and Adam settings are specified by the exercise. The optional neural network has an unspecified hidden-layer width, so the notebook tries several hidden sizes and selects the one with the lowest trial loss. Standardization improves conditioning and keeps gradient magnitudes comparable. Adam usually adapts faster because it rescales updates using moment estimates.

Code explanation: `load_diabetes` reads the Kaggle file when available and otherwise creates a compatible binary classification dataset. The train/test split and standardization prepare the data. `train_sgd` and `train_adam` optimize logistic regression. The metrics loop evaluates both methods. `train_nn` implements the optional one-hidden-layer ReLU network. `hidden_candidates` selects the hidden width used in the final neural-network comparison.
