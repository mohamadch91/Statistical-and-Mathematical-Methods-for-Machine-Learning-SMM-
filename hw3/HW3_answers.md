# Homework 3 Answers and Code Explanation

## Exercise 1

The two Gaussian clusters are linearly separable in most random draws because their centers are far apart. Logistic regression models \(P(y=1|x)=\sigma(\theta^T\tilde x)\), where \(\tilde x=[1,x_1,x_2]\). The decision boundary is \(\theta^T\tilde x=0\), which is a straight line in 2D because the score is linear in the features.

Code explanation: the first block creates class 0 around `[-2,-2]` and class 1 around `[2,2]`. `X_model` adds the bias column. `sigmoid` computes probabilities stably. `bce` computes binary cross-entropy. `grad_bce` computes \(X^T(\sigma(X\theta)-y)/N\). `accuracy` thresholds probabilities at 0.5. The training loop runs full GD. The plot shows the data and line \(\theta_0+\theta_1x_1+\theta_2x_2=0\). The printed line reports parameters, final loss, and accuracy.

## Exercise 2

Small batches make gradients noisier because they estimate the full average with fewer samples. This can make loss and accuracy oscillate, but each update is cheap. The full batch gives the smoothest curve, while batch size 10 is a compromise between speed and stability.

Code explanation: `train_logistic_sgd` initializes parameters, shuffles data, applies mini-batch logistic-gradient updates, and records full-dataset BCE and accuracy after each epoch. `batch_sizes` selects 1, 10, and \(N\). The plotting loop compares loss and accuracy curves. The printed output reports final values and parameters.

## Exercise 3

Threshold 0.5 is the default conversion from probability to class. Lowering the threshold to 0.3 predicts more positives, increasing recall but often decreasing precision. Raising it to 0.7 predicts fewer positives, increasing precision when positive predictions are correct but reducing recall. The best threshold depends on the cost of false positives and false negatives.

Code explanation: `best_theta` takes the batch-size-10 model. `metrics` computes probabilities, thresholds them, counts TP, FP, FN, TN, and derives accuracy, precision, recall, and F1. The loop evaluates thresholds 0.3, 0.5, and 0.7.

## Exercise 4

The diabetes pipeline uses standardized features, a bias term, BCE loss, and SGD or Adam. Normalization improves conditioning and keeps gradient magnitudes comparable across medical measurements. Adam uses first and second moment estimates, so it adapts learning rates coordinate by coordinate and often reduces loss faster than plain SGD. SGD can oscillate more because it uses the same learning rate for all coordinates and sees noisy batches.

Code explanation: `load_diabetes` reads the Kaggle CSV if available; otherwise it creates a compatible binary classification dataset. `train_test_split` makes train and test sets. `mu` and `sigma` standardize the train data and apply the same scaling to test data. `train_sgd` applies mini-batch logistic regression with learning rate \(10^{-3}\). `train_adam` implements the Adam recurrences for `m`, `v`, bias-corrected estimates, and the adaptive update. The plots compare SGD and Adam loss and accuracy. The metrics loop evaluates final test BCE, accuracy, confusion matrix, precision, recall, and F1.

Optional neural network: `train_nn` builds a one-hidden-layer ReLU network with sigmoid output. The forward pass computes hidden activations and probabilities. The backward pass applies BCE derivatives, ReLU derivatives, and gradient descent updates to both layers. The final plot compares logistic regression and the neural network losses.
