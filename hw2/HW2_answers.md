# Homework 2 Answers and Code Explanation

## Exercise 1

The model is \(f_\theta(x)=\theta_0+\theta_1x\), and with the bias design matrix \(X\), the loss is \(L(\theta)=\frac1N\|X\theta-y\|_2^2\). The gradient is \(\frac2N X^T(X\theta-y)\). The batch sizes are specified, but the learning rate and number of epochs are not, so the notebook tries several values and selects the pair with the smallest trial loss before comparing the required batch sizes.

Code explanation: `x`, `y`, `X`, and `Y` build the synthetic regression problem. `l` computes MSE. `grad_l` computes the exact gradient formula. `sgd` shuffles data, loops over mini-batches, updates parameters, and stores full loss and parameter path. `lr_candidates` and `epoch_candidates` perform the unspecified-parameter trials. `selected_lr` and `selected_epochs` are used for the final GD/SGD comparison.

## Exercise 2

The exercise only says to fix a parameter vector, so the notebook tests several possible \(\theta\) values and selects one with a nontrivial full-gradient norm. At that same selected \(\theta\), stochastic gradients are sampled repeatedly. Larger mini-batches average more samples, so the empirical variance decreases and becomes nearly zero when the batch is the full dataset.

Code explanation: `theta_candidates` contains trial vectors. `theta_scores` measures full-gradient norms. `selected_idx` chooses the vector used in the variance experiment. The loop over `batch_sizes` samples 100 mini-batches, stores their gradients, computes their mean, and evaluates the empirical variance formula.

## Exercise 3

The stochastic update uses \(g_k=\nabla L(\theta_k)+\varepsilon_k\). The notebook now tries several initial points and step sizes, selects the best stable pair, and then plots trajectories for different noise levels. It also plots a separate comparison over step sizes. Small noise can help movement through difficult regions; too much noise can prevent settling near a minimizer.

Code explanation: `L` and `grad_L` define the 2D non-convex function. `noisy_gd` adds Gaussian perturbations to the gradient. `theta0_candidates` and `eta_values` test unspecified initial point and step size choices. The contour plot shows selected-step trajectories for multiple noise variances. The second plot compares losses for multiple step sizes.

## Exercise 4

The learning rate and batch sizes are specified, while the number of epochs is left open. The notebook tests several epoch counts using full GD and selects the one with the best final loss and gradient norm. All methods optimize the same standardized linear MSE, so they should converge to the same region; mini-batches oscillate more because their gradients are noisy.

Code explanation: `load_insurance` reads the Kaggle file when available and otherwise creates a compatible local dataset. `X_scaled` and `Y_scaled` standardize the task. `train` implements GD or SGD depending on `batch_size`. `epoch_candidates` tests possible run lengths. The final plots compare MSE and full-gradient norm, and the print block reports parameters and final diagnostics.
