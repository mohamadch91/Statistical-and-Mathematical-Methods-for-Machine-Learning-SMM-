# Homework 2 Answers and Code Explanation

## Exercise 1

The data follow \(y=2x+1+\varepsilon\), and the model is \(f_\theta(x)=\theta_0+\theta_1x\). With the design matrix \(X=[1,x]\), the MSE is \(L(\theta)=\frac1N\|X\theta-y\|_2^2\), and its gradient is \(\nabla L(\theta)=\frac2N X^T(X\theta-y)\).

Full GD uses all samples for every update, so its curve is smooth. Mini-batch SGD uses partial data, so its direction is noisy but cheaper. Batch size 1 has the highest variance, batch size 10 is still noisy but effective, batch size 50 is smoother, and batch size \(N\) recovers GD.

Code explanation: the first lines generate the synthetic data and build `X` with a bias column. `l` computes MSE. `grad_l` computes the analytic gradient. `sgd` shuffles indices each epoch, selects mini-batches, applies \(\theta\leftarrow\theta-\eta g\), and stores the full loss and parameter path. `batch_sizes` contains the required mini-batches. The plots compare loss curves and parameter trajectories. The printed values show the learned intercept, slope, and final loss.

## Exercise 2

For a fixed \(\theta\), stochastic gradients differ because each mini-batch samples different data. The empirical variance \(\frac1{100}\sum_k\|g_k-\bar g\|^2\) decreases as batch size grows, since averaging more samples cancels noise. The batch size \(N\) gives the deterministic full gradient, so the variance is nearly zero.

Code explanation: `batch_sizes` selects the required values. `theta` fixes the point where all gradients are measured. The nested loop samples 100 batches, computes each mini-batch gradient, averages them, and evaluates the empirical variance formula. The plot shows variance against batch size. The printed output gives each variance and mean gradient.

## Exercise 3

The non-convex function has curved valleys and several stationary regions. The simulated stochastic gradient is \(g_k=\nabla L(\theta_k)+\varepsilon_k\). Small noise can help the trajectory avoid poor flat regions, while large noise prevents stable convergence because it keeps perturbing the direction even near a minimizer.

Code explanation: `L` evaluates the 2D non-convex objective. `grad_L` implements its two partial derivatives. `noisy_gd` adds Gaussian noise with variance `sigma2` to the true gradient and updates the parameter. The grid code evaluates the function for contour plotting. The plotting loop overlays trajectories and loss curves for different noise levels. The final print reports the final point and loss for each noise variance.

## Exercise 4

For the insurance task, the model is linear in standardized features. Standardization makes age, BMI, children, and charges comparable in scale, preventing one coordinate from dominating the gradient. GD gives smooth loss and gradient-norm curves because each update uses the whole dataset. SGD oscillates because each mini-batch gives an approximate gradient. Larger batches reduce noise but cost more per update. Since all methods optimize the same convex quadratic MSE, they converge to the same region.

Code explanation: `load_insurance` reads `data/insurance.csv` or `insurance.csv` if present; otherwise it creates a compatible local dataset so the notebook runs. `X_scaled` and `Y_scaled` standardize features and target. `X_model` adds the bias column. `l` and `grad_l` implement MSE and its gradient. `train` performs either GD or SGD depending on `batch_size`, storing full loss and full-gradient norm at the end of each epoch. `methods` defines GD and the three SGD batch sizes. The plots compare MSE and full-gradient norm. The printed lines report final parameters, final loss, and final gradient norm.
