# Homework 1 Answers and Code Explanation

## Exercise 1

The gradient is \(L'(\theta)=2(\theta-3)\), so the update is \(\theta_{k+1}=\theta_k-2\eta(\theta_k-3)\). Since \(L''(\theta)=2>0\), the function is strictly convex and has the unique minimizer \(\theta^*=3\). The notebook now tries several initial values for \(\theta_0\), evaluates them with a stable reference step size, selects the one with the smallest final loss, and then uses that selected value for the required step-size comparison.

For \(\eta=0.05\), convergence is slow because the multiplier around the minimizer is \(0.9\). For \(\eta=0.2\), convergence is faster because the multiplier is \(0.6\). For \(\eta=1.0\), the multiplier is \(-1\), so the method oscillates instead of converging.

Code explanation: `L` evaluates the quadratic loss. `grad_L` returns the analytic derivative. `gradient_descent_1d` stores iterates and losses while applying the gradient update. `theta0_candidates` contains the trial initial points. `theta0_scores` stores the final loss for each trial. `selected_theta0` chooses the best trial. `etas` contains the required step sizes. `runs` stores the final experiments. The plotting loop draws real-line iterates and loss curves. The print block reports the trials, selection, and final values.

## Exercise 2

The derivative is \(L'(\theta)=4\theta^3-6\theta\). The stationary points are \(0\) and \(\pm\sqrt{3/2}\); the two nonzero points are local minima. Because the objective is non-convex, different starts can converge to different minima. The notebook now tries several initial Armijo step values \(\eta_0\), selects the one giving the best average final loss and step count, and uses it for the plotted trajectories.

Code explanation: `L` and `grad_L` define the polynomial and derivative. `backtracking` applies the Armijo condition. `gd_backtracking_1d` runs gradient descent with the selected backtracking rule. `starts` contains the required initial points. `eta0_candidates` and `eta0_scores` test unspecified backtracking choices. The selected value is used to draw the function and all trajectories.

## Exercise 3

For \(L(\theta)=\frac12\theta^TA\theta\), the gradient is \(A\theta\). The matrix has eigenvalues \(1\) and \(25\), so the condition number is \(25\), producing elongated level sets. The notebook tries several initial points and selects one with a large initial loss so the ill-conditioned trajectory is visible. The step size \(0.1\) exceeds the stability bound \(2/25\), so it diverges.

Code explanation: `A` stores the quadratic matrix. `L` computes the quadratic form. `grad_L` computes \(A\theta\). `gradient_descent_2d` stores all 2D iterates. `quad_levelsets` evaluates the loss on a grid and draws contours. `theta0_candidates` and `theta0_scores` test initial points. The selected point is used for all three required step sizes.

## Exercise 4

Exact line search uses \(\eta_k=(g_k^Tg_k)/(g_k^TAg_k)\), the minimizer along the negative-gradient direction. Backtracking instead searches for a step satisfying sufficient decrease. The starting point is specified, so the notebook varies the unspecified backtracking initial step \(\eta_0\), selects the best one, and compares both methods.

Code explanation: `gd_exact` computes the analytic exact-line-search step. `gd_backtracking` uses Armijo backtracking. `eta0_candidates` tests possible initial backtracking step sizes. `selected_eta0` is used in the final comparison. The first plot overlays trajectories on level sets, and the second plot compares loss decay.

## Exercise 5

The Rosenbrock gradient is \([ -2(1-x)-400x(y-x^2),\;200(y-x^2)]^T\). The valley is narrow and curved, so fixed steps can be slow, oscillatory, or unstable. The exercise already specifies several initial points and fixed step sizes; the notebook additionally tries several backtracking initial step values and selects the best average performer.

Code explanation: `rosenbrock` evaluates the objective. `grad_rosenbrock` evaluates the gradient. `gd_constant` runs fixed-step descent. `gd_backtracking` runs adaptive descent. `initial_points` and `etas` contain the required values. `eta0_candidates` tests the unspecified backtracking parameter. The selected value is used for contour trajectories, loss curves, and the final printed results.
