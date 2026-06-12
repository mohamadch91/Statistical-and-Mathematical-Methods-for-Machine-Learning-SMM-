# Homework 1 Answers and Code Explanation

## Exercise 1

The function is \(L(\theta)=(\theta-3)^2+1\). Its derivative is \(L'(\theta)=2(\theta-3)\), so the gradient descent iteration is \(\theta_{k+1}=\theta_k-2\eta(\theta_k-3)\). The unique minimizer is \(\theta^*=3\) because the function is strictly convex and \(L''(\theta)=2>0\).

For \(\eta=0.05\), the multiplier around the minimizer is \(1-2\eta=0.9\), so convergence is monotone but slow. For \(\eta=0.2\), the multiplier is \(0.6\), so convergence is faster. For \(\eta=1.0\), the multiplier is \(-1\), so the method oscillates and does not reduce the error. Convexity guarantees there is only one minimizer, but the step size still controls whether the iteration reaches it efficiently.

Code explanation: `np` and `plt` provide arrays and plots. `L(theta)` evaluates the quadratic loss. `grad_L(theta)` returns the analytic gradient \(2(\theta-3)\). `gradient_descent_1d` stores the current parameter and loss, repeats the update \(\theta-\eta\nabla L\), and returns all iterates. `etas` selects the three required learning rates. The plotting loop shows the iterate positions on a real line and the corresponding loss values against iteration. The final `print` displays the last parameter and loss for each step size.

## Exercise 2

For \(L(\theta)=\theta^4-3\theta^2+2\), the derivative is \(L'(\theta)=4\theta^3-6\theta\). Stationary points solve \(2\theta(2\theta^2-3)=0\), so \(\theta=0\) and \(\theta=\pm\sqrt{3/2}\). The point \(0\) is a local maximum, while the two symmetric points are local minima.

Different initializations can move toward different basins of attraction because the problem is non-convex. Backtracking starts from a candidate step and repeatedly shrinks it until the Armijo decrease condition is satisfied. A constant step can fail if it is too large near high curvature or too small in flatter regions.

Code explanation: `L` and `grad_L` encode the non-convex polynomial and its derivative. `backtracking` implements Armijo by checking whether the proposed update gives enough decrease. `gd_backtracking_1d` repeatedly computes the gradient, chooses a step size, updates the parameter, and stops when the gradient is small. The plotting code draws the function on `[-3,3]` and overlays the iterates from each starting point. The printed values show the final point, final loss, number of steps, and first selected step sizes.

## Exercise 3

Here \(L(\theta)=\frac12\theta^TA\theta\) with \(A=\operatorname{diag}(1,25)\), so \(\nabla L(\theta)=A\theta\). The condition number is \(25\), which gives elongated level sets. The largest eigenvalue controls the stability bound for constant-step gradient descent: stable convergence requires roughly \(0<\eta<2/25=0.08\). Therefore \(0.02\) is stable, \(0.05\) is close to the boundary in the stiff direction, and \(0.1\) diverges.

Code explanation: `A` stores the quadratic matrix. `L` computes \(\frac12\theta^TA\theta\). `grad_L` computes \(A\theta\). `gradient_descent_2d` stores all 2D iterates and losses. `quad_levelsets` evaluates the quadratic on a grid and draws contours. The loop runs the method for each step size and plots the paths on the level sets. The final print reports the last iterate and loss.

## Exercise 4

For a quadratic with positive definite \(A\), exact line search chooses \(\eta_k=(g_k^Tg_k)/(g_k^TAg_k)\), the minimizer of the loss along the negative-gradient direction. Backtracking only enforces sufficient decrease, so it is usually less exact but robust and cheaper to generalize. Exact line search usually decreases the quadratic faster, while backtracking step sizes are piecewise and depend on repeated shrinking.

Code explanation: `A`, `L`, and `grad_L` define the quadratic model. `backtracking` is the Armijo rule in vector form. `gd_exact` computes the optimal quadratic step size at each iteration. `gd_backtracking` chooses an Armijo step instead. `quad_levelsets` plots contours. The two methods are started from `(3,3)`, their paths are overlaid, and a semilog plot compares loss decay. The printed arrays show the early step sizes.

## Exercise 5

The Rosenbrock function \(L(x,y)=(1-x)^2+100(y-x^2)^2\) has global minimizer \((1,1)\). Its gradient is \([ -2(1-x)-400x(y-x^2),\; 200(y-x^2)]^T\). The valley is curved and narrow, so constant-step gradient descent may zig-zag, move very slowly, or diverge if the step size is not compatible with local curvature. Backtracking adapts the step size and is usually more robust, although it may still need many iterations.

Code explanation: `rosenbrock` evaluates the objective. `grad_rosenbrock` evaluates the two partial derivatives. `backtracking` chooses an Armijo step for the current point. `gd_constant` runs fixed-step gradient descent and stops if the loss becomes non-finite or too large. `gd_backtracking` runs the adaptive method until the gradient is small or the epoch limit is reached. `rosenbrock_contour` draws logarithmic level sets to reveal the valley. The first plotting loop overlays constant-step and backtracking trajectories from four initial points. The second plotting loop compares loss curves for three fixed step sizes and backtracking. The printed table gives final iterates, final losses, and iteration counts.
