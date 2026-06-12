# Homework 4 Answers and Code Explanation

## Exercise 1

The singular value decomposition writes \(A=U\Sigma V^T\), where singular values are non-negative and sorted in descending order by NumPy. The reconstruction error \(\|A-U\Sigma V^T\|_F\) is close to machine precision. Small singular values correspond to directions that contribute little to the matrix energy. Exact zeros are rare in floating-point arithmetic because computations are approximate.

Code explanation: `A` creates a non-square matrix. `np.linalg.svd(A, full_matrices=False)` computes the reduced SVD. `Sigma` converts singular values into a diagonal matrix. `A_rec` reconstructs the matrix. `err` measures Frobenius reconstruction error. The plot displays singular values, and the prints show dimensions, error, and values.

## Exercise 2

The rank-\(k\) approximation keeps only the first \(k\) singular triplets. Its error decreases as \(k\) increases and becomes zero at full rank up to numerical precision. The Eckart-Young theorem states that this SVD truncation is the optimal rank-\(k\) approximation in Frobenius and spectral norm.

Code explanation: `rank_k_approx` computes the SVD and returns \(U_k\Sigma_kV_k^T\). `ks` enumerates possible ranks. The loop computes each approximation and Frobenius error. The plot shows error versus rank, and the printed lines report each value.

## Exercise 3

Image compression works because many natural images have most energy in the first singular values. Low \(k\) gives strong compression but blurry reconstructions. Larger \(k\) improves fidelity and lowers error, but stores more values. The compression factor \(1-k(m+n+1)/(mn)\) compares storage for \(U_k\), \(\Sigma_k\), and \(V_k^T\) against the full image.

Code explanation: `load_image` tries to load the cameraman image from `skimage`; if unavailable, it creates a 512 by 512 grayscale test image. `np.linalg.svd` decomposes the image matrix. For each `k`, the code reconstructs \(X_k\), computes Frobenius error, computes compression factor, and displays the reconstruction. The final plots show reconstruction error and compression factor against \(k\).

## Exercise 5

PCA centers the data, computes the SVD of the centered training matrix, and projects onto the leading right singular vectors. For digits 3 and 4, the first two components often show visible separation. Other pairs differ: digits with similar shapes overlap more, while visually distinct digits separate better. With \(k=3\), the third component can reveal extra structure not visible in 2D.

Code explanation: `load_digit_data` reads MNIST-style CSVs if available, otherwise uses scikit-learn digits. `split_data` randomly separates train and test sets. `pca_digits` filters a digit pair, centers training data, computes reduced SVD, chooses the first `k` principal directions, and projects train and test data. The first plot shows digits 3 and 4 in 2D PCA space. The second plot repeats the experiment for other pairs. The 3D plot uses the first three components. The prints show projected shapes and leading singular values.

## Exercise 6

After PCA, a logistic classifier learns a linear boundary in the reduced plane. The centroid classifier assigns each projected test point to the nearest class mean. Logistic regression can adapt the boundary orientation using labels, while the centroid classifier is purely geometric. When clusters are compact and well separated, both work well; when clusters overlap or have different shapes, error patterns differ.

Code explanation: `sigmoid`, `bce`, and `grad_bce` define logistic regression on PCA features. `metric_values` computes accuracy, precision, recall, and confusion matrix. `train_pca_logistic` trains a mini-batch logistic classifier on `[1,z1,z2]`. `run_pca_classifiers` gets PCA features, trains logistic regression, computes centroids, predicts by both models, and returns all objects needed for plotting. The figure shows the linear boundary, centroids, test predictions, and highlighted mistakes. The printed metrics compare linear and centroid classifiers for digits 3 and 4 and then for other digit pairs.
