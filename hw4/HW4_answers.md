# Homework 4 Answers and Code Explanation

## Exercise 1

The matrix dimensions are not fixed, so the notebook tries several non-square shapes and selects one whose reconstruction error is smallest. The SVD reconstruction error is near machine precision. Singular values are returned in descending order, and smaller singular values correspond to lower-energy directions.

Code explanation: `shape_candidates` contains possible matrix sizes. For each shape, the code computes an SVD reconstruction error and stores the matrix. `selected_shape` chooses the final matrix. `U`, `s`, and `VT` are the reduced SVD factors. `Sigma` forms the diagonal singular-value matrix. The plot displays the singular values.

## Exercise 2

The rank-\(k\) approximation keeps the first \(k\) singular triplets. The error decreases as \(k\) increases and reaches numerical zero at full rank. The Eckart-Young theorem explains why truncated SVD is optimal for rank-\(k\) approximation in Frobenius norm.

Code explanation: `rank_k_approx` computes \(U_k\Sigma_kV_k^T\). `ks` tests every possible rank for the selected matrix. The loop computes each approximation and its Frobenius error. The plot shows error versus rank.

## Exercise 3

The values of \(k\) are specified by the exercise. Low rank gives strong compression but lower visual fidelity; larger rank improves the image and reduces error. The compression factor compares the storage required by \(U_k\), \(\Sigma_k\), and \(V_k^T\) with the full image.

Code explanation: `load_image` loads the cameraman image when available and otherwise creates a compatible grayscale image. The SVD is computed once. The loop reconstructs the image for each required `k`, computes error and compression factor, and plots the reconstructions and diagnostic curves.

## Exercise 5

The train/test split ratio is not specified, so the notebook tries several ratios and selects the one with the best PCA-space separation for digits 3 and 4. PCA centers the training data, computes the reduced SVD, and projects onto the leading right singular vectors. Other digit pairs show different separation depending on visual similarity.

Code explanation: `load_digit_data` reads MNIST-style CSVs when available and otherwise uses scikit-learn digits. `split_data` performs a reproducible train/test split. `pca_digits` filters a digit pair, centers the training data, computes SVD, and projects train/test data. `train_ratio_candidates` chooses the split ratio. The plots show 2D and 3D PCA projections.

## Exercise 6

The logistic classifier learning rate is not specified, so each PCA classification run tests several values and selects the one with lowest training BCE. The linear classifier learns a boundary in PCA space, while the centroid classifier assigns by nearest class mean. Their performance differs when clusters are not equally shaped or well separated.

Code explanation: `sigmoid`, `bce`, and `grad_bce` implement logistic regression. `metric_values` computes accuracy, precision, recall, and confusion matrix. `train_pca_logistic_lr` trains with one learning rate. `train_pca_logistic` selects the learning rate. `run_pca_classifiers` prepares PCA data, trains the linear model, computes centroids, predicts test labels, and returns values for plotting and metrics.
