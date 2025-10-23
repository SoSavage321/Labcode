# Labcode

# Machine Learning LAB Assignment

## Question 1

### 1.1)
Supervised learning is a machine learning approach where a model is trained on a dataset containing input-output pairs to predict outcomes for new, unseen data. In the context of books, labeled examples are required because supervised learning relies on explicit mappings between inputs (e.g., book features like genre, author, or text content) and outputs (e.g., categories like "positive" or "negative" for sentiment analysis, or specific genres like "fiction" or "non-fiction"). These labels serve as the ground truth, guiding the model to learn patterns by minimizing prediction errors during training. Without labeled examples, the model lacks a reference to adjust its parameters, making it impossible to learn the relationship between book features and the desired output, resulting in ineffective predictions.

### 1.2) 
Clustering is classified as an unsupervised learning method because it involves grouping data points, such as books, based on their inherent similarities without requiring predefined labels or categories. In unsupervised learning, the algorithm (e.g., k-means or hierarchical clustering) analyzes features like book content, writing style, or metadata to identify patterns and form clusters. Unlike supervised learning, clustering does not rely on labeled data to guide the process; instead, it discovers structure within the data autonomously, making it suitable for exploratory analysis where prior knowledge of categories is absent.

### 1.3)
Reinforcement learning (RL) enhances simple trial-and-error by introducing a structured framework that optimizes decision-making through interaction with an environment. In simple trial-and-error, an agent randomly explores actions without a systematic way to evaluate or improve outcomes, often leading to inefficient learning. RL, however, employs a reward-based system where the agent learns by receiving feedback (rewards or penalties) for its actions, enabling it to develop a policy that maximizes cumulative rewards over time. For example, in a book recommendation system, RL can learn to suggest books by balancing exploration (trying new recommendations) and exploitation (using known preferences), guided by user feedback.

---

## Question 2

### 2.1)
Interpret the meaning of a prediction value of -1 for the book with features [3, 2] (3 marks) The prediction value of -1 for the book with features [3, 2] indicates that the perceptron classifies this book as belonging to the negative class. In the perceptron model, a binary classifier, the output is determined by the net input, calculated as w_0 + w_1*3 + w_2*2 (where w_0 is the bias and w_1, w_2 are weights for the features). If this net input is less than zero, the prediction is -1, as per the predict method (np.where(net_input >= 0.0, 1, -1)). In this context, features [3, 2] (e.g., representing book attributes like page count or publication year) fall on the negative side of the learned decision boundary, suggesting the book is classified as not belonging to the positive class (e.g., not recommended, not in a specific genre, or dissimilar to books labeled +1 in the training set).

### 2.2) 
  Errors per epoch: [2, 1, 2, 1, 1, 1, 0, 0, 0, 0]
Total errors across 10 epochs: 8

Each value shows the number of misclassifications in that epoch. The total of 8 means that during the 10 epochs, the perceptron made 8 weight updates before converging.

### 2.3) 
reaches zero by the seventh epoch, as shown in the output (errors_ = [2, 1, 2, 1, 1, 1, 0, 0, 0, 0]), because the perceptron has converged to a decision boundary that perfectly classifies all training examples (X = [[2,3], [1,1], [4,5]] with y = [1, -1, 1]). The perceptron algorithm updates weights (w_) using the learning rate (eta=0.1) whenever a misclassification occurs, adjusting the linear decision boundary to correctly separate the positive (+1) and negative (-1) classes. By the seventh epoch, the weights have been sufficiently adjusted to classify all three points correctly, resulting in no further updates (errors = 0). This convergence indicates that the dataset is linearly separable, meaning a single hyperplane can separate the classes. The relatively high learning rate (eta=0.1) and small dataset size facilitate rapid convergence, allowing the perceptron to find an optimal boundary by epoch 7, after which no misclassifications occur.

---

## Question 3

### 3.1) 
Both Logistic Regression and Linear SVM likely provide clear separation of the two classes (Setosa and Versicolor) in the Iris dataset, as these classes are known to be linearly separable based on petal length and width (features [2, 3]). The code uses X = iris.data[:100, [2,3]] and y = iris.target[:100], selecting the first two Iris classes, which are well-separated in this feature space. However, Linear SVM typically provides a clearer separation because it optimizes the maximum margin hyperplane, which maximizes the distance between the decision boundary and the nearest points (support vectors) of both classes. This results in a robust boundary that generalizes well, especially for linearly separable data like Setosa and Versicolor. Logistic Regression, while effective, minimizes a log-loss function to estimate class probabilities, which may place the decision boundary closer to the data points, potentially leading to a less distinct separation in some cases. The visualization (contour plots) would show SVM’s boundary as a straight line with wider margins, indicating clearer separation due to its margin maximization, whereas Logistic Regression’s boundary, though linear, may appear slightly less distinct if it prioritizes probability fitting over margin size.

### 3.2) 
Support Vector Machines (SVMs) with kernel functions offer several advantages over Logistic Regression, particularly for complex datasets:
1.	Non-linear separability: Kernel functions (e.g., polynomial or RBF kernels) allow SVMs to transform data into a higher-dimensional space, enabling the separation of non-linearly separable classes. Logistic Regression, being a linear model, cannot handle non-linear relationships without explicit feature engineering, limiting its flexibility.
2.	Robust decision boundaries: SVMs optimize the maximum margin hyperplane, focusing on support vectors (points near the boundary), which makes the model robust to outliers and noise. Logistic Regression, which optimizes log-loss across all data points, may be more sensitive to outliers, as every point influences the loss function.
3.	Flexibility in complex datasets: Kernel SVMs can model intricate patterns (e.g., curved or circular boundaries) by selecting appropriate kernels, making them suitable for datasets where classes are not linearly separable. Logistic Regression requires additional preprocessing (e.g., polynomial features) to achieve similar results, which is less automated.





---

## Question 4

### 4.1)
Feature scaling is critical in model training for algorithms sensitive to the magnitude or range of feature values, such as perceptrons, SVMs, and Logistic Regression, as it ensures optimal performance and convergence. The key reasons for its importance are:
1.	Improved convergence: Algorithms like gradient descent (used in Logistic Regression or neural networks) converge faster when features are scaled to similar ranges (e.g., [0, 1] or [-1, 1]). Unscaled features with large ranges can cause slow or unstable convergence due to disproportionate gradient updates.
2.	Equal feature contribution: Scaling ensures all features contribute equally to the model’s decision-making. For example, in the Iris dataset (petal length in cm vs. petal width in cm), unscaled features with different units or ranges could bias models like SVM, which rely on distance metrics, toward features with larger magnitudes.
3.	Distance-based algorithms: Models like SVM or k-nearest neighbors depend on distance calculations (e.g., Euclidean distance). Without scaling, features with larger scales dominate the distance, skewing results.
4.	Model performance: Proper scaling (e.g., standardization or min-max scaling) can improve accuracy and generalization by aligning data with the assumptions of algorithms like SVM, which assumes balanced feature contributions for optimal margin separation.


### 4.2) 
Categorical variables must be properly encoded (e.g., using one-hot encoding or ordinal encoding) before model fitting, as leaving them in an inappropriate encoded form (e.g., arbitrary numerical labels) can lead to several issues:
1.	Misinterpretation of ordinality: If categorical variables are encoded as integers (e.g., "red"=1, "blue"=2, "green"=3), models like Logistic Regression or SVM may incorrectly assume an ordinal relationship (e.g., green > blue > red), leading to erroneous patterns and poor predictions.
2.	Reduced model performance: Improper encoding can confuse the model, as it learns spurious relationships based on arbitrary numerical assignments. For instance, in the Iris dataset, if species were encoded as integers without one-hot encoding, the model might misinterpret the class relationships, lowering accuracy.


---

## Question 5

### 5.1)
The first two principal components (PCs) in PCA on the Iris dataset (iris.data, 150 samples, 4 features) capture 95% of the variance. PC1 typically explains 70–75% (dominant patterns like size), and PC2 adds 20–25% (shape variations). Assuming standardized data , the explained variance ratio sums to 95%, retaining most variability for 2D representation.

### 5.2)
PCA is effective for visualization because it:
1.	Reduces dimensions (e.g., 4D to 2D in X_pca), enabling 2D scatter plots.
2.	Captures maximum variance (95% for Iris), preserving data structure.
3.	Produces uncorrelated PCs, removing redundancy for clearer plots.
4.	Aligns with class-discriminating features, showing distinct clusters (e.g., Iris classes).
5.	Works with scaling ,ensuring unbiased feature contributions.

---

## Question 6

### 6.1)
Cross-validation offers a more reliable performance estimate than a single train/test split by reducing variance and maximizing data utilization. Unlike a single train/test split, which relies on one potentially unrepresentative data partition due to randomness, cross-validation (e.g., k-fold) averages performance across multiple splits, such as 5 or 10 folds, thereby minimizing variability and bias in the estimate. Additionally, in k-fold cross-validation, each data point is used for both training and testing across different folds, ensuring efficient use of the dataset (e.g., 90% training, 10% testing in 10-fold CV) compared to a single split (e.g., 80/20), which may exclude critical patterns, especially in small datasets like the Iris dataset used in prior questions. This approach provides a robust assessment of model generalization, making cross-validation particularly effective for models like SVM or Logistic Regression 

### 6.2)
Hyperparameter tuning prevents overfitting by optimizing model complexity to achieve a balance between fitting the training data and generalizing to unseen data. By adjusting parameters such as regularization strength or learning rate tuning controls the model’s capacity, preventing it from memorizing training data. Techniques like grid search combined with cross-validation select hyperparameters that minimize validation error, ensuring models generalize well, as seen with the Iris dataset. Proper tuning balances bias and variance, avoiding overly complex models that overfit or overly simple models that underfit, such as by optimizing C in SVM for appropriate regularization. Additionally, parameters like the L2 penalty in Logistic Regression or kernel choice in SVM enforce regularization by penalizing large weights or limiting decision boundary complexity, further reducing overfitting risks.

---

## Question 7

### 7.1)
Ensemble models, such as RandomForestClassifier and AdaBoostClassifier, often outperform individual models because they combine multiple weak learners to create a more robust and accurate predictor. First, ensembles reduce variance by averaging predictions (e.g., bagging in Random Forest), mitigating overfitting common in single models like decision trees, which are sensitive to noise. Second, they reduce bias by iteratively focusing on misclassified instances (e.g., boosting in AdaBoost), improving performance on complex datasets like Wine. Third, ensembles leverage diversity through techniques like random feature selection (Random Forest) or weighted sampling (AdaBoost), capturing varied patterns that a single model might miss. Fourth, they enhance generalization by balancing errors across diverse models, leading to better performance on unseen data. Finally, ensembles are robust to outliers and noise, as errors from individual models are diluted, making them effective for datasets like Wine with multiple features and classes.

### 7.2) 
Both Random Forest (bagging) and AdaBoost (boosting) achieved perfect accuracy of 1.0 on the Wine dataset (178 samples, 13 features, 3 classes), indicating neither model outperformed the other. This equal performance suggests that the Wine dataset, which is relatively clean and well-structured with distinct class boundaries, is well-suited to both ensemble methods. Random Forest, using multiple independent decision trees with random feature subsets, effectively captures the dataset’s high-dimensional patterns and reduces variance, making it robust for the Wine dataset’s correlated features. AdaBoost, by sequentially training weak learners (e.g., decision stumps) and focusing on misclassified instances, also achieves perfect classification, likely due to the dataset’s clear separability and low noise, allowing its iterative corrections to converge effectively. The perfect accuracy for both models indicates that the Wine dataset’s structure aligns well with both bagging’s parallel approach and boosting’s sequential refinement, with no discernible performance difference in this case.

---

## Question 8

### 8.1)
The sentiment prediction for the sentence "The book was good" is likely positive (1). The code trains a Logistic Regression classifier on a small dataset (docs = ["I loved the book", "Terrible story", "Amazing characters"], y = [1, 0, 1]), where 1 represents positive sentiment and 0 represents negative sentiment. The TfidfVectorizer transforms the text into TF-IDF features, capturing word importance (e.g., "loved" and "amazing" for positive sentiment, "terrible" for negative). The sentence "The book was good" contains the word "good," which, while not in the training set, is semantically similar to positive words like "loved" and "amazing." Logistic Regression predicts based on learned feature weights, and since "good" aligns with positive sentiment, the model likely assigns a positive probability, outputting 1. The prediction clf.predict(tfidf.transform(["The book was good"])) relies on the TF-IDF representation, and given the training data’s positive bias for similar terms, the model is expected to classify this sentence as positive.

### 8.2)
Text preprocessing, such as tokenization and stopword removal, is crucial for classification accuracy because it enhances the quality of features used by models like Logistic Regression in the provided code. First, tokenization breaks text into individual words or tokens (e.g., splitting "I loved the book" into ["I", "loved", "the", "book"]), creating meaningful units for feature extraction, allowing TfidfVectorizer to assign weights to relevant terms. Without tokenization, raw text would be uninterpretable by the model. Second, stopword removal eliminates common words (e.g., "the," "was" in "The book was good") that carry little sentiment information, reducing noise and focusing the model on discriminative words like "good" or "terrible." This improves the TF-IDF representation by emphasizing content-rich terms, enhancing class separation (e.g., positive vs. negative sentiment). Third, preprocessing ensures consistency (e.g., lowercasing, removing punctuation), preventing redundant features (e.g., "Book" vs. "book"). Fourth, it reduces dimensionality, making the model computationally efficient and less prone to overfitting, especially with small datasets like the three-document example. Collectively, these steps improve feature quality, leading to higher accuracy in sentiment classification.

---

## Question 9

### 9.1) 
The Random Forest Regressor achieved a lower Root Mean Squared Error (RMSE) of 0.2237 compared to Linear Regression’s RMSE of 0.7241, as shown in the output. Random Forest outperforms Linear Regression because it is an ensemble method that combines multiple decision trees, each trained on random subsets of the data and features, reducing overfitting and capturing complex, non-linear relationships in the dataset (shape: 20640, 8). Linear Regression assumes a linear relationship between the 8 features and the target, which may not hold for datasets with non-linear patterns or interactions, leading to higher prediction errors (RMSE). Random Forest’s ability to model non-linearities and feature interactions, along with its robustness to noise through averaging predictions across trees, results in a significantly lower RMSE, indicating better predictive accuracy on this dataset.

### 9.2)
Random Forest Regressor can outperform Linear Regression on certain datasets due to its flexibility and robustness. First, Random Forest handles non-linear relationships, modeling complex patterns in datasets (e.g., the provided dataset with 8 features) where Linear Regression’s linear assumption fails, leading to underfitting. Second, it captures feature interactions automatically, as decision trees split based on combinations of features, whereas Linear Regression requires explicit interaction terms. Third, Random Forest reduces overfitting through bagging and random feature selection, averaging predictions across diverse trees, unlike Linear Regression, which may struggle with high-dimensional or noisy data. Fourth, it is robust to outliers, as individual tree errors are diluted, while Linear Regression is sensitive to extreme values. Finally, Random Forest scales well with high-dimensional datasets like the one with 20640 samples and 8 features, leveraging ensemble diversity to improve accuracy, making it superior for complex, non-linear, or noisy datasets.

---

## Question 10

### 10.1)
) The K-means clusters likely align closely but not perfectly with the true Iris species labels (Setosa, Versicolor, Virginica). The code applies K-means clustering with n_clusters=3 to iris.data (150 samples, 4 features: sepal length, sepal width, petal length, petal width). The Iris dataset has three species, with Setosa being linearly separable from Versicolor and Virginica, which are partially overlapping. K-means, which groups data based on feature similarity (Euclidean distance), typically forms clusters that correspond well to Setosa (distinct in petal features) but may misclassify some Versicolor and Virginica samples due to their overlap. For example, the output km.labels_[:10] (e.g., [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for the first 10 samples, all Setosa) likely shows one cluster aligning with Setosa, as its features are distinct. However, for the remaining samples (Versicolor and Virginica), K-means may produce some mismatches, as it does not use true labels (iris.target) and relies solely on feature proximity. Standard evaluations (e.g., adjusted Rand index) show K-means achieves ~80–90% alignment with true labels on Iris, with errors in the Versicolor-Virginica boundary, indicating close but imperfect alignment due to the dataset’s structure and K-means’ unsupervised nature.
### 10.2)
Clustering, such as K-means in the provided code, is categorized as an unsupervised learning method because it groups data without using predefined labels. First, K-means operates on iris.data without reference to iris.target, relying solely on feature similarity (e.g., Euclidean distance) to assign samples to clusters, unlike supervised methods (e.g., Logistic Regression) that use labeled data. Second, it discovers inherent patterns or structures in the data (e.g., grouping Iris samples by feature proximity), making it suitable for exploratory analysis where labels are unknown, as reiterated . Third, clustering does not require a target variable, allowing it to handle datasets with no prior class information, unlike supervised learning, which needs labeled examples . Fourth, K-means iteratively optimizes cluster assignments based on centroids, learning without explicit guidance, which defines unsupervised learning. This autonomy enables clustering to identify natural groupings, such as potential species in the Iris dataset, but may lead to misalignments (as in 10.1) without label supervision.


---
