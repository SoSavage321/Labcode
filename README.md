# Labcode

# Machine Learning LAB Assignment

## Question 1

### 1.1) Why supervised learning requires labeled examples of books (3 marks)
Supervised learning is a machine learning approach where a model is trained on a dataset containing input-output pairs to predict outcomes for new, unseen data. In the context of books, labeled examples are required because supervised learning relies on explicit mappings between inputs (e.g., book features like genre, author, or text content) and outputs (e.g., categories like "positive" or "negative" for sentiment analysis, or specific genres like "fiction" or "non-fiction"). These labels serve as the ground truth, guiding the model to learn patterns by minimizing prediction errors during training. Without labeled examples, the model lacks a reference to adjust its parameters, making it impossible to learn the relationship between book features and the desired output, resulting in ineffective predictions.

### 1.2) Why clustering is considered an unsupervised learning method (3 marks)
Clustering is classified as an unsupervised learning method because it involves grouping data points, such as books, based on their inherent similarities without requiring predefined labels or categories. In unsupervised learning, the algorithm (e.g., k-means or hierarchical clustering) analyzes features like book content, writing style, or metadata to identify patterns and form clusters. Unlike supervised learning, clustering does not rely on labeled data to guide the process; instead, it discovers structure within the data autonomously, making it suitable for exploratory analysis where prior knowledge of categories is absent.

### 1.3) How reinforcement learning improves upon simple trial-and-error (4 marks)
Reinforcement learning (RL) enhances simple trial-and-error by introducing a structured framework that optimizes decision-making through interaction with an environment. In simple trial-and-error, an agent randomly explores actions without a systematic way to evaluate or improve outcomes, often leading to inefficient learning. RL, however, employs a reward-based system where the agent learns by receiving feedback (rewards or penalties) for its actions, enabling it to develop a policy that maximizes cumulative rewards over time. For example, in a book recommendation system, RL can learn to suggest books by balancing exploration (trying new recommendations) and exploitation (using known preferences), guided by user feedback.

---

## Question 2

### 2.1) Meaning of a prediction value of -1 for features [3, 2] (3 marks)
A prediction value of -1 indicates that the perceptron classifies this book as belonging to the negative class. The perceptron output is determined by the net input \( w_0 + w_1 \times 3 + w_2 \times 2 \). If the result is less than zero, the model outputs -1. Thus, the book’s features [3, 2] fall on the negative side of the decision boundary, representing a non-positive class (e.g., not recommended or not matching a target category).

### 2.2) Errors per epoch
  Errors per epoch: [2, 1, 2, 1, 1, 1, 0, 0, 0, 0]
Total errors across 10 epochs: 8

Each value shows the number of misclassifications in that epoch. The total of 8 means that during the 10 epochs, the perceptron made 8 weight updates before converging.

### 2.3) Why the error rate reaches zero by the seventh epoch (4 marks)
By epoch 7, the perceptron has perfectly classified all training examples because the dataset is linearly separable. Each weight update during earlier epochs adjusted the boundary closer to optimal separation. Once the boundary correctly separates all points, no further updates are needed (errors = 0).

---

## Question 3

### 3.1) Which classifier provided clearer separation and why (5 marks)
Both Logistic Regression and Linear SVM separate the Iris Setosa and Versicolor classes well, as they are linearly separable using petal length and width. However, the **Linear SVM** provides a clearer separation because it maximizes the margin between classes, making it more robust to outliers. Logistic Regression, while effective, focuses on probability estimation, not margin maximization.

### 3.2) Why SVMs with kernels outperform Logistic Regression
1. **Non-linear separability:** Kernels (e.g., RBF, polynomial) enable SVMs to handle complex, non-linear data, unlike Logistic Regression.  
2. **Robust boundaries:** SVMs depend only on support vectors, making them resistant to noise.  
3. **Flexibility:** Kernel SVMs can model intricate decision boundaries automatically without explicit feature engineering.

---

## Question 4

### 4.1) Importance of feature scaling
Feature scaling ensures that all features contribute equally and speeds up convergence.  
- **Improved convergence:** Gradient-based algorithms like Logistic Regression perform better on scaled data.  
- **Equal feature contribution:** Prevents larger-scale features from dominating.  
- **Essential for distance-based models:** Especially important for SVM and k-NN.  
- **Enhances accuracy:** Produces balanced model performance.

### 4.2) Why categorical variables must be encoded
Categorical data must be encoded (e.g., one-hot encoding) to prevent the model from misinterpreting numeric labels as ordinal. Improper encoding can cause:  
- **Misinterpretation of order** (e.g., “red”=1, “blue”=2).  
- **Reduced model accuracy** due to misleading relationships.  

---

## Question 5

### 5.1) PCA variance explanation
The first two principal components (PCs) in the Iris dataset capture ~95% of total variance — PC1 explains ~70–75% and PC2 adds ~20–25%. This means 2D visualization retains most of the data’s structure.

### 5.2) Why PCA is effective for visualization
1. Reduces 4D data to 2D for easy plotting.  
2. Preserves 95% variance.  
3. Removes redundancy (uncorrelated PCs).  
4. Highlights class structure for visual clarity.  
5. Works best when features are scaled.

---

## Question 6

### 6.1) Why cross-validation gives better performance estimates
Cross-validation reduces variance by testing on multiple splits instead of one train/test division. Each fold uses different training and testing subsets, providing a more reliable generalization estimate.

### 6.2) How hyperparameter tuning prevents overfitting
Grid search with cross-validation finds parameters (e.g., C in SVM) that balance model complexity and generalization. Proper tuning avoids both underfitting and overfitting by selecting optimal regularization values.

---

## Question 7

### 7.1) Why ensemble models outperform single models
Ensemble models like **RandomForest** and **AdaBoost** combine multiple weak learners, reducing bias and variance:  
- **Bagging (RandomForest):** Decreases variance.  
- **Boosting (AdaBoost):** Focuses on misclassified samples to reduce bias.  
- **Improved generalization:** Errors average out across models.  

### 7.2) RandomForest vs AdaBoost on the Wine dataset
Both achieved perfect training accuracy (**1.0**) on the Wine dataset, showing equal performance due to the dataset’s clear separability. RandomForest reduced variance via bagging, while AdaBoost reduced bias through iterative correction. Both methods perfectly fit the clean, structured dataset.

---

## Question 8

### 8.1) Sentiment prediction for “The book was good”
Predicted **positive (1)**.  
TF-IDF identified “good” as semantically similar to “amazing” and “loved,” terms learned from positive samples, leading Logistic Regression to classify the sentence as positive.

### 8.2) Why text preprocessing improves classification
- **Tokenization:** Splits text into interpretable units.  
- **Stopword removal:** Removes irrelevant words (e.g., “the,” “was”).  
- **Lowercasing & punctuation removal:** Ensures consistency.  
- **Dimensionality reduction:** Focuses model on meaningful terms.  

---

## Question 9

### 9.1) Regression model performance comparison
Random Forest Regressor RMSE: **0.2237**  
Linear Regression RMSE: **0.7241**  
→ Random Forest outperforms Linear Regression by capturing non-linear patterns and reducing overfitting via ensemble averaging.

### 9.2) Why Random Forest can outperform Linear Regression
1. Handles non-linear relationships.  
2. Captures feature interactions automatically.  
3. Reduces overfitting through bagging.  
4. Robust to outliers.  
5. Scales effectively for high-dimensional data.

---

## Question 10

### 10.1) K-Means clustering results interpretation
K-Means clusters roughly align with Iris species but misclassify some Versicolor and Virginica due to overlap. Setosa is distinctly separated, while the other two overlap slightly. Alignment is typically **80–90% accurate** with natural clustering.

### 10.2) Why clustering is unsupervised
K-Means groups samples by similarity without labels. It autonomously finds structure using feature distances (e.g., Euclidean), relying solely on input data — making it a classic unsupervised method.

---
