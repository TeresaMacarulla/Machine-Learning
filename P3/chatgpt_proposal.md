# **Project 3 — Understanding Path B: Classification with several methods**

After reviewing Path B in `Project3.ipynb` and the relevant sections of Chapters 2, 3, 5, 7, 8, 9, 10 and 11 of your lecture book, the main requirement is clear:

**Project 3 asks you to take one classification dataset and investigate how different families of machine-learning algorithms perform on exactly the same problem.**

You already developed one of these methods in Project 2: a feed-forward neural network (FFNN). The difference is that Project 3 requires you to compare it against other approaches that learn from data in fundamentally different ways.

## **1. What does the professor mean by "different methods"?**

Path B explicitly requires:

- At least **three different methods** covered in the course.
- At least **one neural network** (feed-forward or convolutional).
- At least **one tree-based ensemble** (such as random forest or gradient boosting).

Therefore, you cannot satisfy the requirement by training three different feed-forward neural-network architectures. Those are variations of the same method.

Here is the most straightforward combination for a dataset containing ordinary numerical features.

THREE-METHOD COMPARISON — SATISFIES PATH B

**Method 1 — Logistic regression**

A relatively simple linear classification model. It establishes a baseline to determine whether more complex algorithms are justified.

Chapter 5

**Method 2 — Feed-forward neural network**

Your Project 2 implementation. It learns nonlinear relationships through hidden layers and backpropagation.

Chapter 8

**Method 3 — Random forest**

An ensemble of decision trees trained on different samples and feature subsets. It combines their predictions to reduce the instability of individual trees.

Chapter 7

This combination would satisfy the requirement without forcing you to learn convolutional or recurrent networks if your chosen dataset does not need them.

There is a small inconsistency in the notebook: its example for tabular data suggests logistic regression, random forest and gradient boosting, but that combination contains no neural network. **I would follow the explicit mandatory requirement**, rather than that example.

## **2. What classification algorithms are available in the lecture book?**

The following table identifies the methods you could use, where they are explained, and when they are appropriate.

| Method                          | Chapter | Basic idea                                                         | Suitable data                           |
| ------------------------------- | ------- | ------------------------------------------------------------------ | --------------------------------------- |
| Logistic regression             | 5       | Learns a linear decision boundary and class probabilities          | Numerical/tabular features; baseline    |
| Decision tree                   | 7       | Repeatedly partitions data using feature thresholds                | Tabular data                            |
| Bagging                         | 7       | Trains many models on bootstrap samples and aggregates predictions | Particularly useful with unstable trees |
| **Random forest**               | **7**   | Combines bootstrapped decision trees with random feature selection | Tabular data                            |
| **Gradient boosting / XGBoost** | **7**   | Builds an additive ensemble, sequentially correcting errors        | Tabular data                            |
| **Feed-forward NN (MLP)**       | **8**   | Learns nonlinear transformations through hidden layers             | Tabular data or flattened inputs        |
| **Convolutional NN (CNN)**      | **10**  | Learns local patterns using shared convolutional filters           | Images and spatial data                 |
| Recurrent NN (RNN/LSTM)         | 11      | Processes ordered sequences while maintaining a hidden state       | Temporal or sequential data             |

A support vector machine (Chapter 6) is also explicitly welcomed as an additional method, but it is not mandatory.

Let's examine the most relevant alternatives to your neural network.

## **3. The three classical methods worth understanding**

### **A. Logistic regression — Chapter 5**

You have already used this method in Project 2, when you trained MNIST with architecture `[784, 10]`.

Its multiclass form is:

\\[ P=\operatorname{softmax}(XW+b) \\]

It has no hidden layers and is trained using cross-entropy.

Logistic regression makes a useful baseline because it has relatively few hyperparameters, its coefficients can be examined, and its optimization problem is convex.

For Project 3, you could reuse your code, although using scikit-learn's `LogisticRegression` would also be acceptable.

### **B. Random forest — Chapter 7**

This would be your main new method.

To understand random forests, first consider a single decision tree.

Suppose we are classifying observations using three numerical variables. A tree might learn rules such as:

Illustrative classification tree; the feature thresholds are learned from the training data.

A decision tree selects splits using a measure such as the **Gini impurity** or entropy, introduced in Section 7.3.

However, a single tree can be unstable: relatively small changes to the training data may result in a different tree.

A random forest addresses this by constructing many trees using bootstrapped training samples and considering a random subset of features at each split.

Each tree makes a prediction, and the forest aggregates their results.

The most important hyperparameters are the number of trees, maximum tree depth, minimum leaf size and number of candidate features per split.

**Why compare it against your neural network?**

Because the two models learn differently. A neural network adjusts continuous weights using gradient-based optimization, whereas a random forest constructs decision rules through recursive splitting and aggregation.

Random forests also generally require much less feature scaling than neural networks.

### **C. Gradient boosting — Chapter 7**

Gradient boosting is another tree-based ensemble, but it works differently from a random forest.

The key distinction is:

**Random forest**

Trees are trained independently and their predictions are aggregated.

**Gradient boosting**

Trees are added sequentially to improve the current model.

This is the distinction between bagging/random forests and boosting discussed in Sections 7.8–7.14.

Gradient boosting is particularly interesting for tabular datasets because it can learn nonlinear interactions and often produces strong predictive performance. Its hyperparameters include the number of boosting iterations, learning rate and tree complexity.

You could compare random forest and gradient boosting as two separate methods because they use different ensemble-learning strategies.

## **4. What about convolutional and recurrent neural networks?**

These are also classification methods, but whether they are appropriate depends heavily on your chosen dataset.

### **Convolutional neural networks — Chapter 10**

[AcaDraw - Outil IA pour dessins scientifiques | Illustrations en quelques secondes](https://images.openai.com/static-rsc-4/DKs-srh42CuSYoBKJCOkuegFPmjI13YHdv5IBf02BC5fq9sqaCfRIgxLGAOSG_JZPrGtjv5S-gsxJ9KKYthKyVMliQaV_1QXiJbzx2pqgL18YQSec4C43LaIcjPXO2qW3S4bbpY-MswedZ3ZbEpPMINUL3WORjAZxaFdtAibDaI?purpose=inline)

In Project 2, you flattened each MNIST image into 784 features. The feed-forward network had no explicit information about which pixels were neighbors.

A CNN instead works with the spatial organization of the image, applying small filters across it. This allows it to reuse learned features, such as edges or shapes, in different locations.

A CNN is a natural additional method if you choose an image-classification dataset.

However, for a dataset consisting of ordinary measurements—for example, 20 physical variables measured for each sample—there is usually no reason to impose a two-dimensional convolutional architecture.

### **Recurrent neural networks — Chapter 11**

RNNs process ordered observations. They maintain an internal state that is updated as they read successive elements of a sequence.

They are useful when the order of observations carries information, for example when classifying a time-dependent sensor signal. Chapter 11 also discusses LSTMs and GRUs, which address some difficulties in learning long-term dependencies.

A recurrent network would make sense for classification of sequences, but it would not be my default choice for independent tabular observations.

### **And Chapter 9?**

**Chapter 9 is not directly relevant to Path B.**

It discusses using neural networks to solve differential equations, including physics-informed neural networks. Its main role in Project 3 is to support Path A.

## **5. What are Chapters 2 and 3 doing in this project?**

These chapters provide the statistical methodology needed to make your comparisons scientifically valid.

| Chapter                        | Relevant content                                                 | Application to Project 3                                                                 |
| ------------------------------ | ---------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| 2 — Probability and statistics | Generalization error, bias–variance, bootstrap, cross-validation | Estimate uncertainty and compare models reliably                                         |
| 3 — Linear regression          | Regularization, scaling, model selection, cross-validation       | Understand regularization and use appropriate preprocessing and hyperparameter selection |
| 5 — Logistic regression        | Classification losses and performance metrics                    | Establish a classifier baseline and measure classification quality                       |
| 7 — Trees and ensembles        | Random forest and boosting                                       | Implement a fundamentally different family of classifiers                                |
| 8 — Neural networks            | MLPs, backpropagation and regularization                         | Reuse and extend Project 2                                                               |
| 9 — Differential equations     | Neural differential-equation solvers                             | Primarily Path A                                                                         |
| 10 — CNNs                      | Convolutions and spatial structure                               | Additional method if your dataset contains images                                        |
| 11 — RNNs                      | Sequential learning and recurrent states                         | Additional method if your dataset contains sequences                                     |

An important distinction: **OLS, Ridge and Lasso as presented in Chapter 3 are regression methods, not your natural classification candidates.** Their ideas about regularization are useful, but multinomial logistic regression is the appropriate linear classification baseline.

Also, the notebook refers to Section 5.12 for classifier quality measures; in the supplied September 2026 lecture book, the relevant section is **5.11, Measuring the quality of a classifier**.

## **6. Which combination should you actually choose?**

The choice depends on the structure of your dataset.

| Dataset type                             | My recommended methods                                                    | Why                                                          |
| ---------------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------ |
| **Tabular numerical data**               | Logistic regression + Random forest + Feed-forward NN                     | Three distinct families; easy to compare                     |
| Tabular data, expanded study             | Logistic regression + Random forest + Gradient boosting + Feed-forward NN | Tests both kinds of tree ensembles                           |
| Images                                   | Feed-forward NN + Random forest + CNN                                     | Compares flattened-input methods with a spatial architecture |
| Time-series or sequential classification | Feed-forward NN + Random forest + RNN/LSTM                                | Tests whether modeling temporal order helps                  |

My preference would be **logistic regression, random forest and feed-forward neural network**, with gradient boosting as an optional fourth method.

It meets every explicit algorithm requirement, provides a strong comparison and reuses your Project 2 work without unnecessarily expanding the workload.

The dataset, however, should be chosen first. A good dataset may justify changing the combination.

## **7. What would you have to do beyond training these models?**

The professor is asking for a complete comparative classification study, not simply three accuracy scores.

Here is how I would organize Path B, following its parts (a)–(e).

1. **Understand the dataset — Part (a)**

   Describe where it comes from, how many samples and features it contains, what the classes represent, how balanced they are, and which preprocessing steps are necessary. Establish training, validation and test partitions.
2. **Study the algorithms — Part (b)**

   Explain the theoretical principles of the three methods, their loss or splitting criteria, optimization or training procedures, and important hyperparameters.
3. **Implement and optimize — Part (c)**

   Reuse your neural network where appropriate, implement or configure the additional algorithms, investigate hyperparameters through validation or cross-validation, verify their behavior, and record training times and model sizes. Present hyperparameter studies using curves or heatmaps.
4. **Compare final predictions — Part (d)**

   Evaluate every selected model on the same held-out test dataset. Report accuracy with uncertainty, confusion matrices, precision, recall, F1, and one-versus-rest ROC curves with AUC. Examine misclassified samples and tree feature importances, and compare your results with published benchmarks.
5. **Critical assessment — Part (e)**

   Decide which algorithm performed best and whether its improvement justifies the computational cost, complexity and lower interpretability. Discuss limitations and make a recommendation for similar problems.

One new requirement compared with Project 2 deserves particular attention: **quantifying uncertainty**.

For example, with \\(n\\) independent test observations and observed accuracy \\(\hat p\\), an approximate standard error is

\\[ \mathrm{SE}(\hat p)=\sqrt{\frac{\hat p(1-\hat p)}{n}}. \\]

For comparing two classifiers on the same observations, a paired bootstrap or another paired comparison can be more informative than treating their accuracies as independent.

You should also compare against published results for the chosen dataset, as the project explicitly requires.

## **8. Two important practical requirements**

The deadline stated in the assignment is **11 December 2026, midnight**. The expected report is approximately 5,000 words (typically 10–12 pages excluding references and appendices), with about 10–15 figures. Your GitHub repository must contain the report PDF, a `Code` directory and a reproducible README.

More importantly, **Project 3 changes the AI-use rules compared with Project 2**. The notebook explicitly limits generative AI assistance with report text to Level 1: spelling, grammar and minor phrasing. AI-generated drafting, rewriting or restructuring of report sections is not permitted. I can still explain the theory, review your analysis, help debug code and provide feedback, but you must write the substantive report text yourself. LLM-assisted code must carry the required inline tags and be understood and tested by you.

## **My recommendation for your next decision**

You already have a working neural-network implementation, experience with validation-based tuning, and code for accuracy, cross-entropy and confusion matrices.

For Project 3, the biggest new learning opportunity is **Chapter 7: decision trees, random forests and gradient boosting**.

I would choose a well-documented tabular classification dataset and compare:

\\[ \boxed{\text{Logistic regression}\quad\text{vs.}\quad\text{Random forest}\quad\text{vs.}\quad\text{Feed-forward NN}} \\]

This would let you investigate the question emphasized in Section 8.15 of the lecture book: *Are flexible neural networks actually the best method for every type of data, or can simpler and tree-based approaches perform equally well or better?*

It is a focused scientific question, and it provides a natural foundation for the critical evaluation that Project 3 expects.

The next useful step is to choose your dataset. Once you have one in mind, we can assess its suitability for Path B before you invest time implementing methods.