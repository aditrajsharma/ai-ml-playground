# AI/ML Playground

A collection of Jupyter notebooks exploring Python, data analysis, classical machine learning, and neural networks through small datasets and worked examples.

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange?logo=jupyter)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-imported%20for%20GPU%20check-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

## Table of contents

- [Repository structure](#repository-structure)
- [Setup and installation](#setup-and-installation)
- [Machine learning notebooks](#machine-learning-notebooks)
  - [Titanic classification, metrics, and cross-validation](#titanic-classification-metrics-and-cross-validation)
  - [GridSearchCV and RandomizedSearchCV](#gridsearchcv-and-randomizedsearchcv)
  - [Ensemble Learning.ipynb](#ensemble-learningipynb)
  - [Stacking,Bagging,Bossting.ipynb](#stackingbaggingbosstingipynb)
  - [Unsupervised learning](#unsupervised-learning)
  - [Heart disease classification and app](#heart-disease-classification-and-app)
  - [NLP preprocessing](#nlp-preprocessing)
  - [Ford car price regression](#ford-car-price-regression)
  - [Insurance cost regression](#insurance-cost-regression)
- [Deep learning notebooks](#deep-learning-notebooks)
  - [Binary classification with Keras layers](#binary-classification-with-keras-layers)
  - [MNIST perceptron, ANN, and CNN](#mnist-perceptron-ann-and-cnn)
  - [Iris classification with a perceptron and Keras](#iris-classification-with-a-perceptron-and-keras)
- [Python and data-analysis notebooks](#python-and-data-analysis-notebooks)
- [Mathematical formulations used](#mathematical-formulations-used)
- [Concepts covered](#concepts-covered)
- [Author and contact](#author-and-contact)

## Repository structure

```text
.
├── README.md                                      Project documentation
├── car_price prediction                           Extensionless JSON notebook with Ford price EDA cells
├── Data_analysis/
│   ├── Game_of_Thrones (1).csv                    Episode data used by pandasProject.ipynb
│   ├── demo.ipynb                                  Jupyter setup, NumPy, and abstract-class exercises
│   ├── matplot.ipynb                               Seaborn distribution and regression plots
│   ├── numpy.ipynb                                 NumPy arrays and standardization examples
│   ├── oop.ipynb                                   Abstraction, encapsulation, and polymorphism examples
│   ├── pandas.ipynb                                Series and DataFrame creation and indexing
│   ├── pandasProject.ipynb                         Episode CSV inspection and year extraction
│   └── stats.ipynb                                 One-sample, two-tailed Z-test example
├── Deep learning/
│   ├── Basic_Layer_making_in_DL.ipynb              Keras binary classifier using a small hand-entered dataset
│   ├── CNN.ipynb                                   MNIST perceptron, dense network, and convolutional network
│   └── DL_iris_prediction.ipynb                    Iris perceptron and dense neural-network classifiers
└── ML_learnings/
    ├── car_price_prediction.ipynb                  Ford used-car price regression and encoding comparison
    ├── Ensemble Learning.ipynb                     Iris stacking, random forest, and boosting comparison
    ├── GridSearch_CV.ipynb                          Iris KNN/SVC baselines and hyperparameter search
    ├── insurance.csv                               Medical insurance charges dataset
    ├── insurance.ipynb                             Insurance data cleaning, feature selection, regression
    ├── logistic.ipynb                              Titanic classification, model comparison, and CV
    ├── NLP.ipynb                                   Emotion-text cleaning and label encoding
    ├── Stacking,Bagging,Bossting.ipynb              Second Iris ensemble exercise
    ├── Unsupervised_learning.ipynb                  K-Means, DBSCAN, and PCA demonstrations
    └── Heart disease model/
        ├── app.py                                  Streamlit heart-disease prediction interface
        ├── columns.pkl                             Saved model feature-column order
        ├── heart.csv                               Heart disease dataset
        ├── heart.ipynb                             Data preparation and classifier comparison
        ├── logistic_regression_model.pkl            Serialized logistic-regression model
        ├── scaler.pkl                              Serialized feature scaler
        └── styles.css                              Streamlit app styling
```

The extensionless `car_price prediction` file contains Ford data exploration cells as notebook JSON but does not have the `.ipynb` suffix; it does not contain the regression section. The runnable, named Ford notebook is `ML_learnings/car_price_prediction.ipynb`.

## Setup and installation

Use Python 3 with a virtual environment. The package list below reflects imports in the notebooks and app; standard-library modules such as `os`, `string`, `abc`, and `warnings` are not installed separately.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install jupyter numpy pandas matplotlib seaborn scipy scikit-learn tensorflow torch nltk xgboost kagglehub streamlit joblib sheryanalysis==0.1.0
```

Start Jupyter from the repository root:

```powershell
jupyter lab
```

Run a notebook from the Jupyter file browser, or launch one directly, for example:

```powershell
jupyter lab "ML_learnings\GridSearch_CV.ipynb"
```

Some notebooks need data that is not committed here: the Ford notebook downloads through KaggleHub (and requires access to that dataset), the MNIST and Iris notebooks fetch CSV files, and `NLP.ipynb` expects a local `train.txt`. The NLP notebook also downloads NLTK stopwords and tokenizer data. The Streamlit app expects to be run with its model artifacts and CSS file in the current directory.

## Machine learning notebooks

### Titanic classification, metrics, and cross-validation

**Notebook:** `ML_learnings/logistic.ipynb`

- **Objective:** Predict Titanic survival, compare classifiers, and demonstrate five-fold cross-validation.
- **Dataset:** Seaborn's Titanic dataset; 891 rows, initially 15 columns. The notebook drops `deck`, `embark_town`, `alive`, `class`, `who`, and `adult_male`; fills missing `age` with its mean and removes rows with missing `embarked`, leaving 889 rows. Target: `survived`. The remaining inputs are `pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, `embarked`, and `alone`.
- **Concepts and formulation:** Scikit-learn `LogisticRegression` uses a sigmoid probability for the binary target. The notebook also fits K-nearest neighbors, Gaussian Naive Bayes, a decision tree, and an RBF SVC. Accuracy, precision, recall, F1, and cross-validation are covered in [Mathematical formulations used](#mathematical-formulations-used).
- **Implementation:** `LabelEncoder` encodes `sex` and `embarked`; the target is split with `test_size=0.2, random_state=42`. Logistic regression is fit on the unscaled training features. The later KNN, decision-tree, and SVC runs use `StandardScaler` fit on the training partition. KNN uses its defaults (`n_neighbors=5`); `GaussianNB` and `DecisionTreeClassifier(random_state=42)` use scikit-learn defaults; SVC uses `kernel='rbf', random_state=42` and other defaults. For the final five-fold SVC `cross_val_score(..., cv=5)` call, the notebook standardizes the full feature set before cross-validation.
- **Results:** The outputs report the following test accuracies and, for the displayed classification reports, the positive-class (survived) scores.

  | Classifier | Accuracy | Precision | Recall | F1 |
  |---|---:|---:|---:|---:|
  | Logistic regression | 0.8034 | 0.74 | 0.77 | 0.75 |
  | K-nearest neighbors | 0.7978 | 0.73 | 0.75 | 0.74 |
  | Gaussian Naive Bayes | 0.7753 | 0.68 | 0.78 | 0.73 |
  | Decision tree | 0.7416 | 0.64 | 0.75 | 0.69 |
  | RBF SVC | 0.8146 | 0.75 | 0.78 | 0.77 |

  The notebook's SVC five-fold accuracy scores are `0.8315`, `0.8202`, `0.8146`, `0.8090`, and `0.8644`.
- **How to run:** `jupyter lab "ML_learnings\logistic.ipynb"`. Seaborn must be able to load its Titanic dataset.

### GridSearchCV and RandomizedSearchCV

**Notebook:** `ML_learnings/GridSearch_CV.ipynb`

- **Objective:** Compare KNN and SVC baselines, then search their hyperparameters using five-fold cross-validation.
- **Dataset:** Seaborn Iris; 150 rows, four measurements (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`), and `species` as the three-class target.
- **Concepts and formulation:** `GridSearchCV` evaluates each supplied parameter combination across five folds and ranks by the classifier's default accuracy score. `RandomizedSearchCV` evaluates the requested number of sampled combinations.
- **Implementation:** The initial split uses `test_size=0.33, random_state=42`. KNN uses `n_neighbors=5`; SVC uses `C=30, gamma='auto', kernel='rbf'`. The SVC grid is `C=[1,10,20,30]` and `kernel=['rbf','linear']` (8 combinations). The KNN grid spans 3 neighbor counts, 4 algorithms, 3 metrics, and 2 weight modes (72 combinations). Both `GridSearchCV` instances are fit on the full Iris dataset after the initial train/test split. The randomized SVC search uses the same SVC grid and `n_iter=4`; it does not set a random seed.
- **Results:** Both baseline models report a held-out score of `0.98`. In the displayed SVC grid results, the mean five-fold score is `0.98` for `C=1` with either kernel and for `C=10` with the RBF kernel. The displayed randomized results include `0.98` for `C=1, kernel='linear'`. The KNN search table is shown, but the notebook does not print a selected best-score summary.
- **How to run:** `jupyter lab "ML_learnings\GridSearch_CV.ipynb"`.

### Ensemble Learning.ipynb

**Notebook:** `ML_learnings/Ensemble Learning.ipynb`

- **Objective:** Compare stacking, a random forest, AdaBoost, gradient boosting, and XGBoost on a multi-class task.
- **Dataset:** Seaborn Iris; 150 rows, four flower measurements, and `species` as the target. An 80/20 stratified split with `random_state=42` gives 120 training and 30 test rows.
- **Concepts and formulation:** Stacking trains a logistic-regression meta-estimator on cross-validated predictions from a decision tree, RBF SVC, and logistic regression. Random forest aggregates tree votes. AdaBoost, gradient boosting, and XGBoost build additive ensembles; the specific aggregation and update descriptions are in [Mathematical formulations used](#mathematical-formulations-used).
- **Implementation:** Base decision tree and `RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)` use the stated settings. The base SVC uses `probability=True, kernel='rbf', random_state=42`; base and final logistic regressions use `max_iter=1000, random_state=42`. Stacking uses `cv=5`. AdaBoost uses 100 estimators and learning rate 1.0 with scikit-learn's default decision-stump base estimator; `GradientBoostingClassifier` uses 100 estimators and learning rate 0.1. XGBoost uses 100 estimators, depth 3, learning rate 0.1, `random_state=42`, and `eval_metric='mlogloss'`; its multi-class objective is selected by the XGBoost classifier default.
- **Results:** Test-set accuracies in the notebook output:

  | Model | Accuracy |
  |---|---:|
  | Stacking | 0.9667 |
  | Random forest | 0.9000 |
  | AdaBoost | 0.9333 |
  | Gradient boosting | 0.9667 |
  | XGBoost | 0.9333 |

- **How to run:** `jupyter lab "ML_learnings\Ensemble Learning.ipynb"`. XGBoost must be installed.

### Stacking,Bagging,Bossting.ipynb

**Notebook:** `ML_learnings/Stacking,Bagging,Bossting.ipynb`

- **Objective:** Repeat the Iris ensemble comparison for stacking, random-forest bagging, and boosting.
- **Dataset:** Seaborn Iris; 150 rows, four measurements, and `species` target; the notebook uses a stratified 80/20 split with `random_state=42`.
- **Concepts and formulation:** Same scikit-learn estimators and ensemble formulations as [Ensemble Learning.ipynb](#ensemble-learningipynb). The filename spelling is retained as it appears in the repository.
- **Implementation:** Stacking uses a decision tree, probabilistic RBF SVC, and logistic regression as base learners, a logistic-regression final estimator, and five-fold internal CV. Random forest has 100 estimators and no depth limit. AdaBoost has 100 estimators and learning rate 1.0 with the default decision-stump base estimator; gradient boosting has 100 estimators and learning rate 0.1; XGBoost has 100 estimators, depth 3, learning rate 0.1, and `mlogloss`, with XGBoost's default multi-class objective.
- **Results:** The notebook reports the same test accuracies as the other ensemble exercise: stacking `0.9667`, random forest `0.9000`, AdaBoost `0.9333`, gradient boosting `0.9667`, and XGBoost `0.9333`.
- **How to run:** `jupyter lab "ML_learnings\Stacking,Bagging,Bossting.ipynb"`.

### Unsupervised learning

**Notebook:** `ML_learnings/Unsupervised_learning.ipynb`

- **Objective:** Demonstrate K-Means clustering, DBSCAN, and dimensionality reduction with PCA.
- **Dataset:** Three generated datasets: 300 two-dimensional `make_blobs` samples (`centers=3, cluster_std=0.60, random_state=42`); 200 `make_moons` samples (`noise=0.05, random_state=42`); and 500 five-dimensional blobs with 3 centers (`random_state=42`). These are synthetic, not external datasets.
- **Concepts and formulation:** K-Means minimizes within-cluster squared distances; DBSCAN groups dense neighborhoods and marks noise; PCA projects standardized data onto high-variance directions.
- **Implementation:** Each clustering input is standardized with `StandardScaler`. K-Means inertia is collected for `k=1` through `10`; the final blobs fit uses `n_clusters=3, random_state=42`. The moons data compares K-Means with 2 clusters against DBSCAN with `eps=0.3, min_samples=5`. PCA retains 2 components from the standardized five-feature data.
- **Results:** The K-Means inertia output for `k=1,2,3,4,5,10` is respectively `600.0000`, `178.6567`, `6.7751`, `5.7675`, `4.7224`, and `2.5998`. DBSCAN and PCA are visualized; no silhouette score or explained-variance result is printed.
- **How to run:** `jupyter lab "ML_learnings\Unsupervised_learning.ipynb"`.

### Heart disease classification and app

**Notebook and app:** `ML_learnings/Heart disease model/heart.ipynb`, `app.py`

- **Objective:** Explore the heart disease dataset, compare binary classifiers, and use a saved logistic-regression model in a Streamlit interface.
- **Dataset:** The local `heart.csv` has 918 rows and 12 columns: 11 inputs (`Age`, `Sex`, `ChestPainType`, `RestingBP`, `Cholesterol`, `FastingBS`, `RestingECG`, `MaxHR`, `ExerciseAngina`, `Oldpeak`, `ST_Slope`) and target `HeartDisease`. The notebook reports no missing values or duplicate rows. Zero cholesterol and resting blood pressure values are replaced with their nonzero-column means.
- **Concepts and formulation:** Dummy encoding, standardization, and binary classification with logistic regression, KNN, Gaussian Naive Bayes, a decision tree, and an RBF SVC. The evaluation metrics use the definitions in [Mathematical formulations used](#mathematical-formulations-used).
- **Implementation:** Categorical fields are one-hot encoded with `drop_first=True`; `Age`, `RestingBP`, `Cholesterol`, `MaxHR`, and `Oldpeak` are standardized on the full dataframe and standardized again after the split using a scaler fit on the training set. The train/test split is `test_size=0.33, random_state=42`. The app loads the serialized model, scaler, and expected feature columns, encodes user inputs, orders columns, scales, and predicts.
- **Results:** The notebook's displayed test-set comparison:

  | Model | Accuracy | F1 |
  |---|---:|---:|
  | Logistic regression | 0.8614 | 0.8772 |
  | KNN | 0.8515 | 0.8688 |
  | Gaussian Naive Bayes | 0.8515 | 0.8673 |
  | Decision tree | 0.7360 | 0.7561 |
  | RBF SVC | 0.8515 | 0.8696 |

- **How to run:** Open a terminal in `ML_learnings\Heart disease model` and run `jupyter lab heart.ipynb` so the relative `heart.csv` path resolves. For the app, run `streamlit run app.py` from the same directory.

### NLP preprocessing

**Notebook:** `ML_learnings/NLP.ipynb`

- **Objective:** Clean emotion-labeled text and map emotion names to integer labels.
- **Dataset:** A semicolon-separated `train.txt` file with no header, read as `text` and `emotion`. The file is not included in the repository; the notebook does not report a dataset source URL or a `shape` result. The shown data has no null values.
- **Concepts and formulation:** Lowercasing, punctuation and digit removal, non-ASCII character removal, NLTK tokenization, English stop-word removal, and integer label mapping. The notebook does not create TF-IDF/count-vector features and does not train or evaluate a classifier.
- **Implementation:** NLTK downloads `stopwords`, `punkt`, and `punkt_tab`. Label integers follow the order in which unique emotion labels appear in the loaded data.
- **Results:** No model metrics are produced. A displayed example is cleaned to `go feeling hopeless damned hopeful around someone cares awake`.
- **How to run:** Place `train.txt` in the working directory, ensure NLTK data can be downloaded, and run `jupyter lab "ML_learnings\NLP.ipynb"`.

### Ford car price regression

**Notebook:** `ML_learnings/car_price_prediction.ipynb`

- **Objective:** Explore Ford used-car listings and compare linear regression after one-hot versus label encoding.
- **Dataset:** KaggleHub dataset `adhurimquku/ford-car-price-prediction`, file `ford.csv`; 17,966 rows and 9 columns. Inputs are `model`, `year`, `transmission`, `mileage`, `fuelType`, `tax`, `mpg`, and `engineSize`; target is `price`. The notebook reports no null values.
- **Concepts and formulation:** `sklearn.linear_model.LinearRegression`; R² and adjusted R². Predictions use a linear combination of the input features. The notebook imports MAE and MSE but does not compute or report them.
- **Implementation:** The target is split from the inputs. One-hot encoding (`drop_first=True`) is compared with label encoding for `model`, `transmission`, and `fuelType`. Numeric features are standardized in both variants before the 80/20 split with `random_state=42`.
- **Results:** The notebook reports:

  | Encoding | R² | Adjusted R² |
  |---|---:|---:|
  | One-hot categorical features | 0.8458 | 0.8444 |
  | Label-encoded categorical features | 0.7366 | 0.7360 |

- **How to run:** Run the named notebook with KaggleHub access to download the Ford dataset: `jupyter lab "ML_learnings\car_price_prediction.ipynb"`.

### Insurance cost regression

**Notebook:** `ML_learnings/insurance.ipynb`

- **Objective:** Clean the local insurance dataset, perform exploratory and feature-selection analyses, and predict medical charges with linear regression.
- **Dataset:** `ML_learnings/insurance.csv`; 1,338 rows, 7 source columns (`age`, `sex`, `bmi`, `children`, `smoker`, `region`, `charges`). The notebook removes one duplicate, leaving 1,337 rows. Target: `charges`.
- **Concepts and formulation:** Linear regression, Pearson correlation, a chi-square independence test on charge quartiles, and MAE, MSE, RMSE, R², and adjusted R². The feature-selection code keeps `age`, `is_female`, `bmi`, `children`, `is_smoker`, and the `region_southeast` indicator for the final model.
- **Implementation:** `sex` and `smoker` are mapped to binary values; `region` is one-hot encoded with `drop_first=True`. The notebook casts the cleaned frame to integers, standardizes `age`, `bmi`, and `children` before an 80/20 split with `random_state=42`, and uses `LinearRegression`.
- **Results:** The test metrics printed in the notebook are MAE `4185.22`, MSE `35763676.15`, and RMSE `5980.27`; R² is `0.8054` and adjusted R² is `0.8009`.
- **How to run:** Run from `ML_learnings` so the relative CSV path resolves, or change the notebook's working directory to that folder; then open `insurance.ipynb`.

## Deep learning notebooks

### Binary classification with Keras layers

**Notebook:** `Deep learning/Basic_Layer_making_in_DL.ipynb`

- **Objective:** Train a small binary classifier to predict whether a plant needs water.
- **Dataset:** A hand-entered DataFrame with 16 examples; inputs are `soil_moisture`, `temperature_c`, and `sunlight_hours`; target is `needs_water`.
- **Concepts and formulation:** Min-max scaling, a dense layer with ReLU, a sigmoid output, binary cross-entropy, and SGD. The model uses standard Keras layers; the notebook does not implement a layer or its gradients from scratch.
- **Implementation:** The features are scaled with `(X - X.min()) / (X.max() - X.min() + 1e-8)`. A stratified 75/25 split uses `random_state=42`, leaving 12 training and 4 validation examples. The network has an 8-unit ReLU hidden layer and a one-unit sigmoid output; it compiles with `optimizer='sgd'`, `loss='binary_crossentropy'`, and accuracy, then trains for 100 epochs with batch size 4. A separate SGD object with learning rate `0.01` and momentum `0.9` is created after training but is not used by the compiled model.
- **Results:** The saved training log ends at epoch 100 with training accuracy `1.0000`, validation accuracy `1.0000`, training loss `0.2991`, and validation loss `0.2926`. The notebook does not call `evaluate` on a separate test set.
- **How to run:** `jupyter lab "Deep learning\Basic_Layer_making_in_DL.ipynb"`.

### MNIST perceptron, ANN, and CNN

**Notebook:** `Deep learning/CNN.ipynb`

- **Objective:** Compare a single dense softmax classifier, a two-hidden-layer dense network, and a convolutional network for digit recognition.
- **Dataset:** MNIST CSV files downloaded from `pjreddie.com`; 60,000 training and 10,000 test examples. Each row contains a digit label and 784 pixel values representing a 28×28 grayscale image. Target: digit class 0–9.
- **Concepts and formulation:** Pixel normalization, one-hot targets, dense and convolutional layers, ReLU, softmax, categorical cross-entropy, max pooling, and dropout. See the formulas below.
- **Implementation:** Pixel values are converted to `float32` and divided by 255; labels are one-hot encoded. All three models train for 5 epochs with batch size 32 and use the test set as `validation_data`. The perceptron uses `Flatten` and `Dense(10, softmax)` with SGD. The ANN uses 128- and 64-unit ReLU layers followed by 10-way softmax with Adam. The CNN uses 32 and 64 filters of size 3×3, 2×2 max-pooling after each convolution, a 128-unit ReLU dense layer, dropout 0.5, and a 10-way softmax output; it uses Adam.
- **Results:** Final test-set accuracy returned by `evaluate`:

  | Model | Test accuracy |
  |---|---:|
  | Dense softmax perceptron | 0.9087 |
  | ANN | 0.9733 |
  | CNN | 0.9924 |

- **How to run:** `jupyter lab "Deep learning\CNN.ipynb"`. The notebook downloads the MNIST CSV files when run.

### Iris classification with a perceptron and Keras

**Notebook:** `Deep learning/DL_iris_prediction.ipynb`

- **Objective:** Compare scikit-learn's linear perceptron with a small dense neural network for three-class Iris classification.
- **Dataset:** Iris CSV fetched from a Gist; 150 rows, four measurements (`sepal.length`, `sepal.width`, `petal.length`, `petal.width`), and `variety` as the target, with 50 examples per class.
- **Concepts and formulation:** Label encoding, training-set standardization, a perceptron, ReLU hidden layers, softmax output, categorical cross-entropy, and Adam.
- **Implementation:** An 80/20 stratified split with `random_state=42` gives 120 training and 30 test examples. `StandardScaler` is fit on training data only. The scikit-learn perceptron uses `max_iter=1000, random_state=42`. The Keras model has two 10-unit ReLU layers and a 3-unit softmax output; it trains for 100 epochs with batch size 8 and `validation_split=0.2`, using Adam and categorical cross-entropy.
- **Results:** Perceptron test accuracy is `0.8667`; Keras test accuracy is `0.9667`, with test loss `0.1120`.
- **How to run:** `jupyter lab "Deep learning\DL_iris_prediction.ipynb"`. The notebook downloads its Iris CSV.

## Python and data-analysis notebooks

These introductory exercises are included in the repository tree but do not train predictive models.

| Notebook / objective | Dataset or inputs | Results / output | How to run |
|---|---|---|---|
| `Data_analysis/demo.ipynb` — Jupyter setup, Python arithmetic, NumPy, and an abstract greeting class. | Hand-written values; no external dataset or target. | Prints a sum of `6` and the array `[10, 30, 50, 70, 90]`. | `jupyter lab "Data_analysis\demo.ipynb"` |
| `Data_analysis/matplot.ipynb` — Distribution and regression plots. | Seaborn `tips` dataset; 244 rows and 7 columns, including bill, tip, and categorical fields. | Produces histograms, joint/pair plots, and regression plots; no predictive metric is calculated. | `jupyter lab "Data_analysis\matplot.ipynb"` |
| `Data_analysis/numpy.ipynb` — Array shape, dtype, reductions, column operations, and z-score standardization. | Small arrays defined in the notebook; no target. | Shows the standardized values and column means/standard deviations; one cell produces a `NameError` when run without the earlier `numpy` import. | `jupyter lab "Data_analysis\numpy.ipynb"` |
| `Data_analysis/oop.ipynb` — Abstraction, encapsulation, area methods, and polymorphic `train()` calls. | Hand-written classes and values; no dataset or target. | Prints square area `25`, circle area `153.86`, and example messages. | `jupyter lab "Data_analysis\oop.ipynb"` |
| `Data_analysis/pandas.ipynb` — Pandas Series/DataFrame creation, indexing, and adding a column. | Hand-written toy dictionaries and lists; no target. | Displays the created series and frames. | `jupyter lab "Data_analysis\pandas.ipynb"` |
| `Data_analysis/pandasProject.ipynb` — Episode CSV inspection and year extraction. | Local `Game_of_Thrones (1).csv`; 73 rows and 19 source columns. The derived `Year` is not a prediction target. | Prints sample air dates and adds a `Year` column. | `jupyter lab "Data_analysis\pandasProject.ipynb"` |
| `Data_analysis/stats.ipynb` — Two-tailed, one-sample Z-test. | Scenario values: claimed mean 70, known standard deviation 15, sample mean 78, and sample size 30; no external dataset. | The code computes a Z statistic and p-value, but no saved numeric output is present. | `jupyter lab "Data_analysis\stats.ipynb"` |

## Mathematical formulations used

The formulas below correspond to operations or metrics present in the notebooks. No TF-IDF vectorizer, silhouette score, or L1-regularized model is used in this repository.

### Logistic classification and regularization

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

Here, $z=w^\top x+b$ is the linear score, $x$ is an input feature vector, $w$ is the learned coefficient vector, $b$ is the intercept, and $\sigma(z)$ is the predicted probability of the positive class.

$$
L_{\mathrm{BCE}}=-\frac{1}{N}\sum_{i=1}^{N}\left[y_i\log(\hat p_i)+(1-y_i)\log(1-\hat p_i)\right]
$$

Here, $N$ is the number of examples, $y_i\in\{0,1\}$ is the true label, and $\hat p_i$ is the predicted positive-class probability. This is the binary cross-entropy form used by the binary Keras model and the binary log-loss form associated with logistic classification.

Scikit-learn `LogisticRegression` uses its default L2 penalty because the notebooks do not override it:

$$
\Omega_2(w)=\lVert w\rVert_2^2=\sum_{j=1}^{d}w_j^2
$$

Here, $w_j$ is the coefficient for feature $j$ and $d$ is the number of features. In scikit-learn, `C` controls inverse regularization strength; the notebook's default logistic-regression calls use `C=1.0`. No L1 penalty is selected.

The SVC examples select the radial-basis-function kernel:

$$
K(x_i,x_j)=\exp\left(-\gamma\lVert x_i-x_j\rVert_2^2\right)
$$

Here, $x_i$ and $x_j$ are two feature vectors and $\gamma$ is the kernel coefficient. The GridSearch notebook sets `gamma='auto'` for its initial SVC; other RBF SVC calls leave the kernel coefficient at the scikit-learn default.

Feature standardization and the min-max scaling used by the small plant example are:

$$
z=\frac{x-\mu}{s},\qquad
x_{\mathrm{scaled}}=\frac{x-x_{\min}}{x_{\max}-x_{\min}+\epsilon}
$$

Here, $x$ is a feature value, $\mu$ and $s$ are the feature mean and standard deviation, and $x_{\min}$ and $x_{\max}$ are its minimum and maximum. The small Keras example uses $\epsilon=10^{-8}$; scikit-learn `StandardScaler` is used in the other notebooks.

### Classification metrics

$$
\mathrm{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN},\qquad
\mathrm{Precision}=\frac{TP}{TP+FP},\qquad
\mathrm{Recall}=\frac{TP}{TP+FN},\qquad
F_1=2\frac{\mathrm{Precision}\cdot\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}
$$

Here, $TP$, $TN$, $FP$, and $FN$ are true positives, true negatives, false positives, and false negatives. Precision, recall, and F1 are reported per class by the classification reports; accuracy is also used in the notebooks' classifier comparisons.

### Regression metrics

Scikit-learn `LinearRegression` fits the coefficients by minimizing the residual sum of squares:

$$
(\hat w,\hat b)=\underset{w,b}{\operatorname{arg\,min}}\sum_{i=1}^{N}\left(y_i-(w^\top x_i+b)\right)^2
$$

Here, $x_i$ is the input vector for example $i$, $y_i$ is its target, $w$ is the coefficient vector, $b$ is the intercept, and $N$ is the number of training examples.

$$
\mathrm{MAE}=\frac{1}{N}\sum_{i=1}^{N}|y_i-\hat y_i|,\qquad
\mathrm{MSE}=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat y_i)^2,\qquad
\mathrm{RMSE}=\sqrt{\mathrm{MSE}}
$$

Here, $y_i$ is the actual value, $\hat y_i$ is its prediction, and $N$ is the number of evaluated examples. The insurance notebook computes all three metrics.

$$
R^2=1-\frac{\sum_{i=1}^{N}(y_i-\hat y_i)^2}{\sum_{i=1}^{N}(y_i-\bar y)^2},\qquad
R^2_{\mathrm{adj}}=1-(1-R^2)\frac{N-1}{N-p-1}
$$

Here, $\bar y$ is the mean of the actual targets, $N$ is the number of test examples, and $p$ is the number of input features. Both regression notebooks compute R² and adjusted R².

### Neural-network activations, losses, and optimization

$$
\mathrm{ReLU}(z)=\max(0,z)
$$

Here, $z$ is a pre-activation value; ReLU is used by the hidden layers in the binary, Iris, MNIST ANN, and MNIST CNN models.

$$
\mathrm{softmax}(z)_k=\frac{e^{z_k}}{\sum_{j=1}^{K}e^{z_j}}
$$

Here, $z_k$ is the score for class $k$ and $K$ is the number of classes. The Keras Iris and MNIST models use softmax output layers.

$$
L_{\mathrm{categorical}}=-\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{K}y_{ik}\log(\hat p_{ik})
$$

Here, $N$ is the batch size, $K$ is the number of classes, $y_{ik}$ is the one-hot target for example $i$ and class $k$, and $\hat p_{ik}$ is the predicted class probability. This is the categorical cross-entropy loss configured for the multi-class Keras models.

$$
\theta_{t+1}=\theta_t-\eta\nabla_\theta L_{\mathcal{B}_t}(\theta_t)
$$

Here, $\theta_t$ is the parameter vector at update $t$, $\eta$ is the learning rate, and $L_{\mathcal{B}_t}$ is the loss over the current mini-batch. This is the SGD update concept applicable to the Keras model compiled with `optimizer='sgd'`; the notebooks do not implement a custom gradient calculation.

For a 2D convolution or pooling operation, the output size along one spatial dimension is:

$$
O=\left\lfloor\frac{N+2P-D(K-1)-1}{S}+1\right\rfloor
$$

Here, $N$ is the input dimension, $O$ is the output dimension, $K$ is the kernel or pooling-window size, $P$ is padding on one side, $S$ is stride, and $D$ is dilation. The MNIST CNN uses valid padding, stride 1, and 3×3 convolution kernels, followed by 2×2 max pooling.

### Ensemble aggregation

For the random forest, class predictions are combined by a majority vote:

$$
\hat y(x)=\underset{k}{\operatorname{arg\,max}}\sum_{m=1}^{M}\mathbf{1}\!\left[h_m(x)=k\right]
$$

Here, $h_m(x)$ is the prediction of tree $m$, $M$ is the number of trees, $k$ is a class, and $\mathbf{1}[\cdot]$ is 1 when its condition is true and 0 otherwise. The notebook uses a 100-tree random forest.

For the multi-class AdaBoost classifier, weighted learner votes are aggregated:

$$
\hat y(x)=\underset{k}{\operatorname{arg\,max}}\sum_{m=1}^{M}\alpha_m\mathbf{1}\!\left[h_m(x)=k\right]
$$

Here, $h_m(x)$ is weak learner $m$, $\alpha_m$ is its learned vote weight, $M$ is the number of boosting stages, and $k$ indexes the classes. This expresses the classifier's weighted-vote prediction, not a metric reported by the notebook.

Gradient boosting and XGBoost construct additive tree ensembles; the gradient-boosting update can be written:

$$
F_m(x)=F_{m-1}(x)+\eta h_m(x)
$$

Here, $F_m(x)$ is the ensemble score after stage $m$, $h_m(x)$ is the new tree fitted to the loss gradient, and $\eta$ is the learning rate. The notebook uses learning rates 0.1 for both gradient boosting and XGBoost.

In stacking, out-of-fold base-estimator predictions become inputs to a final estimator:

$$
z(x)=\left[p_1(1\mid x),\ldots,p_1(K\mid x),\ldots,p_M(1\mid x),\ldots,p_M(K\mid x)\right],\qquad
\hat y(x)=g(z(x))
$$

Here, $p_m(k\mid x)$ is the class-probability output from base estimator $m$ for class $k$, $M$ is the number of base estimators, $K$ is the number of classes, and $g$ is the final logistic-regression estimator. In the notebook, the base learners provide probabilities and stacking uses five-fold cross-validation to form its training features.

### Clustering and dimensionality reduction

K-Means minimizes the within-cluster sum of squared distances:

$$
J=\sum_{i=1}^{N}\min_{1\leq k\leq K}\lVert x_i-\mu_k\rVert_2^2
$$

Here, $x_i$ is sample $i$, $N$ is the number of samples, $K$ is the number of clusters, and $\mu_k$ is the centroid of cluster $k$. The notebook plots this objective as inertia for K-Means.

PCA projects centered data onto orthonormal directions that maximize projected variance:

$$
\max_{V^\top V=I_k}\lVert X_cV\rVert_F^2
$$

Here, $X_c$ is the centered feature matrix, $V$ contains the selected component directions, $I_k$ is the identity matrix for $k$ components, and $\lVert\cdot\rVert_F$ is the Frobenius norm. Scikit-learn's PCA centers its input; the notebook first standardizes the generated features and retains two components.

### Feature statistics and Z-test

The Pearson correlation used in insurance feature exploration is:

$$
r_{xy}=\frac{\sum_{i=1}^{N}(x_i-\bar x)(y_i-\bar y)}
{\sqrt{\sum_{i=1}^{N}(x_i-\bar x)^2\sum_{i=1}^{N}(y_i-\bar y)^2}}
$$

Here, $x_i$ and $y_i$ are paired observations, $\bar x$ and $\bar y$ are their sample means, and $N$ is the number of observations.

The chi-square independence statistic used on categorical feature versus charge-bin contingency tables is:

$$
\chi^2=\sum_{r=1}^{R}\sum_{c=1}^{C}\frac{(O_{rc}-E_{rc})^2}{E_{rc}}
$$

Here, $O_{rc}$ is the observed count in row $r$ and column $c$, $E_{rc}$ is the expected count under independence, and $R$ and $C$ are the numbers of contingency-table rows and columns.

The stats notebook calculates a one-sample Z statistic and a two-tailed normal p-value:

$$
z=\frac{\bar x-\mu_0}{\sigma/\sqrt{n}},\qquad
p=2\left[1-\Phi(|z|)\right]
$$

Here, $\bar x$ is the sample mean, $\mu_0$ is the claimed population mean, $\sigma$ is the known population standard deviation, $n$ is the sample size, and $\Phi$ is the standard normal cumulative distribution function.

## Concepts covered

| Area | Implemented examples |
|---|---|
| Classification | Logistic regression, KNN, Gaussian Naive Bayes, decision trees, SVC, perceptron |
| Model selection | Train/test splitting, five-fold cross-validation, `GridSearchCV`, `RandomizedSearchCV` |
| Ensembles | Stacking, random-forest bagging, AdaBoost, gradient boosting, XGBoost |
| Regression | Linear regression for Ford used-car prices and insurance charges |
| Unsupervised learning | K-Means, DBSCAN, inertia/elbow visualization, PCA |
| NLP | Text normalization and stop-word removal only; no vectorizer or classifier |
| Deep learning | Dense binary classifier, multi-class Iris network, MNIST dense and convolutional networks |
| Data preparation | One-hot and label encoding, missing-value handling, standardization, min-max scaling |
| Foundations | Python OOP, NumPy, Pandas, plotting, descriptive exploration, and a Z-test |

## Author and contact

**GitHub:** [@aditrajsharma](https://github.com/aditrajsharma)
