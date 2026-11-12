# DataVine Analytics – Machine Learning Summative Lab

## Project Overview

This project applies machine learning techniques to three datasets to address classification, recommendation, and clustering problems.

The analysis covers data preprocessing, feature standardization, Principal Component Analysis (PCA), model development, evaluation, and interpretation of results.

## Project Objectives

1. **Wine Classification:** Develop and evaluate a k-Nearest Neighbors (k-NN) model to classify wine categories.
2. **Agricultural Feed Recommendation:** Identify feeds associated with similar chicken weight outcomes using PCA and cosine similarity.
3. **USArrests Clustering:** Group US states based on crime and demographic characteristics using K-Means and Gaussian Mixture Models (GMM).

## Datasets

| Dataset | Description | Observations |
|---|---|---|
| Wine | Chemical characteristics of wines belonging to three classes | 178 |
| Chickwts | Chicken weights recorded under six feed categories | 71 |
| USArrests | Crime arrest rates and urban population percentages across US states | 50 |

## Methods and Tools

**Programming Language:** Python

**Libraries:**
- Pandas and NumPy – Data manipulation and analysis
- Matplotlib and Seaborn – Data visualization
- Scikit-learn – Preprocessing, PCA, classification, similarity analysis, and clustering

**Machine Learning Techniques:**
- StandardScaler
- Principal Component Analysis (PCA)
- k-Nearest Neighbors (k-NN)
- GridSearchCV
- Cosine Similarity
- K-Means Clustering
- Gaussian Mixture Models (GMM)
- Elbow Method, Silhouette Analysis, and Bayesian Information Criterion (BIC)

## Key Findings

### 1. Wine Classification

- PCA retained 10 principal components explaining 96.24% of the variance.
- GridSearchCV selected 11 neighbors and Euclidean distance.
- Cross-validation accuracy: 96.50%.
- Test accuracy: 100% on 36 observations.

### 2. Agricultural Feed Recommendation

- PCA retained one component explaining 100% of the variance.
- Cosine similarity identified two broad groups of feeds based on their mean standardized weight scores.
- Sunflower and casein recorded the closest average chicken weights, at 328.92 and 323.58 respectively.
- One-dimensional cosine similarity was found to have limited ability to distinguish feeds with different average weight outcomes.

### 3. USArrests Clustering

- Three features were selected: Murder, Assault, and UrbanPop.
- PCA retained two components explaining 94.15% of the variance.
- The elbow method suggested four K-Means clusters.
- Silhouette analysis favoured two clusters (0.4317), although four clusters achieved a comparable score (0.4253).
- BIC selected one Gaussian component for GMM.
- K-Means provided four interpretable groups based on crime arrest rates and urbanization.

## Repository Contents

- `DataVine_Analytics.ipynb` – Jupyter Notebook containing the complete analysis
- `wine.csv` – Wine dataset
- `chickwts.csv` – Agricultural feed dataset
- `USArrests.csv` – USArrests dataset
- `README.md` – Project documentation

## How to Run the Project

1. Download or clone this repository.
2. Open `DataVine_Analytics.ipynb` using VS Code or Jupyter Notebook.
3. Ensure the three CSV datasets are available in the notebook's working directory.
4. Install the required Python libraries if necessary.
5. Run the notebook cells sequentially to reproduce the analysis and results.

## Conclusion

The project demonstrates the application of supervised and unsupervised machine learning techniques to different analytical problems.

The Wine classification model achieved strong predictive performance on the test dataset. The feed recommendation analysis highlighted limitations of cosine similarity when applied to one-dimensional data. The clustering analysis demonstrated how different methods and model selection criteria can produce different groupings of the same observations.

The findings highlight the importance of appropriate preprocessing, model selection, evaluation, and interpretation.
