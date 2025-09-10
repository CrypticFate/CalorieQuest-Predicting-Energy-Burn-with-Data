# Calorie Quest: Advanced Calorie Prediction

This repository contains code for an advanced machine learning model to predict calorie expenditure based on physiological data. The solution uses an ensemble of models, extensive feature engineering, and advanced techniques to achieve high accuracy.

## Features

- **Extensive Feature Engineering:** Creates a rich set of features from the base data, including physiological calculations, interaction terms, and polynomial features.
- **Multi-Layer Ensemble Model:** Utilizes a two-level ensemble for robust predictions.
    - **Level 1:** A diverse set of models including LightGBM, XGBoost, Random Forest, and Extra Trees.
    - **Level 2:** A meta-learner (e.g., Bayesian Ridge) that combines the predictions from the Level 1 models.
- **Advanced Target Encoding:** Implements target encoding with multiple statistics to capture categorical feature information effectively.
- **Feature Selection:** Uses statistical tests to select the most relevant features.
- **Pseudo-Labeling:** (Implicit in the ensemble structure) The model can be extended to use pseudo-labeling for semi-supervised learning.

## Notebook

The main logic is contained in the `advanced_calorie_prediction_ensemble.ipynb` notebook. The notebook is structured as follows:

1.  **Imports and Setup:** Imports necessary libraries and sets up the environment.
2.  **Utility Functions:** Contains helper functions, such as the RMSLE calculation.
3.  **Data Loading:** Loads the training and testing datasets.
4.  **Feature Engineering:** Defines the `ultra_feature_engineering` function to create new features.
5.  **Target Encoding:** Defines the `create_advanced_target_encoding` function.
6.  **Feature Selection:** Defines the `select_best_features` function.
7.  **Ensemble Model Training:** Defines the `train_ultra_ensemble` function to train the multi-layer ensemble.
8.  **Main Execution:** The main function that orchestrates the entire pipeline from data loading to submission file generation.

## Usage

1.  Install the required libraries from `requirements.txt`.
2.  Place the `train.csv` and `test.csv` files in the root directory.
3.  Run the cells in the `advanced_calorie_prediction_ensemble.ipynb` notebook.

The notebook will generate a `submission.csv` file with the predicted calorie values for the test set.
