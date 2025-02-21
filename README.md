# Machine Learning for Medical Diagnosis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

Welcome to **Machine Learning for Medical Diagnosis**, a project showcasing the power of machine learning in predicting medical conditions using real-world datasets. This repository contains two robust Python scripts that implement and evaluate Decision Trees, Random Forests, and Naive Bayes classifiers on cancer and diabetes datasets. The models are optimized with varying hyperparameters, and their performance is visualized with intuitive bar charts.

## Project Overview

This project demonstrates the application of supervised machine learning algorithms to classify:
1. **Breast Cancer Diagnosis** (`cancer_data.csv`): Predicts whether a tumor is malignant or benign.
2. **Diabetes Prediction** (`diabetes_data_upload.csv`): Determines the likelihood of diabetes based on symptoms and demographic data.

Each script preprocesses the data, trains multiple models with different configurations, evaluates their accuracy, and generates visualizations to compare performance.

### Key Features
- **Data Preprocessing**: Handles categorical encoding, feature scaling, and dataset splitting.
- **Model Variety**: Implements Decision Trees, Random Forests, and Naive Bayes with diverse hyperparameters.
- **Evaluation**: Computes accuracy scores for each model configuration.
- **Visualization**: Bar charts to compare model performance across parameter sets.

## Datasets

1. **Cancer Dataset** (`cancer_data.csv`):
   - Features: Tumor measurements (e.g., radius, texture, perimeter).
   - Target: Diagnosis (Malignant: 1, Benign: 0).
   - Source: [Adjust based on actual source, e.g., UCI Machine Learning Repository].

2. **Diabetes Dataset** (`diabetes_data_upload.csv`):
   - Features: Age, gender, and symptoms (e.g., Polyuria, Polydipsia).
   - Target: Diabetes class (Positive: 1, Negative: 0).
   - Source: [Adjust based on actual source].

**Note**: Datasets are not included in this repository due to size or licensing constraints. Please provide your own copies of `cancer_data.csv` and `diabetes_data_upload.csv` in the project root or adjust the file paths in the scripts.

## Requirements

To run the scripts, ensure you have the following installed:
- Python 3.8+
- Libraries:
  ```bash
  pip install pandas scikit-learn matplotlib
  ```

## Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/[YourUsername]/machine-learning-medical-diagnosis.git
   cd machine-learning-medical-diagnosis
   ```

2. **Prepare the Datasets**:
   - Place `cancer_data.csv` and `diabetes_data_upload.csv` in the project directory (or update the file paths in the scripts).

3. **Run the Scripts**:
   - For cancer diagnosis:
     ```bash
     python cancer_classification.py
     ```
   - For diabetes prediction:
     ```bash
     python diabetes_classification.py
     ```

4. **Output**:
   - Terminal: Accuracy scores for each model configuration.
   - Visuals: Three bar charts per dataset (Decision Tree, Random Forest, Naive Bayes).

## Model Configurations

### Cancer Classification
- **Decision Trees**: `max_depth=3 (gini)`, `max_depth=5 (entropy)`, `max_depth=None (min_split=5, gini)`.
- **Random Forests**: `n_estimators=50 (max_depth=3)`, `n_estimators=100 (max_depth=5)`, `n_estimators=200 (max_depth=None)`.
- **Naive Bayes**: `var_smoothing=1e-9`, `1e-5`, `1e-3`.

### Diabetes Classification
- Same configurations as above, tailored to the diabetes dataset.

## Results

The scripts generate accuracy metrics and visualizations, allowing you to:
- Compare how hyperparameter tuning affects performance.
- Identify the best-performing model for each dataset.

Sample output:
```
Decision Tree Results:
max_depth=3, criterion='gini': 0.9300
max_depth=5, criterion='entropy': 0.9450
max_depth=None, min_samples_split=5, criterion='gini': 0.9250
```

## Visualizations

Example bar charts:
- **Decision Tree Accuracy**: Compares accuracy across different depths and criteria.
- **Random Forest Accuracy**: Highlights the impact of tree count and depth.
- **Naive Bayes Accuracy**: Shows sensitivity to smoothing parameters.

## Contributing

Contributions are welcome! Feel free to:
- Suggest improvements to preprocessing or model tuning.
- Add new algorithms or datasets.
- Open issues or submit pull requests.

## License

This project is licensed under the [MIT License](LICENSE). Use it freely, and credit the author where applicable.

## Acknowledgments

- Built with ❤️ using Python, scikit-learn, and matplotlib.
- Inspired by the potential of AI in healthcare.
