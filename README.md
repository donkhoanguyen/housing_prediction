# Housing Price Prediction Project

This project focuses on predicting housing prices using various machine learning models. The goal is to achieve a maximum R² score of 0.77 while maintaining a low Mean Absolute Error (MAE) of less than $300,000.

## Project Overview

In this project, I utilized different machine learning models to predict housing prices based on a dataset containing various features such as the number of rooms, location, and other relevant factors. Each model was fine-tuned to optimize performance, with the primary metrics being the R² score and MAE.

## Models Used

- **Linear Regression**
- **Decision Trees**
- **Random Forest**
- **Gradient Boosting Machines**
- **XGBoost**
- **Neural Networks**

## Goals

- **R² Score**: Maximize the coefficient of determination (R²) to 0.77.
- **Mean Absolute Error (MAE)**: Minimize the MAE to be less than $300,000.

## Key Features

- **Data Preprocessing**: Handled missing values, encoded categorical variables, and normalized/standardized the data.
- **Model Tuning**: Hyperparameter tuning was conducted using GridSearchCV and RandomizedSearchCV to find the best-performing models.
- **Cross-Validation**: Employed k-fold cross-validation to validate the models' performance and ensure generalizability.
- **Feature Engineering**: Created new features to improve model accuracy, including interaction terms and polynomial features.

## Results

- **Best Model**: [Model Name] with an R² score of 0.77 and an MAE of $[MAE Value].
- **Model Performance**: Each model's performance was evaluated and compared to identify the best approach for this dataset.

## Tools and Technologies

- **Python**
- **Pandas**
- **Scikit-learn**
- **XGBoost**
- **Matplotlib & Seaborn** (for visualization)

## Conclusion

The project successfully achieved the target R² score and MAE. The models were fine-tuned to provide reliable predictions of housing prices, demonstrating the effectiveness of various machine learning approaches in regression tasks.

## Future Work

- **Feature Selection**: Further refine the feature set to improve model performance.
- **Model Stacking**: Experiment with ensemble methods like stacking for potentially better results.
- **Deployment**: Consider deploying the model using a web service for real-time price prediction.

## Repository Structure

- **data/**: Dataset files
- **notebooks/**: Jupyter notebooks used for analysis and modeling
- **models/**: Saved model files
- **scripts/**: Python scripts for data preprocessing and model training
- **results/**: Visualizations and model performance metrics

## Acknowledgments

---
Feel free to explore the repository and reach out if you have any questions or suggestions!
