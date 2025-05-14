# Used Car Price Prediction with Machine Learning

A machine learning project for **Rusty Bargain**, a used car sales service, to build a predictive model that estimates car prices based on historical data. The goal is to provide a reliable, efficient, and fast pricing engine for integration into their customer-facing app.

---

## Table of Contents
- [Overview](#overview)
- [Dataset Description](#dataset-description)
- [Data Preprocessing](#data-preprocessing)
- [Modeling & Evaluation](#modeling--evaluation)
- [Feature Importance](#feature-importance)
- [Final Model Selection](#final-model-selection)
- [Installation & Usage](#installation--usage)
- [Future Work](#future-work)
- [Project Status](#project-status)

---

## Overview

The project involves building and comparing several regression models to predict used car prices using historical data. The final model is intended to be integrated into Rusty Bargain's mobile app, enabling users to estimate the fair market value of their vehicles.

---

## Dataset Description

- **Source:** `/datasets/car_data.csv`
- **Target Variable:** `Price` (in Euros)
- **Features:**
  - VehicleType
  - RegistrationYear & Month
  - Gearbox
  - Power (horsepower)
  - Model
  - Mileage
  - FuelType
  - Brand
  - NotRepaired (repaired status)
  - Date fields, pictures, and postal codes (removed as irrelevant)

---

## Data Preprocessing

- Handled missing values with mode/imputation
- Removed irrelevant or duplicate columns
- Encoded categorical variables:
  - One-Hot Encoding for models like XGBoost and Random Forest
  - Native handling for LightGBM and CatBoost
- Scaled numerical features using `StandardScaler`
- Ensured no data leakage by applying scaling **after** the train-test split

---

## Modeling & Evaluation

Nine models were trained and evaluated using **Root Mean Squared Error (RMSE)** as the primary metric:

| Model                           | RMSE    | Training Time (s) | Prediction Time (s) |
|--------------------------------|---------|--------------------|----------------------|
| Linear Regression              | 2647.61 | 9.40               | 0.033                |
| Decision Tree                  | 2111.67 | 4.69               | 0.067                |
| Random Forest (Original)       | 1674.44 | 343.72             | 2.70                 |
| Random Forest (Tuned)          | 1685.14 | 787.28             | 3.71                 |
| Random Forest (Feature Select) | 1784.59 | 120.27             | 3.55                 |
| LightGBM                       | 1770.55 | 5.47               | 0.404                |
| XGBoost                        | 1735.30 | 296.78             | 0.543                |
| CatBoost (Original)            | 1682.12 | 27.60              | 0.083                |
| **CatBoost (Tuned)**           | **1658.02** | **29.51**     | **0.089**            |

---

## Feature Importance

Top 5 most influential features:
- **Registration Year** – 0.48
- **Power (HP)** – 0.27
- **Mileage** – 0.06
- **Vehicle Type (e.g., convertible, sedan)** – 0.03
- **Brand and Fuel Type** – 0.01 – 0.005

Unimportant features were excluded to optimize training time.

---

## Final Model Selection

- **Model Chosen:** Tuned CatBoost
- **Reasons:**
  - Best overall RMSE (1658.02)
  - Efficient training and prediction times
  - Native handling of categorical features
  - Excellent generalization performance

---

## Installation & Usage

To reproduce this project:

```bash
# Clone the repository
git clone https://github.com/yourusername/car-price-prediction.git
cd car-price-prediction

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook
