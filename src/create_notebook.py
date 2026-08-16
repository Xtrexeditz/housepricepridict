import os
import json

def create_notebook():
    notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Property Price Prediction - Exploratory Data Analysis & Modeling\n",
    "\n",
    "Welcome to the **Property Price Prediction** project notebook. In this notebook, we will run through a typical data science workflow to analyze housing/property data and train machine learning models to predict prices in **Indian Rupees (INR)**.\n",
    "\n",
    "## Workflow Steps:\n",
    "1. **Load and Inspect Data**: View the structure, types, and summary statistics of the dataset.\n",
    "2. **Exploratory Data Analysis (EDA)**: Visualize distributions, relationships, and correlations in the data.\n",
    "3. **Data Preprocessing**: Prepare the numerical and categorical variables using scikit-learn pipelines.\n",
    "4. **Model Training & Evaluation**: Train multiple regression models and compare their metrics ($R^2$, MAE, RMSE).\n",
    "5. **Feature Importance**: Analyze which features are most influential in predicting property prices.\n",
    "6. **Model Export**: Save the trained pipeline for deployment."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. Imports and environment setup\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.tree import DecisionTreeRegressor\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
    "import joblib\n",
    "\n",
    "# Set visualization style\n",
    "sns.set_theme(style=\"whitegrid\")\n",
    "plt.rcParams[\"figure.figsize\"] = (10, 6)\n",
    "plt.rcParams[\"font.size\"] = 12"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Load and Inspect Data\n",
    "Let's load the synthetic dataset generated in the previous step and view its first few rows."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load dataset\n",
    "df = pd.read_csv('../data/housing_data.csv')\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Basic data properties\n",
    "print(f\"Dataset Shape: {df.shape}\")\n",
    "print(\"\\n--- Data Info ---\")\n",
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Descriptive statistics\n",
    "df.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Exploratory Data Analysis (EDA)\n",
    "Let's visualize the data to understand the underlying distributions and relations."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot the distribution of the target variable (Price in INR)\n",
    "sns.histplot(df['Price'], kde=True, color='teal')\n",
    "plt.title('Distribution of Property Prices (INR)')\n",
    "plt.xlabel('Price (₹)')\n",
    "plt.ylabel('Frequency')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Box plot: PropertyType vs Price\n",
    "sns.boxplot(data=df, x='PropertyType', y='Price', palette='pastel')\n",
    "plt.title('Property Prices (INR) across Property Types')\n",
    "plt.xlabel('Property Type')\n",
    "plt.ylabel('Price (₹)')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Scatter plot: SquareFeet vs Price colored by PropertyType\n",
    "sns.scatterplot(data=df, x='SquareFeet', y='Price', hue='PropertyType', alpha=0.6, palette='viridis')\n",
    "plt.title('Price (INR) vs. Square Footage by Property Type')\n",
    "plt.xlabel('Square Feet')\n",
    "plt.ylabel('Price (₹)')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Box plot: Neighborhood vs Price\n",
    "sns.boxplot(data=df, x='Neighborhood', y='Price', palette='muted')\n",
    "plt.title('Property Prices (INR) across Neighborhoods')\n",
    "plt.xlabel('Neighborhood')\n",
    "plt.ylabel('Price (₹)')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Correlation Heatmap for numeric features\n",
    "numeric_df = df.select_dtypes(include=[np.number])\n",
    "sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=\".2f\", linewidths=0.5)\n",
    "plt.title('Correlation Matrix of Numeric Features')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Data Preprocessing\n",
    "We need to clean and transform our features before fitting models:\n",
    "- Scale numerical features using `StandardScaler` to help model convergence and avoid scale bias.\n",
    "- One-hot encode the categorical features `Neighborhood` and `PropertyType`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Separate features and target\n",
    "X = df.drop(columns=['Price'])\n",
    "y = df['Price']\n",
    "\n",
    "# Define column transformations\n",
    "num_features = ['SquareFeet', 'Bedrooms', 'Bathrooms', 'YearBuilt', 'Floors', 'Garage']\n",
    "cat_features = ['Neighborhood', 'PropertyType']\n",
    "\n",
    "preprocessor = ColumnTransformer(\n",
    "    transformers=[\n",
    "        ('num', StandardScaler(), num_features),\n",
    "        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_features)\n",
    "    ]\n",
    ")\n",
    "\n",
    "# Train-test split (80% train, 20% test)\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "print(f\"Training instances: {len(X_train)}\")\n",
    "print(f\"Testing instances: {len(X_test)}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Model Training & Evaluation\n",
    "We will evaluate three regression models:\n",
    "1. **Linear Regression** (Baseline)\n",
    "2. **Decision Tree Regressor**\n",
    "3. **Random Forest Regressor**\n",
    "\n",
    "We will calculate the evaluation metrics: R-squared ($R^2$), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Helper function to train and evaluate a model pipeline\n",
    "def evaluate_model(model_name, regressor):\n",
    "    pipeline = Pipeline(steps=[\n",
    "        ('preprocessor', preprocessor),\n",
    "        ('regressor', regressor)\n",
    "    ])\n",
    "    \n",
    "    pipeline.fit(X_train, y_train)\n",
    "    y_pred = pipeline.predict(X_test)\n",
    "    \n",
    "    mae = mean_absolute_error(y_test, y_pred)\n",
    "    rmse = np.sqrt(mean_squared_error(y_test, y_pred))\n",
    "    r2 = r2_score(y_test, y_pred)\n",
    "    \n",
    "    return pipeline, {\n",
    "        'Model': model_name,\n",
    "        'MAE (₹)': mae,\n",
    "        'RMSE (₹)': rmse,\n",
    "        'R2': r2\n",
    "    }"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Train and evaluate models\n",
    "models = {\n",
    "    'Linear Regression': LinearRegression(),\n",
    "    'Decision Tree': DecisionTreeRegressor(random_state=42),\n",
    "    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)\n",
    "}\n",
    "\n",
    "results = []\n",
    "pipelines = {}\n",
    "\n",
    "for name, reg in models.items():\n",
    "    pipe, metrics = evaluate_model(name, reg)\n",
    "    pipelines[name] = pipe\n",
    "    results.append(metrics)\n",
    "\n",
    "# Display results as DataFrame\n",
    "results_df = pd.DataFrame(results)\n",
    "results_df"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Feature Importance\n",
    "Let's look at the feature importances for our best performing model (Random Forest)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Extract feature importances\n",
    "best_pipe = pipelines['Random Forest']\n",
    "importances = best_pipe.named_steps['regressor'].feature_importances_\n",
    "\n",
    "# Extract feature names from preprocessing pipeline\n",
    "onehot_cols = list(best_pipe.named_steps['preprocessor']\n",
    "                   .named_transformers_['cat']\n",
    "                   .named_steps['onehot']\n",
    "                   .get_feature_names_out(cat_features))\n",
    "feature_names = num_features + onehot_cols\n",
    "\n",
    "# Plot feature importances\n",
    "importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})\n",
    "importance_df = importance_df.sort_values(by='Importance', ascending=False)\n",
    "\n",
    "sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')\n",
    "plt.title('Random Forest Feature Importance')\n",
    "plt.xlabel('Importance')\n",
    "plt.ylabel('Feature')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Model Export\n",
    "Finally, let's export the trained Random Forest model pipeline to a file using `joblib` so that it can be loaded in our production script or web application."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save pipeline\n",
    "joblib.dump(best_pipe, '../models/house_price_pipeline.joblib')\n",
    "print(\"Model pipeline saved successfully!\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
    
    os.makedirs('notebooks', exist_ok=True)
    with open('notebooks/eda_and_modeling.ipynb', 'w') as f:
        json.dump(notebook_content, f, indent=1)
    print("Jupyter Notebook created at notebooks/eda_and_modeling.ipynb")

if __name__ == '__main__':
    create_notebook()
