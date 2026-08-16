# SmartHouse AI: House Price Prediction Dashboard (INR)

A comprehensive, production-ready machine learning project to predict housing prices in India (**Indian Rupees - INR**). This project features a synthetic data generator, exploratory data analysis (EDA) notebook, CLI training and inference scripts, and a gorgeous glassmorphic web interface served via Flask.

---

## 🚀 Key Features

* **Data Generation Pipeline**: Synthetic data generator producing realistic residential housing properties tailored for India (prices in Lakhs and Crores, Indian numbering commas, and realistic standard of living factors).
* **Jupyter Notebook**: Fully documented workflow detailing EDA, feature scaling, model selection (Linear Regression, Decision Trees, Random Forests), and feature importance analysis.
* **Random Forest ML Pipeline**: Standardized pipeline using `scikit-learn`'s `ColumnTransformer` for robust preprocessing (scaling numerical features and one-hot encoding categorical locations).
* **Production CLI Scripts**: Modular scripts to re-train the pipeline and perform command-line inference (supports arguments or an interactive prompt).
* **Premium Glassmorphic Dashboard**: A high-fidelity, interactive web application featuring real-time sliders, dropdown selectors, responsive card systems, active state changes, and instant API predictions with formatted Rupee outputs.

---

## 📁 Directory Structure

```text
├── data/
│   └── housing_data.csv          # Generated housing dataset (5,000 samples)
├── models/
│   └── house_price_pipeline.joblib # Saved scikit-learn model pipeline artifact
├── notebooks/
│   └── eda_and_modeling.ipynb    # Jupyter Notebook for exploratory analysis & prototyping
├── src/
│   ├── generate_data.py          # Synthetic data generator script
│   ├── train.py                  # CLI training script
│   ├── predict.py                # CLI inference utility (Supports interactive/argument mode)
│   └── create_notebook.py        # Helper to generate the Jupyter Notebook file
├── templates/
│   └── index.html                # Premium HTML dashboard structure
├── static/
│   └── css/
│       └── styles.css            # Custom CSS styling (glassmorphism & animations)
├── app.py                        # Flask backend server exposing the prediction API
├── requirements.txt              # Project package dependencies
└── README.md                     # Project documentation
```

---

## 🛠️ Getting Started & Setup

### 1. Prerequisites
Ensure you have **Python 3.10+** installed on your system.

### 2. Install Dependencies
Install all required libraries using `pip`:
```bash
pip install -r requirements.txt
```

### 3. Generate the Dataset
Create the synthetic housing records (5,000 houses with features: SqFt, Beds, Baths, Neighborhood, Year Built, Floors, Garage):
```bash
python src/generate_data.py
```
This saves `data/housing_data.csv` containing properties priced on a realistic Indian scale (minimum baseline ₹25 Lakhs).

### 4. Train the ML Model
Train the Random Forest regressor pipeline. The script automatically splits the data, applies scaling, fits the model, reports metrics ($R^2 \approx 99.0\%$), and exports the pipeline:
```bash
python src/train.py
```
This generates the saved model file `models/house_price_pipeline.joblib`.

### 5. Launch the Web Dashboard
Start the local Flask development web server:
```bash
python app.py
```
Open your browser and navigate to **`http://127.0.0.1:5000`** to experience the premium valuation dashboard!

---

## 💻 CLI Inference Usage

You can also run prediction queries directly from your terminal using `predict.py`.

### Interactive Mode (Prompt-driven)
Simply run the script with no parameters to be guided through inputs:
```bash
python src/predict.py
```

### Argument Mode
Pass the parameters directly as arguments:
```bash
python src/predict.py --sqft 2500 --bedrooms 4 --bathrooms 3.0 --neighborhood Downtown --year-built 2012 --floors 2 --garage 1
```
**Output Example:**
```text
Predicted Price: Rs. 21,984,210.00
```

---

## 📊 Exploratory Data Analysis & Notebook

To explore the data science workbook, ensure you have Jupyter or VS Code Jupyter Extension running. 
You can open the Jupyter Notebook at:
`notebooks/eda_and_modeling.ipynb`

It guides you through:
1. Data inspection (`info()`, `describe()`)
2. Data Visualization (price distributions, correlation matrices, and location analysis)
3. Comparing **Linear Regression**, **Decision Trees**, and **Random Forest Regressors**
4. Plotting model feature importances (e.g., Square Footage accounts for ~95% of predicted value variation)
