# 💻 Laptop Price Prediction — Machine Learning Project
Overview:
This project predicts laptop prices based on their hardware specifications using various machine learning regression models.
The dataset includes 1300+ laptop records with features like CPU, GPU, RAM, screen resolution, and storage configuration.
The objective was to build a predictive model that accurately estimates a laptop’s price by analyzing its specifications and hardware attributes.

# ⚙️ Tech Stack
Languages: Python
Libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, XGBoost
Tools: Jupyter Notebook

# Dataset:
Key columns:
Company – Manufacturer brand (e.g., Dell, HP, Apple)
TypeName – Category (Notebook, Ultrabook, Gaming, etc.)
Inches – Screen size
Cpu, Gpu – Processor and graphics details
Ram – Memory capacity (GB)
Memory – Type and size of storage (HDD/SSD/Hybrid)
OpSys – Operating System
Weight – Device weight in kilograms
Price – Target variable (in INR)

# 🔧 Data Preprocessing
Removed null and duplicate records.
Extracted and cleaned numerical features from text (e.g., RAM, Weight, Resolution).
Derived new engineered features:
Touchscreen and IPS flags from ScreenResolution
ppi (pixels per inch) = √(X² + Y²) / Inches
CPU/GPU brand encoding
Memory split into HDD, SSD, Hybrid, Flash Storage
Converted categorical variables using One-Hot Encoding.
Normalized target variable using np.log() for regression stability.

# 📊 Exploratory Data Analysis
Seaborn and Matplotlib used for visualizations:
Brand-wise average price comparison
Price vs RAM, PPI, Weight, and GPU trends

# 🧠 Model Development
Multiple regression algorithms were trained and evaluated:
Model	R² Score	Mean Absolute Error
Linear Regression	0.81	0.21
Ridge Regression	0.81	0.20
Lasso Regression	0.80	0.21
KNN Regressor	0.80	0.19
Decision Tree	0.84	0.18
Random Forest	0.89	0.16
Extra Trees	0.88	0.16
Gradient Boosting	0.88	0.16
AdaBoost	0.79	0.23
XGBoost	Trained later with similar high accuracy	
The Random Forest Regressor achieved the best performance with an R² score of 0.887.
Correlation heatmaps between engineered features and price
Observed strong positive correlation between RAM, SSD size, and Price.

# 📈 Results & Insights
Feature engineering (especially ppi and SSD extraction) greatly improved accuracy.
High-end components like Intel i7, SSD > 256GB, and IPS display were strong price indicators.
Light-weight laptops and gaming categories exhibited higher variance in pricing.
