# 🎯 FocusIQ – Student Productivity Prediction

## 📌 Project Overview

FocusIQ is a Machine Learning-based application designed to analyze and predict student productivity based on various factors influencing their daily academic performance.

The project aims to provide insights into student productivity through data analysis, machine learning classification, and an interactive application.

## 🎯 Objectives

- Analyze factors affecting student productivity.
- Preprocess and prepare the dataset for machine learning.
- Build and train a classification model.
- Predict student productivity levels.
- Evaluate model performance using a confusion matrix and accuracy metrics.
- Provide an interactive interface for users.

## 🛠️ Technologies Used

- **Python** – Programming language
- **Pandas** – Data manipulation and analysis
- **NumPy** – Numerical computations
- **Scikit-learn** – Machine Learning model development
- **Matplotlib & Seaborn** – Data visualization
- **Streamlit** – Interactive application interface (if used in app.py)
- **Jupyter Notebook** – Model development and experimentation

## 📂 Project Structure

```text
finalfocus/
│
├── assets/                     # Project assets
├── app.py                      # Application source code
├── final_pro.ipynb             # Machine learning notebook
├── final.pptx                  # Project presentation
├── focusiq_scalerfinal.pkl     # Trained feature scaler
├── focusiqfinal.pkl            # Saved machine learning model
├── requirements.txt            # Required Python libraries
├── student_productivity...csv  # Dataset
├── README.md                   # Project documentation
└── .gitignore                  # Ignored files
```

## ⚙️ Project Workflow

1. **Data Collection:** Collect student productivity-related data.
2. **Data Preprocessing:** Clean the dataset and prepare features.
3. **Exploratory Data Analysis:** Understand patterns and relationships in the data.
4. **Model Development:** Train a machine learning classification model.
5. **Model Evaluation:** Evaluate performance using accuracy and confusion matrix.
6. **Model Deployment:** Integrate the trained model into an interactive application.

## 📊 Model Evaluation

The classification model is evaluated using a confusion matrix to measure its prediction performance across three classes.

**Evaluation Accuracy: 77.13%**

The confusion matrix helps identify correctly classified samples and misclassifications across the different classes.

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

### 2. Navigate to the project folder

```bash
cd YOUR_REPOSITORY
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

If the application is built using Streamlit:

```bash
streamlit run app.py
```

## 📈 Expected Outcome

FocusIQ provides a machine learning-based approach to understanding student productivity patterns and predicting productivity categories. It demonstrates how data science and machine learning can be applied to student performance analysis.

## 🔮 Future Improvements

- Improve model accuracy through hyperparameter tuning.
- Explore additional machine learning algorithms.
- Add more student-related features.
- Enhance the user interface and visualization.
- Deploy the application online.

## 👩‍💻 Author

**Your Name**

Machine Learning | Python | Data Science

---

⭐ If you find this project interesting, feel free to explore the repository!
