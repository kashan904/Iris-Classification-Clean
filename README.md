# 🌸 Iris Flower Classification

## AIL 201 – Artificial Intelligence Lab
### Final Project

---

## Group Members

| Name | Roll Number |
|------|-------------|
| Muhammad Kashan | 01-134212-120 |


---

## Project Description

This project classifies Iris flowers into 3 species (Setosa, Versicolor, Virginica) using machine learning. Users input flower measurements through a web interface and receive instant species prediction.

---

## Technologies Used

| Category | Technology |
|----------|------------|
| Language | Python 3.8+ |
| Web Framework | Streamlit |
| ML Library | scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Data Handling | Pandas, NumPy |

---

## ML Model

- **Algorithm:** Random Forest Classifier
- **Accuracy:** ~95-97%
- **Dataset:** Iris Dataset (150 samples, 4 features)

---

## Features

- ✅ Web-based interface (no installation needed)
- ✅ Instant prediction on input
- ✅ Matplotlib visualizations
- ✅ User-friendly design

---

## Dataset Description

**Source:** UCI Machine Learning Repository / scikit-learn built-in dataset

**Features (4):**
1. Sepal Length (cm)
2. Sepal Width (cm)
3. Petal Length (cm)
4. Petal Width (cm)

**Records:** 150 flowers (50 from each species)

**Species:**
- Iris-Setosa
- Iris-Versicolor
- Iris-Virginica

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Steps

1. **Clone or download the repository**
   ```bash
   git clone https://github.com/[kashan904]/Iris-Classification.git
   cd Iris-Classification
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open in browser**
   - Automatic: Browser opens to `http://localhost:8501`
   - Manual: Go to `http://localhost:8501`

---

## How to Use the App

1. Enter flower measurements in the sidebar:
   - Sepal Length (cm)
   - Sepal Width (cm)
   - Petal Length (cm)
   - Petal Width (cm)

2. Click **"Predict Iris Species"**

3. View the predicted species and visualizations

---

## Project Structure
