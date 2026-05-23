# System Architecture

## Data Flow Diagram

User Input (Sidebar Sliders)
↓
[sepal_length, sepal_width, petal_length, petal_width]
↓
NumPy Array (1, 4)
↓
Random Forest Classifier
↓
├── predict() → Species Class
└── predict_proba() → Confidence Scores
↓
Streamlit GUI Output



## Components
| Component | Technology | Purpose |
|-----------|------------|---------|
| Frontend | Streamlit | User interface |
| ML Model | scikit-learn Random Forest | Classification |
| Data Handling | Pandas/NumPy | Feature processing |
