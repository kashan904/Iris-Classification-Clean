# ============================================
# Iris Flower Classification Project
# AIL 201 - Artificial Intelligence Lab
# Final Project
# 
# Group Members:
# - Muhammad Kashan | Roll No: 01-134212-120
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Page configuration
st.set_page_config(page_title="Iris Flower Classification", page_icon="🌸")

# Title and description
st.title("🌸 Iris Flower Classification")
st.markdown("### AIL 201 Artificial Intelligence Lab - Final Project")
st.write("Enter flower measurements below to predict the Iris species!")

# Sidebar for input
st.sidebar.header("📊 Enter Flower Measurements")

# Input fields
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Load dataset
@st.cache_data
def load_data():
    iris = load_iris()
    return iris

iris = load_data()

# Create DataFrame for visualization
def create_sample_df():
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df

df = create_sample_df()

# Train model
@st.cache_data
def train_model():
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

model, X_test, y_test = train_model()

# Predict button
if st.sidebar.button("🔮 Predict Iris Species"):
    # Create input array
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Predict
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    
    # Display results
    st.markdown("---")
    st.header("🎯 Prediction Result")
    
    # Show predicted species
    species_name = iris.target_names[prediction[0]]
    st.success(f"**Predicted Species:** {species_name}")
    
    # Show confidence
    st.write("**Confidence Score:**")
    confidence = prediction_proba[0] * 100
    for i, species in enumerate(iris.target_names):
        st.write(f"  - {species}: {confidence[i]:.2f}%")
    
    # Show input values
    st.info(f"**Input Values:** Sepal Length={sepal_length}cm, Sepal Width={sepal_width}cm, Petal Length={petal_length}cm, Petal Width={petal_width}cm")
    
    # Evaluation metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    st.markdown("---")
    st.header("📈 Model Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy:.2%}")
    col2.metric("Precision", f"{precision:.2%}")
    col3.metric("Recall", f"{recall:.2%}")
    col4.metric("F1-Score", f"{f1:.2%}")
    
    # Confusion Matrix
    st.markdown("---")
    st.header("🔍 Confusion Matrix")
    
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=iris.target_names, 
                yticklabels=iris.target_names, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    st.pyplot(fig)
    plt.close(fig)
    
    # Decision Boundary Visualization
    st.markdown("---")
    st.header("📊 Decision Boundaries (Petal Features)")
    
    # Use only petal features for 2D visualization
    X_petals = iris.data[:, 2:4]  # Petal Length and Width
    y = iris.target
    
    # Train on petal features
    model_petals = RandomForestClassifier(n_estimators=100, random_state=42)
    model_petals.fit(X_petals, y)
    
    # Create mesh
    h = 0.02
    x_min, x_max = X_petals[:, 0].min() - 1, X_petals[:, 0].max() + 1
    y_min, y_max = X_petals[:, 1].min() - 1, X_petals[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict on mesh
    Z = model_petals.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.Set1)
    
    # Scatter plot with different colors for each species
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for i, species in enumerate(iris.target_names):
        mask = y == i
        ax.scatter(X_petals[mask, 0], X_petals[mask, 1], 
                  c=colors[i], label=species, edgecolors='k', s=50)
    
    ax.set_xlabel(iris.feature_names[2])
    ax.set_ylabel(iris.feature_names[3])
    ax.set_title("Decision Boundaries (Petal Length vs Petal Width)")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

# Dataset information
st.markdown("---")
st.header("📋 Dataset Information")

st.write("**Source:** UCI Machine Learning Repository / scikit-learn built-in dataset")
st.write(f"**Total Records:** {len(df)}")
st.write(f"**Features:** {', '.join(iris.feature_names)}")
st.write(f"**Species:** {', '.join(iris.target_names)}")

# Show sample data
st.subheader("Sample Dataset (First 5 rows)")
st.dataframe(df.head())

# Species distribution
st.subheader("Species Distribution")
species_counts = df['species'].value_counts()
fig, ax = plt.subplots(figsize=(6, 4))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
ax.bar(iris.target_names, species_counts.values, color=colors)
ax.set_title("Number of Samples per Species")
ax.set_ylabel("Count")
for i, v in enumerate(species_counts.values):
    ax.text(i, v + 0.5, str(v), ha='center')
st.pyplot(fig)
plt.close(fig)

