# 🏃‍♂️ Human Activity Recognition Using Smartphones

A machine learning project to classify human activities using smartphone sensor data with a custom-built neural network implementation.

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.19+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-1.2+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-0.24+-blue.svg)

## 📋 Project Overview

This project implements a neural network from scratch to classify human activities using smartphone sensor data. We focus on three key activities:
- Walking
- Walking Downstairs
- Laying

### 🎯 Key Features

- Custom neural network implementation with backpropagation
- Adam optimizer with momentum
- Advanced data preprocessing pipeline
- Comprehensive model evaluation
- Detailed misclassification analysis

## 🔍 Data Understanding

The dataset comes from the UCI Machine Learning Repository's "Human Activity Recognition Using Smartphones" dataset, collected from 30 volunteers performing various activities while carrying a smartphone.

### 📊 Dataset Statistics
- Total samples: 10,299
- Training samples: 7,352 (71.5%)
- Test samples: 2,947 (28.5%)
- Features: Focused on gyroscope data (time domain signals)

## 🛠 Implementation

### Data Preprocessing
- Outlier removal using IQR method
- Feature standardization
- One-hot encoding for labels
- Train-test split maintenance

### Neural Network Architecture
```
Input Layer (26 features)
    ↓
Dense Hidden Layer (64 neurons, ReLU)
    ↓
Output Layer (3 neurons, Softmax)
```

### Key Components
- Adam optimizer for parameter updates
- ReLU activation for hidden layers
- Softmax activation for output layer
- Categorical cross-entropy loss
- L2 regularization

## 📈 Results

The model achieved impressive results across different activities:

### Performance Metrics
- Overall Accuracy: 90%
- LAYING: 100% accuracy
- WALKING: 84% F1-score
- WALKING_DOWNSTAIRS: 79% F1-score

### ROC-AUC Scores
- LAYING: 1.00
- WALKING: 0.96
- WALKING_DOWNSTAIRS: 0.96

## 🔎 Key Insights

1. Perfect Classification for Static Activity
   - The model achieves 100% accuracy in identifying "laying" activity
   - Clear distinction between static and dynamic activities

2. Challenge Areas
   - Some confusion between walking and walking downstairs
   - Dynamic activities show pattern overlap
   - Lower accuracy in distinguishing similar movement patterns

## 👥 Team Members

- Krisna Arinugraha Liantara - 225150207111022
- Muhammad Hasan Fadhlillah - 225150207111026
- Muhammad Husain Fadhlillah - 225150207111027

## 📚 Dependencies

- numpy
- pandas
- seaborn
- matplotlib
- scikit-learn

## 🏃‍♀️ Running the Project

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Jupyter notebook
4. Follow the step-by-step implementation

## 📊 Visualizations

The project includes comprehensive visualizations:
- Activity distribution plots
- Feature analysis
- Learning curves
- Confusion matrix
- ROC curves
- Misclassification analysis

## 🎯 Future Improvements

1. Feature Engineering
   - Explore additional sensor data
   - Implement more sophisticated feature extraction

2. Model Enhancements
   - Test different network architectures
   - Implement dropout layers
   - Experiment with different optimizers

3. Activity Recognition
   - Include more activities
   - Real-time prediction capabilities
