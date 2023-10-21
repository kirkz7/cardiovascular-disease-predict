# 🩺🫀 Cardiovascular Diseases Predictor

## Predicting Cardiovascular Diseases with Machine Learning 👩‍⚕️

Welcome to the **Cardiovascular Diseases Predictor** project repository! In this innovative venture, we harness the power of advanced machine learning models to accurately foresee the presence or absence of cardiovascular diseases. Our analysis delves deep into a comprehensive health dataset encompassing 70,000 patients, providing invaluable insights into the realm of heart health.

### 🌟 Proud Participants of the Borealis AI (RBC Research Institute) LET'S SOLVE IT Summer 2022 Mentorship Program

We are thrilled to announce that our dedicated team members were active participants in the prestigious **Borealis AI (RBC Research Institute) LET'S SOLVE IT Summer 2022** mentorship program. This transformative experience has not only sharpened our skills but has also infused our project with unique perspectives and cutting-edge methodologies. The knowledge gained during this program has played a pivotal role in shaping our approach, making our Cardiovascular Diseases Predictor one of its kind.

Join us on this exciting journey as we endeavor to revolutionize healthcare through the lens of data-driven insights! 🚀

## Getting Started

### Tools

Before you begin, make sure you have the following tools installed on your system:

#### 1. **Jupyter Notebook**

Jupyter Notebook is included in the Anaconda distribution. If you've installed Anaconda (mentioned below), you should already have Jupyter Notebook installed. To start Jupyter Notebook, open your terminal (or Anaconda Prompt on Windows) and type:

```bash
jupyter notebook
```

#### 2. **Anaconda**

Anaconda is a distribution of Python and other scientific libraries for data science and machine learning. You can download and install Anaconda from the official website: [Anaconda Download](https://www.anaconda.com/products/distribution)

#### 3. **Scikit-learn**

Scikit-learn can be installed using pip, which is a package manager for Python. Open your terminal and run:

```bash
pip install scikit-learn
```

#### 4. **Pandas**

Pandas is also installed using pip. Run the following command in your terminal:

```bash
pip install pandas
```

#### 5. **NumPy**

NumPy is a fundamental package for scientific computing with Python. Install it using pip:

```bash
pip install numpy
```

#### 6. **Matplotlib**

Matplotlib is a popular data visualization library. You can install it via pip:

```bash
pip install matplotlib
```

#### 7. **PyTorch**

For installing PyTorch, you can visit the official website and select the appropriate installation command based on your system: [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)

#### 8. **TensorFlow**

Similar to PyTorch, TensorFlow installation commands can be found on the official TensorFlow website: [TensorFlow Installation Guide](https://www.tensorflow.org/install)

#### 9. **Visual Studio Code**

Visual Studio Code is a popular code editor developed by Microsoft. You can download and install it from the official website: [Visual Studio Code Download](https://code.visualstudio.com/download)


## Project Overview

Cardiovascular diseases are a major global health concern, responsible for millions of deaths annually. This project aims to predict the presence or absence of cardiovascular diseases using machine learning models, providing a more affordable and convenient alternative to traditional diagnostic methods.

### Problem Statement

Cardiovascular diseases are often undetected due to the prohibitive costs of diagnostic tests, leading to delayed treatments. This project addresses this issue by developing models that predict cardiovascular diseases based on patient data.

### How AI/ML Solves the Problem

Using a dataset of 70,000 patients, this project employs various machine learning algorithms, including logistic regression, decision tree, random forest, KNN, and XGBoost, to build predictive models. The data includes objective features (age, height, weight, gender), examination features (blood pressure, cholesterol level, glucose level), and lifestyle habits (smoking, alcohol intake, physical activity).

## Project Details

### Preprocessing

The dataset was normalized and balanced, ensuring fair representation of patients with and without cardiovascular diseases. Key features like gender, smoking, and cholesterol levels were analyzed using bar graphs to understand their impact.

### Model Building

Five different models were implemented and optimized using grid search and random search. The models' performance was evaluated using accuracy and AUC ROC scores. Decision tree was chosen for further analysis, including testing gender-based datasets.

### Additional Models

1. **Neural Network (PyTorch):**
   - Implemented a neural network using PyTorch to improve accuracy scores. Explored various layers and loss functions for optimization.
  
2. **Unsupervised Learning (K-Means Clustering):**
   - Utilized the Elbow Method to determine the optimal number of clusters. Trained the model using the decision tree, but accuracy scores showed no significant changes.

### Project Outcomes

The project achieved approximately 72% accuracy using models such as logistic regression, decision tree, random forest, KNN, XGBoost, and PyTorch. Systolic blood pressure emerged as the most important feature correlated with cardiovascular diseases.

### Insights and Conclusions

- **Gender:** Contrary to initial hypotheses, gender did not significantly impact accuracy scores in predicting cardiovascular diseases.
  
- **BMI:** While height and weight were not direct indicators, BMI emerged as an essential feature correlating with cardiovascular diseases.

## Impact of the LSI Program

Participating in the LSI program provided valuable insights into deep learning frameworks, including neural networks and unsupervised learning techniques. The mentor's guidance enhanced the project's organization and effectiveness, facilitating the development of reusable Python functions.

## Additional Resources

- **Project GitHub Repository:** [Cardiovascular Disease Prediction](https://github.com/kirkz7/cardiovascular-disease-predict)
- **Dataset:** [Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)


## Authors

Xuning Zhang (Kirk)

Hanyun Guo (Doris)

Xiaowei Zhang (Vivian)

Shuhan Dong (Bella)

Jia Hu (Judie)
## References

- **Cardiovascular Disease Diagnosis:** [News Medical](https://www.news-medical.net/health/Cardiovascular-Disease-Diagnosis.aspx)
- **Ontario, CA MRI Cost Comparison:** [New Choice Health](https://www.newchoicehealth.com/places/california/ontario/mri)
- **Confusion Matrix:** [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2021/05/in-depth-understanding-of-confusion-matrix/)
- **Gender Differences in Cardiovascular Risk Factors:** [NCBI](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5465115/)
- **PyTorch Documentation:** [PyTorch Linear Layers](https://pytorch.org/docs/stable/nn.html#linear-layers)
