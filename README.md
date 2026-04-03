# Fake News Detection System

This project uses Machine Learning and Natural Language Processing (NLP) to classify news articles as Fake or Real.

## Features
- Text preprocessing using NLTK
- TF-IDF feature extraction
- Multiple model comparison
- Best model selection
- Interactive web interface using Streamlit

## Dataset
WELFake Dataset (72k news articles)

## Models Used
- Logistic Regression
- Linear SVC
- Random Forest
- Multinomial Naive Bayes

## Accuracy
Best Model: Logistic Regression  
Accuracy: 93%+

## Run the Project

Install dependencies

pip install -r requirements.txt

Run training

python train_model.py

Run web app

streamlit run app.py
