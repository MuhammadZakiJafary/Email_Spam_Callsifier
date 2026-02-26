📧 Email Spam Classifier (Logistic Regression)
A Machine Learning-based Email Spam Classifier that predicts whether an email is Spam or Not Spam using Natural Language Processing (NLP) techniques and a Logistic Regression model.

🚀 Project Overview
This project automatically detects spam emails by preprocessing raw text data and applying a supervised learning algorithm. The model converts email text into numerical form using TF-IDF vectorization and then classifies it using Logistic Regression.

🧠 Technologies Used
Python
Pandas
NumPy
Scikit-learn
NLTK
TF-IDF Vectorizer

⚙️ Workflow
Data Collection
Data Cleaning & Preprocessing
Lowercasing
Removing punctuation
Stopword removal
Tokenization
Feature Extraction using TF-IDF
Train-Test Split
Model Training using Logistic Regression
Model Evaluation

🤖 Machine Learning Model
The project uses:
Logistic Regression (Supervised Classification Algorithm)
Logistic Regression is a powerful linear model used for binary classification problems. It estimates the probability that an email belongs to the spam class and makes predictions based on a decision threshold.

📊 Model Performance
Accuracy on testing data: 95%
Accuracy on training data: 96%

📁 Project Structure
Email-Spam-Classifier/
│
├── dataset.csv
├── spam_classifier.py
├── model.pkl
├── requirements.txt
└── README.md
▶️ How to Run the Project

2️⃣ Install required libraries
pip install -r requirements.txt

3️⃣ Run the script
python spam_classifier.py
🎯 Future Improvements
Deploy using Flask
Use advanced vectorization techniques
Compare with Naive Bayes and SVM
Build a simple web interface

👨‍💻 Author
Muhammad Zaki
BS Software Engineering
Aspiring AI & Machine Learning Engineer
