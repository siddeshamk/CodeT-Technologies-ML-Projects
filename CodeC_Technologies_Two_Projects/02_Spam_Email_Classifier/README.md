# 📧 Spam Email Classifier

## Project Overview
A machine-learning text classifier that predicts whether a message is **spam** or **not spam**.

### Technique
- UCI SMS Spam Collection
- TF-IDF text features
- Multinomial Naive Bayes
- Train/test split
- Classification report
- Streamlit interface

The UCI dataset contains 5,574 labeled SMS messages and is intended for classification research.

## Run

```bash
cd 02_Spam_Email_Classifier
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The training script downloads the dataset automatically and saves the trained model under `model/`.

## Command-Line Prediction

After training:

```bash
python predict.py
```

Type a message and the classifier will return the predicted class and confidence.

## How It Works

1. Load labeled messages.
2. Split data into training and test sets.
3. Convert text into TF-IDF features.
4. Train a Multinomial Naive Bayes classifier.
5. Evaluate on unseen test data.
6. Save the trained pipeline.
7. Use the Streamlit UI for predictions.

## Dataset Source

SMS Spam Collection — UCI Machine Learning Repository.

Official dataset page:
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

## Ethical / Practical Note

The model is an educational project and should not be treated as a production email-security system. Real-world deployment should include continuous retraining, adversarial testing, privacy controls, and monitoring.
