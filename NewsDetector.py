import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

def train_model(true_csv_path, fake_csv_path):
    true_df = pd.read_csv(true_csv_path)
    true_df['label'] = 0 

    fake_df = pd.read_csv(fake_csv_path)
    fake_df['label'] = 1  

    df = pd.concat([true_df, fake_df], ignore_index=True)

    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)

    return model

def predict(model, input_text):
    prediction = model.predict([input_text])
    return prediction[0]

if __name__ == "__main__":
    # Example usage
    model = train_model('data/True.csv', 'data/Fake.csv')
    user_input = input("Enter a news article: ")
    prediction = predict(model, user_input)
    if prediction == 0:
        print("The news is likely to be true.")
    else:
        print("The news is likely to be fake.")
