
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import kagglehub


class SentimentAnalysis:
    def __init__(self):
        # Download latest version
        path = kagglehub.dataset_download("snap/amazon-fine-food-reviews")
        
        df = pd.read_csv(f"{path}/Reviews.csv")
        
        df.dropna(inplace=True)
        df.isnull().sum()

        df.describe()
        df["sentiment"] = df["Score"].apply(lambda x: "Negative" if x < 3 else "Neutral" if x < 4 else "Positive")
        df.head()


        X = df["Text"]
        Y = df["sentiment"]

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

        self.text_clf = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", LinearSVC())
        ])

        self.text_clf.fit(X_train, Y_train)


    def Predict_sentence(self, sentence: str) -> str:
        
        test = self.text_clf.predict([sentence])

        return test[0]
    

sa = SentimentAnalysis()