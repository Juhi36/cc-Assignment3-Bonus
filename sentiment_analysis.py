
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

class SentimentAnalysis:
    def __init__(self):
        df = pd.read_csv("../../Module2/Reviews/Reviews.csv")

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