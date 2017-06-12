from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
import pickle

tfidf_model = pickle.load(open("vectorizer.pkl", "rb" ) )
pickle_fname = 'logreg_pickle.model'
pickle_model = pickle.load(open(pickle_fname, 'rb'))

def review_to_words(raw_review):
    review_text = BeautifulSoup(raw_review).get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    cleaned_text = [" ".join(words)]

    return cleaned_text

def predict_data(data):
    pre_data = review_to_words(data)
    tfidf_data = tfidf_model.transform(pre_data)
    result = pickle_model.predict(tfidf_data)

    return result
