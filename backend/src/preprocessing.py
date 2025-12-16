import re
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()

def clean_resume(text):
    text = text.lower()
    text = re.sub('http\\S+', ' ', text)
    text = re.sub('[^a-zA-Z ]', ' ', text)
    text = re.sub('\\s+', ' ', text)

    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)
