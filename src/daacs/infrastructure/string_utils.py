import re
import nltk
from nltk.corpus import stopwords

class StringUtils:
    @staticmethod
    def clean_sentence(val):
        """Removes special characters, lowercases, and removes stop words."""
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('english'))
        regex = re.compile('([^\s\w]|_)+')
        sentence = regex.sub('', val).lower()
        words = sentence.split()
        filtered_words = [word for word in words if word not in stop_words]
        return " ".join(filtered_words)

    @staticmethod
    def encode_sentence(val, remove_stopwords:bool = False) -> list:
        """Removes special characters (except periods and commas), lowercases, 
        and optionally removes stop words.
        """
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('english'))
        regex = re.compile('([^\s\w.,]|_)+')  # Adjusted to keep periods and commas
        sentence = regex.sub('', val).lower()
        words = sentence.split()

        if remove_stopwords :
            return [word for word in words if word not in stop_words]
        else:
            return words  # Return a list of words
    