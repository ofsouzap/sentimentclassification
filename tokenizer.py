from typing import List;
import nltk.tokenize;
from dataset import TokenSequence;

def nltk_tokenize(text: str) -> TokenSequence:
    return nltk.tokenize.word_tokenize(text);
