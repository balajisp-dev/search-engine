"""Text -> tokens.

Produces exactly the same tokens as bm25s.tokenize(..., stopwords="en", stemmer=<english Snowball>,
return_ids=False), which is what lets MyBM25Retriever match the baseline:

    1. lowercase
    2. split with the regex TOKEN_PATTERN (runs of 2+ word characters)
    3. drop STOPWORDS
    4. stem each token with PyStemmer's English Snowball stemmer

Stopwords go before stemming: some words stem into a stopword ("ins" -> "in") and must be kept.
Docs and queries must go through the same Analyzer instance.
"""
import re

import Stemmer

TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")

# bm25s.stopwords.STOPWORDS_EN, copied so the engine doesn't depend on the baseline library
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
    "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these", "they",
    "this", "to", "was", "will", "with",
})


class Analyzer:
    def __init__(self, language: str = "english") -> None:
        self.stemmer=Stemmer.Stemmer(language)

    def analyze(self, text: str) -> list[str]:
        """Return the tokens for one text, in order (duplicates kept — BM25 needs the counts)."""
        text=text.lower();
        words=TOKEN_PATTERN.findall(text)
        filtered_words=[word for word in words if word not in STOPWORDS]
        tokens=self.stemmer.stemWords(filtered_words)
        return tokens

