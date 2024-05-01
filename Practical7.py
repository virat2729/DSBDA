import nltk
nltk.download('punkt')

-----------------------------------------------------------------------------------------------------

import nltk
from nltk.tokenize import word_tokenize
text = "Welcome to the Python Programming at Indeed Insprining Infotech"
print(word_tokenize(text))

------------------------------------------------------------------------------------------------------

from nltk.tokenize import sent_tokenize
text = "Hello Everyone. Welcome to the Python Programming"
print(sent_tokenize(text))

----------------------------------------------------------------------------------------------------

from nltk.stem import PorterStemmer
# words = ['Wait','Waiting','Waited','Waits']
words = ['clean','cleaning','cleans','cleaned']
ps = PorterStemmer()
for w in words:
    words=ps.stem(w)
    print(words)

-------------------------------------------------------------------------------------------------

import nltk
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
text  = "studies studying floors cry"
tokenization = nltk.word_tokenize(text)
for w in tokenization:
    print('Stemming for ', w,'is',porter_stemmer.stem(w))

--------------------------------------------------------------------------------------------------

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

--------------------------------------------------------------------------------------------------

import nltk
from nltk.stem import WordNetLemmatizer
Wordnet_lemmatizer = WordNetLemmatizer()
text  = "studies study floors cry"
tokenization = nltk.word_tokenize(text)
for w in tokenization:
    print('Lemma for ', w,'is',Wordnet_lemmatizer.lemmatize(w))

-------------------------------------------------------------------------------------------------

nltk.download('stopwords')

--------------------------------------------------------------------------------------------------

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
data = 'AI was introduced in the year 1956 but it gained popularity recently.'
stopwords = set(stopwords.words('english'))
words = word_tokenize(data)
wordsFiltered = []
for w in words:
    if w not in stopwords:
        wordsFiltered.append(w)
print(wordsFiltered)

----------------------------------------------------------------------------------------------------

print(len(stopwords))
print(stopwords)

----------------------------------------------------------------------------------------------------

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

----------------------------------------------------------------------------------------------------

document = "This is an example document that we will use to demonstrate document preprocessing."

----------------------------------------------------------------------------------------------------

tokens = word_tokenize(document)

----------------------------------------------------------------------------------------------------

tokens

----------------------------------------------------------------------------------------------------

import nltk
nltk.download('averaged_perceptron_tagger')

----------------------------------------------------------------------------------------------------

pos_tags = pos_tag(tokens)

----------------------------------------------------------------------------------------------------

pos_tags

----------------------------------------------------------------------------------------------------

stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if not word.lower() in stop_words]

----------------------------------------------------------------------------------------------------

ps = PorterStemmer()
stemmed_tokens = [ps.stem(word) for word in filtered_tokens]

----------------------------------------------------------------------------------------------------

wnl = WordNetLemmatizer()
lemmatized_tokens = [wnl.lemmatize(word) for word in filtered_tokens]

----------------------------------------------------------------------------------------------------

print("Tokens: ", tokens)
print("POS tags: ", pos_tags)
print("Filtered tokens: ", filtered_tokens)
print("Stemmed tokens: ", stemmed_tokens)
print("Lemmatized tokens: ", lemmatized_tokens)

----------------------------------------------------------------------------------------------------

import math
from collections import Counter

----------------------------------------------------------------------------------------------------

corpus = [
    'The quick brown fox jumps over the lazy dog',
    'The brown fox is quick',
    'The lazy dog is sleeping'
]

----------------------------------------------------------------------------------------------------

tokenized_docs = [doc.lower().split() for doc in corpus]

----------------------------------------------------------------------------------------------------

tf_docs = [Counter(tokens) for tokens in tokenized_docs]

----------------------------------------------------------------------------------------------------

n_docs = len(corpus)
idf = {}
for tokens in tokenized_docs:
    for token in set(tokens):
        idf[token] = idf.get(token, 0) + 1
for token in idf:
    idf[token] = math.log(n_docs / idf[token])

----------------------------------------------------------------------------------------------------

tfidf_docs = []
for tf_doc in tf_docs:
    tfidf_doc = {}
    for token, freq in tf_doc.items():
        tfidf_doc[token] = freq * idf[token]
    tfidf_docs.append(tfidf_doc)

----------------------------------------------------------------------------------------------------

for i, tfidf_doc in enumerate(tfidf_docs):
    print(f"Document {i+1}: {tfidf_doc}")