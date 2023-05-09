import pandas as pd
import numpy as np
import nltk
import re
import string

from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB

import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from nltk.metrics.scores import precision

# nltk.download('wordnet')
# from bs4 import BeautifulSoup

# ! pip install bs4 # in case you don't have it installed
# Dataset: https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Beauty_v1_00.tsv.gz

""" Helper Function"""

# Average Length Function
def average_length(review_column) -> str:
  return review_column.apply(len).mean()

def average_length_print(before_cleaning, review_column) -> str:
  return str(before_cleaning) + ", " + str(review_column.apply(len).mean())

""" Read Data"""
""" Dataset Path"""
# data = r"D:\University of Southern California\CSCI544 APPLIED NATURAL LANGUAGE PROCESSING\HW1\amazon_reviews_us_Beauty_v1_00.tsv.gz"
data = 'data.tsv'
dataset = pd.read_table(data, on_bad_lines='skip', low_memory=False)

""" Keep Reviews and Ratings"""
InputDataFrame = dataset[['star_rating','review_body']].copy()
# print("Done")

""" We form three classes and select 20000 reviews randomly from each class """
# InputDataFrame.head()
# InputDataFrame.isnull().sum()

# InputDataFrame.info(verbose = True, show_counts = True)
InputDataFrame.dropna(inplace = True)

InputDataFrame['star_rating'] = InputDataFrame['star_rating'].astype('float').astype('int')
# InputDataFrame.isnull().sum()

def assign_class(value):
    if value == 1 or value == 2:
        return 1
    if value == 3:
        return 2
    if value == 4 or value == 5:
        return 3
 
InputDataFrame['class'] = InputDataFrame['star_rating'].map(assign_class)

# InputDataFrame['class'].value_counts()
# display(InputDataFrame)
# InputDataFrame.head()

# print(average_length(InputDataFrame["review_body"]))

InputDataFrame_1 = InputDataFrame.loc[InputDataFrame['class'] == 1].sample(n = 20000)
InputDataFrame_2 = InputDataFrame.loc[InputDataFrame['class'] == 2].sample(n = 20000)
InputDataFrame_3 = InputDataFrame.loc[InputDataFrame['class'] == 3].sample(n = 20000)

InputDataFrames = [InputDataFrame_1, InputDataFrame_2, InputDataFrame_3]
InputDataFrameFinal = pd.concat(InputDataFrames)

# del InputDataFrame_1, InputDataFrame_2, InputDataFrame_3
# InputDataFrameFinal['class'].value_counts()

before_cleaning = average_length(InputDataFrameFinal["review_body"])
# print("Review Average Length : " + str(before_cleaning))

""" Data Cleaning """

class DataCleaning:

  _url = 'https?://\S+|www\.\S+'
  _html = '<[^<]+?>'
  _alphnumeric = '\w*\d\w*'
  _number = '^[\d\s]+'
  _linebresks = '\n'
  _whitespace = ' +'

  def __init__(self, InputDataFrameFinal) -> None:
     self.InputDataFrameFinal = InputDataFrameFinal
  
  def clean(self):
     print("Data Cleaning Process")

class Contraction(DataCleaning):
  def contraction_function(self, review):

      review = re.sub(r"won\'t", "will not", review)
      review = re.sub(r"can\'t", "can not", review)

      review = re.sub(r"n\'t", " not", review)
      review = re.sub(r"\'re", " are", review)
      review = re.sub(r"\'s", " is", review)
      review = re.sub(r"\'d", " would", review)
      review = re.sub(r"\'ll", " will", review)
      review = re.sub(r"\'t", " not", review)
      review = re.sub(r"\'ve", " have", review)
      review = re.sub(r"\'m", " am", review)
      return review

  def clean(self) -> None:
    InputDataFrameFinal['review_body'] = InputDataFrameFinal['review_body'].apply(lambda text: self.contraction_function(text))

class Lowercase(DataCleaning):
  def clean(self) -> None:
    InputDataFrameFinal['review_body'] = InputDataFrameFinal['review_body'].apply(lambda text: str(text).lower())

class RemoveHTMLURL(DataCleaning):
  def html_url(self, review):
    review = re.sub('https?://\S+|www\.\S+', '', review) # html
    review = re.sub('<[^<]+?>', '', review)              # url
    return review

  def clean(self) -> None:
    # InputDataFrameFinal['review_body'] = InputDataFrameFinal['review_body'].apply(lambda text: re.sub(self._html + '|' + self._url , '', text))
    InputDataFrameFinal['review_body'] = InputDataFrameFinal['review_body'].apply(lambda text: self.html_url(text))

class RemoveNonAlphabeticalCharacter(DataCleaning):
  def non_alphabetical_function(self, review):
    # review = re.sub('[!\"#\$%&\'\(\)\*\+,-\./:;<=>\?@\[\\\]\^_`{\|}~]', '', review) # punctuation
    # review = re.sub('[^ \w+]', '', review)                                          # alphnumeric
    # review = re.sub('^[\d\s]+', '', review)                                         # number
    # review = re.sub('\n', '', review)                                               # linebreaks
    return review

  def clean(self) -> None:
    InputDataFrameFinal['review_body'] = InputDataFrameFinal['review_body'].apply(lambda text: re.sub('[^a-zA-Z\s]' , '', text))
    # InputDataFrameFinal['review_body'] = InputDataFrameFinal['review_body'].apply(lambda text: self.non_alphabetical_function(text))

class RemoveExtraSpaces(DataCleaning):
  def clean(self) -> None:
    InputDataFrameFinal['review_body'] = InputDataFrameFinal['review_body'].apply(lambda text: re.sub(' +', ' ', text))

data_cleaning = DataCleaning(InputDataFrameFinal)

contraction = Contraction(InputDataFrameFinal)
contraction.clean()
# print("Review Average Length : " + str(average_length(InputDataFrameFinal["review_body"])))
# print(average_length_print(before_cleaning, InputDataFrameFinal["review_body"]))

data_cleaning_lowercase = Lowercase(InputDataFrameFinal)
data_cleaning_lowercase.clean()
# print("Review Average Length : " + str(average_length(InputDataFrameFinal["review_body"])))
# print(average_length_print(before_cleaning, InputDataFrameFinal["review_body"]))

data_cleaning_remove_html_url = RemoveHTMLURL(InputDataFrameFinal)
data_cleaning_remove_html_url.clean()
# print("Review Average Length : " + str(average_length(InputDataFrameFinal["review_body"])))
# print(average_length_print(before_cleaning, InputDataFrameFinal["review_body"]))

remove_non_alphabetical_character = RemoveNonAlphabeticalCharacter(InputDataFrameFinal)
remove_non_alphabetical_character.clean()
# print("Review Average Length : " + str(average_length(InputDataFrameFinal["review_body"])))
# print(average_length_print(before_cleaning, InputDataFrameFinal["review_body"]))

remove_white_space = RemoveExtraSpaces(InputDataFrameFinal)
remove_white_space.clean()
# print("Review Average Length : " + str(average_length(InputDataFrameFinal["review_body"])))
# print(average_length_print(before_cleaning, InputDataFrameFinal["review_body"]))

# display(InputDataFrameFinal)
# InputDataFrameFinal.head()

# InputDataFrameFinal = InputDataFrameFinal.sample(frac = 1)

# display(InputDataFrameFinal)
# InputDataFrameFinal.head()

# print("Average length of reviews before and after data cleaning")
print(average_length_print(before_cleaning, InputDataFrameFinal["review_body"]))
# print("Done")

""" Pre-processing"""

before_pre_processing = average_length(InputDataFrameFinal["review_body"])

# before_stop_words = average_length(InputDataFrameFinal["review_body"])
# print("Review Average Length : " + str(before_stop_words))

""" remove the stop words """

## Reference - https://www.geeksforgeeks.org/removing-stop-words-nltk-python/

# from nltk.corpus import stopwords

# nltk.download('stopwords')
# stop_words=stopwords.words('english')

# stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with'}

# def remove_stopword() -> None:
#   InputDataFrameFinal['review_body'] = InputDataFrameFinal['review_body'].apply(lambda text: ' '.join([word for word in text.split() if word not in (stop_words)]))

# remove_stopword()
# display(InputDataFrameFinal)
# InputDataFrameFinal.head()
# print(average_length_print(before_stop_words, InputDataFrameFinal["review_body"]))

""" perform lemmatization  """

# before_lemmatization = average_length(InputDataFrameFinal["review_body"])
# print("Review Average Length : " + str(before_lemmatization))

# InputDataFrameFinal.head()

def lemmatization(review) -> None:
  lemmatizer = WordNetLemmatizer()
  # print(review)
  lemmatized_sentence = []
  for word, tag in pos_tag(review):
    if (review is None):
        return review
    else:
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
  # print(lemmatized_sentence)
  return lemmatized_sentence

InputDataFrameFinal['review_body'] = InputDataFrameFinal['review_body'].apply(lambda text: ' '.join(lemmatization(text.split()) ))

# InputDataFrameFinal.head()
# print("Average length of reviews before and after data preprocessing")
print(average_length_print(before_pre_processing, InputDataFrameFinal["review_body"]))
# print("Done")

""" TF-IDF Feature Extraction"""

# max_features=5000, ngram_range=(2,2)

TF_IDF = TfidfVectorizer(ngram_range=(1,4))
X = TF_IDF.fit_transform(InputDataFrameFinal["review_body"])
Y = InputDataFrameFinal['class']

x_train ,x_test, y_train, y_test=train_test_split(X, Y, test_size=0.2, random_state=100)

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

def Evaluation_Metric(y_test,y_pred) -> None:
  precision = precision_score(y_test,y_pred, average=None)
  recall = recall_score(y_test,y_pred, average=None)
  f1 = f1_score(y_test,y_pred, average=None)

  for index in range(3):
    print("{}, {}, {}".format(precision[index], recall[index], f1[index]))
  print("{}, {}, {}".format(precision.mean(), recall.mean(), f1.mean()))

""" Perceptron """

# model_perceptron = Perceptron(tol=1e-3, random_state=0)
model_perceptron = Perceptron()
model_perceptron.fit(x_train,y_train)
y_pred = model_perceptron.predict(x_test)
# print('Precision, Recall, and f1-score for the testing split for Perceptron')
Evaluation_Metric(y_test,y_pred)
# print(accuracy_score(y_test,y_pred))
# print(metrics.classification_report(y_test,y_pred))
# print(precision_score(y_test,y_pred, average=None))

""" SVM """

# model_svm = LinearSVC(C=0.5, max_iter=2000)
model_svm = LinearSVC()
model_svm.fit(x_train,y_train)
y_pred = model_svm.predict(x_test)
# print('Precision, Recall, and f1-score for the testing split for SVM')
Evaluation_Metric(y_test,y_pred)
# print(accuracy_score(y_test,y_pred))
# print(metrics.classification_report(y_test,y_pred))

""" Logistic Regression """

model_logistic_regression=LogisticRegression(max_iter = 10000)
model_logistic_regression.fit(x_train,y_train)
y_pred = model_logistic_regression.predict(x_test)
# print('Precision, Recall, and f1-score for the testing split for Logistic Regression')
Evaluation_Metric(y_test,y_pred)
# print(accuracy_score(y_test,y_pred))
# print(metrics.classification_report(y_test,y_pred))

""" Naive Bayes """

model_naive_bayes = MultinomialNB()
model_naive_bayes.fit(x_train,y_train)
y_pred = model_naive_bayes.predict(x_test)
# print('Precision, Recall, and f1-score for the testing split for Naive Bayes')
Evaluation_Metric(y_test,y_pred)
# print(accuracy_score(y_test,y_pred))
# print(metrics.classification_report(y_test,y_pred))