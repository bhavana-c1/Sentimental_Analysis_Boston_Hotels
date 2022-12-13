# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 20:31:23 2022

@author: Bhavana
"""
from inspect import signature

import nltk
import pandas as pd
from nltk.corpus import wordnet
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Lasso
from wordcloud import WordCloud
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
#from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
#from sklearn.metrics import average_precision_score, precision_recall_curve,r2_score
from sklearn.metrics import confusion_matrix

# Trim dataset for sentimental analysis #
def initial_feature_selection(hotel_reviews):
    # append the positive and negative text reviews
    hotel_reviews["review_text"] = hotel_reviews["review_text_positive"] + "" + hotel_reviews["review_text_negative"]
    # create a new label
    # bad reviews have overall ratings <= 5
    # good reviews have overall ratings > 5
    hotel_reviews.info()
    hotel_reviews["is_bad_review"] = hotel_reviews["review_score_badge"].apply(lambda x: 1 if x < 5 else 0)
    # select only review text and category column
    #return hotel_reviews[["review_text", "is_bad_review"]]
    return hotel_reviews



def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def text_cleaning(text):
    # lower text
    text = text.lower()
    # tokenize text and remove punctuation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # perform lemmatization
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return text


# Data Cleaning #
def data_cleaning(hotel_reviews):
    # remove NaNs
    print(hotel_reviews["review_text"].isna().sum())
    hotel_reviews = hotel_reviews.dropna()
    print("Na dropped")
    hotel_reviews.info()
    # remove 'N/As' from text
    hotel_reviews['review_text'] = hotel_reviews['review_text'].apply(lambda x: x.replace("N/A", " "))
    # clean text reviews
    hotel_reviews['raw_clean_text'] = hotel_reviews['review_text'].apply(lambda x: text_cleaning(x))
    return hotel_reviews


# Feature engineering #
def feature_engineering(hotel_reviews):
    # add sentiment analysis columns

    sentiment_analyzer = SentimentIntensityAnalyzer()
    hotel_reviews["sentiments"] = hotel_reviews["review_text"].apply(lambda x: sentiment_analyzer.polarity_scores(x))
    hotel_reviews = pd.concat(
        [hotel_reviews.drop(['sentiments'], axis=1), hotel_reviews['sentiments'].apply(pd.Series)], axis=1)

    # add total number of characters as column
    hotel_reviews["total_num_chars"] = hotel_reviews["review_text"].apply(lambda x: len(x))

    # add total number of words as column
    hotel_reviews["total_num_words"] = hotel_reviews["review_text"].apply(lambda x: len(x.split(" ")))

    documents = [TaggedDocument(doc, [i]) for i, doc in
                 enumerate(hotel_reviews["raw_clean_text"].apply(lambda x: x.split(" ")))]

    # train a Doc2Vec model with our text data
    model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

    # transform each document into a vector data
    doc2vec_df = hotel_reviews["raw_clean_text"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
    doc2vec_df.columns = ["vector" + str(x) for x in doc2vec_df.columns]
    hotel_reviews = pd.concat([hotel_reviews, doc2vec_df], axis=1)

    # add tf-idfs columns

    tfidf = TfidfVectorizer(min_df=10)
    tfidf_result = tfidf.fit_transform(hotel_reviews["raw_clean_text"]).toarray()
    tfidf_df = pd.DataFrame(tfidf_result, columns=tfidf.get_feature_names())
    tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
    tfidf_df.index = hotel_reviews.index
    return pd.concat([hotel_reviews, tfidf_df], axis=1)


def show_wordcloud(reviews,color, title=None):
    # excluding words from wordcloud. 
    stopwords=["as","is","has","have","did","with","nNot","Th","it","","The","Name","dtype","review_text","was","of","and","our","in","to","review_text_negative","review_text_positive","oppor","Length","object"]
    wordcloud = WordCloud(
        stopwords=stopwords,
        background_color='white',
        max_words=200,
        max_font_size=40,
        scale=3,
        colormap=color,
        random_state=42
    ).generate(str(reviews))

    fig = plt.figure(1, figsize=(20, 20))
    plt.axis('off')
    if title:
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()


# Exploratory data analysis #
def data_analysis(hotel_reviews):
    # show is_bad_review distribution
    print(hotel_reviews.head())
    print("is_bad_review", hotel_reviews["is_bad_review"].value_counts(normalize=True))

    # print wordcloud for all reviews
    show_wordcloud(hotel_reviews["review_text"],"brg")
    # highest positive sentiment reviews (with more than 5 words)
    hotel_reviews[hotel_reviews["total_num_words"] >= 5].sort_values("pos", ascending=False)[
        ["review_text", "pos"]].head(10)

    # lowest negative sentiment reviews (with more than 5 words)
    hotel_reviews[hotel_reviews["total_num_words"] >= 5].sort_values("neg", ascending=False)[
        ["review_text", "neg"]].head(10)

    # plot sentiment distribution for positive and negative reviews
    for x in [0, 1]:
        subset = hotel_reviews[hotel_reviews['is_bad_review'] == x]

        # Draw the density plot
        if x == 0:
            label = "Good reviews"
        else:
            label = "Bad reviews"
        sns.displot(subset['compound'], label=label)


# Press the green button in the gutter to run the script.
def modeling_using_RT(hotel_reviews, X_train, X_test, y_train, y_test):
    # train a random forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=65)
    rf = rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    # Print the confusion matrix using Matplotlib
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
     
    labels=['Good Review','Bad Review']
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels,rotation=90)
    ax.xaxis.set_label_position('top')
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()
    print("Score for random forest", rf.score(X_test, y_test))


def modeling_using_logistic(raw_clean_text, X_train, X_test, y_train, y_test):
    # Define the model
    model = LogisticRegression(random_state=0, solver='lbfgs',
                               multi_class='multinomial')
    # Use the logistic regression model to make sentiment label predictions
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    # Print the confusion matrix using Matplotlib
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    
    labels=['Good Review','Bad Review']
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels,rotation=90)
    ax.xaxis.set_label_position('top')
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()
    # score
    print("Score for logistic regression", model.score(X_test, y_test))


def download_packages():
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('vader_lexicon')



if __name__ == '__main__':
    download_packages()
    # load data into df
    hotel_reviews_df = pd.read_csv("H:\Brandeis\Study\Sem 3(Fall22)\Marketing analytics\Final project\codes\hotel_reviews.csv")
    print(hotel_reviews_df.columns)
    print("Before cleaning")
    hotel_reviews_df.info()
    hotel_reviews_df = initial_feature_selection(hotel_reviews_df)
    print(hotel_reviews_df.head(25))
    hotel_reviews_df = data_cleaning(hotel_reviews_df)
    print("After cleaning")
    # print wordcloud for positive reviews
    show_wordcloud(hotel_reviews_df["review_text_positive"],"Blues")
    # print wordcloud for negative reviews
    show_wordcloud(hotel_reviews_df["review_text_negative"],"Reds")
    hotel_reviews_df=hotel_reviews_df[["raw_clean_text","review_text", "is_bad_review"]]
    hotel_reviews_df.info()
    hotel_reviews_df = feature_engineering(hotel_reviews_df)
    hotel_reviews_df.head()
    hotel_reviews_df.shape
    data_analysis(hotel_reviews_df)
    # feature selection
    y = hotel_reviews_df['is_bad_review']
    X = hotel_reviews_df.drop(columns=['is_bad_review', 'review_text', 'raw_clean_text'])
    
    # split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=65)
    modeling_using_RT(hotel_reviews_df, X_train, X_test, y_train, y_test)
    modeling_using_logistic(hotel_reviews_df, X_train, X_test, y_train, y_test)
    