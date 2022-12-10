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
from wordcloud import WordCloud
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve


# Trim dataset for sentimental analysis #
def initial_feature_selection(hotel_reviews):
    # append the positive and negative text reviews
    hotel_reviews["review_text"] = hotel_reviews["review_text_positive"] + "" + hotel_reviews["review_text_negative"]
    # create  a new label
    # bad reviews have overall ratings < 5
    # good reviews have overall ratings >= 5
    hotel_reviews["is_bad_review"] = hotel_reviews["review_score_badge"].apply(lambda x: 1 if x < 5 else 0)
    # select only review text and category column
    return hotel_reviews[["review_text", "is_bad_review"]]


# Data Sampling #
def data_sampling(hotel_reviews):
    return hotel_reviews.sample(frac=0.1, replace=False, random_state=40)


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
    hotel_reviews = hotel_reviews.dropna()
    # remove 'N/As' from text
    hotel_reviews.loc[: "review_text"] = hotel_reviews["review_text"].apply(lambda x: x.replace("N/A", ""))
    # clean text reviews
    hotel_reviews.loc[:"raw_clean_text"] = hotel_reviews["review_text"].apply(lambda x: text_cleaning(x))
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
    tfidf_df = pd.DataFrame(tfidf_result, columns=tfidf.get_feature_names_out())
    tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
    tfidf_df.index = hotel_reviews.index
    return pd.concat([hotel_reviews, tfidf_df], axis=1)


def show_wordcloud(reviews, title=None):
    wordcloud = WordCloud(
        background_color='white',
        max_words=200,
        max_font_size=40,
        scale=3,
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
    hotel_reviews["is_bad_review"].value_counts(normalize=True)
    # print wordcloud
    show_wordcloud(hotel_reviews["review_text"])
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
def modeling(hotel_reviews):
    # feature selection
    label = "is_bad_review"
    ignore_cols = [label, "review_text", "raw_clean_text"]
    features = [c for c in hotel_reviews.columns if c not in ignore_cols]

    # split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(hotel_reviews[features], hotel_reviews[label], test_size=0.20,
                                                        random_state=40)

    # train a random forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=40)
    rf.fit(X_train, y_train)

    # show feature importance
    important_features_df = pd.DataFrame({"feature": features, "importance": rf.feature_importances_}).sort_values(
        "importance", ascending=False)
    important_features_df.head(20)
    # y_prediction = [x[1] for x in rf.predict_proba(X_test)]
    # fpr, tpr, thresholds = roc_curve(y_test, y_prediction, pos_label=1)
    #
    # roc_auc = auc(fpr, tpr)
    #
    # plt.figure(1, figsize=(15, 10))
    # lw = 2
    # plt.plot(fpr, tpr, color='darkred',
    #          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.show()
    #
    # average_precision = average_precision_score(y_test, y_prediction)
    #
    # precision, recall, _ = precision_recall_curve(y_test, y_prediction)
    #
    # # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    # step_kwargs = ({'step': 'post'}
    #                if 'step' in signature(plt.fill_between).parameters
    #                else {})
    #
    # plt.figure(1, figsize=(15, 10))
    # plt.step(recall, precision, color='b', alpha=0.2,
    #          where='post')
    # plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    #
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))


def download_packages():
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('vader_lexicon')


if __name__ == '__main__':
    download_packages()
    # load data into df
    hotel_reviews_df = pd.read_csv("hotel_reviews.csv")
    hotel_reviews_df = initial_feature_selection(hotel_reviews_df)
    hotel_reviews_df = data_sampling(hotel_reviews_df)
    hotel_reviews_df = data_cleaning(hotel_reviews_df)
    hotel_reviews_df = feature_engineering(hotel_reviews_df)
    hotel_reviews_df.head()
    hotel_reviews_df.shape
    data_analysis(hotel_reviews_df)
    modeling(hotel_reviews_df)
