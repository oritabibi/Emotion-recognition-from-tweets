import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import twint
import re
import emoji
import os
import matplotlib.pyplot as plt


@st.cache(allow_output_mutation=True)
def load_tokenizer(path: str):
    """
    load tokenizer for processing words corpus
    :parameter
    (str) : path to tokenizer location
    :returns
    (tokenizer) : tokenizer for all corpus
    """
    try:
        with open(path, 'rb') as handle:
            tok = pickle.load(handle)
        return tok
    except IOError:
        print('tokenizer not found at path - ', path)


def get_sequences(tok, tweets: list, max_len: int = 50):
    """
    make all sequences the same input size for our model, len is 50
    :parameter
    (tokenizer) tok : tokenizer of our corpus
    (list) tweets : list of tweets
    (int) max_len : max length of padded sequence
    :returns
    (list) : list of all padded tweets
    """
    sequences = tok.texts_to_sequences(tweets)  # get all tweets as sequences (numbers that define each word)
    padded = pad_sequences(sequences, truncating='post', padding='post', maxlen=max_len)
    return padded


@st.cache
def get_classes():
    """ return list of classes"""
    return ['anger', 'fear', 'joy', 'sadness', 'surprise', 'love']


@st.cache
def get_classes_to_index():
    """return dictionary of emotions and the corresponding index"""
    return {'anger': 0, 'fear': 1, 'joy': 2, 'sadness': 3, 'surprise': 4, 'love': 5}


@st.cache
def get_index_to_classes():
    """return dictionary of indexes and the corresponding emotion"""
    return {0: 'anger', 1: 'fear', 2: 'joy', 3: 'sadness', 4: 'surprise', 5: 'love'}


def get_model(path: str):
    """load model"""
    return tf.keras.models.load_model(path)


def scrap_tweets(key_to_search: str, start_time, end_time, num_of_tweets: int = 200, lang: str = 'en'
                 , json_output: str = 'output.json'):
    """scarp tweets and save the data with json format"""
    if os.path.isfile(json_output):
        os.remove(json_output)
    # Configure
    c = twint.Config()
    c.Search = key_to_search  # Here, I am searching for tweets mentioning “covid19”.
    c.Lang = lang  # Searching tweets which are only english language.
    c.Store_json = True
    c.Output = json_output
    c.Limit = num_of_tweets  # I am scraping 100 tweets.

    if start_date != 0 and end_date != 0:
        c.since = start_time
        c.until = end_time

    twint.run.Search(c)


def cleaner(tweet: str):
    """
    clean tweet for model usage
    :returns
    (str) : cleaned tweet
    """
    tweet = re.sub("@[A-Za-z0-9]+","",tweet) #Remove @ sign
    tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet)  # Remove http links
    tweet = " ".join(tweet.split())
    tweet = emoji.get_emoji_regexp().sub(u'', tweet)  # remove emojis from text
    tweet = tweet.replace("#", "").replace("_", " ")  # Remove hashtag sign but keep the text
    tweet = tweet.lower() # upper case to lower case
    tweet = ''.join([i for i in tweet if not i.isdigit()])  # remove digits
    return " ".join(tweet.split())


def get_scrapped_tweets(df: pd.DataFrame):
    """
    read tweets from data frame to a list
    :returns
    (list) tweets_list : tweets as list of strings
    """
    data.reset_index(drop=True, inplace=True)
    tweets_list = []
    for i in range(df.shape[0]):
        tweets_list.append(df['tweet'][i])
    return tweets_list


def get_count_predictions(classes_list: list):
    """
    create a dictionary of each class as keys initialize with 0
    :parameter
    (list) classes_list : list of all classes(emotions)
    :returns
    (dictionary) : dictionary of each class as keys initialize with 0
    """
    return dict((k, 0) for k in classes_list)


def predict_all_tweets(tweets_seq: list, predictions_dict: dict):
    """
    create prediction of emotion for each tweet
    :parameter
    (list) tweets_seq : list of sequences representing each tweets
    (dict) predictions_dict : dictionary to count how many predictions for each class(emotion)
    :returns
    (list) predictions : list of predictions for tweets sorted by the index of each tweet
    """
    predictions = []
    for i in range(len(tweets_seq)):
        # expand dim to make the input shape correct to only 1 tweet
        pred = model.predict(np.expand_dims(tweets_seq[i], axis=0))[0]
        pred_class = index_to_class[np.argmax(pred).astype('uint8')]
        predictions.append(pred_class)
        predictions_dict[pred_class] += 1

    return predictions


def get_pie_chart(count_dict: dict):
    """create pie chart from dictionary, labels is keys, data is values"""
    labels = count_dict.keys()
    sizes = count_dict.values()
    explode = (0.1, 0.1, 0.1, 0.1, 0.2, 0.1)

    fig1 = plt.figure()
    # plt.rcParams['font.size'] = 9.0
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', explode=explode)
    return fig1


def get_hist_chart(predictions: list):
    """create an histogram from values of a list """
    fig1 = plt.figure()
    plt.hist(predictions, bins=11)
    return fig1


# load variables
classes = get_classes()
class_to_index = get_classes_to_index()
index_to_class = get_index_to_classes()
tokenizer = load_tokenizer('tokenizer.pickle')
model = model = get_model('Tweet_Emotion_model.h5')

# side bar view
start_date = 0
end_date = 0
key_search = st.sidebar.text_input('search:')
date_check = st.sidebar.checkbox('select time range')

if date_check:
    # show date input for user if check_date is true
    start_date = st.sidebar.date_input('start date:', help='starting date for tweets search.')
    end_date = st.sidebar.date_input('end date:', help='end date for tweets search.')
else:
    # hide date input for user if check_date is false
    start_date = 0
    end_date = 0

is_clicked = st.sidebar.button('predict', help='press to scrap tweets and get predictions for them.')

# main view
st.title('Emotion recognition from tweets')

if start_date > end_date:
    # check that the user enter correct range of dates
    st.sidebar.warning('Please check your date input ')

if is_clicked:
    # check if the user enter a search word
    if key_search == "":
        st.sidebar.warning('Please enter a search word ')
    if start_date <= end_date and key_search != '':
        # if all variables checked out preform scraping and show results

        # scrap tweets according to user input
        scrap_tweets(key_to_search=key_search, start_time=start_date, end_time=end_date)
        # read the tweets using dataframe
        data = pd.read_json('output.json', lines=True)
        # keep only tweets in english
        data = data.loc[lambda df: df['language'] == 'en']
        data = data.head(100)
        # clean tweets for model
        data['tweet'] = data['tweet'].map(lambda x: cleaner(x))

        # process tweets for the model and get predictions
        scraped_tweets = get_scrapped_tweets(data)
        scraped_tweets_seq = get_sequences(tokenizer, scraped_tweets)
        count_predictions_dict = get_count_predictions(classes)
        all_predictions = predict_all_tweets(scraped_tweets_seq, count_predictions_dict)

        # show pie chart
        pie = get_pie_chart(count_predictions_dict)
        st.pyplot(pie)

        st.write('')
        st.write('')

        # show hist chart
        h = get_hist_chart(all_predictions)
        st.pyplot(h)

        st.write('')
        st.write('')

        # show tweets and predictions as dataframe
        st.table(
            pd.DataFrame({
            'tweet': scraped_tweets,
            'predicted emotion': all_predictions,
         }))

