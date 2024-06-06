import telebot
import joblib
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer

TOKEN = 'my_token'
bot = telebot.TeleBot(TOKEN)


def expand_contractions(text):
    contractions = {
        "won't": "will not",
        "can't": "cannot",
        "n't": " not",
        "'re": " are",
        "'s": " is",
        "'d": " would",
        "'ll": " will",
        "'ve": " have",
        "'m": " am"
    }
    for contraction, expansion in contractions.items():
        text = re.sub(contraction, expansion, text)
    return text


stop_words = set(nltk_stopwords.words('english'))
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()


def process_synopsis(synopsis, stop_words, stemmer, lemmatizer):
    synopsis = re.sub(r"http\S+", "", synopsis)
    synopsis = BeautifulSoup(synopsis, 'lxml').get_text()
    synopsis = expand_contractions(synopsis)
    synopsis = re.sub(r"\S*\d\S*", "", synopsis).strip()
    synopsis = re.sub(r'[^A-Za-z]+', ' ', synopsis)

    processed_words = [
        stemmer.stem(lemmatizer.lemmatize(word.lower())).encode('utf8')
        for word in synopsis.split() if word.lower() not in stop_words
    ]

    return b' '.join(processed_words)


def load_model(model_path):
    model = joblib.load(model_path)
    return model


@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id,
                     f'Hi, {message.from_user.first_name}! Send me a movie description and I will predict the tags.')


@bot.message_handler(func=lambda message: True)
def handle_message(message):
    model = load_model('best_model.pkl')
    vectorizer = load_model('tfidf_vectorizer.pkl')
    mlb = load_model('mlb.pkl')

    user_text = message.text
    processed_text = process_synopsis(user_text, stop_words, stemmer, lemmatizer)
    processed_text_decoded = processed_text.decode("utf-8")
    vectorized_text = vectorizer.transform([processed_text_decoded])

    prediction = model.predict(vectorized_text)
    tags = mlb.inverse_transform(prediction)

    tags_list = [tag for sublist in tags for tag in sublist]
    tags_str = ', '.join(tags_list)

    bot.send_message(message.chat.id, f'Predicted tags: {tags_str}')


if __name__ == '__main__':
    bot.polling(none_stop=True)
