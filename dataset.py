import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re

# Load the dataset
dataset_path = "twcs.csv"
data = pd.read_csv(dataset_path)

# Tokenization
data['tokens'] = data['tweet'].apply(lambda x: x.split())

# Lowercasing
data['tokens_lower'] = data['tokens'].apply(lambda x: [token.lower() for token in x])

# Punctuation Removal
translator = str.maketrans("", "", string.punctuation)
data['tokens_punct_removed'] = data['tokens_lower'].apply(lambda x: [token.translate(translator) for token in x])

# Stop words Removal
stop_words = set(stopwords.words('english'))
data['tokens_no_stop'] = data['tokens_punct_removed'].apply(lambda x: [token for token in x if token not in stop_words])

# Stemming
stemmer = PorterStemmer()
data['stemmed_tokens'] = data['tokens_no_stop'].apply(lambda x: [stemmer.stem(token) for token in x])

# Lemmatization
lemmatizer = WordNetLemmatizer()
data['lemmatized_tokens'] = data['tokens_no_stop'].apply(lambda x: [lemmatizer.lemmatize(token) for token in x])

# Removal of emojis, emoticons, and URLs
def remove_special_characters(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    return text

data['cleaned_tweet'] = data['tweet'].apply(remove_special_characters)

# Now, you can use data['cleaned_tweet'] for your further analysis

# Save the pre-processed data to a new CSV file
preprocessed_data_path = "path_to_save_preprocessed_data.csv"
data.to_csv(preprocessed_data_path, index=False)
