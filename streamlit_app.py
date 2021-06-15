import torch
import streamlit as st
import bs4 as bs
import urllib.request
import re
import time
import nltk
import gensim
import validators

from gensim.summarization import summarize as textrank_summarize

from nltk.tokenize import word_tokenize
from transformers import pipeline

# from transformers import BartTokenizer, BartForConditionalGeneration
# from transformers import T5Tokenizer, T5ForConditionalGeneration

nltk.download('punkt')
nltk.download('stopwords')

st.title('Text Summarization')
st.markdown('With this app you can summarize whatever you want!')

customize = st.checkbox('Customize the settings')
if customize:
    model_name = st.selectbox('Select the model', ('BART', 'T5', 'PEGASUS'))

    if model_name == 'BART':
        _model = "facebook/bart-large-cnn"
        _max_input_length = 512
    elif model_name == 'T5':
        _model = "t5-small"
        _max_input_length = 512
    elif model_name == 'PEGASUS':
        sub_model_name = st.selectbox('Select the pre-trained model', ('Large', 'Bigbird', 'CNN / Dailymail', 'x-sum'))

        if sub_model_name == 'Bigbird':
            _model = "google/bigbird-pegasus-large-arxiv"
            _max_input_length = 4096
        elif sub_model_name == 'CNN / Dailymail':
            _model = "google/pegasus-cnn_dailymail"
            _max_input_length = 1024
        elif sub_model_name == 'x-sum':
            _model = "google/pegasus-xsum"
            _max_input_length = 512
        else:
            _model = "google/pegasus-large"
            _max_input_length = 1024

    _min_length = 150
    _max_length = 300
    col1, col2 = st.beta_columns(2)
    _min_length = col1.number_input("min_length", value=_min_length)
    _max_length = col2.number_input("max_length", value=_max_length)
else:
    model_name = "PEGASUS"
    _model = "google/pegasus-large"
    _max_input_length = 1024
    _min_length = 150
    _max_length = 300

text = st.text_area('Text Input or URL')


def scrap_web(link):
    scraped_data = urllib.request.urlopen(link)
    article = scraped_data.read()

    parsed_article = bs.BeautifulSoup(article, 'lxml')

    paragraphs = parsed_article.find_all('p')

    article_text = ""

    for p in paragraphs:
        article_text += p.text

    return article_text


def preprocessing(article_text):
    # Removing Square Brackets
    article_text = re.sub(r'\[[0-9]*\]', '', article_text)
    article_text = re.sub(r'\[[a-zA-Z]*\]', '', article_text)

    # Removing Extra Spaces
    article_text = re.sub(r'\s+', ' ', article_text)  # 1 ore more whitespace characters

    return article_text


def extractive_summary(input_text, num_tokens, tokenizer):
    num_tokens_aux = num_tokens

    reduced_text = textrank_summarize(input_text, word_count=num_tokens)

    while len(tokenizer.tokenize(reduced_text)) >= num_tokens:
        num_tokens_aux = num_tokens_aux - (len(tokenizer.tokenize(reduced_text)) - num_tokens)
        reduced_text = textrank_summarize(input_text, word_count=num_tokens_aux)

    return reduced_text


def print_time(seconds):
    minutes = seconds // 60
    seconds %= 60
    if int(minutes) == 0:
        st.info(f'Generated in {int(seconds)} s')
    else:
        st.info(f'Generated in {int(minutes)} min, {int(seconds)} s')


def run_model(input_text):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_name == "BART":
        from transformers import BartTokenizer, BartForConditionalGeneration
        model = BartForConditionalGeneration.from_pretrained(_model)
        tokenizer = BartTokenizer.from_pretrained(_model)

    elif model_name == "T5":
        from transformers import T5Tokenizer, T5ForConditionalGeneration

        model = T5ForConditionalGeneration.from_pretrained(_model)
        tokenizer = T5Tokenizer.from_pretrained(_model)

    elif model_name == "PEGASUS":
        from transformers import PegasusTokenizer, PegasusForConditionalGeneration

        model = PegasusForConditionalGeneration.from_pretrained(_model)
        tokenizer = PegasusTokenizer.from_pretrained(_model)

    if len(input_text) > _max_input_length:
        input_text = extractive_summary(input_text, _max_input_length, tokenizer)

    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    output = summarizer(input_text, min_length=_min_length, max_length=_max_length)

    st.write('Summary')
    st.success(output[0]['summary_text'])


if st.button('Submit'):
    if validators.url(text):
        try:
            text = scrap_web(text)
        except urllib.error.HTTPError:
            st.warning("The URL is incorrect. Please try again.")
            st.stop()
        except urllib.error.URLError:
            st.warning("The URL is incorrect. Please try again.")
            st.stop()

    st.info("Your text is being summarized...")

    start = time.time()
    text = preprocessing(text)
    run_model(text)
    end = time.time()

    time = end - start
    print_time(time)