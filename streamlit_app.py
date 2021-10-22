import torch
import streamlit as st
import bs4 as bs
import urllib.request
import re
import time
import nltk
import validators

from gensim.summarization import summarize as textrank_summarize

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer

from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.kl import KLSummarizer

from transformers import pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration

nltk.download('punkt')
nltk.download('stopwords')

st.title('Summarizer')
st.markdown('Paste a text or link below to summarize it...')

customize = st.checkbox('Customize the settings')
if customize:
    st.caption('Extractive: Creates the summary using sentences from the original text.')
    st.caption('Abstractive: Creates the summary making new sentences.')
    summary_type = st.selectbox('Select the type of summary', ('Extractive', 'Abstractive'))
    if summary_type == 'Extractive':
        extractive_name = st.selectbox('Select the algorithm', ('TextRank', 'Luhn', 'LexRank', 'LSA', 'KL-Sum'))

        if extractive_name == 'TextRank':
            _length_words = 200
            _length_words = st.number_input("length (in words)", value=_length_words)
        else:
            _length_sentences = 5
            _length_sentences = st.number_input("length (in sentences)", value=_length_sentences)
    else:
        st.caption('* : Due to limited resources, some models are only available in the unlimited version in my GitHub. Here you can experiment with the smaller ones (T5). To run the other models follow the instructions in the README.')
        model_name = st.selectbox('Select the model', ('T5', 'BART (not available)*', 'Longformer (not available)*', 'PEGASUS (not available)*'))

        if model_name == 'T5':
            sub_model_name = st.selectbox('Select the pre-trained model', ('Small', 'Base (not available)*', 'Large (not available)*'))

            if sub_model_name == 'Small':
                _model = "t5-small"
                _max_input_length = 512
            elif sub_model_name == 'Base (not available)*':
                _model = "t5-small" #instead of 't5-base' -> to avoid a resources error
                _max_input_length = 512
            else:
                _model = "t5-small " #instead of 't5-large' -> to avoid a resources error
                _max_input_length = 512

        elif model_name == 'BART (not available)*':
            sub_model_name = st.selectbox('Select the pre-trained model', ('BART', 'DistilBART'))
            st.stop()
            if sub_model_name == 'BART':
                _model = "facebook/bart-large-cnn"
                _max_input_length = 512
            else:
                _model = "sshleifer/distilbart-cnn-12-6"
                _max_input_length = 1024
                
        elif model_name == 'Longformer (not available)*':
            st.stop()
            _model = "patrickvonplaten/longformer2roberta-cnn_dailymail-fp16"
            _model_tokenizer = "allenai/longformer-base-4096"
            _max_input_length = 4096

        elif model_name == 'PEGASUS (not available)*':
            sub_model_name = st.selectbox('Select the pre-trained model',
                                          ('Large', 'CNN / Dailymail', 'x-sum'))                        
            st.stop()
            if sub_model_name == 'Bigbird':  # currently disabled
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
        _min_length = col1.number_input("min_length (in words)", value=_min_length)
        _max_length = col2.number_input("max_length (in words)", value=_max_length)

else:  # default
    summary_type = 'Extractive'
    extractive_name = 'TextRank'
    _length_words = 200

text = st.text_area(label='Text Input or URL', height=200, value='https://en.wikipedia.org/wiki/Artificial_intelligence')


def scrap_web(link):
    scraped_data = urllib.request.urlopen(link)
    article = scraped_data.read()

    parsed_article = bs.BeautifulSoup(article, 'lxml')

    paragraphs = parsed_article.find_all('p')

    article_text = ""

    for p in paragraphs:
        article_text += p.text

    return article_text


def scrap_web2(link):
    scraped_data = urllib.request.urlopen(link)
    article = scraped_data.read()
    soup = bs.BeautifulSoup(article, 'html.parser')
    for s in soup(['script', 'style']):
        s.extract()
    return (soup.text.strip()).encode('ascii', 'ignore').decode("utf-8")


def preprocessing(text_input):
    # Removing Square Brackets
    text_input = re.sub(r'\[[0-9]*\]', '', text_input)
    text_input = re.sub(r'\[[a-zA-Z]*\]', '', text_input)

    # Removing Extra Spaces
    text_input = re.sub(r'\s+', ' ', text_input)  # 1 ore more whitespace characters

    return text_input


def postprocessing(text_input):
    text_input = re.sub(r'[<][a-zA-Z][>]', ' ', text_input)
    return text_input


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
    if summary_type == 'Abstractive':
        if model_name == "T5":
            #from transformers import T5Tokenizer, T5ForConditionalGeneration

            model = T5ForConditionalGeneration.from_pretrained(_model)
            tokenizer = T5Tokenizer.from_pretrained(_model)
        elif model_name == "BART":
            from transformers import BartTokenizer, BartForConditionalGeneration

            model = BartForConditionalGeneration.from_pretrained(_model)
            tokenizer = BartTokenizer.from_pretrained(_model)

        elif model_name == "Longformer":
            from transformers import LongformerTokenizer, EncoderDecoderModel

            model = EncoderDecoderModel.from_pretrained(_model)
            tokenizer = LongformerTokenizer.from_pretrained(_model_tokenizer)

        elif model_name == "PEGASUS":
            from transformers import PegasusTokenizer, PegasusForConditionalGeneration

            model = PegasusForConditionalGeneration.from_pretrained(_model)
            tokenizer = PegasusTokenizer.from_pretrained(_model)

        if len(input_text) > _max_input_length:
            input_text = extractive_summary(input_text, _max_input_length, tokenizer)

        summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
        output = summarizer(input_text, min_length=_min_length, max_length=_max_length)

        st.write('Summary')
        st.success(postprocessing(output[0]['summary_text']))

    else:
        if extractive_name == 'TextRank':
            output = textrank_summarize(input_text, word_count=_length_words)
        else:
            # Initializing the parser
            parser = PlaintextParser.from_string(input_text, Tokenizer('english'))

            if extractive_name == 'Luhn':
                summarizer = LuhnSummarizer()
                output_list = summarizer(parser.document, sentences_count=_length_sentences)
            elif extractive_name == 'Luhn':
                summarizer = LexRankSummarizer()
                output_list = summarizer(parser.document, sentences_count=_length_sentences)
            elif extractive_name == 'Luhn':
                summarizer = LsaSummarizer()
                output_list = summarizer(parser.document, sentences_count=_length_sentences)
            else:
                summarizer = KLSummarizer()
                output_list = summarizer(parser.document, sentences_count=_length_sentences)

            sent_list = []
            for sent in output_list:
                sent_list.append(str(sent))
            output = ' '.join(sent_list)

        st.write('Summary')
        st.success(postprocessing(output))


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
        # except ValueError:
        # print("ValueError")
        # text = scrap_web2(text)

    st.info("Your text is being summarized...")

    start = time.time()
    text = preprocessing(text)
    run_model(text)
    end = time.time()

    time = end - start
    print_time(time)

st.caption('https://github.com/YanisNC/Text-Summarization-NLP')
