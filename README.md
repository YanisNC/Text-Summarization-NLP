# Text-Summarization-NLP

I've developed an API for text summarization, which can be used through streamlit.

To use the unlimited version with a lot of different transformer models, follow the instructions:

### Steps to run the app on your local machine:
#### 1. Install the requirements.txt packages.
#### 2. Run the following command in your local terminal (same directory as the .py file)
#####    $ *streamlit run streamlit_app_unlimited.py*

<br /><br />
----
<br /><br />

<p align="center">
    Abstract - Final Master's Thesis
</p>

<p align="justify">
Natural Language Processing (NLP) applications have boomed in recent years. Its use for
automatic text summarization (ATS) has been showing surprising results, that are getting
closer to those of humans. This paper aims to provide a historical overview to get a good
context of the subject, starting from extractive algorithms, which generate summaries by
concatenating the most important sentences, to abstractive algorithms, which generate the
summary using sentences that do not appear in the original text, in the same way as humans
do. The paper shows the performance of some of the SOTA algorithms and offers a solution
to one of their major problems: The limited size of text they can support as input. Given this
limitation, it has been opted for the combination of extractive and abstractive techniques to
achieve a robust algorithm that can deal with any input size, where the extractive algorithm
is responsible for selecting the most relevant sentences of the original text and passes this
selection to the abstractive model, with an adjusted size to the number of tokens that this
model is able to process. In this way, an abstractive summary is obtained. Using this
algorithm, an application has been developed so that any user can perform summaries in a
simple way. Moreover, the user can choose among the different algorithms and models that
are available and adjust the summary to his needs, customizing different parameters such as
the desired summary length.
</p>
