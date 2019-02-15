from flask import Flask, render_template, request, url_for
from flask_bootstrap import Bootstrap
from textblob import TextBlob, Word
import random
import spacy

###
app = Flask(__name__)
Bootstrap(app)


@app.route('/')
def index():
	return render_template('index.html')

@app.route('/image_classification')
def image_classification():
	return render_template('image_classification.html')

"""
Spell Correction
"""
@app.route('/spell_correction')
def spell_correction():
	return render_template('spell_correction.html')



@app.route('/spell_correction_analyse', methods=['POST'])
def spell_correction_analyse():
	if request.method == 'POST':
		text = request.form['text']
		correction = TextBlob(text).correct()
	return render_template('spell_correction.html', correction=correction)

"""
Document Similarity Analysis
"""
@app.route('/document_similarity')
def document_similarity():
	return render_template('document_similarity.html')


@app.route('/document_similarity_analyse', methods=['POST'])
def document_similarity_analyse():
	if request.method == 'POST':
		text1 = request.form['text1']
		text2 = request.form['text2']
		nlp = spacy.load('en')
		doc1 = nlp(text1)
		doc2 = nlp(text2)
		similarity = doc1.similarity(doc2)
	return render_template('document_similarity.html', similarity=similarity)
	

"""
Sentiment Analysis
"""
@app.route('/nlp')
def nlp():
	return render_template('nlp.html')

@app.route('/nlp_analyse', methods=['POST'])
def nlp_analyse():
	if request.method == 'POST':
		rawtext = request.form['rawtext']
		#NLP Stuff
		blob = TextBlob(rawtext)
		received_text2 = blob
		blob_sentiment, blob_subjectivity = blob.sentiment.polarity, blob.sentiment.subjectivity
		number_of_tokens = len(list(blob.words))
		nouns = list()
		for word, tag in blob.tags:
			if tag == 'NN':
				nouns.append(word.lemmatize())
				len_of_words = len(nouns)
				rand_words = random.sample(nouns, len(nouns))
				final_word = list()
				for item in rand_words:
					word = Word(item).pluralize()
					final_word.append(word)
	return render_template('nlp.html', received_text=received_text2, number_of_tokens=number_of_tokens, blob_sentiment=blob_sentiment, blob_subjectivity=blob_subjectivity)


if __name__ == '__main__':
   app.run(debug = True)