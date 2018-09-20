from __future__ import division

# import a newspaper article 
from newspaper import Article
url = 'https://www.nytimes.com/2018/09/19/world/europe/blind-brexit-theresa-may-european-union.html'
article = Article(url)
article.download()
article.parse()
txt = article.text

# IGNORE: paragraphs will be taken into account in the next version of the code 
paragraphs = txt.split(sep = "\n\n")
paragraphs = [paragraph for paragraph in paragraphs if len(paragraph) > 1]

# split the article's text into sentences 
import spacy 
nlp = spacy.load('en')
doc = nlp(txt)
sents = []
for sent in doc.sents:
    sents.append(sent)

# import packages for cleaning 
import re 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.corpus import wordnet as wn

# find the WordNet tag equivalent of the nltk part of speech tag
def pos_to_wordnet(tag):
    if tag.startswith('NN'):
        return wn.NOUN
    elif tag.startswith('VB'):
        return wn.VERB
    elif tag.startswith('JJ'):
        return wn.ADJ
    elif tag.startswith('RB'):
        return wn.ADV
    else:
        return 'n'

# lemmatize a given word based on its part of speech tag
def lemmatize(word):
    word_doc = nlp(word)
    word_tag = ''
    for token in word_doc:
        word_tag = pos_to_wordnet(token.tag_)
    return lemmatizer.lemmatize(word, pos=word_tag)

# clean the sentences: remove stopwords, lemmatize, remove punctuation/odd characters (keeping digits)
def clean_sentence(sent):
    sent = re.sub('[^a-zA-Z0-9]', ' ', sent)
    sent = sent.lower()
    sent = sent.split()
    sent = [lemmatize(word) for word in sent if not word in set(stopwords.words('english'))]
    sent = ' '.join(sent)
    return sent

# clean the sentences, store in alternate list 
clean_sents = [clean_sentence(sent.text) for sent in sents]
for i in range((len(clean_sents) - 1), -1, -1):
    if len(clean_sents[i]) <= 1:
        del sents[i]
        del clean_sents[i]

# create a vector representation of the cleaned sentences
# load glove's pre-set word vectors in, store the words, vectors, and a dictionary of them 
filename = 'glove.6B/glove.6B.300d.txt'
words = []
vectors = []
word_vectors = {} 
file = open(filename, 'r', encoding='UTF-8')
for line in file.readlines():
    row = line.strip().split(' ')
    word = row[0]
    vector = [float(i) for i in row[1:]]
    words.append(word)
    vectors.append(vector)
    word_vectors[word] = vector
file.close()
    
# get the sentence vector for a given sentence (input = list of words), negation words act as subtraction
import numpy as np 
negation_words = ['no', 'not']
def get_sentence_vector(sent):
    sent = sent.split()
    sent = [word for word in sent if not len(word) < 2]
    sentence_vector = np.zeros((300, ), dtype='float64')
    for i in range(0, len(sent)):
        word = sent[i]
        if word in negation_words and (i + 1) < len(sent):
            word = sent[i + 1]
            vector = np.array(word_vectors[word]) if word in words else np.zeros((300,))
            sentence_vector = sentence_vector - vector 
            i = i + 1
        else:
            vector = np.array(word_vectors[word]) if word in words else np.zeros((300,))
            sentence_vector = sentence_vector + vector 
    return sentence_vector
sentence_vects = [get_sentence_vector(sent) for sent in clean_sents]

# create a matrix of computed cosine similarities between sentence vectors
from scipy import spatial 
def compute_cosine_sim(vect1, vect2):
    sim = 1 - spatial.distance.cosine(vect1, vect2)
    return sim
def get_similarity_matrix(vectors):
    sim_matrix = np.zeros((len(vectors), len(vectors)))
    for i in range(0, len(vectors)):
        main_vect = vectors[i]
        for j in range(0, len(vectors)):
            other_vect = vectors[j]
            sim_matrix[i][j] = compute_cosine_sim(main_vect, other_vect)
    return sim_matrix
sim_matrix = get_similarity_matrix(sentence_vects)

# create a matrix of 1's and 0's. 1 if the sentence's similarity to the other sentence is greater than
# or equal to the mean similarity score of the set, 0 if it is less than
def get_sim_matrix_thresholds(sim_matrix):
    mean = (sim_matrix.sum() - len(sim_matrix)) / (len(sim_matrix) * (len(sim_matrix) - 1))
    threshold_matrix = np.zeros((len(sim_matrix), len(sim_matrix)))
    for i in range(0, len(sim_matrix)):
        for j in range(0, len(sim_matrix)):
            if i == j or sim_matrix[i][j] < mean:
                threshold_matrix[i][j] = 0
            else:
                threshold_matrix[i][j] = 1
    return threshold_matrix
threshold_matrix = get_sim_matrix_thresholds(sim_matrix)

# true or false if the sentence has a similarity greater than or equal to the mean similarity score 
# of the given similarity matrix 
def set_covers_sentence(set_indices, other_index, threshold_matrix):
    for index in set_indices:
        if index != other_index and threshold_matrix[index][other_index] == 1:
            return True
    return False

# get the percentage of sentences that a given set covers above threshold similarity (mean)
def get_set_threshold_percent(summary_indices, threshold_matrix):
    sent_sum = 0
    for i in range(0, len(threshold_matrix)):
        sent_sum = (sent_sum + 1) if set_covers_sentence(summary_indices, i, threshold_matrix) else sent_sum
    return sent_sum / len(threshold_matrix)

# get the average similarity score of the sentences in a set and a separate sentence 
def get_average_sim_score(summary_indices, other_index, sim_matrix):
    score_sum = 0.0
    num_scores = 0
    for set_index in summary_indices:
        if not set_index == other_index:
            score_sum += sim_matrix[set_index][other_index]
            num_scores += 1
    return score_sum / num_scores

# get the average similarity score of the sentences in a set and all sentences 
def get_set_average_score(summary_indices, sim_matrix):
    set_score_sum = 0.0
    for i in range(0, len(sim_matrix)):
        set_score_sum += get_average_sim_score(summary_indices, i, sim_matrix)
    return set_score_sum / len(sim_matrix)

# compute a sentence set's total score using this function (takes into account similarity score of sentences
# in set with other sentences as well as the amount of unique information covered). Change alpha/beta as needed.
def compute_set_score(summary_indices, sim_matrix, threshold_matrix):
    alpha = 1.0 # test with different values
    beta = 1.0 # test with different values 
    score = (alpha * get_set_average_score(summary_indices, sim_matrix)) + (beta * get_set_threshold_percent(summary_indices, threshold_matrix))
    return score

# Do exhaustive search to find set of N sentences with maximum similarity + threshold aggregate score 
num_sents = 3 # put in the number of sentences to extract
def search_combos(num_sents, indices, sents, sim_matrix, threshold_matrix):
    max_score = 0
    highest_combo = {}
    if not len(indices) + 1 == num_sents:
        start_index = (max(indices) + 1) if len(indices) > 0 else 0
        end_index = len(clean_sents) - (num_sents - len(indices) - 1)
        for index in range(start_index, end_index):
            current_set = search_combos(num_sents, indices + [index], sents, sim_matrix, threshold_matrix)
            score = max(current_set.keys())
            if score > max_score:
                if len(highest_combo) > 0:
                    del highest_combo[max_score]
                max_score = score
                highest_combo[max_score] = current_set[score]
    else:
        for index in range(max(indices) + 1, len(clean_sents)):
            score = compute_set_score(indices + [index], sim_matrix, threshold_matrix)
            if score > max_score:
                if len(highest_combo) > 0:
                    del highest_combo[max_score]
                highest_combo[score] = indices + [index]
                max_score = score
    return highest_combo
top_combos = search_combos(num_sents, [], sents, sim_matrix, threshold_matrix)

# Print the sentence summary 
summary_sentences = [sents[index].text.strip() for index in top_combos[max(top_combos.keys())]]
summary_sentences = '\n'.join(summary_sentences)
print(summary_sentences)


    
    

    




