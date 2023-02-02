import numpy as np
import matplotlib.pyplot as plt
from rake_nltk import Rake
from create_dataset import *
import re
import pickle
from collections import Counter
import random
import math
from create_dataset import *
from nltk.corpus import wordnet as wn
from feature_vector import *
import time
from quick_query_sim_ev import *
from scipy import stats


def new_word_probability(d):
    mu = 8
    return (float(mu)/float(d+mu))*(float(1)/float(d))

def make_candidate_QS3plus():
    (doc_collection_lm, doc_collection_lm_binary) = pickle.load(open("../TREC_Robust_data/all_doc_language_model.pk", "rb"))
    d = sum(doc_collection_lm.values())
    total_num_words = sum(doc_collection_lm.values())
    doc_collection_lm_dist = {term:float(doc_collection_lm[term])/float(total_num_words) for term in doc_collection_lm}
    topic_descs = read_topic_descs()
    candidate_queries = {}
    for topic_num in topic_descs:
        topic_desc = topic_descs[topic_num]
        topic_desc = preprocess(topic_desc, lemmatizing = True)
        topic_desc2 = preprocess(topic_desc, "../Session_track_2014/clueweb_snippet/lemur-stopwords.txt", lemmatizing = True)
        IN = Counter(topic_desc2.split())
        topic_IN = {x:float(IN[x])/float(sum(IN.values())) for x in IN}
        text = dict()
        text[1] = topic_desc
        bigram_topic_IN = get_bigram_word_lm(text)
        all_possible_bigrams = {}
        word_sequence = topic_desc2.split()
        for idx in range(len(word_sequence)-1):
            bigram = word_sequence[idx] + " " + word_sequence[idx+1] 
            all_possible_bigrams[bigram] = 1
        '''
        for word1 in self.topic_IN:
            for word2 in self.topic_IN:
                if(word1!=word2):
                    bigram = word1 + " " + word2
                    all_possible_bigrams[bigram] = 1
        '''
        bigrams_scores = []
        for bigram in all_possible_bigrams:
            bigram_score = 0
            try:
                bigram_score += bigram_topic_IN[bigram]
            except KeyError:
                pass
            bigrams_scores += [(bigram,bigram_score)]
        bigrams_scores = sorted(bigrams_scores, key = lambda l: l[1], reverse = True)[:10]
        all_queries = []
        for (bigram,score) in bigrams_scores:
            query_score = 0
            for word in bigram.split():
                try:
                    query_score += math.log(topic_IN[word]/doc_collection_lm_dist[word])
                except:
                    print ("Coming here")
                    query_score += math.log(topic_IN[word]/new_word_probability(d))
            for word in topic_IN:
                try:
                    word_score = math.log(topic_IN[word]/doc_collection_lm_dist[word])
                except:
                    print ("Coming here")
                    word_score = math.log(topic_IN[word]/new_word_probability(d)) 
                all_queries += [(bigram.split() + [word],query_score + word_score)]
        all_queries = sorted(all_queries, key = lambda l: l[1], reverse=True)
        candidate_queries[topic_num] = all_queries
        #print ('TOPIC NUM: ', topic_num)
        #print ('CANDIDATE QUERIES: ', candidate_queries)
    return candidate_queries

candidate_queries = make_candidate_QS3plus()
pickle.dump(candidate_queries, open("../simulated_sessions/candidate_QS3plus_baseline_queries_testing.pk", "wb"))
