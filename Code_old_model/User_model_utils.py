#Contain all general functions to suppert the usermodel
import numpy as np
import matplotlib.pyplot as plt
from rake_nltk import Rake
from create_dataset import *
import re
import pickle
from collections import Counter
import random
import math
from nltk.corpus import wordnet as wn
from feature_vector import *
import time
from quick_query_sim_ev import *
from scipy import stats

class Session():
    def __init__(self, act_session = None, document_content=None, topic_num= None):
        self.topic_num = topic_num
        self.interactions = []
        self.session_num = None
        if act_session != None:
            self.get_actual_session(act_session, document_content)
        
    def add_sim_interaction(self, query, results, clicked_results, action_code):
        inte = Interaction()
        inte.query = query
        inte.results = []
        for result in results:
            result_dict = {}
            result_dict["docid"] = result["docid"]
            result_dict["content"] = result["title"] + " " + result["content"]
            inte.results += [result_dict]
        inte.clicks = []
        for rank,(result_doc_id, click) in enumerate(clicked_results):
            if click == 1:
                inte.clicks += [(result_doc_id, rank+1, 35)]
        if (action_code == 1):
            inte.type = "reformulate"
        else:
            inte.type = "page"
        self.interactions += [inte]
    def get_actual_session(self, session_xml, doc_collection):
        self.topic_num = session_xml.getElementsByTagName("topic")[0].getAttribute("num")
        self.session_num = session_xml.getAttribute("num")
        interactions = session_xml.getElementsByTagName("interaction")
        self.interactions = []
        #print ("MAKING SESSION...")
        for interaction in interactions:
            inte = Interaction()
            query_text = getText(interaction.getElementsByTagName("query")[0].childNodes)
            inte.query = query_text.split()
            results = interaction.getElementsByTagName("result")
            clicks = []
            try:
                clicked_items = interaction.getElementsByTagName("click")
                for click in clicked_items:
                    #print ("GETTING CLICK TIME...")
                    time = float(click.getAttribute("endtime")) - float(click.getAttribute("starttime"))
                    #print ("GOT CLICK TIME...")
                    rank = int(getText(click.getElementsByTagName("rank")[0].childNodes))-int(results[0].getAttribute("rank"))
                    clicks += [(getText(click.getElementsByTagName("docno")[0].childNodes),rank+1, time)]
            except:
                pass
            inte.clicks = clicks
            #print ("GETTING RESULTS...")
            inte.results = []
            for result in results:
                result_dict = {}
                result_dict["docid"] = getText(result.getElementsByTagName("clueweb12id")[0].childNodes)
                if (doc_collection!=None):
                    #result_dict["title"] = doc_collection[result_dict["doc_id"]]["title"]
                    result_dict["content"] = doc_collection[result_dict["docid"]]
                #result_dict["snippet"] = getText(result.getElementsByTagName("clueweb12id")[0].childNodes)
                inte.results += [result_dict]
            #print ("GOT RESULTS...")
            inte.type = interaction.getAttribute("type")    
            self.interactions += [inte]

class Interaction():
    def __init__(self):
        self.query = None
        self.results = []
        self.clicks = []
        self.type = None


def dirichlet_smoothing(prob1, prob2, mu = 8):
    smooth_prob1 = {}
    d = sum(prob1.values())
    for word in list(prob2.keys()):
        try:
            there = prob1[word]
            smooth_prob1[word] = (float(d)/float(d+mu))*prob1[word] + (float(mu)/float(d+mu))*prob2[word]  
        except KeyError:
            smooth_prob1[word] = (float(mu)/float(d+mu))*prob2[word]
    return smooth_prob1

def new_word_probability(d):
    mu = 8
    return (float(mu)/float(d+mu))*(float(1)/float(d))

def plot_frequencies(frequencies):
    x = list(range(len(frequencies)))
    plt.plot(x, frequencies)
    plt.ylabel("frequencies")
    plt.savefig("word_frequencies.png")

def frequency_based_topic_divisions(doc_collection_lm):
    terms = sorted(doc_collection_lm.items(),key = lambda l:l[1], reverse=True)
    frequencies = [t[1] for t in terms]
    #more populated frequency range
    bin_edges = [i for i in range(0,max(frequencies), 1000)]
    hists, bin_edges = np.histogram(frequencies, bins = bin_edges)
    selected_bins = []
    for idx,h in enumerate(hists):
        if h > 500:
            selected_bins += [bin_edges[idx]]
            selected_bins += [bin_edges[idx+1]]
    min_fre,max_fre = min(selected_bins), max(selected_bins)
    selected_frequencies = []
    for f in frequencies:
        if f < max_fre and f > min_fre:
            selected_frequencies += [f]
    plot_frequencies(selected_frequencies)
    hist, bin_edges = np.histogram(selected_frequencies, bins = 5)
    print ("BIN EDGES: ", hist, bin_edges)
    p_A_i = [0.1, (0.4-0.000001), 0.3, 0.2, 0.000001]
    p_A_i_distribution = [{},{},{},{},{}]
    for term in terms:
        i = 0
        if term[1] < bin_edges[1]:
            i = 4
        elif term[1] < bin_edges[2]:
            i = 3
        elif term[1] < bin_edges[3]:
            i = 2
        elif term[1] <= bin_edges[4]:
            i = 1
        elif term[1] <= bin_edges[5]:
            i = 0
        else:
            pass
            #print ("NOT IN ANY LIST??")
        p_A_i_distribution[i][term[0]] = term[1]
    for i,dist in enumerate(p_A_i_distribution):
        total_num_words = sum(dist.values())
        dist = {term:float(dist[term])/float(total_num_words) for term in dist}
        p_A_i_distribution[i] = dist
        print ("{}th topic".format(i))
        print ("top 10 words: ", sorted(dist.items(),key = lambda l:l[1], reverse=True)[:10])
    return (p_A_i, p_A_i_distribution)
    
def wordnet_topic_division(doc_collection_lm):
    #for word in doc_collection_lm
    #get word generality using word net
    terms = sorted(doc_collection_lm.items(),key = lambda l:l[1], reverse=True)
    term_depths = []
    for term in terms:
        if (wn.synsets(term[0])!= []):
            term_depths += [(term[0],wn.synsets(term[0])[0].min_depth())]
    max_depth, min_depth = max([t[1] for t in term_depths]), min([t[1] for t in term_depths])
    print (max_depth, min_depth)
    x = 1
    m = max_depth
    bin_edges=[]
    while(m>min_depth):
        bin_edges += [m]
        x = x + 1
        m = max_depth - x
    bin_edges += [min_depth]
    print (bin_edges)
    hist, bin_edges = np.histogram([t[1] for t in term_depths], bins = bin_edges[::-1])
    print ("BIN EDGES: ", hist, bin_edges, len(bin_edges))
    p_A_i_distribution = [{} for i in range(len(bin_edges)-1)]
    term_depths = dict(term_depths)
    for term in terms:
        i = 0
        try:
            term_depths[term[0]]
        except:
            continue
        for j in range(1,len(bin_edges)):
            if term_depths[term[0]] < bin_edges[j]:
                i = j-1 
                break
        if (term[1] == bin_edges[len(bin_edges)-1]):
            i = len(bin_edges)-2
            #print ("NOT IN ANY LIST??")
        p_A_i_distribution[i][term[0]] = term[1]
    for i,dist in enumerate(p_A_i_distribution):
        total_num_words = sum(dist.values())
        dist = {term:float(dist[term])/float(total_num_words) for term in dist}
        p_A_i_distribution[i] = dist
        print ("{}th topic".format(i))
        print ("top 10 words: ", sorted(dist.items(),key = lambda l:l[1], reverse=True)[:10])

    return (p_A_i_distribution)

def get_keywords(text, sentences = None):
    #get keywords using rake
    r = Rake()
    if sentences !=None:
        r.extract_keywords_from_sentences(sentences)
    else:
        r.extract_keywords_from_text(text)    
    phrases = r.get_ranked_phrases_with_scores()
    return phrases

def get_keywords_binary(topic_desc):
    keywords = get_keywords(topic_desc)
    keywords = list(filter(lambda l: (l[1] !="find web pages") and (l[1]!="find webpages"), keywords))
    print (keywords)
    text = preprocess(topic_desc,"../Session_track_2014/clueweb_snippet/lemur-stopwords.txt", lemmatizing = True)
    IN = Counter(text.split())
    print ("IN: ", IN)
    IN = {x:float(IN[x])/float(sum(IN.values())) for x in IN}
    keyword_IN = {}
    for (score,keyword) in keywords:
        words = preprocess(keyword,"../Session_track_2014/clueweb_snippet/lemur-stopwords.txt", lemmatizing = True).split()
        for word in words:
            if word in IN:
                try:
                    keyword_IN[word] += 1 
                except:
                    keyword_IN[word] = 1
    return keyword_IN
def get_topic_IN_with_keywords(topic_desc):
    keywords = get_keywords(topic_desc)
    keywords = list(filter(lambda l: (l[1] !="find web pages") and (l[1]!="find webpages"), keywords))
    print (keywords)
    text = preprocess(topic_desc,"../Session_track_2014/clueweb_snippet/lemur-stopwords.txt", lemmatizing = True)
    IN = Counter(text.split())
    print ("IN: ", IN)
    for (score,keyword) in keywords:
        words = preprocess(keyword,"../Session_track_2014/clueweb_snippet/lemur-stopwords.txt", lemmatizing = True).split()
        for word in words:
            if word in IN:
                IN[word] = IN[word] + score 
    IN = {x:float(IN[x])/float(sum(IN.values())) for x in IN}
    print ("IN after: " , sorted(IN.items(),key = lambda l :l[1], reverse=True))
    return IN

def get_topic_IN_with_keywords_2(topic_desc):
    keywords = get_keywords(topic_desc)
    keywords = list(filter(lambda l: (l[1] !="find web pages") and (l[1]!="find webpages"), keywords))
    print (keywords)
    text = preprocess(topic_desc,"../Session_track_2014/clueweb_snippet/lemur-stopwords.txt", lemmatizing = True)
    IN = Counter(text.split())
    print ("IN: ", IN)
    IN = {x:float(IN[x])/float(sum(IN.values())) for x in IN}
    keyword_IN = {}
    for (score,keyword) in keywords:
        words = preprocess(keyword,"../Session_track_2014/clueweb_snippet/lemur-stopwords.txt", lemmatizing = True).split()
        for word in words:
            if word in IN:
                keyword_IN[word] = (score/float(len(keyword.split()))) 
    keyword_IN = {x:float(keyword_IN[x])/float(sum(keyword_IN.values())) for x in keyword_IN}
    for word in IN:
        try:
            IN[word] = 0.5*IN[word] + 0.5*keyword_IN[word]
        except:
            IN[word] = 0.5*IN[word]
    return IN

def get_keyword_word_scores():
    topic_word_scores = {}
    topic_descs = read_topic_descs()
    #for topic in topic_descs:
        #sentences = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        #print (sentences)
        #print ("TOPIC NUM:{} TOPIC DESC:{}".format(topic, topic_descs[topic]))
        #get_topic_IN_with_keywords(topic_descs[topic])
    (doc_collection_lm, doc_collection_lm_binary) = pickle.load(open("../TREC_Robust_data/all_doc_language_model.pk", "rb"))
    d = sum(doc_collection_lm.values())
    topic_word_distribution = wordnet_topic_division(doc_collection_lm)
    total_num_words = sum(doc_collection_lm.values())
    doc_collection_lm_dist = {term:float(doc_collection_lm[term])/float(total_num_words) for term in doc_collection_lm}
    topic_proportion = []
    topic_descs = read_topic_descs()
    for topic_num in topic_descs:
        print ("TOPIC NUM:{} TOPIC DESC:{}".format(topic_num, topic_descs[topic_num]))
        topic_IN = get_topic_IN_with_keywords(topic_descs[topic_num])
        word_scores = []
        for word in topic_IN:
            word_topic_idx = -1
            for idx,topic in enumerate(topic_word_distribution):
                try:
                    there = topic[word]
                    word_topic_idx = idx
                    break
                except:
                    pass
            try:
                word_score = math.log(topic_IN[word]/doc_collection_lm_dist[word])
            except:
                word_score = 0 #math.log(topic_IN[word]/new_word_probability(d))
            word_scores += [(word,word_score)]
            print ("word:{} word_score: {} word_topic_index:{}".format(word, word_score, word_topic_idx))
            try:
                print ("Doc collection word score: ", doc_collection_lm[word])
            except:
                print ("Doc collection word score: Not available")
        word_scores = sorted(word_scores, key = lambda l :l[1], reverse=True)
        topic_word_scores[topic_num] = word_scores
        print (word_scores)
    return topic_word_scores

def get_basic_word_scores_2():
    (doc_collection_lm, doc_collection_lm_binary) = pickle.load(open("../TREC_Robust_data/all_doc_language_model.pk", "rb"))
    d = sum(doc_collection_lm.values())
    total_num_words = sum(doc_collection_lm.values())
    doc_collection_lm_dist = {term:float(doc_collection_lm[term])/float(total_num_words) for term in doc_collection_lm}
    topic_descs = read_topic_descs()
    feature_weights = [3.0/4.0,1.0/4.0]
    topic_word_scores = {}
    for topic_num in topic_descs:
        topic_desc = topic_descs[topic_num]
        topic_desc = preprocess(topic_desc, "../Session_track_2014/clueweb_snippet/lemur-stopwords.txt", lemmatizing = True)
        topic_IN = Counter(topic_desc.split())
        topic_IN = {x:float(topic_IN[x])/float(sum(topic_IN.values())) for x in topic_IN}
        word_scores = []
        for word in topic_IN:
            try:
                word_score =  feature_weights[0]*math.log(topic_IN[word]) + feature_weights[1]*math.log(topic_IN[word]/doc_collection_lm_dist[word])
            except:
                word_score = feature_weights[0]*math.log(topic_IN[word])#math.log(topic_IN[word]/new_word_probability(d))
            word_scores += [(word,word_score)]
            print ("word:{} word_score: {}".format(word, word_score))
            try:
                print ("Doc collection word score: ", doc_collection_lm[word])
            except:
                print ("Doc collection word score: Not available")
        word_scores = sorted(word_scores, key = lambda l :l[1], reverse=True)
        print ("WORD SCORES: ", word_scores)
        topic_word_scores[topic_num] = word_scores
    return topic_word_scores

def query_formulation_list(word_scores, C_u_1, all_doc_index, min_th, max_th, topic_num):
    max_threshold = max_th
    min_threshold = min_th
    l = len(word_scores)
    Q = []
    word_index = -1 
    Q = []
    R_q_docs = C_u_1.keys()
    query_index_values = ""
    p_a = len(C_u_1.keys())
    max_length = 4
    possible_queries = [([], list(C_u_1.keys()), 0, 0)]
    queries = []
    max_candidates_queries = 100
    while possible_queries != []:
        q = possible_queries[0][0]
        doc_list = possible_queries[0][1]
        score = possible_queries[0][2]
        q_index = possible_queries[0][3]
        print (q, len(doc_list), score)
        #if (score < score_threshold) and (q != []):
        #   possible_queries = possible_queries[1:]
        if (len(doc_list)) >= min_threshold:
            if (len(doc_list)) < max_threshold:
                queries += [(q,doc_list, score)]
                if (len(queries) >= max_candidates_queries):
                    break
            for idx,(w2,word_score) in enumerate(word_scores[q_index:]):
                try:
                    there = all_doc_index[w2]
                    intersection_docs = list(set(doc_list).intersection(all_doc_index[w2].keys()))
                    new_score = float(score*len(q) + word_score) / float(len(q)+1)
                    #if (new_score >= score_threshold):
                    if (len(intersection_docs) >= min_threshold):
                        possible_queries += [ (q+[w2], intersection_docs, new_score, q_index+idx+1)]    
                except KeyError:
                    pass
                    
        possible_queries = possible_queries[1:]
    query_scores = [(q[0],q[1],q[2]) for q in queries]
    ranked_queries = sorted(query_scores,key = lambda l:l[2], reverse=True)
    return ranked_queries

def query_formulation_list_stochastic(word_scores, C_u_1, all_doc_index, min_th, max_th, topic_num):
    max_threshold = max_th
    min_threshold = min_th
    l = len(word_scores)
    Q = []
    word_index = -1 
    Q = []
    R_q_docs = C_u_1.keys()
    query_index_values = ""
    p_a = len(C_u_1.keys())
    max_length = 4
    possible_queries = [([], C_u_1.keys(), 0, 0)]
    queries = []
    max_candidates_queries = 100
    while len(queries)<100:
        doc_list = C_u_1.keys()
        candidate_query = []
        avg_word_score = 0
        for idx,(w2,word_score) in enumerate(word_scores):
            try:
                there = all_doc_index[w2]
                if ((len(doc_list)) >= min_threshold) and ((len(doc_list)) < max_threshold):
                    break
                if (random.random()) >= 0.5:
                    candidate_query += [w2]
                    avg_word_score += word_score
                    doc_list = list(set(doc_list).intersection(all_doc_index[w2].keys()))
            except KeyError:
                pass
        if len(candidate_query) != 0:
            print ("Candidate query: ", candidate_query)
            queries += [(candidate_query, doc_list, float(avg_word_score)/float(len(candidate_query)))]
    queries_set = {}
    for query in queries:
        queries_set[" ".join(query[0])] = [query[1],query[2]]
    query_scores = [(query.split(),queries_set[query][0],queries_set[query][1]) for query in queries_set]
    print ("NUM CANDIDATE QUERIES: ", len(query_scores))
    '''
    while possible_queries != []:
        q = possible_queries[0][0]
        doc_list = possible_queries[0][1]
        score = possible_queries[0][2]
        q_index = possible_queries[0][3]
        print (q, len(doc_list), score)
        #if (score < score_threshold) and (q != []):
        #   possible_queries = possible_queries[1:]
        if (len(doc_list)) >= min_threshold:
            if (len(doc_list)) < max_threshold:
                queries += [(q,doc_list, score)]
                if (len(queries) >= max_candidates_queries):
                    break
            for idx,(w2,word_score) in enumerate(word_scores[q_index:]):
                try:
                    there = all_doc_index[w2]
                    intersection_docs = list(set(doc_list).intersection(all_doc_index[w2]))
                    new_score = float(score*len(q) + word_score) / float(len(q)+1)
                    #if (new_score >= score_threshold):
                    if (len(intersection_docs) >= min_threshold):
                        possible_queries += [ (q+[w2], intersection_docs, new_score, q_index+idx+1)]    
                except KeyError:
                    pass
    
        possible_queries = possible_queries[1:]
    '''
    #query_scores = [(q[0],q[1],q[2]) for q in queries]
    ranked_queries = sorted(query_scores,key = lambda l:l[2], reverse=True)
    return ranked_queries

bigram_topic_lm = read_bigram_topic_lm()
bigram_topic_lm_trec_robust = read_bigram_topic_lm_trec_robust()
def query_bigram_score(query, topic_num):
    num_phrases = 0
    '''
    for idx,word1 in enumerate(query):
        for word2 in query[idx+1:]:
            if (word1+" "+word2) in bigram_topic_lm[topic_num]:
                num_phrases += bigram_topic_lm[topic_num][word1+" "+word2]

                #if (word1 + " " +word2) in keywords:
                #   num_phrases += 1
    '''
    '''
    all_bigrams = sum(bigram_topic_lm[topic_num].values())
    bigram_topic_lm_dist = {bigram:float(bigram_topic_lm[topic_num][bigram])/float(all_bigrams) for bigram in bigram_topic_lm[topic_num]}
    log_probability = 0
    for idx in range(len(query)-1):
        word1 = query[idx]
        word2 = query[idx+1]
        if (word1+" "+word2) in bigram_topic_lm_dist:
            log_probability += math.log(bigram_topic_lm_dist[word1+" "+word2])
    return log_probability
    '''
    for idx in range(len(query)-1):
        word1 = query[idx]
        word2 = query[idx+1]
        if (word1+" "+word2) in bigram_topic_lm[topic_num]:
            num_phrases += bigram_topic_lm[topic_num][word1+" "+word2]
    
    return float(1)/float(math.exp(-(float(num_phrases)/float(len(query)))+1))
    

def query_formulation_list_2(word_scores, C_u_1, all_doc_index, min_th, max_th, topic_num):
    max_threshold = max_th
    min_threshold = min_th
    l = len(word_scores)
    Q = []
    word_index = -1 
    Q = []
    R_q_docs = C_u_1.keys()
    query_index_values = ""
    p_a = len(C_u_1.keys())
    possible_queries = [([], C_u_1.keys(), 0, 0, 0)]
    queries = []
    max_candidates_queries = 100
    alpha = 0.8
    while possible_queries != []:
        q = possible_queries[0][0]
        doc_list = possible_queries[0][1]
        word_score = possible_queries[0][2]
        total_query_score = possible_queries[0][3]
        q_index = possible_queries[0][4]
        print (q, len(doc_list), word_score, total_query_score)
        if (len(doc_list)) >= min_threshold:
            if (len(doc_list)) < max_threshold:
                queries += [(q,doc_list,word_score,total_query_score)]
                if (len(queries) >= max_candidates_queries):
                    break
            for idx,(w2,word_score) in enumerate(word_scores[q_index:]):
                try:
                    there = all_doc_index[w2]
                    intersection_docs = list(set(doc_list).intersection(set(all_doc_index[w2].keys())))
                    new_word_score = float(word_score*len(q) + word_score) / float(len(q)+1)
                    new_total_query_score = alpha*new_word_score/float(3.0) + (1-alpha)*query_bigram_score(q+[w2],topic_num)
                    if (len(intersection_docs) >= min_threshold):
                        possible_queries += [ (q+[w2], intersection_docs, new_word_score, new_total_query_score,q_index+idx+1)]    
                except KeyError:
                    pass
                
        possible_queries = possible_queries[1:]
        possible_queries = sorted(possible_queries, key = lambda l :l[3], reverse=True)
    query_scores = [(q[0],q[1],q[3]) for q in queries]
    ranked_queries = sorted(query_scores,key = lambda l:l[2], reverse=True)
    print (ranked_queries[:10])

    return ranked_queries

def get_supervised_word_scores(model_filename):
    topic_descs = read_topic_descs()
    topic_word_scores = {}
    for topic_num in topic_descs:
        print ("TOPIC NUM ", topic_num)
        training_instances = {}
        training_instances[topic_num] = {}
        word_features = make_word_feature_vectors_better(training_instances, topic_descs)
        word_list = [f[-3] for f in word_features]
        word_scores = predict_word_score_1(word_features, model_filename)
        word_scores = list(zip(word_list, word_scores))
        word_scores = sorted(word_scores, key = lambda l :l[1], reverse=True)
        print ("WORD SCORES: ", word_scores)
        topic_word_scores[topic_num] = word_scores
    return topic_word_scores

def get_supervised_word_scores_trec_robust(model_filename):
    topic_descs = read_trec_robust_topic_descs()
    topic_word_scores = {}
    for topic_num in topic_descs:
        print ("TOPIC NUM ", topic_num)
        training_instances = {}
        training_instances[topic_num] = {}
        word_features = make_word_feature_vectors_better(training_instances, topic_descs)
        word_list = [f[-3] for f in word_features]
        word_scores = predict_word_score_1(word_features, model_filename)
        word_scores = list(zip(word_list, word_scores))
        word_scores = sorted(word_scores, key = lambda l :l[1], reverse=True)
        print ("WORD SCORES: ", word_scores)
        topic_word_scores[topic_num] = word_scores
    return topic_word_scores

def get_unsupervised_word_scores():
    topic_descs = read_trec_robust_topic_descs()
    topic_word_scores = {}
    for topic_num in topic_descs:
        print ("TOPIC NUM ", topic_num)
        training_instances = {}
        training_instances[topic_num] = {}
        word_features = make_word_feature_vectors_better(training_instances, topic_descs)
        word_list = [f[-3] for f in word_features]
        word_scores = []
        alpha = 0.75
        for feature in word_features:
            word_scores += [alpha*feature[0] + (1-alpha)*feature[2]]
        word_scores = list(zip(word_list, word_scores))
        word_scores = sorted(word_scores, key = lambda l :l[1], reverse=True)
        print ("WORD SCORES: ", word_scores)
        topic_word_scores[topic_num] = word_scores
    return topic_word_scores

def get_reformulated_greedy_soln_queries(min_threshold, updated_topic_IN, precision_lm, updated_not_topic_IN, results_word_lm, reference_lm, gamma, topic_num):
    words_set = [(word,updated_topic_IN[word]) for word in updated_topic_IN]
    words_set = sorted(words_set,key=lambda l:l[1],reverse=True)[:30]
    words_set = [word[0] for word in words_set]
    word_scores = []
    for word in words_set:
        feature0 = math.log(updated_topic_IN[word])
        if precision_lm[word]["den"] == 0:
            feature2 = math.log(0.00000001)
        else:
            feature2 = math.log(float(precision_lm[word]["num"])/float(precision_lm[word]["den"]))
        '''
        try:
            feature2 = gamma*math.log(updated_topic_IN[word]/reference_lm[word]) + (1-gamma)*math.log(updated_topic_IN[word]/(updated_topic_IN[word]+updated_not_topic_IN[word]))
        except:
            feature2 = gamma*math.log(0.00000001)
        if word in updated_not_topic_IN:
            feature2 += (1-gamma)*math.log(updated_topic_IN[word]/(updated_topic_IN[word]+updated_not_topic_IN[word]))
        else:
            feature2 += (1-gamma)*math.log(updated_topic_IN[word]/(updated_topic_IN[word]+0))  
        '''  
        word_scores += [(word, feature0, feature2)]
    word_scores = sorted(word_scores, key= lambda l: l[2], reverse=True)
    a1 = 0.6
    a2 = 0.2
    a3 = 0.6
    possible_queries = []
    candidate_queries = {}
    current_list_dict = {}
    for word in word_scores:
        query_dict = {}
        query_dict[word[0]] = 1
        max_query = [[word[0]], word[1], word[2], math.log(query_bigram_score([word[0]],topic_num)), query_dict]
        possible_queries += [max_query]
        s2 = word[2]
        if (s2>=min_threshold): #and (s2<max_threshold):
            current_list_dict[max_query[0][0]] = 1
            candidate_queries[max_query[0][0]] = [max_query[0], a1*max_query[1]+a2*max_query[2]+a3*max_query[3]]
    #candidate_queries = sorted(candidate_queries, key = lambda l:l[1], reverse = True)
    #print ("FIRST CANDIDATE QUERIES: ", candidate_queries)
    print (len(word_scores))       
    for i in range(len(word_scores)-1):
        for idx,query in enumerate(possible_queries):
            max_score = math.log(0.00000001)
            max_query = None
            for word in word_scores:
                try:
                    there = query[4][word[0]] 
                except:                        
                    s1 = ((query[1]*len(query[0]))+word[1])/(len(query[0])+1)
                    s2 = ((query[2]*len(query[0]))+word[2])/(len(query[0])+1)
                    s3 = query_bigram_score(query[0]+[word[0]], topic_num)
                    if (s3 == 0):
                        s3 = math.log(0.00000001)
                    else:
                        s3 = math.log(s3)
                    score = a1*s1+a2*s2+a3*s3
                    if (score > max_score):
                        max_score = score
                        new_query_dict = query[4].copy() 
                        new_query_dict[word[0]] = 1
                        max_query = [query[0]+[word[0]], s1, s2, s3, new_query_dict] 
                    #max_query_list = 
            if max_query!=None:
                possible_queries[idx] = max_query
            if (max_query!=None):
                if (max_query[2]>=min_threshold): #and (max_query[2]<max_threshold):
                    s = max_query[0].copy()
                    s.sort()
                    try:
                        there = current_list_dict[" ".join(s)]
                        if a1*max_query[1]+a2*max_query[2]+a3*max_query[3] > candidate_queries[" ".join(s)][1]:
                            candidate_queries[" ".join(s)]= [max_query[0], a1*max_query[1]+a2*max_query[2]+a3*max_query[3]]
                    except:
                        candidate_queries[" ".join(s)]= [max_query[0], a1*max_query[1]+a2*max_query[2]+a3*max_query[3]]
                        current_list_dict[" ".join(s)] = 1
        #candidate_queries = sorted(candidate_queries, key = lambda l:l[1], reverse = True)
    candidate_queries = sorted(candidate_queries.items(), key = lambda l:l[1][1], reverse = True)
    candidate_queries = [(c[1][0],c[1][1]) for c in candidate_queries]
    print ("CANDIDATE QUERIES: ", candidate_queries)        
    return candidate_queries

def get_greedy_sol_queries_unsupervised(min_threshold, max_threshold):
    topic_descs = read_topic_descs()
    topic_word_scores = {}
    a1 = 0.6
    a2 = 0.2
    a3 = 0.6
    for topic_num in topic_descs:
        print ("TOPIC NUM ", topic_num)
        training_instances = {}
        training_instances[topic_num] = {}
        word_features = make_word_feature_vectors_better(training_instances, topic_descs)
        word_list = [(f[-3],f[0],f[2]) for f in word_features]
        possible_queries = []
        candidate_queries = {}
        current_list_dict = {}

        for word in word_list:
            query_dict = {}
            query_dict[word[0]] = 1
            max_query = [[word[0]], word[1], word[2], math.log(query_bigram_score([word[0]],topic_num)), query_dict]
            possible_queries += [max_query]
            s2 = word[2]
            if (s2>min_threshold): #and (s2<max_threshold):
                current_list_dict[max_query[0][0]] = 1
                candidate_queries[max_query[0][0]] = [max_query[0], a1*max_query[1]+a2*max_query[2]+a3*max_query[3]]
        possible_queries_bigrams = []
        for idx,query in enumerate(possible_queries):
            for word in word_list:
                try:
                    there = query[4][word[0]] 
                except:                        
                    s1 = ((query[1]*len(query[0]))+word[1])/(len(query[0])+1)
                    s2 = ((query[2]*len(query[0]))+word[2])/(len(query[0])+1)
                    s3 = query_bigram_score(query[0]+[word[0]], topic_num)
                    if (s3 == 0):
                        s3 = math.log(0.00000001)
                    else:
                        s3 = math.log(s3)
                    score = a1*s1+a2*s2+a3*s3
                    new_query_dict = query[4].copy() 
                    new_query_dict[word[0]] = 1
                    max_query = [query[0]+[word[0]], s1, s2, s3, new_query_dict] 
                    possible_queries_bigrams += [max_query]
                    if (s2>min_threshold): #and (s2<max_threshold):
                        s = max_query[0].copy()
                        s.sort()
                        try:
                            there = current_list_dict[" ".join(s)]
                            if a1*max_query[1]+a2*max_query[2]+a3*max_query[3] > candidate_queries[" ".join(s)][1]:
                                candidate_queries[" ".join(s)]= [max_query[0], a1*max_query[1]+a2*max_query[2]+a3*max_query[3]]
                        except:
                            candidate_queries[" ".join(s)]= [max_query[0], a1*max_query[1]+a2*max_query[2]+a3*max_query[3]]
                            current_list_dict[" ".join(s)] = 1

        possible_queries = possible_queries_bigrams
        word_list = sorted(word_list, key= lambda l: l[2], reverse=True)
        #candidate_queries = sorted(candidate_queries, key = lambda l:l[1], reverse = True)
        print (len(word_list))       
        for i in range(len(word_list)-2):
            for idx,query in enumerate(possible_queries):
                max_score = math.log(0.00000001)
                max_query = None
                for word in word_list:
                    try:
                        there = query[4][word[0]] 
                    except:                        
                        s1 = ((query[1]*len(query[0]))+word[1])/(len(query[0])+1)
                        s2 = ((query[2]*len(query[0]))+word[2])/(len(query[0])+1)
                        s3 = query_bigram_score(query[0]+[word[0]], topic_num)
                        if (s3 == 0):
                            s3 = math.log(0.00000001)
                        else:
                            s3 = math.log(s3)
                        score = a1*s1+a2*s2+a3*s3
                        if (score > max_score):
                            max_score = score
                            new_query_dict = query[4].copy() 
                            new_query_dict[word[0]] = 1
                            max_query = [query[0]+[word[0]], s1, s2, s3, new_query_dict] 
                            #max_query_list = 
                if max_query!=None:
                    possible_queries[idx] = max_query
                if (max_query!=None):
                    if (max_query[2]>min_threshold): #and (max_query[2]<max_threshold):
                        s = max_query[0].copy()
                        s.sort()
                        try:
                            there = current_list_dict[" ".join(s)]
                            #print (candidate_queries[" ".join(s)][1])
                            if a1*max_query[1]+a2*max_query[2]+a3*max_query[3] > candidate_queries[" ".join(s)][1]:
                            #    print ('coming here')
                                candidate_queries[" ".join(s)]= [max_query[0], a1*max_query[1]+a2*max_query[2]+a3*max_query[3]]
                        except:
                            candidate_queries[" ".join(s)]= [max_query[0], a1*max_query[1]+a2*max_query[2]+a3*max_query[3]]
                            current_list_dict[" ".join(s)] = 1
            #candidate_queries = sorted(candidate_queries, key = lambda l:l[1], reverse = True)
        candidate_queries = sorted(candidate_queries.items(), key = lambda l:l[1][1], reverse = True)
        candidate_queries = [(c[1][0],c[1][1]) for c in candidate_queries]
        print ("CANDIDATE QUERIES: ", candidate_queries)        
        topic_word_scores[topic_num] = candidate_queries
    return topic_word_scores

def get_greedy_sol_queries_unsupervised_new(min_threshold, max_threshold):
    topic_descs = read_topic_descs()
    topic_word_scores = {}
    a1 = 0.6
    a2 = 0.2
    a3 = 0.6
    for topic_num in topic_descs:
        print ("TOPIC NUM ", topic_num)
        training_instances = {}
        training_instances[topic_num] = {}
        word_features = make_word_feature_vectors_better(training_instances, topic_descs)
        word_list = [(f[-3],f[0],f[2]) for f in word_features]
        possible_queries = []
        candidate_queries = {}
        current_list_dict = {}

        for word in word_list:
            query_dict = {}
            query_dict[word[0]] = 1
            max_query = [[word[0]], word[1], word[2], math.log(query_bigram_score([word[0]],topic_num)), query_dict]
            possible_queries += [max_query]
            s2 = word[2]
            #if (s2>min_threshold): #and (s2<max_threshold):
                #current_list_dict[max_query[0][0]] = 1
                #candidate_queries[max_query[0][0]] = [max_query[0], a1*max_query[1]+a2*max_query[2]+a3*max_query[3]]
            #candidate_queries += [(max_query[0], a1*max_query[1]+a2*max_query[2]+a3*max_query[3])] 
        word_list = sorted(word_list, key= lambda l: l[2], reverse=True)
        #candidate_queries = sorted(candidate_queries, key = lambda l:l[1], reverse = True)
        print (len(word_list))       
        for i in range(2):
            new_possible_queries = []
            for idx,query in enumerate(possible_queries):
                for word in word_list:
                    try:
                        there = query[4][word[0]] 
                    except:
                        #print (query)                        
                        s1 = ((query[1]*len(query[0]))+word[1])/(len(query[0])+1)
                        s2 = ((query[2]*len(query[0]))+word[2])/(len(query[0])+1)
                        s3 = query_bigram_score(query[0]+[word[0]], topic_num)
                        if (s3 == 0):
                            s3 = math.log(0.00000001)
                        else:
                            s3 = math.log(s3)
                        score = a1*s1+a2*s2+a3*s3
                        new_query_dict = query[4].copy() 
                        new_query_dict[word[0]] = 1
                        max_query = [query[0]+[word[0]], s1, s2, s3, new_query_dict] 
                        new_possible_queries += [max_query]
                    #max_query_list = 

            new_candidate_queries = [(max_query, a1*max_query[1]+a2*max_query[2]+a3*max_query[3]) for max_query in new_possible_queries] 
            if i == 0:
                new_candidate_queries = sorted(new_candidate_queries, key = lambda l:l[1], reverse = True)
            else:
                new_candidate_queries = sorted(new_candidate_queries, key = lambda l:l[1], reverse = True)    
            print ("COMING HERE")
            possible_queries = [query[0] for query in new_candidate_queries]
            if i==1:
                for query in new_candidate_queries:
                    max_query = query[0]
                    s = max_query[0].copy()
                    s.sort()
                    try:
                        there = current_list_dict[" ".join(s)]
                        if a1*max_query[1]+a2*max_query[2]+a3*max_query[3] > candidate_queries[" ".join(s)][1]:
                            candidate_queries[" ".join(s)]= [max_query[0], a1*max_query[1]+a2*max_query[2]+a3*max_query[3]]
                    except:
                        candidate_queries[" ".join(s)]= [max_query[0], a1*max_query[1]+a2*max_query[2]+a3*max_query[3]]
                        current_list_dict[" ".join(s)] = 1
                        
            #candidate_queries = sorted(candidate_queries, key = lambda l:l[1], reverse = True)
        candidate_queries = sorted(candidate_queries.items(), key = lambda l:l[1][1], reverse = True)
        candidate_queries = [(c[1][0],c[1][1]) for c in candidate_queries]
        #candidate_queries = sorted(candidate_queries, key = lambda l:l[1], reverse = True)
        print ("CANDIDATE QUERIES: ", candidate_queries)        
        topic_word_scores[topic_num] = candidate_queries
    return topic_word_scores



def get_reformulated_word_scores_clueweb(updated_collection_lm, updated_topic_IN, updated_not_topic_IN, alpha2, model_filename):
    word_features = []
    words_set = list(updated_topic_IN.keys())
    for word in words_set:
        features = []
        try:
            features += [math.log(updated_topic_IN[word])]
        except KeyError:
            features += [math.log(0.00000001)]              #possible if we expand topic_iIN with synonyms
        '''
        try:
            features += [math.log(keywords_IN[word])]
        except KeyError:
            features += [math.log(0.00000001)]  
        '''         
        '''
        try:
            features += [keyword_binary[word]]
        except KeyError:
            features += [0]
        '''
        try:
            features += [math.log(updated_collection_lm[word])]
        except KeyError:
            features += [math.log(0.00000001)]
        try:
            alpha = alpha2
            if word in updated_not_topic_IN:
                features += [math.log(alpha*(updated_topic_IN[word]/updated_collection_lm[word]) + (1-alpha)*(float(updated_topic_IN[word])/float(updated_topic_IN[word]+updated_not_topic_IN[word])))]
            else:
                features += [math.log(updated_topic_IN[word]/updated_collection_lm[word])]
        except KeyError:
            features += [math.log(0.00000001)]
        features += [0]
        features += [word,0,[]]
        word_features += [features]
    word_list = [f[-3] for f in word_features]
    word_scores = predict_word_score_1(word_features, model_filename)
    word_scores = list(zip(word_list, word_scores))
    word_scores = sorted(word_scores, key = lambda l :l[1], reverse=True)
    print ("REFORMULATE UPDATED WORD SCORES: ", word_scores[:50])
    return word_scores

def get_unsup_reformulated_word_scores_clueweb(updated_collection_lm, updated_topic_IN, updated_not_topic_IN, alpha2):
    word_features = []
    words_set = list(updated_topic_IN.keys())
    for word in words_set:
        features = []
        try:
            features += [math.log(updated_topic_IN[word])]
        except KeyError:
            features += [math.log(0.00000001)]              #possible if we expand topic_iIN with synonyms
        '''
        try:
            features += [math.log(keywords_IN[word])]
        except KeyError:
            features += [math.log(0.00000001)]  
        '''         
        '''
        try:
            features += [keyword_binary[word]]
        except KeyError:
            features += [0]
        '''
        try:
            features += [math.log(updated_collection_lm[word])]
        except KeyError:
            features += [math.log(0.00000001)]
        try:
            alpha = alpha2
            if word in updated_not_topic_IN:
                features += [math.log(alpha*(updated_topic_IN[word]/updated_collection_lm[word]) + (1-alpha)*(float(updated_topic_IN[word])/float(updated_topic_IN[word]+updated_not_topic_IN[word])))]
            else:
                features += [math.log(updated_topic_IN[word]/updated_collection_lm[word])]
        except KeyError:
            features += [math.log(0.00000001)]
        features += [0]
        features += [word,0,[]]
        word_features += [features]
    word_list = [f[-3] for f in word_features]
    word_scores = [(0.75*f[0]+0.25*f[2]) for f in word_features]
    word_scores = list(zip(word_list, word_scores))
    word_scores = sorted(word_scores, key = lambda l :l[1], reverse=True)
    print ("REFORMULATE UPDATED WORD SCORES: ", word_scores[:50])
    return word_scores


def get_reformulated_word_scores(updated_collection_lm, updated_topic_IN, updated_not_topic_IN, alpha2):
    word_features = []
    words_set = list(updated_topic_IN.keys())
    for word in words_set:
        features = []
        try:
            features += [math.log(updated_topic_IN[word])]
        except KeyError:
            features += [math.log(0.00000001)]              #possible if we expand topic_iIN with synonyms
        '''
        try:
            features += [math.log(keywords_IN[word])]
        except KeyError:
            features += [math.log(0.00000001)]  
        '''         
        '''
        try:
            features += [keyword_binary[word]]
        except KeyError:
            features += [0]
        '''
        try:
            features += [math.log(updated_collection_lm[word])]
        except KeyError:
            features += [math.log(0.00000001)]
        try:
            alpha = alpha2
            if word in updated_not_topic_IN:
                features += [math.log(alpha*(updated_topic_IN[word]/updated_collection_lm[word]) + (1-alpha)*(float(updated_topic_IN[word])/float(updated_topic_IN[word]+updated_not_topic_IN[word])))]
            else:
                features += [math.log(updated_topic_IN[word]/updated_collection_lm[word])]
        except KeyError:
            features += [math.log(0.00000001)]
        features += [0]
        features += [word,0,[]]
        word_features += [features]
    word_list = [f[-3] for f in word_features]
    word_scores = predict_word_score_1(word_features, "../supervised_models/log_reg_model_word_type_1_var_session_nums_all.pk")
    word_scores = list(zip(word_list, word_scores))
    word_scores = sorted(word_scores, key = lambda l :l[1], reverse=True)
    print ("REFORMULATE UPDATED WORD SCORES: ", word_scores[:50])
    return word_scores

def topic_session_queries_mle(training_sessions_topics):
    topic_mle = {}
    p_t_w = {}
    T = list(training_sessions_topics.keys())
    p_t_s_w = {}
    for topic_num in training_sessions_topics:
        print ("TOPIC NUM: ", topic_num)
        topic_sessions = training_sessions_topics[topic_num]
        queries = []
        for session in topic_sessions:
            queries += [preprocess(' '.join(interaction.query),lemmatizing=True) for interaction in session.interactions if interaction.type == "reformulate"]
        queries = [(query.split(),len(query.split())) for query in queries]
        p_l_t = Counter([query[1] for query in queries])
        p_l_t = {l:float(p_l_t[l])/float(sum(p_l_t.values())) for l in p_l_t}
        p_w_l_t = {}
        print ("coming here")
        for l in p_l_t:
            filtered_queries = list(filter(lambda x: x[1] == l, queries))
            p_w_l_t[l] = {}
            for query in filtered_queries:
                print ("Query: ", set(query[0]))
                for word in set(query[0]):
                    try:
                        p_w_l_t[l][word] += 1  
                    except KeyError:
                        p_w_l_t[l][word] = 1  
                    try:
                        p_t_w[word][topic_num] += 1
                    except:
                        p_t_w[word] = {}
                        for t in T:
                            p_t_w[word][t] = 0
                        p_t_w[word][topic_num] += 1
        print (p_l_t)
        print (p_w_l_t)
        topic_mle[topic_num] = [p_l_t,p_w_l_t]
        contents = []
        for session in topic_sessions:        
            for interaction in session.interactions:
                contents += [result["content"] for result in interaction.results]
        terms_lm = Counter(" ".join(contents).split()) 
        p_t_s_w[topic_num] = terms_lm
    print (len(p_t_w))
    print (len(p_t_s_w))
    p_t_s_w_final = {}
    for topic_num in p_t_s_w:
        for word in p_t_s_w[topic_num]:
            try:
                p_t_s_w_final[word][topic_num] = p_t_s_w[topic_num][word] 
            except KeyError:
                p_t_s_w_final[word] = {}
                for t in T:
                    p_t_s_w_final[word][t] = 0    
                p_t_s_w_final[word][topic_num] = p_t_s_w[topic_num][word] 
    for word in p_t_s_w_final:
        p_t_s_w_final[word]["total"] = sum(p_t_s_w_final[word].values())
    for word in p_t_w:
        p_t_w[word]["total"] = sum(p_t_w[word].values())
    return topic_mle,p_t_w,p_t_s_w_final

def make_candidate_queries():
    all_doc_index = pickle.load(open("../TREC_Robust_data/all_doc_index_lemmatized.pk", "rb"))
    C_u_1 = all_doc_index[1]
    all_doc_index = all_doc_index[0]
    print ("coming here")
    ##INPUT CHANGE
    #supervised
    #topic_word_scores = get_supervised_word_scores("../supervised_models/log_reg_model_word_type_1_var_session_nums_5.pk")  #get_basic_word_scores_2()
    #unsupervised
    topic_word_scores = get_unsupervised_word_scores()  #get_basic_word_scores_2()
    print ("coming here")
    candidate_queries = {}
    print ("coming here")
    for topic_num in topic_word_scores:
        word_scores = topic_word_scores[topic_num]
        print ("TOPIC NUM: {}".format(topic_num))
        start_time = time.time()
        candidate_queries[topic_num] = query_formulation_list(word_scores, C_u_1, all_doc_index, 1, 500, topic_num)
        print ("Time taken: ", (time.time()-start_time))
    ##OUTPUT CHANGE
    pickle.dump(candidate_queries, open("../simulated_sessions_TREC_robust/final_results/candidate_queries_basic_recall_precision.pk", "wb"))

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
    return candidate_queries

def main(): 
    document_content, document_content_wo_stopwords = load_preprocess_robust_data()
    topic_rel_docs = read_judgements()
    ideal_user_sessions,precise_user_sessions,recall_user_sessions,all_sessions = select_sessions(topic_rel_docs, read_full=True)
    print ("Num sessions: ",len(all_sessions))
    all_session_topic_nums = {}
    training_session_nums_1 = pickle.load(open("../supervised_models/train_test_splits/training_session_nums_5.pk", "rb"))
    session_nums_dict = dict(zip(training_session_nums_1, range(len(training_session_nums_1))))
    for session in all_sessions:
        if session.getAttribute("num") in session_nums_dict:
            topic_num = session.getElementsByTagName("topic")[0].getAttribute("num")
            act_session = Session(session, document_content)
            try:
                all_session_topic_nums[topic_num] += [act_session] 
            except KeyError:
                all_session_topic_nums[topic_num] = [act_session]
    print ("training sessions: ", len(session_nums_dict), len(all_sessions))
    topic_mle,p_t_w,p_t_s_w = topic_session_queries_mle(all_session_topic_nums)
    with open('../supervised_models/baseline_session_lms/topic_mle_training_session_nums_5.json', 'w') as outfile:
        json.dump(topic_mle, outfile)
    with open('../supervised_models/baseline_session_lms/p_t_w_training_session_nums_5.json', 'w') as outfile:
        json.dump(p_t_w, outfile)
    with open('../supervised_models/baseline_session_lms/p_t_s_w_training_session_nums_5.json', 'w') as outfile:
        json.dump(p_t_s_w, outfile)

if __name__ == "__main__":
    starttime = time.time()
    topic_word_scores = get_greedy_sol_queries_unsupervised(math.log(0.0001), -1) 
    print ("TIME TAKEN FOR GREEDY SOLN QUERIES: ", time.time()-starttime)
    topic_rel_docs = read_judgements()
    clueweb_snippet_collection,clueweb_snippet_collection_2 = read_clueweb_snippet_data()
    print ("Num docs: ", len(clueweb_snippet_collection_2))
    i = 0
    for docid in clueweb_snippet_collection_2:
        clueweb_snippet_collection_2[docid] = preprocess(clueweb_snippet_collection_2[docid], "../Session_track_2014/clueweb_snippet/lemur-stopwords.txt", lemmatizing = True) 
        i += 1
        if (i%100000 == 0):
            print (i)
    ideal_user_sessions,precise_user_sessions,recall_user_sessions,all_sessions = select_sessions(topic_rel_docs,read_full=True)
    
    act_sessions_full_topics = {}
    #testing_session_nums_1 = pickle.load(open("../supervised_models/train_test_splits/testing_session_nums_1.pk", "rb"))
    #session_nums_dict = dict(zip(testing_session_nums_1, range(len(testing_session_nums_1)))) 
    act_sessions = []
    act_session_topics ={}
    for session in all_sessions:
        act_session = Session(session, clueweb_snippet_collection_2)
        act_sessions+=[act_session]
        try:
            act_session_topics[act_session.topic_num] += [act_session] 
        except:
            act_session_topics[act_session.topic_num] = [act_session] 
    query_similarities = []
    query_similarities_2 = []
    query_similarities_topics = {}
    query_similarities_topics_first_q = {}
    query_similarities_first_q = []
    query_similarities_2_first_q = []
    for topic_num in topic_word_scores:
        print ('TOPIC NUM: ', topic_num)
        queries_list_sim = [query[0] for query in topic_word_scores[topic_num]][:10]
        queries_list_act = [list(filter(lambda l: l.type =="reformulate", session.interactions)) for session in act_session_topics[topic_num]]
        queries_list_act = [interaction.query for session in queries_list_act for interaction in session]
        queries_list_act = [preprocess(" ".join(query), lemmatizing = True).split(" ") for query in queries_list_act]
        queries_list_act2 = [list(filter(lambda l: l.type =="reformulate", session.interactions))[:1] for session in act_session_topics[topic_num]]
        queries_list_act2 = [interaction.query for session in queries_list_act2 for interaction in session]
        queries_list_act2 = [preprocess(" ".join(query), lemmatizing = True).split(" ") for query in queries_list_act2]
        print ('Real queries: ', queries_list_act)
        print ("Simulated queries: ", queries_list_sim)
        (simulated_to_act_sim,act_to_simulated_sim,simulated_to_act_sim_list,act_to_simulated_sim_list) = query_similarity_evaluation(queries_list_sim, queries_list_act)
        query_similarities += [float(simulated_to_act_sim+act_to_simulated_sim)/float(2)]
        query_similarities_2 += [simulated_to_act_sim]
        query_similarities_topics[topic_num] = simulated_to_act_sim_list
        (simulated_to_act_sim,act_to_simulated_sim,simulated_to_act_sim_list,act_to_simulated_sim_list) = query_similarity_evaluation(queries_list_sim, queries_list_act2)
        query_similarities_first_q += [float(simulated_to_act_sim+act_to_simulated_sim)/float(2)]
        query_similarities_2_first_q += [simulated_to_act_sim]
        query_similarities_topics_first_q[topic_num] = simulated_to_act_sim_list
    print (query_similarities)
    print (query_similarities_2)
    print ("QUERY Similarities: ", float(sum(query_similarities))/float(len(query_similarities)))  
    print ("QUERY Similarities_2 : ", float(sum(query_similarities_2))/float(len(query_similarities_2)))  
    print ("Average similarity @1: ", sum([sum(query_similarities_topics[x][:1])/float(len(query_similarities_topics[x][:1])) for x in query_similarities_topics])/float(len(query_similarities_topics)) )
    print ("Average similarity @5: ", sum([sum(query_similarities_topics[x][:5])/float(len(query_similarities_topics[x][:5])) for x in query_similarities_topics])/float(len(query_similarities_topics)) )
    print ("Average similarity @10: ", sum([sum(query_similarities_topics[x][:10])/float(len(query_similarities_topics[x][:10])) for x in query_similarities_topics])/float(len(query_similarities_topics)) )
    print ("Average similarity @1: ", sum([sum(query_similarities_topics_first_q[x][:1])/float(len(query_similarities_topics_first_q[x][:1])) for x in query_similarities_topics_first_q])/float(len(query_similarities_topics_first_q)) )
    print ("Average similarity @5: ", sum([sum(query_similarities_topics_first_q[x][:5])/float(len(query_similarities_topics_first_q[x][:5])) for x in query_similarities_topics_first_q])/float(len(query_similarities_topics_first_q)) )
    print ("Average similarity @10: ", sum([sum(query_similarities_topics_first_q[x][:10])/float(len(query_similarities_topics_first_q[x][:10])) for x in query_similarities_topics_first_q])/float(len(query_similarities_topics_first_q)) )
    act_sessions_first_q = []
    for session in act_sessions:
        new_session = Session()
        new_session.topic_num = session.topic_num
        new_session.session_num = session.session_num
        new_session.interactions = list(filter(lambda l: l.type =="reformulate", session.interactions))[:1]
        act_sessions_first_q += [new_session]
    dtc_similarities,qs3_similarities = quick_evaluation(act_sessions)
    dtc_similarities_first,qs3_similarities_first = quick_evaluation(act_sessions_first_q)
    similarity1 = []
    similarity2 = []
    similarity3 = []
    similarity4 = []
    similarity5 = []
    similarity6 = [] 
    for topic_num in dtc_similarities:
        print (len(query_similarities_topics[topic_num]), len(dtc_similarities[topic_num]))
        similarity1 += [x for x in query_similarities_topics[topic_num]]
        similarity2 += [x for x in dtc_similarities[topic_num]]
        similarity3 += [x for x in query_similarities_topics[topic_num][:5] ]
        similarity4 += [x for x in dtc_similarities[topic_num][:5]]
        similarity5 += [x for x in query_similarities_topics[topic_num][:1]]
        similarity6 += [x for x in dtc_similarities[topic_num][:1]]
    print (stats.ttest_ind(similarity1, similarity2, equal_var=False))
    print (stats.ttest_ind(similarity3, similarity4, equal_var=False))
    print (stats.ttest_ind(similarity5, similarity6, equal_var=False))

    similarity1 = []
    similarity2 = []
    similarity3 = []
    similarity4 = []
    similarity5 = []
    similarity6 = [] 
    for topic_num in qs3_similarities:
        print (len(query_similarities_topics[topic_num]), len(qs3_similarities[topic_num]))
        similarity1 += [x for x in query_similarities_topics[topic_num]]
        similarity2 += [x for x in qs3_similarities[topic_num]]
        similarity3 += [x for x in query_similarities_topics[topic_num][:5] ]
        similarity4 += [x for x in qs3_similarities[topic_num][:5]]
        similarity5 += [x for x in query_similarities_topics[topic_num][:1]]
        similarity6 += [x for x in qs3_similarities[topic_num][:1]]

    print (stats.ttest_ind(similarity1, similarity2, equal_var=False))
    print (stats.ttest_ind(similarity3, similarity4, equal_var=False))
    print (stats.ttest_ind(similarity5, similarity6, equal_var=False))


    #FIRST
    similarity1 = []
    similarity2 = []
    similarity3 = []
    similarity4 = []
    similarity5 = []
    similarity6 = [] 
    for topic_num in dtc_similarities_first:
        print (len(query_similarities_topics_first_q[topic_num]), len(dtc_similarities_first[topic_num]))
        similarity1 += [x for x in query_similarities_topics_first_q[topic_num]]
        similarity2 += [x for x in dtc_similarities_first[topic_num]]
        similarity3 += [x for x in query_similarities_topics_first_q[topic_num][:5] ]
        similarity4 += [x for x in dtc_similarities_first[topic_num][:5]]
        similarity5 += [x for x in query_similarities_topics_first_q[topic_num][:1]]
        similarity6 += [x for x in dtc_similarities_first[topic_num][:1]]
    print (stats.ttest_ind(similarity1, similarity2, equal_var=False))
    print (stats.ttest_ind(similarity3, similarity4, equal_var=False))
    print (stats.ttest_ind(similarity5, similarity6, equal_var=False))

    similarity1 = []
    similarity2 = []
    similarity3 = []
    similarity4 = []
    similarity5 = []
    similarity6 = [] 
    for topic_num in qs3_similarities_first:
        print (len(query_similarities_topics_first_q[topic_num]), len(qs3_similarities_first[topic_num]))
        similarity1 += [x for x in query_similarities_topics_first_q[topic_num]]
        similarity2 += [x for x in qs3_similarities_first[topic_num]]
        similarity3 += [x for x in query_similarities_topics_first_q[topic_num][:5] ]
        similarity4 += [x for x in qs3_similarities_first[topic_num][:5]]
        similarity5 += [x for x in query_similarities_topics_first_q[topic_num][:1]]
        similarity6 += [x for x in qs3_similarities_first[topic_num][:1]]

    print (stats.ttest_ind(similarity1, similarity2, equal_var=False))
    print (stats.ttest_ind(similarity3, similarity4, equal_var=False))
    print (stats.ttest_ind(similarity5, similarity6, equal_var=False))
