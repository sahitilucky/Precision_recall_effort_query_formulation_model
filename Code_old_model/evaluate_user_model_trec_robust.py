#select dataset for the specific task and then evaluate
from create_dataset import *
from User_model_trec_robust import User_model
from User_model_utils import *
from BM25_ranker import BM25_ranker, InvertedIndex
import random
from collections import Counter
import math
import pickle

def format_results(results, document_content):
    formatted_results = []
    new_results = []
    for result in results:
        try:
            there = document_content[result[0]]
            new_results += [result]
        except KeyError:
            pass
    for result in new_results:
        formatted_result={} 
        formatted_result["docid"] = result[0]
        formatted_result["content"] = document_content[result[0]]
        formatted_result["title"] = ""
        formatted_results += [formatted_result]
    return formatted_results
all_doc_index_bm25 = pickle.load(open("../TREC_Robust_data/all_doc_index_lemmatized.pk", "rb") )
bm25_ranker = BM25_ranker(k1 = 1.2, b = 0.75, k3 = 500)
bm25_ranker.index = InvertedIndex()
bm25_ranker.index.index = all_doc_index_bm25[0]
bm25_ranker.dlt = all_doc_index_bm25[1]
bm25_ranker.avg_dl = float(sum(list(bm25_ranker.dlt.values())))/float(len(bm25_ranker.dlt.values()))
    
def simulate_trec_robust_sessions(topic_number, topic_desc, topic_target_items, robust_doc_content, parameters, all_doc_bigram_lm):    
    #bm25_ranker.make_inverted_index("../Session_Track_2014/clueweb_snippet/clueweb_snippet.dat", "../Session_track_2014/clueweb_snippet/lemur-stopwords.txt")
    print ("simulating sessions")
    sim_sessions = []
    for session_num in range(1):
        print ("Simulating Session num:", session_num)
        sim_session = Session(topic_num = topic_number)
        user_model = User_model(parameters, {}, robust_doc_content, topic_desc, topic_number, topic_target_items, all_doc_bigram_lm)
        query = user_model.first_interaction()
        candidate_queries = user_model.candidate_queries
        results = bm25_ranker.score(query, 50)
        formatted_results = format_results(results, robust_doc_content)
        i = 0
        max_session_length = 20
        session_length = 1
        action_code = 1
        while (session_length < max_session_length):
            print ("Session length: ", session_length)
            (next_action_code,next_query,clicked_results) = user_model.next_interaction(formatted_results[i:i+10])
            sim_session.add_sim_interaction(query.split(), formatted_results[i:i+10], clicked_results, action_code)
            print ("action_code:{} , next_query:{}, clicked_results:{}". format(action_code,next_query,clicked_results))
            if (next_action_code == 2):
                print ("Ending session")
                break
            elif (next_action_code==1):
                query = next_query
                action_code = next_action_code
                results = bm25_ranker.score(next_query, 10)
                formatted_results = format_results(results, robust_doc_content)
                i = 0
            elif (next_action_code == 0):
                query = query
                action_code = next_action_code
                i = i + 10
            session_length += 1
        sim_sessions += [sim_session]
    return sim_sessions,candidate_queries

def jaccard_similarity(list1, list2):
    print (list1, list2, len(set(list1).intersection(set(list2))), len(set(list1).union(set(list2))))
    return float(len(set(list1).intersection(set(list2))))/float(len(set(list1).union(set(list2))))

def evaluate_sessions(sim_sessions, real_sessions):
    print ("Num sim_sessions: {} Num real_sessions {}", len(sim_sessions), len(real_sessions))
    avg_session_length_sim = [len(session.interactions) for session in sim_sessions]
    avg_session_length_sim = float(sum(avg_session_length_sim))/float(len(avg_session_length_sim))
    avg_session_length_act = [len(session.interactions) for session in real_sessions]
    avg_session_length_act = float(sum(avg_session_length_act))/float(len(avg_session_length_act))
    num_queries_sim = [len(list(filter(lambda l: l.type =="reformulate", session.interactions))) for session in sim_sessions]
    num_queries_act = [len(list(filter(lambda l: l.type =="reformulate", session.interactions))) for session in real_sessions]
    avg_num_queries_sim = float(sum(num_queries_sim))/float(len(num_queries_sim))
    avg_num_queries_act = float(sum(num_queries_act))/float(len(num_queries_act))
    num_clicks_sim = [sum([len(l.clicks) for l in session.interactions]) for session in sim_sessions]
    num_clicks_act = [sum([len(l.clicks) for l in session.interactions]) for session in real_sessions]
    avg_num_clicks_sim = float(sum(num_clicks_sim))/float(len(num_clicks_sim))
    avg_num_clicks_act = float(sum(num_clicks_act))/float(len(num_clicks_act))
    queries_list_sim = [list(filter(lambda l: l.type =="reformulate", session.interactions)) for session in sim_sessions]
    queries_list_sim = [interaction.query for session in queries_list_sim for interaction in session]
    for session in real_sessions:
        for interaction in session.interactions:
            print (interaction.type)
    queries_list_act = [list(filter(lambda l: l.type =="reformulate", session.interactions)) for session in real_sessions]
    queries_list_act = [interaction.query for session in queries_list_act for interaction in session]
    queries_list_act = [preprocess(" ".join(query), lemmatizing = True).split(" ") for query in queries_list_act]
    print ("Simulated queries: ", queries_list_sim[:10])
    print ("real queries: ", queries_list_act[:10])
    act_to_simulated_sim = 0
    simulated_to_act_sim = 0
    print ("Simulated session avg length: ", avg_session_length_sim)
    print ("Real session avg length: ", avg_session_length_act)
    print ("Simulated session avg num queries: ", avg_num_queries_sim)
    print ("Real session avg num queries: ", avg_num_queries_act)
    print ("Simulated session avg num clicks: ", avg_num_clicks_sim)
    print ("Real session avg num clicks: ", avg_num_clicks_act)
    simulated_to_act_sim = [[] for q1 in queries_list_sim]
    act_to_simulated_sim = []
    for q2 in queries_list_act:
        list1 = []
        for idx,q1 in enumerate(queries_list_sim):
            sim = jaccard_similarity(q1, q2)
            simulated_to_act_sim[idx] += [sim]
            list1 += [sim]
        act_to_simulated_sim += [list1]
    #simulated_to_act_sim = [[jaccard_similarity(q1, q2) for q2 in queries_list_act]for q1 in queries_list_sim]
    #act_to_simulated_sim = [[jaccard_similarity(q1, q2) for q1 in queries_list_sim]for q2 in queries_list_act] 
    print (simulated_to_act_sim, act_to_simulated_sim)
    simulated_to_act_sim = [max(x) for x in simulated_to_act_sim]
    act_to_simulated_sim = [max(x) for x in act_to_simulated_sim]
    simulated_to_act_sim = float(sum(simulated_to_act_sim))/float(len(simulated_to_act_sim))
    act_to_simulated_sim = float(sum(act_to_simulated_sim))/float(len(act_to_simulated_sim))
    print ("Query similarity sim_to_act, act_to_sim, avg_sim: ", act_to_simulated_sim, act_to_simulated_sim, (act_to_simulated_sim+simulated_to_act_sim)/2.0)
    
    return ([avg_session_length_sim, avg_session_length_act, avg_num_queries_sim, avg_num_queries_act, act_to_simulated_sim, simulated_to_act_sim, avg_num_clicks_sim, avg_num_clicks_act] )

def simulate_trec_robust_sessions_noreformulate(topic_number, topic_desc, topic_target_items, robust_doc_content, parameters, all_doc_bigram_lm):    
    user_model = User_model(parameters, {}, robust_doc_content, topic_desc, topic_number, topic_target_items, all_doc_bigram_lm)
    query_sequence = user_model.no_reformulation_query_sequence()
    results_sequence = []
    for query in query_sequence:
        results = bm25_ranker.score(query, 50)
        formatted_results = format_results(results, robust_doc_content)
        results_sequence += [formatted_results[:10]]
    clicks_sequence = user_model.click_sequence(results_sequence)
    for idx,clicks in enumerate(clicks_sequence):
        sim_session = Session(topic_num = topic_number)
        sim_session.add_sim_interaction(query_sequence[idx].split(), results_sequence[idx][0:10], clicks, 1)
    return [sim_session]
    
def query_similarity_evaluation(queries_list_sim, queries_list_act):
    simulated_to_act_sim = [[] for q1 in queries_list_sim]
    act_to_simulated_sim = []
    for q2 in queries_list_act:
        list1 = []
        for idx,q1 in enumerate(queries_list_sim):
            sim = jaccard_similarity(q1, q2)
            simulated_to_act_sim[idx] += [sim]
            list1 += [sim]
        act_to_simulated_sim += [list1]
    #simulated_to_act_sim = [[jaccard_similarity(q1, q2) for q2 in queries_list_act]for q1 in queries_list_sim]
    #act_to_simulated_sim = [[jaccard_similarity(q1, q2) for q1 in queries_list_sim]for q2 in queries_list_act] 
    print (simulated_to_act_sim, act_to_simulated_sim)
    simulated_to_act_sim = [max(x) for x in simulated_to_act_sim]
    act_to_simulated_sim = [max(x) for x in act_to_simulated_sim]
    simulated_to_act_sim = float(sum(simulated_to_act_sim))/float(len(simulated_to_act_sim))
    act_to_simulated_sim = float(sum(act_to_simulated_sim))/float(len(act_to_simulated_sim))
    return (simulated_to_act_sim,act_to_simulated_sim)
print ("started reading...")
#robust_doc_content = read_robust_data_collection()

all_doc_bigram_lm = pickle.load(open("../TREC_Robust_data/all_doc_bigram_language_model.pk", "rb"))
print ("ended reading...")
with open('../TREC_Robust_data/robust_data_collection_preprocessed.json', 'r') as infile:
    robust_doc_content = json.load(infile)

topic_descs = read_trec_robust_topic_descs()
topic_rel_docs = read_trec_robust_judgements()
topic_descs_query = read_trec_robust_queries()
filter_topic_rel_docs = {}
for topic_num in topic_rel_docs:
    filter_topic_rel_docs[topic_num] = {}
    for docid in topic_rel_docs[topic_num]:
        try:
            there = robust_doc_content[docid]
            filter_topic_rel_docs[topic_num][docid] = topic_rel_docs[topic_num][docid]
        except KeyError:
            pass
topic_rel_docs = filter_topic_rel_docs
#simulated sessions for the topic
#parameter_setting
parameters = [1.0, 20]
num_topics = 0
simulated_sessions = []
avg_topic_performances = [[],[],[],[],[],[],[],[]]
avg_topic_performances_2 = [[],[],[],[],[],[],[],[]]
print ("NUM TOPICS:", len(topic_rel_docs))
topic_num_nums = [int(s) for s in topic_rel_docs.keys()]
topic_num_nums.sort()
print (topic_num_nums)
query_similarities = []
query_similarities_2 = []
candidate_queries = {}
for topic_num in topic_num_nums:
    #if (topic_num == '36'):
    topic_num = str(topic_num)
    try:
        topic_desc = topic_descs[topic_num]
        topic_target_items = topic_rel_docs[topic_num].copy()
    except:
        continue
    num_topics += 1
    print ("Topic num: {} Topic desc: {}".format(topic_num, preprocess(topic_desc)))
    print ("Topic target docs: ", topic_target_items)
    sim_sessions,ranked_queries = simulate_trec_robust_sessions(topic_num, preprocess(topic_desc), topic_target_items, robust_doc_content, parameters, all_doc_bigram_lm)
    ranked_queries = sorted(ranked_queries,key = lambda l:l[1], reverse=True)
    candidate_queries[topic_num] = ranked_queries
    simulated_sessions += sim_sessions
    #if(num_topics == 10):
    #   break
    queries_list_sim = [list(filter(lambda l: l.type =="reformulate", session.interactions)) for session in sim_sessions]
    queries_list_sim = [interaction.query for session in queries_list_sim for interaction in session]
    (simulated_to_act_sim,act_to_simulated_sim) = query_similarity_evaluation(queries_list_sim, [preprocess(topic_descs_query[topic_num], lemmatizing = True).split(" ")])
    query_similarities += [float(simulated_to_act_sim+act_to_simulated_sim)/float(2)]
    query_similarities_2 +=[simulated_to_act_sim]
    print ("Query similarity: ", simulated_to_act_sim, act_to_simulated_sim)
#session_reformulate_trends = evaluate_sim_sessions_NDCG(simulated_sessions, topic_rel_docs)  
print ("QUERY Similarities: ", float(sum(query_similarities))/float(len(query_similarities)))  
print ("QUERY Similarities simulated_to_act_sim: ", float(sum(query_similarities_2))/float(len(query_similarities_2)))  
avg_session_length_sim = [len(session.interactions) for session in simulated_sessions]
avg_session_length_sim = float(sum(avg_session_length_sim))/float(len(avg_session_length_sim))
print ('Simulated sessions Avg Session length: ', avg_session_length_sim)
print ("Dumping simulated sessions")

#pickle.dump(simulated_sessions, open("../simulated_sessions_TREC_robust/reform_variations/trec_robust_simulated_session_word_sup_1_query_unsup_1_1000_reform_0.9_0.9_0.9_session_nums_all_0_50.pk", "wb"))
#pickle.dump(session_reformulate_trends, open("../simulated_sessions_TREC_robust/trec_robust_session_reformuate_trends_word_sup_1_query_unsup_reform_session_nums_all.pk", "wb"))
pickle.dump(simulated_sessions, open("../simulated_sessions_TREC_robust/final_results/simulated_session_all_topics_QS3plus_baseline_more_clicks.pk", "wb"))
pickle.dump(candidate_queries, open("../simulated_sessions_TREC_robust/final_results/candidate_queries_QS3plus_baseline.pk", "wb"))

'''
document_content = {}
with open("../Session_Track_2014/clueweb_snippet_data.txt", "r") as infile:
    line = infile.readline()
    while(line.strip() != ""):
        clueweb_id = line.strip().split("# ")[1]
        document_content[clueweb_id] = {}
        line = infile.readline()
        document_content[clueweb_id]["title"]  = preprocess(line.strip(), lemmatizing = True)
        line = infile.readline()
        document_content[clueweb_id]["content"] = preprocess(line.strip(), lemmatizing = True)
        line = infile.readline()
topic_rel_docs = read_judgements()
ideal_user_sessions,precise_user_sessions,recall_user_sessions,all_sessions = select_sessions(topic_rel_docs,read_full=True)
act_sessions = []
for session in all_sessions:
    try:
        topic_rel_docs[session.getElementsByTagName("topic")[0].getAttribute("num")]
    except:
        continue
    #if session.getAttribute("num") in session_nums_dict:
    act_session = Session(session, document_content)
    act_sessions += [act_session]
session_reformulate_trends = evaluate_sim_sessions_NDCG(act_sessions, topic_rel_docs)
pickle.dump(session_reformulate_trends, open("../simulated_sessions_TREC_robust/trec_session_track_session_reformulate_trends.pk", "wb"))
#print ("QUERY Similarities: ", float(sum(query_similarities))/float(len(query_similarities)))
'''
