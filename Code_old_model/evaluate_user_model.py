#select dataset for the specific task and then evaluate
from create_dataset import *
from User_model_unsup import User_model
from User_model_utils import *
from BM25_ranker import BM25_ranker
import random
from collections import Counter
import math
import pickle

def format_results_old(results, doci_id_to_clueweb_id, document_content):
    formatted_results = []
    for result in results:
        formatted_result={} 
        formatted_result["docid"] = doci_id_to_clueweb_id[result[0]]
        formatted_result["content"] = document_content[formatted_result["docid"]]["content"]
        formatted_result["title"] = document_content[formatted_result["docid"]]["title"]
        formatted_results += [formatted_result]
    return formatted_results

def format_results(results, document_content):
    formatted_results = []
    for result in results:
        formatted_result={} 
        formatted_result["docid"] = result[0]
        formatted_result["content"] = document_content[formatted_result["docid"]]
        formatted_result["title"] = ""
        formatted_results += [formatted_result]
    return formatted_results

clueweb_snippet_collection,clueweb_snippet_collection_2 = read_clueweb_snippet_data()
print ("Num docs: ", len(clueweb_snippet_collection_2))
i = 0
for docid in clueweb_snippet_collection_2:
    clueweb_snippet_collection_2[docid] = preprocess(clueweb_snippet_collection_2[docid], "../Session_track_2014/clueweb_snippet/lemur-stopwords.txt", lemmatizing = True) 
    i += 1
    if (i%100000 == 0):
        print (i)
bm25_ranker = BM25_ranker(k1 = 1.2, b = 0.75, k3 = 500)
bm25_ranker.make_inverted_index_2(clueweb_snippet_collection_2, "../Session_track_2014/clueweb_snippet/lemur-stopwords.txt")
#bm25_ranker.make_inverted_index("../Session_Track_2014/clueweb_snippet/clueweb_snippet.dat", "../Session_track_2014/clueweb_snippet/lemur-stopwords.txt")
    
def simulate_sessions(topic_number, topic_desc, topic_target_items, document_content, robust_doc_content, parameters, all_doc_bigram_lm):
    '''
    doci_id_to_clueweb_id = {}
    with open("../Session_Track_2014/clueweb_snippet/clueweb_line_corpus_id_mapping.txt", "r") as infile:
        for line in infile:
            doci_id_to_clueweb_id[int(line.strip().split("\t")[1])-1] = line.strip().split("\t")[0]
    '''
    print ("simulating sessions")
    sim_sessions = []
    for session_num in range(1):
        print ("Simulating Session num:", session_num)
        sim_session = Session(topic_num = topic_number)
        user_model = User_model(parameters, document_content, robust_doc_content, topic_desc, topic_number, topic_target_items, all_doc_bigram_lm)
        query = user_model.first_interaction()
        candidate_queries = user_model.candidate_queries
        results = bm25_ranker.score(query, 50)[:20]
        formatted_results = format_results(results, clueweb_snippet_collection_2)
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
                formatted_results = format_results(results, clueweb_snippet_collection_2)
                i = 0
            elif (next_action_code == 0):
                query = query
                action_code = next_action_code
                i = i + 10
            session_length += 1
        sim_sessions += [sim_session]
    return sim_sessions,candidate_queries

def simulate_sessions_noreformulate(topic_number, topic_desc, topic_target_items, robust_doc_content, parameters, all_doc_bigram_lm):    
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


def find_session_likelihood(topic_desc, topic_target_items, document_content, parameters, actual_session):
    print ("To find likelihood of a given session")
    total_log_likelihood = 0
    user_model = User_model(parameters, document_content, topic_desc, topic_target_items)
    for interaction in actual_session:
        if (interaction.type == "reformulate"):
            log_likelihood = user_model.query_likelihood(interaction.query)
            print ("Query: likelihood: ", interaction.query, log_likelihood)
            total_log_likelihood += log_likelihood 
    return total_log_likelihood


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
print ("started this")
print ("Num docs:", len(document_content.keys()))
print ("started reading...")
#robust_doc_content = read_robust_data_collection()
all_doc_bigram_lm = pickle.load(open("../TREC_Robust_data/all_doc_bigram_language_model.pk", "rb"))
print ("ended reading...")
topic_descs = read_topic_descs()
topic_rel_docs = read_judgements()
filter_topic_rel_docs = {}
for topic_num in topic_rel_docs:
    filter_topic_rel_docs[topic_num] = {}
    for docid in topic_rel_docs[topic_num]:
        try:
            there = document_content[docid]
            filter_topic_rel_docs[topic_num][docid] = topic_rel_docs[topic_num][docid]
        except KeyError:
            pass
topic_rel_docs = filter_topic_rel_docs
'''
ideal_user_sessions,precise_user_sessions,recall_user_sessions,all_sessions = select_sessions(topic_rel_docs,read_full=True)

precise_session_topic_nums = {}
for session in precise_user_sessions:
    topic_num = session.getElementsByTagName("topic")[0].getAttribute("num")
    try:
        precise_session_topic_nums[topic_num] += [session] 
    except KeyError:
        precise_session_topic_nums[topic_num] = [session] 

all_session_topic_nums = {}
for session in all_sessions:
    topic_num = session.getElementsByTagName("topic")[0].getAttribute("num")
    try:
        all_session_topic_nums[topic_num] += [session] 
    except KeyError:
        all_session_topic_nums[topic_num] = [session] 
'''
act_session_topics = {}
ideal_user_sessions,precise_user_sessions,recall_user_sessions,all_sessions = select_sessions(topic_rel_docs,read_full=True)
act_session_queries = []
act_sessions_full_topics = {}
testing_session_nums_1 = pickle.load(open("../supervised_models/train_test_splits/testing_session_nums_1.pk", "rb"))
session_nums_dict = dict(zip(testing_session_nums_1, range(len(testing_session_nums_1)))) 
for session in all_sessions:
    if session.getAttribute("num") in session_nums_dict:
        act_session = Session(session, clueweb_snippet_collection_2)
        try:
            act_session_topics[act_session.topic_num] += [act_session] 
        except:
            act_session_topics[act_session.topic_num] = [act_session] 
    act_session = Session(session, clueweb_snippet_collection_2)
    try:
        act_sessions_full_topics[act_session.topic_num] += [act_session] 
    except:
        act_sessions_full_topics[act_session.topic_num] = [act_session] 

#simulated sessions for the topic
#parameter_setting
parameters = [1.0, 20]
num_topics = 0
simulated_sessions = []
query_similarities = []
query_similarities_full = []
query_similarities_2 = []
query_similarities_full_2 = []
#avg_topic_performances = [[],[],[],[],[],[],[],[]]
#avg_topic_performances_2 = [[],[],[],[],[],[],[],[]]
candidate_queries = {}
topic_num_nums = [int(s) for s in topic_rel_docs.keys()]
topic_num_nums.sort()
print (topic_num_nums)
query_similarities_topics = {}
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
    sim_sessions,ranked_queries = simulate_sessions(topic_num, preprocess(topic_desc), topic_target_items, {}, {}, parameters, all_doc_bigram_lm)
    ranked_queries = sorted(ranked_queries,key = lambda l:l[1], reverse=True)
    candidate_queries[topic_num] = ranked_queries
    '''
    act_sessions = []
    training_session_nums_1 = pickle.load(open("../supervised_models/train_test_splits/training_session_nums_1.pk", "rb"))
    session_nums_dict = dict(zip(training_session_nums_1, range(len(training_session_nums_1))))
    for session in all_session_topic_nums[topic_num]:
        if session.getAttribute("num") in session_nums_dict:
            act_session = Session(session, document_content)
            act_sessions += [act_session]
    
    act_sessions_first_click = []
    for session in act_sessions:
        pruned_interactions = session.interactions
        for idx,inte in enumerate(session.interactions):
            if len(inte.clicks)>0:
                pruned_interactions = session.interactions[:idx+1]
                break
        pruned_session = Session()
        pruned_session.interactions = pruned_interactions
        pruned_session.topic_num = session.topic_num
        act_sessions_first_click += [pruned_session]
    '''
    simulated_sessions += sim_sessions

    '''
    print ("Num act sessions: ", len(act_sessions))
    topic_performance = evaluate_sessions(sim_sessions, act_sessions_first_click)    
    topic_performance_2 = evaluate_sessions(sim_sessions, act_sessions)
    for idx,performance in enumerate(avg_topic_performances):
        performance += [topic_performance[idx]]
    for idx,performance in enumerate(avg_topic_performances_2):
        performance += [topic_performance_2[idx]]
    '''
    #if(num_topics == 10):
    #   break
    print ("Topic num: {} Topic desc: {}".format(topic_num, preprocess(topic_desc)))
    queries_list_sim = [list(filter(lambda l: l.type =="reformulate", session.interactions)) for session in sim_sessions]
    queries_list_sim = [interaction.query for session in queries_list_sim for interaction in session]
    queries_list_act = [list(filter(lambda l: l.type =="reformulate", session.interactions)) for session in act_session_topics[topic_num]]
    queries_list_act = [interaction.query for session in queries_list_act for interaction in session]
    queries_list_act = [preprocess(" ".join(query), lemmatizing = True).split(" ") for query in queries_list_act]
    (simulated_to_act_sim,act_to_simulated_sim) = query_similarity_evaluation(queries_list_sim, queries_list_act)
    query_similarities += [float(simulated_to_act_sim+act_to_simulated_sim)/float(2)]
    query_similarities_2 += [simulated_to_act_sim]
    queries_list_act = [list(filter(lambda l: l.type =="reformulate", session.interactions)) for session in act_sessions_full_topics[topic_num]]
    queries_list_act = [interaction.query for session in queries_list_act for interaction in session]
    queries_list_act = [preprocess(" ".join(query), lemmatizing = True).split(" ") for query in queries_list_act]
    print ("Simulated queries: ", queries_list_sim)
    print ("Real queries: ", queries_list_act)
    (simulated_to_act_sim,act_to_simulated_sim) = query_similarity_evaluation(queries_list_sim, queries_list_act)
    query_similarities_full += [float(simulated_to_act_sim+act_to_simulated_sim)/float(2)]    
    query_similarities_full_2 += [simulated_to_act_sim]
    query_similarities_topics[topic_num] = simulated_to_act_sim
    print ("Query similarity: ", simulated_to_act_sim, act_to_simulated_sim)
print ('Query_similarities_topics: ', query_similarities_topics)
print ("QUERY Similarities testing set: ", float(sum(query_similarities))/float(len(query_similarities)))  
print ("QUERY Similarities Full: ", float(sum(query_similarities_full))/float(len(query_similarities_full)))  
print ("QUERY Similarities_2 testing set: ", float(sum(query_similarities_2))/float(len(query_similarities_2)))  
print ("QUERY Similarities Full 2: ", float(sum(query_similarities_full_2))/float(len(query_similarities_full_2))) 
avg_session_length_sim = [len(session.interactions) for session in simulated_sessions]
avg_session_length_sim = float(sum(avg_session_length_sim))/float(len(avg_session_length_sim))
print ('Simulated sessions Avg Session length: ', avg_session_length_sim)
avg_session_length_sim_first_click = []
for session in simulated_sessions:
    pruned_interactions = session.interactions
    for idx,inte in enumerate(session.interactions):
        if len(inte.clicks)>0:
            pruned_interactions = session.interactions[:idx+1]
            break
    avg_session_length_sim_first_click += [len(pruned_interactions)]
avg_session_length_sim_first_click = float(sum(avg_session_length_sim_first_click))/float(len(avg_session_length_sim_first_click))
print ('Simulated sessions Avg Session length first click: ', avg_session_length_sim_first_click)
print ("Dumping simulated sessions")
#pickle.dump(simulated_sessions, open("../simulated_sessions/reform_variations/simulated_session_all_topics_greedy_soln_query_reform_0.9_0.7_ratio_upd_more_clicks.pk", "wb"))

#pickle.dump(simulated_sessions, open("../simulated_sessions/final_results/simulated_session_all_topics_basic_recall_precision_query_unsup_0.8_reform_1.0_0.7_0.7_more_clicks.pk", "wb"))
#pickle.dump(simulated_sessions, open("../simulated_sessions/final_results/simulated_session_all_topics_word_sup_1_1_500_query_unsup_0.8_reform_no_reform_session_nums_5_more_clicks.pk", "wb"))
pickle.dump(simulated_sessions, open("../simulated_sessions/final_results/simulated_session_all_topics_QS3plus_baseline_more_clicks_final.pk", "wb"))
pickle.dump(candidate_queries, open("../simulated_sessions/final_results/candidate_QS3plus_baseline_queries_final.pk", "wb"))
#pickle.dump(simulated_sessions, open("../simulated_sessions/final_results/simulated_session_all_topics_dtc_baseline_with_reform_session_nums_5_more_clicks.pk", "wb"))
#pickle.dump(candidate_queries, open("../simulated_sessions/final_results/candidate_dtc_baseline_queries_session_nums_5.pk", "wb"))
        


'''
performance_names = ["avg_avg_session_length_sim","avg_avg_session_length_act", "avg_avg_num_queries_sim", "avg_avg_num_queries_act", "avg_act_to_simulated_sim", "avg_act_to_simulated_sim", "avg_avg_num_clicks_sim","avg_avg_num_clicks_act"]
for idx,performance_name in enumerate(performance_names):
    avg_topic_performances[idx] = float(sum(avg_topic_performances[idx]))/float(len(avg_topic_performances[idx]))
    print ("{} first click: {}".format(performance_name, avg_topic_performances[idx]))
for idx,performance_name in enumerate(performance_names):
    avg_topic_performances_2[idx] = float(sum(avg_topic_performances_2[idx]))/float(len(avg_topic_performances_2[idx]))
    print ("{} whole session: {}".format(performance_name, avg_topic_performances_2[idx]))
'''
#break
    #topic_performance = find_session_likelihood(preprocess(topic_desc, stopwords), topic_target_items, document_content, parameters, precise_session_topic_nums[topic_num])
#real sessions likelihood for th topic
#for topic_num in :