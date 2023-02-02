#select dataset for the specific task and then evaluate
from create_dataset import *
from User_model_unsup import User_model, Session
from User_model_utils import *
from BM25_ranker import BM25_ranker
import random
from collections import Counter
import math
import pickle
import scipy

def select_sessions(topic_rel_docs):
    session_data = xml.dom.minidom.parse('../Session_Track_2014/sessiontrack2014.xml')
    sessions = session_data.getElementsByTagName("session")
    ideal_user_sessions = []
    precise_user_sessions = []
    recall_user_sessions = []
    all_sessions = []
    for session in sessions:
        topic_num = session.getElementsByTagName("topic")[0].getAttribute("num")
        try:
            rel_docs = topic_rel_docs[topic_num]
        except:
            rel_docs = {}
        interactions = session.getElementsByTagName("interaction")
        result_doc_ids = []
        click_doc_ids = []
        for interaction in interactions:
            results = interaction.getElementsByTagName("result")
            try:
                clicked_items = interaction.getElementsByTagName("click")
                for click in clicked_items:
                    click_doc_ids += [getText(click.getElementsByTagName("docno")[0].childNodes)]
            except:
                pass
            for result in results:
                result_doc_ids += [getText(result.getElementsByTagName("clueweb12id")[0].childNodes)]
            
        precision = 0
        for doc_id in click_doc_ids:
            if doc_id in rel_docs:
                precision += 1
        if click_doc_ids!=[]:
            precision = float(precision)/float(len(click_doc_ids))
        else:
            print ("no clicks")
        recall = 0
        recall_total = 0 
        for doc_id in result_doc_ids:
            if doc_id in rel_docs:
                recall_total += 1
                if doc_id in click_doc_ids:
                    recall += 1
        if (recall_total!=0):
            recall = float(recall)/float(recall_total)
        else:
            print ("no rel docs in result")
        if (precision == 1) and recall==1:
            ideal_user_sessions += [session]
        if (precision == 1):
            print ( click_doc_ids)
            precise_user_sessions += [session]
        if (recall == 1):
            recall_user_sessions += [session]
        all_sessions += [session]
        print (session.getAttribute("num"), precision, recall)

    return ideal_user_sessions,precise_user_sessions,recall_user_sessions,all_sessions

def format_results(results, doci_id_to_clueweb_id, document_content):
    formatted_results = []
    for result in results:
        formatted_result={} 
        formatted_result["docid"] = doci_id_to_clueweb_id[result[0]]
        formatted_result["content"] = document_content[formatted_result["docid"]]["content"]
        formatted_result["title"] = document_content[formatted_result["docid"]]["title"]
        formatted_results += [formatted_result]
    return formatted_results


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
    print ( "NUM CLICKS: ", num_clicks_act, sum(num_clicks_act),  len(num_clicks_act))
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

topic_descs = read_topic_descs()
topic_rel_docs = read_judgements()
ideal_user_sessions,precise_user_sessions,recall_user_sessions,all_sessions = select_sessions(topic_rel_docs)

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

document_content = {}
with open("../Session_Track_2014/clueweb_snippet_data.txt", "r") as infile:
    line = infile.readline()
    while(line.strip() != ""):
        clueweb_id = line.strip().split("# ")[1]
        document_content[clueweb_id] = ""
        line = infile.readline()
        document_content[clueweb_id]  += preprocess(line.strip())
        line = infile.readline()
        document_content[clueweb_id] += " " + preprocess(line.strip())
        line = infile.readline()
print ("started this")
print ("Num docs:", len(document_content.keys()))


simulated_sessions = pickle.load(open("../simulated_sessions/simulated_session_all_topics_word_sup_1_query_unsup_session_nums_1_fast.pk", "rb"))
simulated_sessions_topics = {}
for session in simulated_sessions:
    try:
        simulated_sessions_topics[session.topic_num] += [session]
    except:
        simulated_sessions_topics[session.topic_num] = [session]
avg_topic_performances = [[],[],[],[],[],[],[],[]]
avg_topic_performances_2 = [[],[],[],[],[],[],[],[]]
avg_topic_performances_3 = [[],[],[],[],[],[],[],[]]
avg_topic_performances_4 = [[],[],[],[],[],[],[],[]]
training_session_nums_1 = pickle.load(open("../supervised_models/train_test_splits/testing_session_nums_1.pk", "rb"))
session_nums_dict = dict(zip(training_session_nums_1, range(len(training_session_nums_1))))
for topic_num in all_session_topic_nums:
    #if (topic_num == '36'):
    try:
        topic_desc = topic_descs[topic_num]
        topic_target_items = topic_rel_docs[topic_num]
    except:
        continue
    print ("Topic num: {} Topic desc: {}".format(topic_num, preprocess(topic_desc)))
    sim_sessions = simulated_sessions_topics[topic_num]
    act_sessions = []
    for session in all_session_topic_nums[topic_num]:
        act_session = Session(session, document_content)
        act_sessions += [act_session]
    
    act_sessions_first_click = []
    for session in act_sessions:
        pruned_interactions = session.interactions
        print ("ORIGINAL SESSION")
        for idx,inte in enumerate(session.interactions):
            print ("INTE CLICKS: ", inte.clicks)
            if len(inte.clicks)>0:
                pruned_interactions = session.interactions[:idx+1]
                break
        pruned_session = Session()
        pruned_session.interactions = pruned_interactions
        pruned_session.topic_num = session.topic_num
        pruned_session.session_num = session.session_num
        print ("PRUNED SESSION")
        for inte in pruned_session.interactions:
            print ("INTE CLICKS: ", inte.clicks)
        act_sessions_first_click += [pruned_session]

    print ("Num act sessions: ", len(act_sessions))
    topic_performance = evaluate_sessions(sim_sessions, act_sessions_first_click)
    topic_performance_2 = evaluate_sessions(sim_sessions, act_sessions)
    for idx,performance in enumerate(avg_topic_performances):
        performance += [topic_performance[idx]]
    for idx,performance in enumerate(avg_topic_performances_2):
        performance += [topic_performance_2[idx]]

    act_sessions = []
    for session in all_session_topic_nums[topic_num]:
        if session.getAttribute("num") in session_nums_dict:
            act_session = Session(session, document_content)
            act_sessions += [act_session]
    
    act_sessions_first_click = []
    for session in act_sessions:
        pruned_interactions = session.interactions
        print ("ORIGINAL SESSION")
        for idx,inte in enumerate(session.interactions):
            print ("INTE CLICKS: ", inte.clicks)
            if len(inte.clicks)>0:
                pruned_interactions = session.interactions[:idx+1]
                break
        pruned_session = Session()
        pruned_session.interactions = pruned_interactions
        pruned_session.topic_num = session.topic_num
        pruned_session.session_num = session.session_num
        print ("PRUNED SESSION")
        for inte in pruned_session.interactions:
            print ("INTE CLICKS: ", inte.clicks)
        act_sessions_first_click += [pruned_session]
    print ("Num act sessions: ", len(act_sessions))
    topic_performance = evaluate_sessions(sim_sessions, act_sessions_first_click)
    topic_performance_2 = evaluate_sessions(sim_sessions, act_sessions)
    for idx,performance in enumerate(avg_topic_performances_3):
        performance += [topic_performance[idx]]
    for idx,performance in enumerate(avg_topic_performances_4):
        performance += [topic_performance_2[idx]]
    #if(num_topics == 10):
    #   break
all_avg_topic_performances = [avg_topic_performances, avg_topic_performances_2, avg_topic_performances_3, avg_topic_performances_4]
test_session_names = ["All sessions first click", "All sessions whole sesssion", "Testing session first click", "Testing session whole sesssion"]
for s,avg_topic_performances in enumerate(all_avg_topic_performances):
    print ("EVALUATION SET: ",test_session_names[s])
    session_len_corr = scipy.stats.pearsonr(avg_topic_performances[1] , avg_topic_performances[0])
    num_queries_corr = scipy.stats.pearsonr(avg_topic_performances[3] , avg_topic_performances[2])
    clicks_corr = scipy.stats.pearsonr(avg_topic_performances[7] , avg_topic_performances[6])
    session_len_diff = [avg_topic_performances[1][x]-avg_topic_performances[0][x]  for x in range(len(avg_topic_performances[0]))]
    num_queries_diff = [avg_topic_performances[3][x]-avg_topic_performances[2][x]  for x in range(len(avg_topic_performances[0]))]
    clicks_diff = [avg_topic_performances[7][x]-avg_topic_performances[6][x]  for x in range(len(avg_topic_performances[0]))]
    session_len_diff = float(sum(session_len_diff))/float(len(session_len_diff))
    num_queries_diff = float(sum(num_queries_diff))/float(len(num_queries_diff))
    clicks_diff = float(sum(clicks_diff))/float(len(clicks_diff))
    performance_names = ["avg_avg_session_length_sim","avg_avg_session_length_act", "avg_avg_num_queries_sim", "avg_avg_num_queries_act", "avg_act_to_simulated_sim", "avg_simulated_to_act_sim", "avg_avg_num_clicks_sim","avg_avg_num_clicks_act"]
    for idx,performance_name in enumerate(performance_names):
        avg_topic_performances[idx] = float(sum(avg_topic_performances[idx]))/float(len(avg_topic_performances[idx]))
        print ("{}: {}".format(performance_name, avg_topic_performances[idx]))
    performance_names = ["session_len_pearson_corr", "num_queries_pearson_corr", "clicks_pearson_corr", "session_len_diff", "num_queries_diff", "clicks_diff"]
    performances = [session_len_corr, num_queries_corr, clicks_corr, session_len_diff, num_queries_diff, clicks_diff]
    for idx,performance_name in enumerate(performance_names):
        print ("{}: {}".format(performance_name, performances[idx]))

#break
    #topic_performance = find_session_likelihood(preprocess(topic_desc, stopwords), topic_target_items, document_content, parameters, precise_session_topic_nums[topic_num])
#real sessions likelihood for th topic
#for topic_num in :