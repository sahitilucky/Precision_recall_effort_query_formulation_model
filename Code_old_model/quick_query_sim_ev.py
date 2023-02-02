
from create_dataset import *
#print ("where is it coming from")
from User_model_utils import *
#print ("where is it coming from")
from BM25_ranker import BM25_ranker, InvertedIndex
import random
from collections import Counter
import math
import pickle
import matplotlib.pyplot as plt
from statistics import mean 

def evaluate_sim_sessions_NDCG(sim_sessions, topic_rel_docs):
    all_sessions_trends = {}
    for session_idx,session in enumerate(sim_sessions):
        session_ndcgs = []
        for interaction in session.interactions:
            results = [(result["docid"],1) for result in interaction.results]
            session_ndcgs += [compute_NDCG(results,topic_rel_docs[session.topic_num], 10, list(topic_rel_docs[session.topic_num].keys()))]
        #print ("Session ndcgs idx: ", session_idx, session.topic_num, session_ndcgs)
        num_increments = 0 
        num_decrements = 0
        num_equals = 0
        for i in range(len(session_ndcgs)):
            if (i>0):
                if (session_ndcgs[i]>session_ndcgs[i-1]):
                    num_increments +=1
                if (session_ndcgs[i]<session_ndcgs[i-1]):
                    num_decrements += 1
                if (session_ndcgs[i]==session_ndcgs[i-1]):
                    num_equals += 1
        if (len(session_ndcgs))<3:
            session_ndcgs_2 = session_ndcgs + [session_ndcgs[-1]]*(3-len(session_ndcgs))
        else:
            session_ndcgs_2 = session_ndcgs
        session_part1 = session_ndcgs_2[0: int(float(len(session_ndcgs_2))/float(3)) ]
        session_part2 = session_ndcgs_2[int(float(len(session_ndcgs_2))/float(3)): 2*int(float(len(session_ndcgs_2))/float(3))]
        session_part3 = session_ndcgs_2[2*int(float(len(session_ndcgs_2))/float(3)): ]
        session_average_ndcgs = [float(sum(session_part1))/float(len(session_part1)),float(sum(session_part2))/float(len(session_part2)),float(sum(session_part3))/float(len(session_part3))  ]
        all_sessions_trends[session_idx] = [num_increments, num_decrements, num_equals, session_average_ndcgs, session_ndcgs]
    return (all_sessions_trends)

def get_first_non_zero_ndcgs(sim_sessions, topic_rel_docs, number):
    first_non_zero_ndcgs = {}
    for session_idx,session in enumerate(sim_sessions):
        ndcg_idx = 0
        rel_docs = topic_rel_docs[session.topic_num].copy()
        for idx,interaction in enumerate(session.interactions):
            success = False
            results = [(result["docid"],1) for result in interaction.results][:10]
            for docid,score in results: 
                try:
                    there = rel_docs[docid]
                    if (number == 1):
                        del(rel_docs[docid])
                    success = True
                except:
                    pass
            if (success):
                ndcg_idx += 1
                try:
                    first_non_zero_ndcgs[ndcg_idx] += [idx+1]
                except:
                    first_non_zero_ndcgs[ndcg_idx] = [idx+1]
    for ndcg_idx in first_non_zero_ndcgs:
        first_non_zero_ndcgs[ndcg_idx] = float(sum(first_non_zero_ndcgs[ndcg_idx]))/float(len(first_non_zero_ndcgs[ndcg_idx]))
    print (first_non_zero_ndcgs)
 

def time_series_similarity(session1, session2):
    return

def get_some_avg_session_trends(all_sessions_trends):
    avg_session_trends = [[],[],[],[[],[],[]], [], [[],[],[]]]
    for s in all_sessions_trends:
        total_changes = all_sessions_trends[s][0] + all_sessions_trends[s][1] + all_sessions_trends[s][2]
        if total_changes != 0:
            avg_session_trends[5][0] += [float(all_sessions_trends[s][0])/float(total_changes)] 
            avg_session_trends[5][1] += [float(all_sessions_trends[s][1])/float(total_changes)]
            avg_session_trends[5][2] += [float(all_sessions_trends[s][2])/float(total_changes)]
            avg_session_trends[0] += [all_sessions_trends[s][0]]
            avg_session_trends[1] += [all_sessions_trends[s][1]]
            avg_session_trends[2] += [all_sessions_trends[s][2]]
        for idx in range(3):
            avg_session_trends[3][idx] += [all_sessions_trends[s][3][idx]]
    total_changes = float(sum(avg_session_trends[0])) + float(sum(avg_session_trends[1])) + float(sum(avg_session_trends[2]))
    avg_session_trends[4] = [float(sum(avg_session_trends[0]))/float(total_changes),float(sum(avg_session_trends[1]))/float(total_changes),float(sum(avg_session_trends[2]))/float(total_changes)]
    avg_session_trends[0] = float(sum(avg_session_trends[0]))/float(len(avg_session_trends[0]))
    avg_session_trends[1] = float(sum(avg_session_trends[1]))/float(len(avg_session_trends[1]))
    avg_session_trends[2] = float(sum(avg_session_trends[2]))/float(len(avg_session_trends[2]))
    avg_session_trends[5][0] = float(sum(avg_session_trends[5][0]))/float(len(avg_session_trends[5][0]))
    avg_session_trends[5][1] = float(sum(avg_session_trends[5][1]))/float(len(avg_session_trends[5][1]))
    avg_session_trends[5][2] = float(sum(avg_session_trends[5][2]))/float(len(avg_session_trends[5][2]))
    
    #print ("AVERAGE NUM INCREMENTS: ", avg_session_trends[0])
    #print ("AVERAGE NUM DECREMENTS: ", avg_session_trends[1])
    #print ("AVERAGE NUM EQUALS: ", avg_session_trends[2])
    for idx in range(3):
        avg_session_trends[3][idx] = sum(avg_session_trends[3][idx])/float(len(avg_session_trends[3][idx]))
    length_wise_avg_trends = {}
    session_length_number = {}
    for session_idx in all_sessions_trends:
        length = len(all_sessions_trends[session_idx][4])
        try:
            length_wise_avg_trends[length] = list(map(lambda x,y: x+y , all_sessions_trends[session_idx][4], length_wise_avg_trends[length]))
            session_length_number[length] += 1
        except:
            length_wise_avg_trends[length] = all_sessions_trends[session_idx][4]
            session_length_number[length] = 1
    for length in length_wise_avg_trends:
        length_wise_avg_trends[length] = list(map(lambda l:float(l)/float(session_length_number[length]), length_wise_avg_trends[length]))
    #length_wise_inc_dec = {}
    #print ("AVERAGE AVG NDCG Trends: ", avg_session_trends)    
    return (avg_session_trends,length_wise_avg_trends)

def plot_avg_session_ndcgs(avg_session_trends, avg_session_trends_sim_list, length_wise_avg_trends, length_wise_avg_trends_sim_list, measure):
    x = list(range(len(avg_session_trends[3])))
    plt.clf()
    colors_list = ['c', 'b', 'g','y','m']
    sim_session_labels = ["sim_sessions_with_reform_nc", "sim_sessions_with_reform", "sim_sessions_without_reform", "dtc_sim_sessions", "qs3+_sim_sessions"]
    sim_session_labels = sim_session_labels[1:]
    colors_list = colors_list[1:]
    plt.plot(x,avg_session_trends[3], label = "real_sessions", color = 'r', linewidth = 1.0)
    for label_idx,avg_session_trends_sim in enumerate(avg_session_trends_sim_list):
        plt.plot(x,avg_session_trends_sim[3], label=sim_session_labels[label_idx], color = colors_list[label_idx], linewidth = 1.0)
    plt.xlabel('part of the sessions')
    plt.ylabel('Average ' + measure)
    if measure =='NDCG':
        plt.ylim((0,0.6))
    else:
        plt.ylim((0,0.2))    
    plt.xticks(x, ['first part','middle part','last part'])
    plt.legend()
    plt.title("Average " + measure + " in first ,middle, last part of session")
    plt.savefig("../reformulation_trends_plots/more_clicks_new_2/Average_" + measure + "_3_parts_more_clicks.jpg")
    plt.clf()
    all_lengths = []
    for length_wise_avg_trends_sim in length_wise_avg_trends_sim_list:
        all_lengths += list(length_wise_avg_trends_sim.keys())
    all_lengths += list(length_wise_avg_trends.keys())
    all_lengths = list(set(all_lengths))
    for l in all_lengths:
        plt.clf()
        try:
            there = length_wise_avg_trends[l]
            plt.plot(list(range(l)),length_wise_avg_trends[l], label = "real_sessions", color = 'r', linewidth = 1.0)
        except KeyError:
            pass
        for label_idx,length_wise_avg_trends_sim in enumerate(length_wise_avg_trends_sim_list):
            try:
                length_wise_avg_trends_sim[l]
                plt.plot(list(range(l)),length_wise_avg_trends_sim[l], label = sim_session_labels[label_idx], color = colors_list[label_idx], linewidth = 1.0)
            except KeyError:
                pass
        plt.xlabel('log number')
        plt.ylabel('Average ' + measure)
        if measure =='NDCG':
            plt.ylim((0,0.6))
        else:
            plt.ylim((0,0.2))    
        plt.legend()
        plt.xticks(list(range(l)))
        plt.title("Average " + measure + " of sessions of length "+ str(l))
        plt.savefig("../reformulation_trends_plots/more_clicks_new_2/Average_" + measure + "_len_" + str(l) + "_more_clicks.jpg")
        plt.clf()


def jaccard_similarity(list1, list2):
    #print (list1, list2, len(set(list1).intersection(set(list2))), len(set(list1).union(set(list2))))
    return float(len(set(list1).intersection(set(list2))))/float(len(set(list1).union(set(list2))))

#TWO THINGS
#QUERY SIMILARITY ONLY USING CANDIDATE QUEIRES DIRECTLY for trec robust, trec sessions, both baselines queries comparision.
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
    #print (simulated_to_act_sim, act_to_simulated_sim)
    simulated_to_act_sim_list = [max(x) for x in simulated_to_act_sim]
    act_to_simulated_sim_list = [max(x) for x in act_to_simulated_sim]
    simulated_to_act_sim = float(sum(simulated_to_act_sim_list))/float(len(simulated_to_act_sim_list))
    act_to_simulated_sim = float(sum(act_to_simulated_sim_list))/float(len(act_to_simulated_sim_list))
    return (simulated_to_act_sim,act_to_simulated_sim,simulated_to_act_sim_list,act_to_simulated_sim_list)

#topic similarity based relevance trends for normal queries, reformulated queries, real queries, one baseline queries
def judge(results, topic_num, target_documents_details_2,target_topic_vectors_2,documents_rels):
    results_relevance = []
    lda,comm_dict = target_documents_details_2["all_topics"][1][0],target_documents_details_2["all_topics"][1][1]
    #print (len(target_topic_vectors_2))
    for result in results:
        weighted_sum_num = 0
        weighted_sum_den = 0
        a = lda[comm_dict.doc2bow(result["content"].split())]
        topic_dist_vector = [0]*100
        for v in a:
            topic_dist_vector[v[0]] = v[1]
        for idx,(docid,doc_rel) in enumerate(documents_rels): 
            weighted_sum_num += doc_rel*np.dot(topic_dist_vector, target_topic_vectors_2[idx])
            weighted_sum_den += documents_rels[idx][1]
        #cos_sim = sum([vector[i]*topic_dist_vector[i] for i in range(len(topic_dist_vector))])
        #weighted_sum_num += documents_rels[idx][1]*cos_sim
        result_relevance = float(weighted_sum_num)/float(weighted_sum_den)
        #normalization of result relevance
        results_relevance += [result_relevance]
    return results_relevance

def evaluate_sim_sessions_avg_sim(simulated_sessions, target_documents_details_2, target_document_vectors, middle_rel):
    all_sessions_trends = {}
    for session_idx,session in enumerate(simulated_sessions):
        session_avg_sim = []
        #print ("Session avg_sims idx dng judging...: ", session_idx, session.topic_num)
        documents_rels = target_documents_details_2[session.topic_num][3]
        target_topic_vectors_2 = []
        documents_rels += [("topic_desc", middle_rel)] 
        for docid,doc_rel in documents_rels:
            if (docid == "topic_desc"):
                target_topic_vectors_2 += [target_document_vectors[docid+"_" + str(session.topic_num)]]
            else:
                target_topic_vectors_2 += [target_document_vectors[docid]]  
        start_time = time.time()  
        for interaction in session.interactions:
            #print (len(interaction.results))
            sims = judge(interaction.results, session.topic_num,target_documents_details_2, target_topic_vectors_2, documents_rels)
            if (len(sims) == 0):
                session_avg_sim += [0]
            else:
                session_avg_sim += [float(sum(sims))/float(len(sims))]
        #print ("TIME TAKEN: ", time.time()-start_time)
        #print ("Session avg_sims idx: ", session_idx, session.topic_num, session_avg_sim)
        num_increments = 0 
        num_decrements = 0
        num_equals = 0
        session_average_ndcgs = []
        for i in range(len(session_avg_sim)):
            if (i>0):
                if (session_avg_sim[i]>session_avg_sim[i-1]):
                    num_increments +=1
                if (session_avg_sim[i]<session_avg_sim[i-1]):
                    num_decrements += 1
                if (session_avg_sim[i]==session_avg_sim[i-1]):
                    num_equals += 1
        #for i in range(0,len(session_avg_sim),3):
        #    session_average_ndcgs += [float(sum(session_avg_sim[i:i+3]))/float(len(session_avg_sim[i:i+3]))]
        if (len(session_avg_sim))<3:
            session_avg_sim_2 = session_avg_sim + [session_avg_sim[-1]]*(3-len(session_avg_sim))
        else:
            session_avg_sim_2 =  session_avg_sim
        session_part1 = session_avg_sim_2[0: int(float(len(session_avg_sim_2))/float(3)) ]
        session_part2 = session_avg_sim_2[int(float(len(session_avg_sim_2))/float(3)): 2*int(float(len(session_avg_sim_2))/float(3))]
        session_part3 = session_avg_sim_2[2*int(float(len(session_avg_sim_2))/float(3)): ]
        session_average_svg_sim = [float(sum(session_part1))/float(len(session_part1)),float(sum(session_part2))/float(len(session_part2)),float(sum(session_part3))/float(len(session_part3))  ]
        all_sessions_trends[session_idx] = [num_increments, num_decrements, num_equals, session_average_svg_sim, session_avg_sim]
    return (all_sessions_trends)

def evaluate_click_log_trends(sessions,sim_sessions_list):
    avg_click_log_indeces = {}
    for session in sessions:
        click_log_indeces = []
        for idx,interaction in enumerate(session.interactions):
            if(len(interaction.clicks)) > 0:
                click_log_indeces += [idx+1]
        for idx in range(len(click_log_indeces)):
            try:
                avg_click_log_indeces[idx] += [click_log_indeces[idx]]
            except KeyError:
                avg_click_log_indeces[idx] = [click_log_indeces[idx]]
    avg_click_log_indeces_sim_list = []
    for sim_sessions in sim_sessions_list:
        avg_click_log_indeces_sim = {}
        for session in sim_sessions:
            click_log_indeces = []
            for idx,interaction in enumerate(session.interactions):
                if(len(interaction.clicks)) > 0:
                    click_log_indeces += [idx+1]
            for idx in range(len(click_log_indeces)):
                try:
                    avg_click_log_indeces_sim[idx] += [click_log_indeces[idx]]
                except KeyError:
                    avg_click_log_indeces_sim[idx] = [click_log_indeces[idx]]
        avg_click_log_indeces_sim_list += [avg_click_log_indeces_sim]

    #print ("Click times: ", avg_click_log_indeces)
    #print ("Click times: ", avg_click_log_indeces_sim)
    titles = ['Distribution_of_first_clicks', 'distribution_of_second_clicks', 'distribution_of_third_clicks', 'distribution_of_fourth_clicks']
    max1 = 0
    for avg_click_log_indeces_sim in avg_click_log_indeces_sim_list:
        max1 =  max([max1,len(avg_click_log_indeces_sim.keys())])
    click_times = max([len(avg_click_log_indeces.keys()), max1])
    click_times = range(click_times)
    sim_session_labels = ["sim_sessions_with_reform_nc","sim_sessions_with_reform", "sim_sessions_without_reform", "dtc_sim_sessions", "qs3+_sim_sessions"]
    colors_list = ['c', 'b', 'g','y','m']
    sim_session_labels = sim_session_labels[1:]
    colors_list = colors_list[1:]
    for idx in click_times:
        plt.clf()
        try:
            distribution_of_first_clicks = Counter(avg_click_log_indeces[idx])
            distribution_of_first_clicks = {x:float(distribution_of_first_clicks[x])/float(len(avg_click_log_indeces[idx])) for x in distribution_of_first_clicks}
            distribution_of_first_clicks = sorted(distribution_of_first_clicks.items(), key=lambda l: l[0])
            plt.plot([s[0] for s in distribution_of_first_clicks], [s[1] for s in distribution_of_first_clicks], 'r',label='real_sessions', linewidth = 1.0)
            click_mean = round(mean(avg_click_log_indeces[idx]),3)
            print ("CLICK MEAN real sessions: ", idx, click_mean)
            plt.axvline(click_mean, ymax=0.7,color = 'r', linewidth = 1.0)
            plt.text(click_mean+0.05,0.7,str(round(click_mean,2)),color = 'r', fontsize = 'x-small')
        except KeyError:
            pass
        for label_idx,avg_click_log_indeces_sim in enumerate(avg_click_log_indeces_sim_list):
            try:
                distribution_of_first_clicks_sim = Counter(avg_click_log_indeces_sim[idx])
                distribution_of_first_clicks_sim = {x:float(distribution_of_first_clicks_sim[x])/float(len(avg_click_log_indeces_sim[idx])) for x in distribution_of_first_clicks_sim}
                distribution_of_first_clicks_sim = sorted(distribution_of_first_clicks_sim.items(), key=lambda l: l[0])
                plt.plot([s[0] for s in distribution_of_first_clicks_sim], [s[1] for s in distribution_of_first_clicks_sim],  color = colors_list[label_idx], label = sim_session_labels[label_idx], linewidth = 1.0)
                click_mean = round(mean(avg_click_log_indeces_sim[idx]),3)
                plt.axvline(click_mean, ymax=0.7, color = colors_list[label_idx], linewidth = 1.0)
                plt.text(click_mean+0.1,0.7, str(round(click_mean,2)),color = colors_list[label_idx], fontsize = 'x-small')
                print ("CLICK MEAN " +sim_session_labels[label_idx] + ": ", idx, click_mean)
            except KeyError:
                pass
        plt.xlabel('Log number')
        plt.ylabel('probability of click')
        plt.title('Distribution_of_' +str(idx+1) + '_time_clicks')
        plt.legend()
        plt.xticks(list(range(19)))
        plt.xlim((0,19)) 
        plt.ylim((0,1))
        plt.savefig('../reformulation_trends_plots/more_clicks_new_2/Distribution_of_' +str(idx+1) + '_time_clicks_more_clicks.jpg')
        plt.clf()
    #plot(distribution_of_first_clicks)

def just_session_query_simiarity_session_track(sim_sessions, act_sessions, test_file):
    act_session_topics = {}
    act_sessions_full_topics = {}
    testing_session_nums_1 = pickle.load(open(test_file, "rb"))
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

    sim_sessions_full_topics = {}
    for sim_session in sim_sessions:
        try:
            sim_sessions_full_topics[sim_session.topic_num] += [sim_session] 
        except:
            sim_sessions_full_topics[sim_session.topic_num] = [sim_session] 
    query_similarities = []
    query_similarities_2 = []
    query_similarities_full = []
    query_similarities_full_2 = []
    for topic_num in sim_sessions_full_topics:
        queries_list_sim = [list(filter(lambda l: l.type =="reformulate", session.interactions)) for session in sim_sessions_full_topics[topic_num]]
        queries_list_sim = [interaction.query for session in queries_list_sim for interaction in session]
        queries_list_act = [list(filter(lambda l: l.type =="reformulate", session.interactions)) for session in act_session_topics[topic_num]]
        queries_list_act = [interaction.query for session in queries_list_act for interaction in session]
        queries_list_act = [preprocess(" ".join(query), lemmatizing = True).split(" ") for query in queries_list_act]
        #print (len(queries_list_sim), len(queries_list_act))
        (simulated_to_act_sim,act_to_simulated_sim,simulated_to_act_sim_list, act_to_simulated_sim_list) = query_similarity_evaluation(queries_list_sim, queries_list_act)
        query_similarities += [float(simulated_to_act_sim+act_to_simulated_sim)/float(2)]
        query_similarities_2 += [simulated_to_act_sim]
        queries_list_act = [list(filter(lambda l: l.type =="reformulate", session.interactions)) for session in act_sessions_full_topics[topic_num]]
        queries_list_act = [interaction.query for session in queries_list_act for interaction in session]
        queries_list_act = [preprocess(" ".join(query), lemmatizing = True).split(" ") for query in queries_list_act]
        #print (len(queries_list_act))
        (simulated_to_act_sim,act_to_simulated_sim,simulated_to_act_sim_list,act_to_simulated_sim_list) = query_similarity_evaluation(queries_list_sim, queries_list_act)
        query_similarities_full += [float(simulated_to_act_sim+act_to_simulated_sim)/float(2)]    
        query_similarities_full_2 += [simulated_to_act_sim]
    print ("QUERY Similarities testing set: ", float(sum(query_similarities))/float(len(query_similarities)))  
    print ("QUERY Similarities Full: ", float(sum(query_similarities_full))/float(len(query_similarities_full)))  
    print ("QUERY Similarities_2 testing set: ", float(sum(query_similarities_2))/float(len(query_similarities_2)))  
    print ("QUERY Similarities Full 2: ", float(sum(query_similarities_full_2))/float(len(query_similarities_full_2))) 
    return float(sum(query_similarities_2))/float(len(query_similarities_2), )

def just_session_query_simiarity_trec_robust(simulated_sessions):
    topic_descs_query = read_trec_robust_queries()
    topic_num_nums = [int(s) for s in topic_descs_query.keys()]
    topic_num_nums.sort()
    query_similarities_2 = []
    sim_sessions_topics = {}
    for session in simulated_sessions:
        try:
            sim_sessions_topics[session.topic_num] += [session]
        except:
            sim_sessions_topics[session.topic_num] = [session]
    avg_session_length_sim = []
    for topic_num in topic_num_nums[:100]:
        topic_num = str(topic_num)
        queries_list_sim = [list(filter(lambda l: l.type =="reformulate", session.interactions)) for session in sim_sessions_topics[topic_num]]
        queries_list_sim = [interaction.query for session in queries_list_sim for interaction in session]
        (simulated_to_act_sim,act_to_simulated_sim) = query_similarity_evaluation(queries_list_sim, [preprocess(topic_descs_query[topic_num], lemmatizing = True).split(" ")])
        query_similarities_2 += [simulated_to_act_sim]
        avg_session_length_sim += [len(session.interactions) for session in sim_sessions_topics[topic_num]]
    avg_session_length_sim = float(sum(avg_session_length_sim))/float(len(avg_session_length_sim))
    print ('Simulated sessions Avg Session length: ', avg_session_length_sim)
    print ('query_similarities_2: ', float(sum(query_similarities_2))/float(len(query_similarities_2)))

#QUERY SIMILARITY with trec robust queries
def just_query_simiarity_trec_robust(candidate_queries):
    print ("Getting similarity")
    print ("TREC Robust")
    topic_descs_query = read_trec_robust_queries()
    query_similarities = []
    query_similarities_2 = []
    for topic_num in candidate_queries:
        queries_list_sim = [query[0] for query in candidate_queries[topic_num]][:10]
        (simulated_to_act_sim,act_to_simulated_sim) = query_similarity_evaluation(queries_list_sim, [preprocess(topic_descs_query[topic_num], lemmatizing = True).split(" ")])
        query_similarities += [float(simulated_to_act_sim+act_to_simulated_sim)/float(2)]
        query_similarities_2 += [simulated_to_act_sim]
        #print ("Query similarity: ", simulated_to_act_sim, act_to_simulated_sim)
    print ("QUERY Similarities: ", float(sum(query_similarities))/float(len(query_similarities))) 
    print ("QUERY Similarities_2: ", float(sum(query_similarities_2))/float(len(query_similarities_2)))  
    return

#QUERY SIMILARITY Uwith clueweb queries
def just_query_simiarity_session_track(candidate_queries, test_file, sessions_Sessions = None):
    query_similarities = []
    query_similarities_full = []
    query_similarities_2 = []
    query_similarities_full_2 = []
    document_content, clueweb_snippet_collection_2 = load_preprocess_robust_data()
    topic_rel_docs = read_judgements()
    act_session_topics = {}
    if sessions_Sessions == None:
        ideal_user_sessions,precise_user_sessions,recall_user_sessions,all_sessions = select_sessions(topic_rel_docs,read_full=True)
    else:
        all_sessions = sessions_Sessions
    act_session_queries = []
    act_sessions_full_topics = {}
    testing_session_nums_1 = pickle.load(open(test_file, "rb"))
    session_nums_dict = dict(zip(testing_session_nums_1, range(len(testing_session_nums_1)))) 
    if sessions_Sessions == None:
        for session in all_sessions:
            act_session = Session(session, clueweb_snippet_collection_2)
            if session.getAttribute("num") in session_nums_dict:
                try:
                    act_session_topics[act_session.topic_num] += [act_session] 
                except:
                    act_session_topics[act_session.topic_num] = [act_session] 
            try:
                act_sessions_full_topics[act_session.topic_num] += [act_session] 
            except:
                act_sessions_full_topics[act_session.topic_num] = [act_session] 
    else:
        for act_session in all_sessions:
            if act_session.session_num in session_nums_dict:
                try:
                    act_session_topics[act_session.topic_num] += [act_session] 
                except:
                    act_session_topics[act_session.topic_num] = [act_session] 
            try:
                act_sessions_full_topics[act_session.topic_num] += [act_session] 
            except:
                act_sessions_full_topics[act_session.topic_num] = [act_session] 

    query_similarity_topics = {}
    query_similarity_topics_full = {}
    for topic_num in candidate_queries:
        queries_list_sim = [query[0] for query in candidate_queries[topic_num]][:10]    
        queries_list_act = [list(filter(lambda l: l.type =="reformulate", session.interactions)) for session in act_session_topics[topic_num]]
        queries_list_act = [interaction.query for session in queries_list_act for interaction in session]
        queries_list_act = [preprocess(" ".join(query), lemmatizing = True).split(" ") for query in queries_list_act]
        #print (len(queries_list_sim), len(queries_list_act))
        (simulated_to_act_sim,act_to_simulated_sim,simulated_to_act_sim_list,act_to_simulated_sim_list) = query_similarity_evaluation(queries_list_sim, queries_list_act)
        query_similarities += [float(simulated_to_act_sim+act_to_simulated_sim)/float(2)]
        query_similarities_2 += [simulated_to_act_sim]
        query_similarity_topics[topic_num] = simulated_to_act_sim_list
        queries_list_act = [list(filter(lambda l: l.type =="reformulate", session.interactions)) for session in act_sessions_full_topics[topic_num]]
        queries_list_act = [interaction.query for session in queries_list_act for interaction in session]
        queries_list_act = [preprocess(" ".join(query), lemmatizing = True).split(" ") for query in queries_list_act]
        #print (len(queries_list_act))
        (simulated_to_act_sim,act_to_simulated_sim,simulated_to_act_sim_list,act_to_simulated_sim_list) = query_similarity_evaluation(queries_list_sim, queries_list_act)
        query_similarities_full += [float(simulated_to_act_sim+act_to_simulated_sim)/float(2)]    
        query_similarities_full_2 += [simulated_to_act_sim]
        query_similarity_topics_full[topic_num] = simulated_to_act_sim_list
    print ("QUERY Similarities testing set: ", float(sum(query_similarities))/float(len(query_similarities)))  
    print ("QUERY Similarities Full: ", float(sum(query_similarities_full))/float(len(query_similarities_full)))  
    print ("QUERY Similarities_2 testing set: ", float(sum(query_similarities_2))/float(len(query_similarities_2)))  
    print ("QUERY Similarities Full 2: ", float(sum(query_similarities_full_2))/float(len(query_similarities_full_2))) 
    #return float(sum(query_similarities_2))/float(len(query_similarities_2))
    return query_similarity_topics, query_similarity_topics_full
clueweb_snippet_collection,clueweb_snippet_collection_2 = read_clueweb_snippet_data()
def prune_simulated_sessions(simulated_sessions):
    topic_rel_docs = read_judgements()
    filter_topic_rel_docs = {}
    for topic_num in topic_rel_docs:
        filter_topic_rel_docs[topic_num] = {}
        for docid in topic_rel_docs[topic_num]:
            try:
                there = clueweb_snippet_collection_2[docid]
                filter_topic_rel_docs[topic_num][docid] = topic_rel_docs[topic_num][docid]
            except KeyError:
                pass
    topic_rel_docs = filter_topic_rel_docs
    for session in simulated_sessions:
        target_docs = topic_rel_docs[session.topic_num].copy()
        for idx,interaction in enumerate(session.interactions):
            for click in interaction.clicks:
                del(target_docs[click[0]])
            if (len(target_docs) == 0):
                break
        session.interactions = session.interactions[:(idx+1)]
    return simulated_sessions

def filter_real_sessions(real_sessions):
    pruned_sessions = []
    for session in real_sessions:
        clicks = [len(interaction.clicks) for interaction in session.interactions]
        clicks = list(filter(lambda l : l > 0, clicks))
        if (len(clicks) == 3):
            pruned_sessions += [session]
    print ('PRUNED REAL SESSIONS: ', len(pruned_sessions))
    return pruned_sessions
def candidate_query_scoring(candidate_queries, alpha, bigram_topic_lm):
    candidate_queries_new = {}
    for topic_num in candidate_queries:
        query_scores = []
        #alpha = float(3)/float(4)
        for (query,doc_list, score) in candidate_queries[topic_num]:
            num_phrases = 0
            for idx,word1 in enumerate(query):
                for word2 in query[idx+1:]:
                    if (word1+" "+word2) in bigram_topic_lm[topic_num]:
                        num_phrases += bigram_topic_lm[topic_num][word1+" "+word2]

                    #if (word1 + " " +word2) in keywords:
                    #    num_phrases += 1
            if (len(query)>1) and (num_phrases>0):
                p_bigrams = float(num_phrases)/float(len(query)*(len(query)-1))
            else:
                p_bigrams = 0.000001
            score1 = alpha*score/float(3.0) + (1-alpha)*math.log(p_bigrams) 
            score2 = alpha*score/float(3.0) + (1-alpha)*float(1)/float(math.exp(-float(num_phrases)/float(len(query)))+1)
            query_scores += [(query, score, float(1)/float(math.exp(-float(num_phrases)/float(len(query)))+1),score2)]
        query_scores = sorted(query_scores, key =lambda l:l[3],reverse=True)
        candidate_queries_new[topic_num] = query_scores
    candidate_queries = candidate_queries_new
    return candidate_queries

def quick_evaluation(sessions):
    candidate_queries = pickle.load(open("../simulated_sessions/final_results/candidate_dtc_baseline_queries_session_nums_1.pk", "rb"))
    sim_dict1,_ = just_query_simiarity_session_track(candidate_queries,"../supervised_models/train_test_splits/testing_session_nums_1.pk", sessions)
    candidate_queries = pickle.load(open("../simulated_sessions/final_results/candidate_dtc_baseline_queries_session_nums_2.pk", "rb"))
    sim_dict2,_ = just_query_simiarity_session_track(candidate_queries,"../supervised_models/train_test_splits/testing_session_nums_2.pk", sessions)
    candidate_queries = pickle.load(open("../simulated_sessions/final_results/candidate_dtc_baseline_queries_session_nums_3.pk", "rb"))
    sim_dict3,_ = just_query_simiarity_session_track(candidate_queries,"../supervised_models/train_test_splits/testing_session_nums_3.pk", sessions)
    candidate_queries = pickle.load(open("../simulated_sessions/final_results/candidate_dtc_baseline_queries_session_nums_4.pk", "rb"))
    sim_dict4,_ = just_query_simiarity_session_track(candidate_queries,"../supervised_models/train_test_splits/testing_session_nums_4.pk", sessions)
    candidate_queries = pickle.load(open("../simulated_sessions/final_results/candidate_dtc_baseline_queries_session_nums_5.pk", "rb"))
    sim_dict5,_ = just_query_simiarity_session_track(candidate_queries,"../supervised_models/train_test_splits/testing_session_nums_5.pk", sessions)
    similarities = {}
    for topic_num in sim_dict1:
        similarities[topic_num] = []
        for idx in range(len(sim_dict1[topic_num])):
            similarities[topic_num] += [(sim_dict1[topic_num][idx]+sim_dict2[topic_num][idx]+sim_dict3[topic_num][idx]+sim_dict4[topic_num][idx]+sim_dict5[topic_num][idx])/5]
    print ('Average similarity @1: ', float(sum([sum(similarities[x][:1])/float(len(similarities[x][:1])) for x in similarities]))/float(len(similarities)))         
    print ('Average similarity @5: ', float(sum([sum(similarities[x][:5])/float(len(similarities[x][:5])) for x in similarities]))/float(len(similarities)))         
    print ('Average similarity @10: ', float(sum([sum(similarities[x][:10])/float(len(similarities[x][:10])) for x in similarities]))/float(len(similarities)))         
    print ('SIMILARITIES:' , similarities) 

    candidate_queries4 = pickle.load(open("../simulated_sessions/final_results/candidate_QS3plus_baseline_queries_final.pk", "rb"))
    _,sim_dict1 = just_query_simiarity_session_track(candidate_queries4,"../supervised_models/train_test_splits/testing_session_nums_1.pk", sessions)
    qs3_similarities = {}
    for topic_num in sim_dict1:
        qs3_similarities[topic_num] = []
        for idx in range(len(sim_dict1[topic_num])):
            qs3_similarities[topic_num] += [sim_dict1[topic_num][idx]]
    print ('Average similarity @1: ', float(sum([sum(qs3_similarities[x][:1])/float(len(qs3_similarities[x][:1])) for x in qs3_similarities]))/float(len(qs3_similarities)))         
    print ('Average similarity @5: ', float(sum([sum(qs3_similarities[x][:5])/float(len(qs3_similarities[x][:5])) for x in qs3_similarities]))/float(len(qs3_similarities)))         
    print ('Average similarity @10: ', float(sum([sum(qs3_similarities[x][:10])/float(len(qs3_similarities[x][:10])) for x in qs3_similarities]))/float(len(qs3_similarities)))         
    print ('SIMILARITIES:' , qs3_similarities) 

    
    return similarities, qs3_similarities

def query_similarity_comparision(sessions):
    simulated_sessions_dtc = []
    simulated_sessions_dtc += pickle.load(open("../simulated_sessions/final_results/simulated_session_all_topics_dtc_baseline_with_reform_session_nums_1_more_clicks.pk", "rb"))
    simulated_sessions_dtc += pickle.load(open("../simulated_sessions/final_results/simulated_session_all_topics_dtc_baseline_with_reform_session_nums_2_more_clicks.pk", "rb"))
    simulated_sessions_dtc += pickle.load(open("../simulated_sessions/final_results/simulated_session_all_topics_dtc_baseline_with_reform_session_nums_3_more_clicks.pk", "rb"))
    simulated_sessions_dtc += pickle.load(open("../simulated_sessions/final_results/simulated_session_all_topics_dtc_baseline_with_reform_session_nums_4_more_clicks.pk", "rb"))
    simulated_sessions_dtc += pickle.load(open("../simulated_sessions/final_results/simulated_session_all_topics_dtc_baseline_with_reform_session_nums_5_more_clicks.pk", "rb"))
    print ("DTC sessions length: ", len(simulated_sessions_dtc))
    #query_similarities_topics = []
    simulated_sessions_QS3plus = pickle.load(open("../simulated_sessions/final_results/simulated_session_all_topics_QS3plus_baseline_more_clicks.pk", "rb"))
    #for session in simulated_sessions_dtc:
    for session in simulated_sessions_dtc:
	    queries_list_sim = [list(filter(lambda l: l.type =="reformulate", session.interactions)) for session in sim_sessions]
	    queries_list_sim = [interaction.query for session in queries_list_sim for interaction in session]
	    queries_list_act = [list(filter(lambda l: l.type =="reformulate", session.interactions)) for session in act_session_topics[topic_num]]
	    queries_list_act = [interaction.query for session in queries_list_act for interaction in session]
	    queries_list_act = [preprocess(" ".join(query), lemmatizing = True).split(" ") for query in queries_list_act]
    
    queries_list_sim = [list(filter(lambda l: l.type =="reformulate", session.interactions)) for session in sim_sessions]
    queries_list_sim = [interaction.query for session in queries_list_sim for interaction in session]
    queries_list_act = [list(filter(lambda l: l.type =="reformulate", session.interactions)) for session in act_session_topics[topic_num]]
    queries_list_act = [interaction.query for session in queries_list_act for interaction in session]
    queries_list_act = [preprocess(" ".join(query), lemmatizing = True).split(" ") for query in queries_list_act]
    

def main():
    candidate_queries1 = pickle.load(open("../simulated_sessions_TREC_robust/final_results/candidate_queries_basic_precision.pk", "rb"))
    candidate_queries2 = pickle.load(open("../simulated_sessions_TREC_robust/final_results/candidate_queries_basic_recall.pk", "rb"))
    candidate_queries3 = pickle.load(open("../simulated_sessions_TREC_robust/final_results/candidate_queries_basic_recall_precision.pk", "rb"))
    #candidate_queries4 = pickle.load(open("../simulated_sessions/final_results/candidate_queries_type_2_word_sup_1_var_1_500_session_nums_1.pk", "rb"))
    #candidate_queries4 = pickle.load(open("../simulated_sessions/final_results/candidate_QS3plus_baseline_queries.pk", "rb"))
    #candidate_queries = pickle.load(open("../simulated_sessions_TREC_robust/final_results/candidate_queries_QS3plus_baseline.pk", "rb"))
    #bigram_topic_lm = read_bigram_topic_lm_trec_robust()
    #print ("NUM topics now: ", len(candidate_queries1))
    #just_query_simiarity_session_track(candidate_queries1)
    #just_query_simiarity_session_track(candidate_queries2)
    #print ("NUM topics now: ", len(candidate_queries3))
    #just_query_simiarity_session_track(candidate_queries3)
    '''
    print ("NUM topics now: ", len(candidate_queries1))
    print ("precision")
    just_query_simiarity_trec_robust(candidate_queries1)
    print ("recall")
    just_query_simiarity_trec_robust(candidate_queries2)
    print ("recall + precision")
    just_query_simiarity_trec_robust(candidate_queries3)
    '''
    #just_query_simiarity_session_track(candidate_queries1)
    bigram_topic_lm = read_bigram_topic_lm_trec_robust()
    candidate_queries = pickle.load(open("../simulated_sessions_TREC_robust/criteria_query_scoring_variations/candidate_queries_word_sup_1_var_1_500_session_nums_all.pk", "rb"))
    candidate_queries = candidate_query_scoring(candidate_queries, 0.8, bigram_topic_lm)
    #print (just_query_simiarity_session_track(candidate_queries, "../supervised_models/train_test_splits/testing_session_nums_1.pk"))
    just_query_simiarity_trec_robust(candidate_queries)

    '''
    candidate_queries_new = {}
    for topic_num in candidate_queries:
        candidate_queries_new[topic_num] = []
        for (query,doc_list, score) in candidate_queries[topic_num]:
            if len(doc_list)>=10:
                candidate_queries_new[topic_num] += [(query,doc_list, score)]
    #candidate_queries = candidate_queries_new
    '''
    #alphas = [0.0,0.25,0.5,0.75,0.8,0.85,0.9,0.95,1.0]
    '''
    print ("DTC VARIATION")
    bigram_topic_lm = read_bigram_topic_lm()
    sim = []
    candidate_queries = pickle.load(open("../simulated_sessions/final_results/candidate_dtc_baseline_queries_session_nums_1.pk", "rb"))
    sim += [just_query_simiarity_session_track(candidate_queries,"../supervised_models/train_test_splits/testing_session_nums_1.pk")]
    candidate_queries = pickle.load(open("../simulated_sessions/final_results/candidate_dtc_baseline_queries_session_nums_2.pk", "rb"))
    sim += [just_query_simiarity_session_track(candidate_queries,"../supervised_models/train_test_splits/testing_session_nums_2.pk")]
    candidate_queries = pickle.load(open("../simulated_sessions/final_results/candidate_dtc_baseline_queries_session_nums_3.pk", "rb"))
    sim += [just_query_simiarity_session_track(candidate_queries,"../supervised_models/train_test_splits/testing_session_nums_3.pk")]
    candidate_queries = pickle.load(open("../simulated_sessions/final_results/candidate_dtc_baseline_queries_session_nums_4.pk", "rb"))
    sim += [just_query_simiarity_session_track(candidate_queries,"../supervised_models/train_test_splits/testing_session_nums_4.pk")]
    candidate_queries = pickle.load(open("../simulated_sessions/final_results/candidate_dtc_baseline_queries_session_nums_5.pk", "rb"))
    sim += [just_query_simiarity_session_track(candidate_queries,"../supervised_models/train_test_splits/testing_session_nums_5.pk")]
    print ("average similarity: ", float(sum(sim))/float(len(sim)))
    print ("SUPERVISED METHOD + QUERY SCORING")
    candidate_queries = pickle.load(open("../simulated_sessions/final_results/candidate_queries_word_sup_1_var_1_500_session_nums_1.pk", "rb"))
    candidate_queries = candidate_query_scoring(candidate_queries, 0.8,bigram_topic_lm)
    sim += [just_query_simiarity_session_track(candidate_queries, "../supervised_models/train_test_splits/testing_session_nums_1.pk")]
    candidate_queries = pickle.load(open("../simulated_sessions/final_results/candidate_queries_word_sup_1_var_1_500_session_nums_2.pk", "rb"))
    candidate_queries = candidate_query_scoring(candidate_queries, 0.8,bigram_topic_lm)
    sim += [just_query_simiarity_session_track(candidate_queries, "../supervised_models/train_test_splits/testing_session_nums_2.pk")]
    candidate_queries = pickle.load(open("../simulated_sessions/final_results/candidate_queries_word_sup_1_var_1_500_session_nums_3.pk", "rb"))
    candidate_queries = candidate_query_scoring(candidate_queries, 0.8,bigram_topic_lm)
    sim += [just_query_simiarity_session_track(candidate_queries, "../supervised_models/train_test_splits/testing_session_nums_3.pk")]
    candidate_queries = pickle.load(open("../simulated_sessions/final_results/candidate_queries_word_sup_1_var_1_500_session_nums_4.pk", "rb"))
    candidate_queries = candidate_query_scoring(candidate_queries, 0.8,bigram_topic_lm)
    sim += [just_query_simiarity_session_track(candidate_queries, "../supervised_models/train_test_splits/testing_session_nums_4.pk")]
    candidate_queries = pickle.load(open("../simulated_sessions/final_results/candidate_queries_word_sup_1_var_1_500_session_nums_5.pk", "rb"))
    candidate_queries = candidate_query_scoring(candidate_queries, 0.8,bigram_topic_lm)
    sim += [just_query_simiarity_session_track(candidate_queries, "../supervised_models/train_test_splits/testing_session_nums_5.pk")]
    print ("average similarity: ", float(sum(sim))/float(len(sim)))
    print ("ONLY SUPERVISED METHOD")
    candidate_queries = pickle.load(open("../simulated_sessions/final_results/candidate_queries_word_sup_1_var_1_500_session_nums_1.pk", "rb"))
    #candidate_queries = candidate_query_scoring(candidate_queries, 0.8)
    sim += [just_query_simiarity_session_track(candidate_queries, "../supervised_models/train_test_splits/testing_session_nums_1.pk")]
    candidate_queries = pickle.load(open("../simulated_sessions/final_results/candidate_queries_word_sup_1_var_1_500_session_nums_2.pk", "rb"))
    #candidate_queries = candidate_query_scoring(candidate_queries, 0.8)
    sim += [just_query_simiarity_session_track(candidate_queries, "../supervised_models/train_test_splits/testing_session_nums_2.pk")]
    candidate_queries = pickle.load(open("../simulated_sessions/final_results/candidate_queries_word_sup_1_var_1_500_session_nums_3.pk", "rb"))
    #candidate_queries = candidate_query_scoring(candidate_queries, 0.8)
    sim += [just_query_simiarity_session_track(candidate_queries, "../supervised_models/train_test_splits/testing_session_nums_3.pk")]
    candidate_queries = pickle.load(open("../simulated_sessions/final_results/candidate_queries_word_sup_1_var_1_500_session_nums_4.pk", "rb"))
    #candidate_queries = candidate_query_scoring(candidate_queries, 0.8)
    sim += [just_query_simiarity_session_track(candidate_queries, "../supervised_models/train_test_splits/testing_session_nums_4.pk")]
    candidate_queries = pickle.load(open("../simulated_sessions/final_results/candidate_queries_word_sup_1_var_1_500_session_nums_5.pk", "rb"))
    #candidate_queries = candidate_query_scoring(candidate_queries, 0.8)
    sim += [just_query_simiarity_session_track(candidate_queries, "../supervised_models/train_test_splits/testing_session_nums_5.pk")]
    print ("average similarity: ", float(sum(sim))/float(len(sim)))
    '''
    '''
    simulated_sessions = pickle.load(open("../simulated_sessions_TREC_robust/final_results/simulated_session_all_topics_QS3plus_baseline_more_clicks.pk", "rb"))
    topic_descs_query = read_trec_robust_queries()
    topic_num_nums = [int(s) for s in topic_descs_query.keys()]
    topic_num_nums.sort()
    query_similarities_2 = []
    sim_sessions_topics = {}
    for session in simulated_sessions:
        try:
            sim_sessions_topics[session.topic_num] += [session]
        except:
            sim_sessions_topics[session.topic_num] = [session]
    avg_session_length_sim = []
    for topic_num in topic_num_nums[:100]:
        topic_num = str(topic_num)
        queries_list_sim = [list(filter(lambda l: l.type =="reformulate", session.interactions)) for session in sim_sessions_topics[topic_num]]
        queries_list_sim = [interaction.query for session in queries_list_sim for interaction in session]
        (simulated_to_act_sim,act_to_simulated_sim) = query_similarity_evaluation(queries_list_sim, [preprocess(topic_descs_query[topic_num], lemmatizing = True).split(" ")])
        query_similarities_2 += [simulated_to_act_sim]
        avg_session_length_sim += [len(session.interactions) for session in sim_sessions_topics[topic_num]]
    avg_session_length_sim = float(sum(avg_session_length_sim))/float(len(avg_session_length_sim))
    print ('Simulated sessions Avg Session length: ', avg_session_length_sim)
    print ('query_similarities_2: ', float(sum(query_similarities_2))/float(len(query_similarities_2)))
    '''

    #simulated_sessions1 = pickle.load(open("../simulated_sessions_TREC_robust/trec_robust_simulated_session_word_sup_1_query_unsup_1_1000_reform_session_nums_all_10.pk", "rb"))
    #simulated_sessions2 = pickle.load(open("../simulated_sessions_TREC_robust/trec_robust_simulated_session_word_sup_1_query_unsup_1_1000_reform_session_nums_all_10_30.pk", "rb"))
    #simulated_sessions3 = pickle.load(open("../simulated_sessions_TREC_robust/trec_robust_simulated_session_word_sup_1_query_unsup_1_1000_reform_session_nums_all_30_50.pk", "rb"))
    #simulated_sessions = simulated_sessions1+simulated_sessions2+simulated_sessions3
    '''
    simulated_sessions = pickle.load(open("../simulated_sessions/final_results/simulated_session_all_topics_basic_recall_precision_query_unsup_0.8_reform_1.0_0.9_0.9_more_clicks.pk", "rb"))
    print ("sessions with reform sessions length: ", len(simulated_sessions))
    simulated_sessions = prune_simulated_sessions(simulated_sessions)
    print ('REFORM')
    avg_session_length_sim = [len(session.interactions) for session in simulated_sessions]
    avg_session_length_sim = float(sum(avg_session_length_sim))/float(len(avg_session_length_sim))
    print ('Simulated sessions Avg Session length: ', avg_session_length_sim)

    simulated_sessions_no_reform = pickle.load(open("../simulated_sessions/final_results/simulated_session_all_topics_basic_recall_precision_query_unsup_0.8_reform_no_reform_more_clicks.pk", "rb"))
    print ("sessions with reform sessions length: ", len(simulated_sessions_no_reform))
    simulated_sessions_no_reform = prune_simulated_sessions(simulated_sessions_no_reform)
    print ('NO REFORM')
    avg_session_length_sim = [len(session.interactions) for session in simulated_sessions_no_reform]
    avg_session_length_sim = float(sum(avg_session_length_sim))/float(len(avg_session_length_sim))
    print ('Simulated sessions Avg Session length: ', avg_session_length_sim)
    '''
    '''
    simulated_sessions_new_cadt = []
    simulated_sessions_new_cadt += pickle.load(open("../simulated_sessions/final_results/simulated_session_all_topics_word_sup_1_1_500_query_unsup_0.8_reform_big_1.0_0.7_0.7_session_nums_1_more_clicks_0_10.pk", "rb"))
    simulated_sessions_new_cadt += pickle.load(open("../simulated_sessions/final_results/simulated_session_all_topics_word_sup_1_1_500_query_unsup_0.8_reform_big_1.0_0.7_0.7_session_nums_1_more_clicks_10_20.pk", "rb"))
    simulated_sessions_new_cadt += pickle.load(open("../simulated_sessions/final_results/simulated_session_all_topics_word_sup_1_1_500_query_unsup_0.8_reform_big_1.0_0.7_0.7_session_nums_1_more_clicks_20_30.pk", "rb"))
    simulated_sessions_new_cadt += pickle.load(open("../simulated_sessions/final_results/simulated_session_all_topics_word_sup_1_1_500_query_unsup_0.8_reform_big_1.0_0.7_0.7_session_nums_1_more_clicks_30_40.pk", "rb"))
    simulated_sessions_new_cadt += pickle.load(open("../simulated_sessions/final_results/simulated_session_all_topics_word_sup_1_1_500_query_unsup_0.8_reform_big_1.0_0.7_0.7_session_nums_1_more_clicks_40_end.pk", "rb"))
    print ("sessions with reform with new candidates sessions length: ", len(simulated_sessions_new_cadt))
    simulated_sessions_new_cadt = prune_simulated_sessions(simulated_sessions_new_cadt)
    simulated_sessions = []
    simulated_sessions += pickle.load(open("../simulated_sessions/final_results/simulated_session_all_topics_word_sup_1_1_500_query_unsup_0.8_reform_1.0_0.9_0.9_session_nums_1_more_clicks.pk", "rb"))
    simulated_sessions += pickle.load(open("../simulated_sessions/final_results/simulated_session_all_topics_word_sup_1_1_500_query_unsup_0.8_reform_1.0_0.9_0.9_session_nums_2_more_clicks.pk", "rb"))
    simulated_sessions += pickle.load(open("../simulated_sessions/final_results/simulated_session_all_topics_word_sup_1_1_500_query_unsup_0.8_reform_1.0_0.9_0.9_session_nums_3_more_clicks.pk", "rb"))
    simulated_sessions += pickle.load(open("../simulated_sessions/final_results/simulated_session_all_topics_word_sup_1_1_500_query_unsup_0.8_reform_1.0_0.9_0.9_session_nums_4_more_clicks.pk", "rb"))
    simulated_sessions += pickle.load(open("../simulated_sessions/final_results/simulated_session_all_topics_word_sup_1_1_500_query_unsup_0.8_reform_1.0_0.9_0.9_session_nums_5_more_clicks.pk", "rb"))
    print ("sessions with reform sessions length: ", len(simulated_sessions))
    simulated_sessions = prune_simulated_sessions(simulated_sessions)
    simulated_sessions_no_reform = []
    simulated_sessions_no_reform += pickle.load(open("../simulated_sessions/final_results/simulated_session_all_topics_word_sup_1_1_500_query_unsup_0.8_reform_no_reform_session_nums_1_more_clicks.pk", "rb"))
    simulated_sessions_no_reform += pickle.load(open("../simulated_sessions/final_results/simulated_session_all_topics_word_sup_1_1_500_query_unsup_0.8_reform_no_reform_session_nums_2_more_clicks.pk", "rb"))
    simulated_sessions_no_reform += pickle.load(open("../simulated_sessions/final_results/simulated_session_all_topics_word_sup_1_1_500_query_unsup_0.8_reform_no_reform_session_nums_3_more_clicks.pk", "rb"))
    simulated_sessions_no_reform += pickle.load(open("../simulated_sessions/final_results/simulated_session_all_topics_word_sup_1_1_500_query_unsup_0.8_reform_no_reform_session_nums_4_more_clicks.pk", "rb"))
    simulated_sessions_no_reform += pickle.load(open("../simulated_sessions/final_results/simulated_session_all_topics_word_sup_1_1_500_query_unsup_0.8_reform_no_reform_session_nums_5_more_clicks.pk", "rb"))
    print ("session no reform sessions length: ", len(simulated_sessions_no_reform))
    simulated_sessions_no_reform = prune_simulated_sessions(simulated_sessions_no_reform)
    '''
    '''
    simulated_sessions_dtc = []
    simulated_sessions_dtc += pickle.load(open("../simulated_sessions/final_results/simulated_session_all_topics_dtc_baseline_with_reform_session_nums_1_more_clicks.pk", "rb"))
    simulated_sessions_dtc += pickle.load(open("../simulated_sessions/final_results/simulated_session_all_topics_dtc_baseline_with_reform_session_nums_2_more_clicks.pk", "rb"))
    simulated_sessions_dtc += pickle.load(open("../simulated_sessions/final_results/simulated_session_all_topics_dtc_baseline_with_reform_session_nums_3_more_clicks.pk", "rb"))
    simulated_sessions_dtc += pickle.load(open("../simulated_sessions/final_results/simulated_session_all_topics_dtc_baseline_with_reform_session_nums_4_more_clicks.pk", "rb"))
    simulated_sessions_dtc += pickle.load(open("../simulated_sessions/final_results/simulated_session_all_topics_dtc_baseline_with_reform_session_nums_5_more_clicks.pk", "rb"))
    print ("DTC sessions length: ", len(simulated_sessions_dtc))
    simulated_sessions_dtc = prune_simulated_sessions(simulated_sessions_dtc)
    simulated_sessions_QS3plus = pickle.load(open("../simulated_sessions/final_results/simulated_session_all_topics_QS3plus_baseline_more_clicks.pk", "rb"))
    simulated_sessions_QS3plus = prune_simulated_sessions(simulated_sessions_QS3plus)
    #simulated_sessions_new_cadt
    simulated_sessions_list = [simulated_sessions, simulated_sessions_no_reform, simulated_sessions_dtc, simulated_sessions_QS3plus]

    print ("Num sessions", len(simulated_sessions))
    target_documents_details_trec_robust = pickle.load(open("../TREC_Robust_data/trec_robust_topic_rel_doc_details.pk","rb"))
    target_document_vectors_trec_robust = pickle.load(open("../TREC_Robust_data/trec_robust_target_doc_topic_vectors.pk", 'rb'))
    target_document_vectors = pickle.load(open("../TREC_Robust_data/target_doc_topic_vectors.pk", 'rb'))
    target_documents_details = pickle.load(open("../TREC_Robust_data/topic_rel_doc_details.pk","rb"))
    '''
    '''
    topic_descs_query = read_trec_robust_queries()
    query_similarities = []
    for session in simulated_sessions:
        queries_list_sim = [inte.query for inte in session.interactions]
        (simulated_to_act_sim,act_to_simulated_sim) = query_similarity_evaluation(queries_list_sim, [preprocess(topic_descs_query[session.topic_num], lemmatizing = True).split(" ")])
        query_similarities += [float(simulated_to_act_sim+act_to_simulated_sim)/float(2)]
        #print ("Query similarity: ", simulated_to_act_sim, act_to_simulated_sim)
    print ("QUERY Similarities: ", float(sum(query_similarities))/float(len(query_similarities))) 

    print("AVG SIM evaluation ...")
    all_sessions_trends_avg_sim = evaluate_sim_sessions_avg_sim(simulated_sessions, target_documents_details_trec_robust, target_document_vectors_trec_robust, 1)
    pickle.dump(all_sessions_trends_avg_sim, open('../reformulation_trends_plots/trec_robust_sim_50_session_trends_avg_sim.pk', 'wb'))
    #all_sessions_trends_avg_sim = pickle.load(open('../reformulation_trends_plots/trec_robust_sim_30_session_trends_avg_sim_more_clicks.pk', 'rb'))
    topic_rel_docs = read_trec_robust_judgements()
    print("NDCG evaluation ...")
    all_sessions_trends = evaluate_sim_sessions_NDCG(simulated_sessions, topic_rel_docs)
    avg_session_trends,length_wise_avg_trends = get_some_avg_session_trends(all_sessions_trends)
    avg_session_trends_2,length_wise_avg_trends_2 = get_some_avg_session_trends(all_sessions_trends_avg_sim)
    '''
    '''
    avg_session_trends_list = []
    length_wise_avg_trends_list = []
    avg_session_trends_2_list = []
    length_wise_avg_trends_2_list = []
    # 'sim_sessions_basic_recall_precision_1_500_query_unsup_0.8_reform_1.0_0.9_0.9_more_clicks_trends_avg_sim', 'sim_sessions_basic_recall_precision_1_500_query_unsup_0.8_reform_no_reform_more_clicks_trends_avg_sim',
    #'sim_sessions_word_sup_1_1_500_query_unsup_0.8_reform_1.0_0.9_0.9_more_clicks_trends_avg_sim', 'sim_sessions_word_sup_1_1_500_query_unsup_0.8_reform_no_reform_more_clicks_trends_avg_sim',
    #names = ['sim_sessions_word_sup_1_1_500_query_unsup_0.8_reform_big_1.0_0.7_0.7_more_clicks_trends_avg_sim', 'sim_sessions_word_sup_1_1_500_query_unsup_0.8_reform_1.0_0.9_0.9_more_clicks_trends_avg_sim', 'sim_sessions_word_sup_1_1_500_query_unsup_0.8_reform_no_reform_more_clicks_trends_avg_sim', 'sim_sessions_dtc_baseline_with_reform_session_nums_1_more_clicks_trends_avg_sim', 'sim_sessions_QS3plus_baseline_more_clicks_trends_avg_sim' ]
    names = [ 'sim_sessions_basic_recall_precision_1_500_query_unsup_0.8_reform_1.0_0.9_0.9_more_clicks_trends_avg_sim', 'sim_sessions_basic_recall_precision_1_500_query_unsup_0.8_reform_no_reform_more_clicks_trends_avg_sim', 'sim_sessions_dtc_baseline_with_reform_session_nums_1_more_clicks_trends_avg_sim', 'sim_sessions_QS3plus_baseline_more_clicks_trends_avg_sim' ]
    all_sessions_trends_list = []
    for idx,simulated_sessions in enumerate(simulated_sessions_list):
        all_sessions_trends_avg_sim = evaluate_sim_sessions_avg_sim(simulated_sessions, target_documents_details, target_document_vectors, 3)
        pickle.dump(all_sessions_trends_avg_sim, open('../reformulation_trends_plots/' +names[idx] + '.pk', 'wb'))
        #all_sessions_trends_avg_sim = pickle.load(open('../reformulation_trends_plots/' +names[idx] + '.pk', 'rb'))
        topic_rel_docs = read_judgements()
        print("NDCG evaluation ...")
        all_sessions_trends = evaluate_sim_sessions_NDCG(simulated_sessions, topic_rel_docs)
        avg_session_trends,length_wise_avg_trends = get_some_avg_session_trends(all_sessions_trends)
        avg_session_trends_2,length_wise_avg_trends_2 = get_some_avg_session_trends(all_sessions_trends_avg_sim)
        avg_session_trends_list += [avg_session_trends]
        length_wise_avg_trends_list += [length_wise_avg_trends]
        avg_session_trends_2_list += [avg_session_trends_2]
        length_wise_avg_trends_2_list += [length_wise_avg_trends_2]
        all_sessions_trends_list += [all_sessions_trends]

    '''
    clueweb_snippet_collection,clueweb_snippet_collection_2 = read_clueweb_snippet_data()
    print ("Num docs: ", len(clueweb_snippet_collection_2))
    i = 0
    for docid in clueweb_snippet_collection_2:
        clueweb_snippet_collection_2[docid] = preprocess(clueweb_snippet_collection_2[docid],"../Session_track_2014/clueweb_snippet/lemur-stopwords.txt" ,lemmatizing = True)
        i += 1
        if (i%100000 == 0):
            print (i)

    topic_rel_docs = read_judgements()
    ideal_user_sessions,precise_user_sessions,recall_user_sessions,all_sessions = select_sessions(topic_rel_docs,read_full=True)
    act_sessions = []
    for session in all_sessions:
        try:
            topic_rel_docs[session.getElementsByTagName("topic")[0].getAttribute("num")]
        except:
            continue
        if int(session.getElementsByTagName("topic")[0].getAttribute("num"))>33:
            #if session.getAttribute("num") in session_nums_dict:
            act_session = Session(session, clueweb_snippet_collection_2)
            act_sessions += [act_session]
    act_sessions_full = act_sessions
    #act_sessions = filter_real_sessions(act_sessions)
    print ('AVG SIM evaluation ...')
    print ('Num sessions: ', len(act_sessions))
    all_sessions_trends_avg_sim_act = evaluate_sim_sessions_avg_sim(act_sessions, target_documents_details, target_document_vectors, 3)
    pickle.dump(all_sessions_trends_avg_sim_act, open('../reformulation_trends_plots/track_2013_real_sessions_session_trends_avg_sim.pk', 'wb'))
    #all_sessions_trends_avg_sim_act = pickle.load(open('../reformulation_trends_plots/real_sessions_873_session_trends_avg_sim.pk', 'rb'))

    topic_rel_docs = read_judgements()
    print("NDCG evaluation ...")
    all_sessions_trends_act = evaluate_sim_sessions_NDCG(act_sessions, topic_rel_docs)
    avg_session_trends_act,length_wise_avg_trends_act = get_some_avg_session_trends(all_sessions_trends_act)
    avg_session_trends_act_2,length_wise_avg_trends_act_2 = get_some_avg_session_trends(all_sessions_trends_avg_sim_act)

    print ('ACTUAL SESSIONS: ')
    get_first_non_zero_ndcgs(act_sessions, topic_rel_docs,1)
    get_first_non_zero_ndcgs(act_sessions, topic_rel_docs,0)

    print ("ACTUAL: ")
    print ('ndcg')
    print ("AVERAGE NUM INCREMENTS: ", avg_session_trends_act[0])
    print ("AVERAGE NUM DECREMENTS: ", avg_session_trends_act[1])
    print ("AVERAGE NUM EQUALS: ", avg_session_trends_act[2])
    print ("AVG Probability of inc,dec,eq: ", avg_session_trends_act[5])
    print ("total count Probability of inc,dec,eq: ", avg_session_trends_act[4])
    print ('avg sim')
    print ("AVERAGE NUM INCREMENTS: ", avg_session_trends_act_2[0])
    print ("AVERAGE NUM DECREMENTS: ", avg_session_trends_act_2[1])
    print ("AVERAGE NUM EQUALS: ", avg_session_trends_act_2[2])
    print ("AVG Probability of inc,dec,eq: ", avg_session_trends_act_2[5])
    print ("total count Probability of inc,dec,eq: ", avg_session_trends_act_2[4])

    avg_click_log_indeces = {}
    for session in sessions:
        click_log_indeces = []
        for idx,interaction in enumerate(session.interactions):
            if(len(interaction.clicks)) > 0:
                click_log_indeces += [idx+1]
        for idx in range(len(click_log_indeces)):
            try:
                avg_click_log_indeces[idx] += [click_log_indeces[idx]]
            except KeyError:
                avg_click_log_indeces[idx] = [click_log_indeces[idx]]
    click_times = max([len(avg_click_log_indeces.keys()), 0])
    click_times = range(click_times)
    for idx in click_times:
        try:
            distribution_of_first_clicks = Counter(avg_click_log_indeces[idx])
            distribution_of_first_clicks = {x:float(distribution_of_first_clicks[x])/float(len(avg_click_log_indeces[idx])) for x in distribution_of_first_clicks}
            distribution_of_first_clicks = sorted(distribution_of_first_clicks.items(), key=lambda l: l[0])
            #plt.plot([s[0] for s in distribution_of_first_clicks], [s[1] for s in distribution_of_first_clicks], 'r',label='real_sessions', linewidth = 1.0)
            click_mean = round(mean(avg_click_log_indeces[idx]),3)
            print ("CLICK MEAN real sessions: ", idx, click_mean)
            print ('idx, distribution: ', idx, distribution_of_first_clicks)
        except KeyError:
            pass


    '''
    names_short = ["RPC REFORM NEW CANDIDATES", "RPC REFORM", "RPC NO REFORM", "DTC", "QS3plus"]
    names_short = names_short[1:]
    for name_idx,simulated_sessions in enumerate(simulated_sessions_list):
        print (names_short[name_idx])
        get_first_non_zero_ndcgs(simulated_sessions, topic_rel_docs, 1) 
        get_first_non_zero_ndcgs(simulated_sessions, topic_rel_docs, 0)

    evaluate_click_log_trends(act_sessions[:873], simulated_sessions_list)

    plot_avg_session_ndcgs(avg_session_trends_act, avg_session_trends_list, length_wise_avg_trends_act, length_wise_avg_trends_list, 'NDCG')
    plot_avg_session_ndcgs(avg_session_trends_act_2, avg_session_trends_2_list, length_wise_avg_trends_act_2, length_wise_avg_trends_2_list, 'topic_similarity')
    #plot session trends


    names_short = ["RPC REFORM NEW CANDIDATES", "RPC REFORM", "RPC NO REFORM", "DTC", "QS3plus"]
    names_short = names_short[1:]
    for idx,avg_session_trends in enumerate(avg_session_trends_list):
        print ('NAME: ', names_short[idx])
        print ("SIM: " )
        print ('ndcg')
        print ("AVERAGE NUM INCREMENTS: ", avg_session_trends[0])
        print ("AVERAGE NUM DECREMENTS: ", avg_session_trends[1])
        print ("AVERAGE NUM EQUALS: ", avg_session_trends[2])
        print ("AVG Probability of inc,dec,eq: ", avg_session_trends[5])
        print ("total count Probability of inc,dec,eq: ", avg_session_trends[4])
        print ('avg sim')
        print ("AVERAGE NUM INCREMENTS: ", avg_session_trends_2_list[idx][0])
        print ("AVERAGE NUM DECREMENTS: ", avg_session_trends_2_list[idx][1])
        print ("AVERAGE NUM EQUALS: ", avg_session_trends_2_list[idx][2])
        print ("AVG Probability of inc,dec,eq: ", avg_session_trends_2_list[idx][5])
        print ("total count Probability of inc,dec,eq: ", avg_session_trends_2_list[idx][4])

    print ('REAL sessions')
    avg_session_length_sim = [len(session.interactions) for session in act_sessions]
    avg_session_length_sim = float(sum(avg_session_length_sim))/float(len(avg_session_length_sim))
    print ('Simulated sessions Avg Session length: ', avg_session_length_sim)

    names_short = ["RPC REFORM NEW CANDIDATES", "RPC REFORM", "RPC NO REFORM", "DTC", "QS3plus"]
    names_short = names_short[1:]
    for idx,simulated_sessions in enumerate(simulated_sessions_list):
        print (names_short[idx])
        avg_session_length_sim = [len(session.interactions) for session in simulated_sessions]
        avg_session_length_sim = float(sum(avg_session_length_sim))/float(len(avg_session_length_sim))
        print ('Simulated sessions Avg Session length: ', avg_session_length_sim)

    #just_session_query_simiarity_session_track(simulated_sessions_new_cadt, act_sessions_full, "../supervised_models/train_test_splits/testing_session_nums_1.pk")

    '''

if __name__ == "__main__":
    quick_evaluation()