#select dataset for the specific task and then evaluate
from utils import *
from User_model_new import *
import argparse
#from indri_rankers import *
from evaluation_methods import Evaluation_queries,Session_operations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import pyndri
#stopwords_file = "../Session_track_2014/clueweb_snippet/lemur-stopwords.txt"
stopwords_file = "../lemur-stopwords.txt"
old_stdout_target = sys.stdout
def format_results(results, index):
    formatted_results = []
    for result in results:
        formatted_result={} 
        formatted_result["docid"] = result[0]
        formatted_result["title"] = ""
        formatted_result["snippet"] = result[2]
        formatted_result["full_text_lm"] = ""
        formatted_results += [formatted_result]
    return formatted_results

def find_effort_constraint_plots(dataset_name):
    print ('EFFORT PLOTS...')
    dataset = dataset_name
    session_dir = "../simulated_sessions/" 
    if not os.path.exists(session_dir): os.makedirs(session_dir)
    data_dir = os.path.join(session_dir, dataset)
    if not os.path.exists(data_dir): os.makedirs(data_dir)
    
    if dataset == "Session_track_2014" or dataset == "Session_track_2013" or dataset == "Session_track_2012":
        (doc_collection_lm, doc_collection_lm_binary) = pickle.load(open("../TREC_Robust_data/all_doc_language_model.pk", "rb"))
        total_noof_words = sum(doc_collection_lm.values())
        doc_collection_lm_dist = {term:float(doc_collection_lm[term])/float(total_noof_words) for term in doc_collection_lm}
    '''
    else:
    index = pyndri.Index('../../ClueWeb09_catB_sample_index')
    token2id, id2token, id2df = index.get_dictionary()
    id2tf = index.get_term_frequencies()
    doc_collection_lm_dist = {}
    for tokenid in id2tf:
        doc_collection_lm_dist[id2token[tokenid]] = id2tf[tokenid]
    total_noof_words = float(sum(doc_collection_lm_dist.values()))
    doc_collection_lm_dist = {token: float(doc_collection_lm_dist[token])/float(total_noof_words) for token in doc_collection_lm_dist}
    '''
    session_ops = Session_operations()
    act_sessions = session_ops.get_actual_sessions(dataset)
    validation_session_queries = session_ops.get_validation_set(act_sessions)
    save_dir = os.path.join(data_dir, 'constraint_variations')
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    topic_descs = read_topic_descs(dataset)
    topic_rel_docs = read_judgements(dataset)
    Reformulation_comp =  Reformulation_component_multinomial(collection_lm = doc_collection_lm_dist)
    result_judgement_component = Result_judgement_qrels(collection_lm = doc_collection_lm_dist, document_rels_all_topics = topic_rel_docs)    
    '''
    effort_list = [1,2,3,4,5,6,7,8,9,10]
    min_threshold = math.log(0.000000001)
    a1,a2,a3 = 1.0,0.81,0.09
    #a3_list = [float(i)*0.5 for i in [0.5,1]]
    parameters_list = itertools.product(effort_list)
    bigram_scoring_parameter = 2
    data_points = []
    best_perf = 0
    for effort in effort_list: 
        method = 'CCQF_' + str(a1) + '_' + str(a2) + '_' + str(a3) + '_' + str(bigram_scoring_parameter) + '_' + str(effort)
        sys.stdout = open(os.path.join(save_dir, method + '_out.txt'), 'w') #CCQF_0.6_0.2_0.6_reform_0.9_0.9
        do_reformulation = True
        user_model_parameters = []
        user_model_parameters += [do_reformulation]
        Formulation_comp = Query_formulation_CCQF(parameters = (a1,a2,a3,bigram_scoring_parameter,effort,min_threshold), collection_lm = doc_collection_lm_dist, print_info = True)
        #Formulation_comp = Query_formulation_QS3plus(collection_lm = doc_collection_lm_dist, total_num_words = total_noof_words)
        candidate_queries_topics = {}
        for topic_num in topic_descs:
            print ("TOPIC NUM ", topic_num)
            user_model = User_model(user_model_parameters, topic_num, {} , dataset, Formulation_comp, Reformulation_comp, result_judgement_component)
            candidate_queries = user_model.candidate_queries[:]
            candidate_queries_topics[topic_num] = candidate_queries
        Formulation_comp.intl_cand_qurs_all_tpcs = None
        print ('PARAMETER SETTING: ', a1,a2,a3, bigram_scoring_parameter, effort, min_threshold)
        ev = Evaluation_queries(act_sessions, None, candidate_queries_topics)
        print ("QF performance top-5:")
        avg_jaccard_similarity_vali, avg_jaccard_similarity_2_vali = ev.query_similarity_evaluation(onlyqueries = True, top_k = 5, validation_set = True, act_session_queries = validation_session_queries)
        avg_jaccard_similarity_test, avg_jaccard_similarity_2_test = ev.query_similarity_evaluation(onlyqueries = True, top_k = 5)        
        data_points += [((effort,min_threshold), avg_jaccard_similarity_2_vali, avg_jaccard_similarity_2_test)]
        if (avg_jaccard_similarity_2_vali > best_perf):
            best_point = ((effort,min_threshold), avg_jaccard_similarity_2_vali, avg_jaccard_similarity_2_test) 
            best_perf = avg_jaccard_similarity_2_vali   
        #for topic_num in candidate_queries_topics:
        #    candidate_queries_topics[topic_num] = candidate_queries_topics[topic_num][:200] 
        #pickle.dump(candidate_queries_topics, open(os.path.join(save_dir, method + "_queries.pk"), "wb"))
        
    #write_performances
    with open(os.path.join(save_dir, 'write_performances.txt'), 'w') as outfile:
        #outfile.write('PARAMETER SETTING: BIGRAM SCORING METHOD: ' + str(bigram_scoring_parameter) + ' CONSTRAINT: noeffort-min-threshold' + ' MIN-THRESHOLD: ' +str(math.log(0.0001)) + '\n')
        outfile.write('PARAMETER SETTING: BIGRAM SCORING METHOD: ' + str(bigram_scoring_parameter) + ' A1: ' + str(a1) + ' A2: ' +str(a2) + ' A3: ' +str(a3) + '\n')
        outfile.write('EF M-TH VALI_perf_top-5 total_perf_top-5\n')
        for point in data_points:
            outfile.write(str(point[0][0]) + ' ' + str(point[0][1]) + ' ' + str(point[1]) + ' ' +str(point[2]) + '\n')
        outfile.write('BEST PARAMETER: ' + str(best_point[0][0]) + ' ' + str(best_point[0][1]) + ' ' + str(best_point[1]) + ' ' + str(best_point[2]) +'\n')
    
    fig_dir = os.path.join(save_dir, 'effort_parameter_figures')
    if not os.path.exists(fig_dir): os.makedirs(fig_dir)
    #plots
    plt.clf()
    plt.cla()
    label = 'CCQF'
    query_length = [x[0][0] for x in data_points] 
    similarity = [x[2] for x in data_points]
    plt.step(query_length, similarity, where='post', label=label)
    plt.xlabel('Query length')
    plt.ylabel('Query similarity')
    plt.ylim([0.0, 1.0])
    #plt.xlim([0.0, 10])
    plt.title('Amount of Effort(AOE) effect on Query formulation')
    plt.legend(loc="best")
    plt.savefig(os.path.join(fig_dir, '.'.join(['CCQF_' + str(a1) + '_' + str(a2) + '_' + str(a3) + '_' + str(bigram_scoring_parameter) + '_effort.png'])), facecolor='white',
                edgecolor='none', bbox_inches="tight")
    '''

    effort = -1
    min_threshold_list = [math.log(0.001*pow(10,i)) for i in range(3)]
    min_threshold_list_2 = [math.log(0.001*pow(10,i)*5) for i in range(3)]
    min_threshold_list_2 += [math.log(0.5), math.log(0.7), math.log(0.8), math.log(0.9), math.log(0.99)]
    min_threshold_list = min_threshold_list + min_threshold_list_2
    min_threshold_list.sort()
    a1,a2,a3 = 1.0,0.81,0.09
    #a3_list = [float(i)*0.5 for i in [0.5,1]]
    parameters_list = itertools.product(min_threshold_list)
    bigram_scoring_parameter = 2
    data_points = []
    best_perf = 0
    avg_query_lengths = []
    for min_threshold in min_threshold_list: 
        method = 'CCQF_' + str(a1) + '_' + str(a2) + '_' + str(a3) + '_' + str(bigram_scoring_parameter) + '_' + str(round(min_threshold,2))
        sys.stdout = open(os.path.join(save_dir, method + '_out.txt'), 'w') #CCQF_0.6_0.2_0.6_reform_0.9_0.9
        do_reformulation = True
        user_model_parameters = []
        user_model_parameters += [do_reformulation]
        Formulation_comp = Query_formulation_CCQF(parameters = (a1,a2,a3,bigram_scoring_parameter,effort,min_threshold), collection_lm = doc_collection_lm_dist, print_info = True)
        #Formulation_comp = Query_formulation_QS3plus(collection_lm = doc_collection_lm_dist, total_num_words = total_noof_words)
        candidate_queries_topics = {}
        for topic_num in topic_descs:
            print ("TOPIC NUM ", topic_num)
            user_model = User_model(user_model_parameters, topic_num, {} , dataset, Formulation_comp, Reformulation_comp, result_judgement_component)
            candidate_queries = user_model.candidate_queries[:]
            candidate_queries_topics[topic_num] = candidate_queries
        Formulation_comp.intl_cand_qurs_all_tpcs = None
        print ('PARAMETER SETTING: ', a1,a2,a3, bigram_scoring_parameter, effort, min_threshold)
        ev = Evaluation_queries(act_sessions, None, candidate_queries_topics)
        print ("QF performance top-5:")
        avg_jaccard_similarity_vali, avg_jaccard_similarity_2_vali = ev.query_similarity_evaluation(onlyqueries = True, top_k = 5, validation_set = True, act_session_queries = validation_session_queries)
        avg_jaccard_similarity_test, avg_jaccard_similarity_2_test = ev.query_similarity_evaluation(onlyqueries = True, top_k = 5)
        data_points += [((effort,min_threshold), avg_jaccard_similarity_2_vali, avg_jaccard_similarity_2_test)]
        if (avg_jaccard_similarity_2_vali > best_perf):
            best_point = ((effort,min_threshold), avg_jaccard_similarity_2_vali, avg_jaccard_similarity_2_test) 
            best_perf = avg_jaccard_similarity_2_vali   
        lengths = []
        for topic_num in candidate_queries_topics:
            lengths += [len(query[0]) for query in candidate_queries_topics[topic_num][:10]]
        length = float(sum(lengths))/float(len(lengths))
        avg_query_lengths += [(min_threshold,length)]
        #for topic_num in candidate_queries_topics:
        #    candidate_queries_topics[topic_num] = candidate_queries_topics[topic_num][:200] 
        #pickle.dump(candidate_queries_topics, open(os.path.join(save_dir, method + "_queries.pk"), "wb"))
        
    print ('AVERAGE QUERY LEGNTHS: ', avg_query_lengths)
    with open(os.path.join(save_dir, 'write_performances_min_threshold.txt'), 'w') as outfile:
        outfile.write('PARAMETER SETTING: BIGRAM SCORING METHOD: ' + str(bigram_scoring_parameter) + ' A1: ' + str(a1) + ' A2: ' +str(a2) + ' A3: ' +str(a3) + '\n')
        outfile.write('EF M-TH VALI_perf_top-5 total_perf_top-5\n')
        for point in data_points:
            outfile.write(str(point[0][0]) + ' ' + str(point[0][1]) + ' ' + str(point[1]) + ' ' + str(point[2]) + '\n')
        outfile.write('BEST PARAMETER: ' + str(best_point[0][0]) + ' ' + str(best_point[0][1]) + ' ' + str(best_point[1]) + ' ' + str(best_point[2]) +'\n')
    
    fig_dir = os.path.join(save_dir, 'effort_parameter_figures')
    if not os.path.exists(fig_dir): os.makedirs(fig_dir)
    #plots
    plt.clf()
    plt.cla()
    label = 'CCQF'
    query_length = [x[0][1] for x in data_points] 
    similarity = [x[2] for x in data_points]
    plt.step(query_length, similarity, where='post', label=label)
    plt.xlabel('Minimum threshold for effort')
    plt.ylabel('Query similarity')
    plt.ylim([0.0, 1.0])
    #plt.xlim([0.0, 10])
    plt.title('Amount of effort(AOE) effect on Query formulation')
    plt.legend(loc="best")
    plt.savefig(os.path.join(fig_dir, '.'.join(['CCQF_' + str(a1) + '_' + str(a2) + '_' + str(a3) + '_' + str(bigram_scoring_parameter) + '_min_th.png'])), facecolor='white',
                edgecolor='none', bbox_inches="tight")

    plt.clf()
    plt.cla()
    label = 'CCQF'
    query_length = [x[0] for x in avg_query_lengths] 
    similarity = [x[1] for x in avg_query_lengths]
    plt.step(query_length, similarity, where='post', label=label)
    plt.xlabel('Minimum threshold for effort')
    plt.ylabel('Average query length')
    #plt.xlim([0.0, 10])
    plt.title('Minimum threshold vs Average query length effect')
    plt.legend(loc="best")
    plt.savefig(os.path.join(fig_dir, '.'.join(['CCQF_' + str(a1) + '_' + str(a2) + '_' + str(a3) + '_' + str(bigram_scoring_parameter) + '_min_th_avg_len.png'])), facecolor='white',
                edgecolor='none', bbox_inches="tight")

    return


#bm25_ranker.make_inverted_index("../Session_track_2014/clueweb_snippet/clueweb_snippet.dat", "../Session_track_2014/clueweb_snippet/lemur-stopwords.txt")
def simulate_sessions():
    '''
    Getting background lm from a sample index, simulate only first queries for all topics given the parameters.
    :return:
    '''
    print ('SIMULATING SESSIONS..')
    dataset = "Session_track_2012"    
    '''
    if dataset == "Session_track_2014" or dataset == "Session_track_2013" or dataset == "Session_track_2012":
        (doc_collection_lm, doc_collection_lm_binary) = pickle.load(open("../TREC_Robust_data/all_doc_language_model.pk", "rb"))
        total_noof_words = sum(doc_collection_lm.values())
        doc_collection_lm_dist = {term:float(doc_collection_lm[term])/float(total_noof_words) for term in doc_collection_lm}
        noof_docs =  527775
        doc_collection_lm_binary_dist = {term:float(doc_collection_lm_binary[term])/float(noof_docs) for term in doc_collection_lm_binary}
        doc_collection_lm_dist = doc_collection_lm_binary_dist
    
    elif dataset == "Session_track_2012":
    '''
    '''
    index = pyndri.Index('../../ClueWeb09_catB_sample_index')
    token2id, id2token, id2df = index.get_dictionary()
    id2tf = index.get_term_frequencies()
    doc_collection_lm_dist = {}
    for tokenid in id2tf:
        doc_collection_lm_dist[id2token[tokenid]] = id2tf[tokenid]
    total_noof_words = float(sum(doc_collection_lm_dist.values()))
    doc_collection_lm_dist = {token: float(doc_collection_lm_dist[token])/float(total_noof_words) for token in doc_collection_lm_dist}
    doc_collection_lm_binary_dist = {}
    for tokenid in id2df:
        doc_collection_lm_binary_dist[id2token[tokenid]] = id2df[tokenid]
    total_noof_docs = float(max(doc_collection_lm_dist.values()))
    '''
    #doc_collection_lm_binary_dist = {token: float(doc_collection_lm_binary_dist[token])/float(total_noof_docs) for token in doc_collection_lm_binary_dist}
    #doc_collection_lm_dist = doc_collection_lm_binary_dist
    '''
    index = pyndri.Index('../../ClueWeb09_catB_sample_index')
    token2id, id2token, id2df = index.get_dictionary()
    id2tf = index.get_term_frequencies()
    '''    
    session_dir = "../simulated_sessions/" 
    if not os.path.exists(session_dir): os.makedirs(session_dir)
    save_dir = os.path.join(session_dir, dataset)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    do_reformulation = True
    user_model_parameters = []
    user_model_parameters += [do_reformulation]
    bigram_scoring_parameter = 1
    #a1,a2,a3 = 0.125,0.645,0.23
    a1,a2,a3 = 0.5,0.5,0.5
    method = 'CCQF_' + str(a1) + '_' + str(a2) + '_' + str(a3) + '_' + 'new_ins'
    print ('METHOD: ', method)
    sys.stdout = open(os.path.join(save_dir, method + '_out.txt'), 'w') #CCQF_0.6_0.2_0.6_reform_0.9_0.9
    topic_descs = read_topic_descs(dataset)
    topic_rel_docs = read_judgements(dataset)
    Reformulation_comp =  Reformulation_component_multinomial(parameters = (0.9,0.9), collection_lm = doc_collection_lm_dist, print_info = True)
    #Reformulation_comp =  Reformulation_component_binary(parameters = (0.95,0.95), collection_lm = doc_collection_lm_dist, print_info = True)
    #Formulation_comp = Query_formulation_CCQF(parameters = (a1,a2,a3,bigram_scoring_parameter,-1,math.log(0.0000001)), collection_lm = doc_collection_lm_dist)
    Formulation_comp = Query_formulation_CCQF_new(parameters = (a1,a2,a3,-1,math.log(0.0000001)), collection_lm = doc_collection_lm_dist)
    #Formulation_comp = Query_formulation_QS3plus(collection_lm = doc_collection_lm_dist, total_num_words = total_noof_words)
    result_judgement_component = Result_judgement_qrels(collection_lm = doc_collection_lm_dist, document_rels_all_topics = topic_rel_docs)
    topic_num_nums = [int(s) for s in topic_rel_docs.keys()]
    topic_num_nums.sort()
    topic_num_nums = [str(s) for s in topic_num_nums]
    print (topic_num_nums)
    simulated_sessions = []
    candidate_queries_topics = {}
    for topic_num in topic_descs:
        print ("TOPIC NUM ", topic_num)
        user_model = User_model(user_model_parameters, topic_num, {} , dataset, Formulation_comp, Reformulation_comp, result_judgement_component)
        candidate_queries = user_model.candidate_queries[:]
        candidate_queries_topics[topic_num] = candidate_queries
    '''
    for topic_num in topic_num_nums:
        print ("TOPIC NUM ", topic_num)
        sim_session = Session(topic_num = topic_num)
        user_model = User_model(user_model_parameters, topic_num, topic_rel_docs[topic_num], dataset, Formulation_comp, Reformulation_comp, result_judgement_component)
        #candidate_queries = user_model.candidate_queries[:]
        query = user_model.first_interaction()
        results = indri_ranker(query, index)[:20]
        formatted_results = format_results(results, index)
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
                results = indri_ranker(next_query, index)
                formatted_results = format_results(results, index)
                i = 0
            elif (next_action_code == 0):
                query = query
                action_code = next_action_code
                i = i + 10
            session_length += 1
        simulated_sessions += [sim_session]
        #candidate_queries_topics[topic_num] = candidate_queries
    for topic_num in candidate_queries_topics:
        candidate_queries_topics[topic_num] = candidate_queries_topics[topic_num][:400]     
    pickle.dump(simulated_sessions, open(os.path.join(save_dir, method + "_sessions.pk"), "wb"))
    '''
    pickle.dump(candidate_queries_topics, open(os.path.join(save_dir, method + "_queries.pk"), "wb"))
    session_ops = Session_operations()
    act_sessions = session_ops.get_actual_sessions(dataset)
    #save_dir = "../simulated_sessions/Session_track_2014"
    sim_sessions = None
    #sim_sessions =  pickle.load(open(os.path.join(save_dir, method + "_sessions.pk"), "rb"))
    candidate_queries_topics = pickle.load(open(os.path.join(save_dir, method + "_queries.pk"), "rb"))
    ev = Evaluation_queries(act_sessions, sim_sessions, candidate_queries_topics)
    #print ("QR performance:")
    #print (ev.query_similarity_evaluation(onlyqueries = False))
    print ("QF performance top-1:")
    print (ev.query_similarity_evaluation(onlyqueries = True, top_k = 1))
    print ("QF performance top-5:")
    print (ev.query_similarity_evaluation(onlyqueries = True, top_k = 5))
    print ("QF performance top-10:")
    print (ev.query_similarity_evaluation(onlyqueries = True, top_k = 10))

def find_QF_parameters(dataset_name):
    '''
    gets background lm from sample index
    Given a Configuration setting (eg: R3_P2,R12_P1,...), generates initial queries for all topic for all parameters combinations.
    pickle dump initial queries, also computed performance for each parameter value on single validation set (not cross validation)
    uses Query_formulation_CCQF
    :param dataset_name:
    :return:
    '''
    print ('FITTING 3 PARAMETERS...')
    dataset = dataset_name
    session_dir = "../simulated_sessions/" 
    if not os.path.exists(session_dir): os.makedirs(session_dir)
    data_dir = os.path.join(session_dir, dataset)
    if not os.path.exists(data_dir): os.makedirs(data_dir)
    '''
    if dataset == "Session_track_2014" or dataset == "Session_track_2013" or dataset == "Session_track_2012":
        (doc_collection_lm, doc_collection_lm_binary) = pickle.load(open("../TREC_Robust_data/all_doc_language_model.pk", "rb"))
        total_noof_words = sum(doc_collection_lm.values())
        doc_collection_lm_dist = {term:float(doc_collection_lm[term])/float(total_noof_words) for term in doc_collection_lm}
    
    else:
    '''
    '''
    index = pyndri.Index('../../ClueWeb09_catB_sample_index')
    token2id, id2token, id2df = index.get_dictionary()
    id2tf = index.get_term_frequencies()
    doc_collection_lm_dist = {}
    for tokenid in id2tf:
        doc_collection_lm_dist[id2token[tokenid]] = id2tf[tokenid]
    total_noof_words = float(sum(doc_collection_lm_dist.values()))
    doc_collection_lm_dist = {token: float(doc_collection_lm_dist[token])/float(total_noof_words) for token in doc_collection_lm_dist}
    '''
    session_ops = Session_operations()
    act_sessions = session_ops.get_actual_sessions(dataset)
    validation_session_queries = session_ops.get_validation_set(act_sessions)
    x_list = [float(i) for i in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]]  #0.0,0.1,0.2,0.3,0.4,0.5,0.6,
    y_list = [float(i) for i in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]]
    #x_list = [float(i) for i in [0.0,0.025,0.05,0.075,0.1,0.125,0.15]]  #0.0,0.1,0.2,0.3,0.4,0.5,0.6,
    #y_list = [float(i) for i in [0.85,0.875,0.9,0.925,0.95,0.975,1.0]]
    #a3_list = [float(i)*0.5 for i in [0.5,1]]
    parameters_list = itertools.product(x_list,y_list)
    bigram_scoring_parameter = 2
    effort = 6
    w_len = 0
    min_threshold = math.log(0.0000001)
    save_dir = os.path.join(data_dir, 'fit_3_parameters_' + str(bigram_scoring_parameter) + '_noef_new_r12c_100_pts_wo_len')
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    print ("SAVE DIR: ", save_dir)
    data_points = []
    best_perf = 0
    topic_descs = read_topic_descs(dataset)
    topic_rel_docs = read_judgements(dataset)
    Reformulation_comp =  Reformulation_component_multinomial(collection_lm = doc_collection_lm_dist)
    result_judgement_component = Result_judgement_qrels(collection_lm = doc_collection_lm_dist, document_rels_all_topics = topic_rel_docs)    
    (topic_bigram_ct, topic_unigram_ct, topic_bigram_prob) = read_bigram_topic_lm(dataset)
    topic_descs = read_topic_descs(dataset)
    topic_INs = {}
    for iter_idx,(x,y) in enumerate(parameters_list): 
        '''
        a1 = x
        a2 = y*(1.0-x)
        a3 = (1.0-y)*(1.0-x)
        '''
        a1 = x
        a2 = y
        a3 = 1.0-y
        method = 'CCQF_' + str(a1) + '_' + str(a2) + '_' + str(a3) + '_' + str(bigram_scoring_parameter) + '_' + 'noef'
        sys.stdout = old_stdout_target
        print ('METHOD: ', method)
        sys.stdout = open(os.path.join(save_dir, method + '_out.txt'), 'w') #CCQF_0.6_0.2_0.6_reform_0.9_0.9
        do_reformulation = True
        user_model_parameters = []
        user_model_parameters += [do_reformulation]
        if iter_idx == 0:
            Formulation_comp = Query_formulation_CCQF(parameters = (a1,a2,a3,bigram_scoring_parameter,effort,min_threshold,w_len), collection_lm = doc_collection_lm_dist, print_info = True, topic_bi_probs = (topic_bigram_ct, topic_unigram_ct, topic_bigram_prob), topic_descs = topic_descs, recall_combine = True)
        else:
            Formulation_comp = Query_formulation_CCQF(parameters = (a1,a2,a3,bigram_scoring_parameter,effort,min_threshold,w_len), collection_lm = doc_collection_lm_dist, print_info = True, topic_bi_probs = (topic_bigram_ct, topic_unigram_ct, topic_bigram_prob), topic_descs = topic_descs, topic_INs = topic_INs, recall_combine = True)    
        #Formulation_comp = Query_formulation_QS3plus(collection_lm = doc_collection_lm_dist, total_num_words = total_noof_words)
        candidate_queries_topics = {}
        start_time = time.time()
        for topic_num in topic_descs:
            print ("TOPIC NUM ", topic_num)
            user_model = User_model(user_model_parameters, topic_num, {} , dataset, Formulation_comp, Reformulation_comp, result_judgement_component)
            candidate_queries = user_model.candidate_queries[:]
            if (iter_idx == 0):
                topic_INs[topic_num] = user_model.topic_IN
            candidate_queries_topics[topic_num] = candidate_queries
        Formulation_comp.intl_cand_qurs_all_tpcs = None
        print ('PARAMETER SETTING: ', a1,a2,a3)
        print ('TIME TAKEN FOR QF: ' ,  (time.time()-start_time))
        start_time = time.time()
        ev = Evaluation_queries(act_sessions, None, candidate_queries_topics)
        print ("QF performance top-5:")
        #avg_jaccard_similarity_vali, avg_jaccard_similarity_2_vali_0 = ev.query_similarity_evaluation(onlyqueries = True, top_k = 1, validation_set = True, act_session_queries = validation_session_queries)
        #avg_jaccard_similarity_test, avg_jaccard_similarity_2_test_0 = ev.query_similarity_evaluation(onlyqueries = True, top_k = 1)
        avg_jaccard_similarity_vali, avg_jaccard_similarity_2_vali = ev.query_similarity_evaluation(onlyqueries = True, top_k = 5, validation_set = True, act_session_queries = validation_session_queries)
        avg_jaccard_similarity_test, avg_jaccard_similarity_2_test = ev.query_similarity_evaluation(onlyqueries = True, top_k = 5)
        #avg_jaccard_similarity_vali, avg_jaccard_similarity_2_vali_2 = ev.query_similarity_evaluation(onlyqueries = True, top_k = 10, validation_set = True, act_session_queries = validation_session_queries)
        #avg_jaccard_similarity_test, avg_jaccard_similarity_2_test_2 = ev.query_similarity_evaluation(onlyqueries = True, top_k = 10)
        for topic_num in candidate_queries_topics:
            candidate_queries_topics[topic_num] = candidate_queries_topics[topic_num][:200] 
        pickle.dump(candidate_queries_topics, open(os.path.join(save_dir, method + "_queries.pk"), "wb"))
        data_points += [((a1,a2,a3), avg_jaccard_similarity_2_vali, avg_jaccard_similarity_2_test)]
        if (avg_jaccard_similarity_2_vali > best_perf):
            best_point = ((a1,a2,a3), avg_jaccard_similarity_2_vali, avg_jaccard_similarity_2_test) 
            best_perf = avg_jaccard_similarity_2_vali   
        print ('TIME TAKEN FOR EV: ' ,  (time.time()-start_time))

    #fig_dir = os.path.join(data_dir, 'fit_3_parameters_figures')
    #if not os.path.exists(fig_dir): os.makedirs(fig_dir)
    #write_performances
    with open(os.path.join(save_dir, 'write_performances.txt'), 'w') as outfile:
        outfile.write('PARAMETER SETTING: BIGRAM SCORING METHOD: ' + str(bigram_scoring_parameter) + ' CONSTRAINT: ' + str(effort) + ' MIN-THRESHOLD: ' +str(min_threshold) + '\n')
        outfile.write('A1 A2 A3 VALI_perf_top-5 total_perf_top-5\n')
        for point in data_points:
            outfile.write(str(point[0][0]) + ' ' + str(point[0][1]) + ' ' + str(point[0][2]) + ' ' + str(point[1]) + ' ' + str(point[2]) + '\n')
        outfile.write('BEST PARAMETER: ' + str(best_point[0][0]) + ' ' + str(best_point[0][1]) + ' ' + str(best_point[0][2]) + ' ' + str(best_point[1]) + ' ' + str(best_point[2]) +'\n')
    return


def find_QF_parameters_new(dataset_name, setting, parameters = ["alpha", "beta"], effort = 6):
    '''
    THIS IS THE LATEST METHOD
    gets background lm from sample index
    Given a Configuration setting (eg: R3_P2,R12_P1,...), generates initial queries for all topic for all parameters combinations.
    pickle dump initial queries, also computed performance for each parameter value on single validation set (not cross validation)
    uses Query_formulation_CCQF_new
    :param dataset_name:
    :return:
    '''
    print ('FITTING 3 PARAMETERS CCQF NEW...')
    dataset = dataset_name
    session_dir = "../simulated_sessions/" 
    if not os.path.exists(session_dir): os.makedirs(session_dir)
    data_dir = os.path.join(session_dir, dataset)
    if not os.path.exists(data_dir): os.makedirs(data_dir)
    '''
    index = pyndri.Index('../../ClueWeb09_catB_sample_index')
    token2id, id2token, id2df = index.get_dictionary()
    id2tf = index.get_term_frequencies()
    doc_collection_lm_dist = {}
    for tokenid in id2tf:
        doc_collection_lm_dist[id2token[tokenid]] = id2tf[tokenid]
    total_noof_words = float(sum(doc_collection_lm_dist.values()))
    doc_collection_lm_dist = {token: float(doc_collection_lm_dist[token])/float(total_noof_words) for token in doc_collection_lm_dist}
    '''
    doc_collection_lm_dist = {}
    with open("../unigram_freq.csv", "r") as infile:
        i = 0
        for line in infile:
            if i==0:
                i = i + 1
                continue
            doc_collection_lm_dist[line.strip().split(",")[0]] = float(line.strip().split(",")[1])
    total_noof_words = float(sum(doc_collection_lm_dist.values()))
    print(len(doc_collection_lm_dist), doc_collection_lm_dist["the"])
    doc_collection_lm_dist = {token: float(doc_collection_lm_dist[token]) / float(total_noof_words) for token in doc_collection_lm_dist}
    print (len(doc_collection_lm_dist), doc_collection_lm_dist["the"], min(list(doc_collection_lm_dist.values())))

    session_ops = Session_operations()
    act_sessions = session_ops.get_actual_sessions(dataset)
    validation_session_queries = session_ops.get_validation_set(act_sessions)
    if "only_recall" in parameters:
        x_list = [1.0]
    elif "only_precision" in parameters:
        x_list = [0.0]
    elif "alpha" in parameters:
        x_list = [float(i) for i in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]] #alpha #0.0,0.1,0.2,0.3,0.4,0.5,0.6,
    else:
        x_list = [1.0]

    if "beta" in parameters:
        y_list = [float(i) for i in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]] #beta
    else:
        y_list = [1.0]
    #x_list = [float(i) for i in [0.0]]  #0.0,0.1,0.2,0.3,0.4,0.5,0.6,
    #y_list = [float(i) for i in [0.1]]
    #a3_list = [float(i)*0.5 for i in [0.5,1]]
    parameters_list = itertools.product(x_list,y_list)
    #effort = 6
    min_threshold = math.log(0.0000001)
    w_len = 0
    #setting = ["Cr_R12g", "Cp"]
    #setting = "com_r_p3"
    setting_string = "_".join(setting)
    save_dir = os.path.join(data_dir, 'fit_3_parameters_PRE_' + setting_string + '_ef' + str(effort) + '_100_pts_wo_len')
    #save_dir = os.path.join(data_dir, "QS3plus_new")
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    print ("SAVE DIR: ", save_dir)
    data_points = []
    best_perf = 0
    topic_descs = read_topic_descs(dataset)
    topic_rel_docs = read_judgements(dataset)
    Reformulation_comp =  Reformulation_component_multinomial(collection_lm = doc_collection_lm_dist)
    result_judgement_component = Result_judgement_qrels(collection_lm = doc_collection_lm_dist, document_rels_all_topics = topic_rel_docs)    
    (topic_bigram_ct, topic_unigram_ct, topic_bigram_prob) = read_bigram_topic_lm(dataset)
    topic_INs = {}
    for iter_idx,(x,y) in enumerate(parameters_list): 
        alpha = x
        beta = y
        method = 'CCQF_' + str(alpha) + '_' + str(beta) + '_' + 'noef'
        sys.stdout = old_stdout_target
        print ('METHOD: ', method, save_dir)
        print('PARAMETER SETTING: ', alpha, beta)
        sys.stdout = open(os.path.join(save_dir, method + '_out.txt'), 'w') #CCQF_0.6_0.2_0.6_reform_0.9_0.9
        do_reformulation = False
        user_model_parameters = []
        user_model_parameters += [do_reformulation]
        if iter_idx == 0:
            Formulation_comp = Query_formulation_PRE(parameters = (alpha, beta, effort, min_threshold, w_len), collection_lm = doc_collection_lm_dist, topic_bi_probs = (topic_bigram_ct, topic_unigram_ct, topic_bigram_prob), topic_descs = topic_descs, print_info = False, setting = setting)
        else:
            Formulation_comp = Query_formulation_PRE(parameters = (alpha, beta, effort, min_threshold, w_len), collection_lm = doc_collection_lm_dist, topic_bi_probs = (topic_bigram_ct, topic_unigram_ct, topic_bigram_prob), topic_descs = topic_descs, topic_INs = topic_INs, print_info = False, setting = setting)
        #Formulation_comp = Query_formulation_QS3plus(collection_lm = doc_collection_lm_dist, total_num_words = total_noof_words)
        candidate_queries_topics = {}
        for topic_num in topic_descs:
            #print ("TOPIC NUM ", topic_num)
            user_model = User_model(user_model_parameters, topic_num, {} , dataset, Formulation_comp, Reformulation_comp, result_judgement_component)
            candidate_queries = user_model.candidate_queries[:]
            if (iter_idx == 0):
                topic_INs[topic_num] = user_model.topic_IN
            candidate_queries_topics[topic_num] = candidate_queries
        Formulation_comp.intl_cand_qurs_all_tpcs = None
        #ev = Evaluation_queries(act_sessions, None, candidate_queries_topics)
        #print ("QF performance top-5:")
        #avg_jaccard_similarity_vali, avg_jaccard_similarity_2_vali = ev.query_similarity_evaluation(onlyqueries = True, top_k = 5, validation_set = True, act_session_queries = validation_session_queries)
        #avg_jaccard_similarity_test, avg_jaccard_similarity_2_test = ev.query_similarity_evaluation(onlyqueries = True, top_k = 5)
        for topic_num in candidate_queries_topics:
            candidate_queries_topics[topic_num] = candidate_queries_topics[topic_num][:200] 
        pickle.dump(candidate_queries_topics, open(os.path.join(save_dir, method + "_queries.pk"), "wb"))
        #data_points += [((a1,a2,a3), avg_jaccard_similarity_2_vali, avg_jaccard_similarity_2_test)]
        '''
        if (avg_jaccard_similarity_2_vali > best_perf):
            best_point = ((a1,a2,a3), avg_jaccard_similarity_2_vali, avg_jaccard_similarity_2_test) 
            best_perf = avg_jaccard_similarity_2_vali   
        '''
    '''
    with open(os.path.join(save_dir, 'write_performances2.txt'), 'w') as outfile:
        outfile.write('PARAMETER SETTING:' + ' CONSTRAINT: ' + str(effort) + ' MIN-THRESHOLD: ' +str(min_threshold) + '\n')
        outfile.write('A1 A2 A3 VALI_perf_top-5 total_perf_top-5\n')
        for point in data_points:
            outfile.write(str(point[0][0]) + ' ' + str(point[0][1]) + ' ' + str(point[0][2]) + ' ' + str(point[1]) + ' ' + str(point[2]) + '\n')
        outfile.write('BEST PARAMETER: ' + str(best_point[0][0]) + ' ' + str(best_point[0][1]) + ' ' + str(best_point[0][2]) + ' ' + str(best_point[1]) + ' ' + str(best_point[2]) +'\n')
    '''
    return


def fit_QF_parameters_diff_topics_users(dataset_name, foldername, outfilename = None, parameters=["alpha", "beta"],
                                           simtype="jsim", diff_users = False):
    '''
    Given the generated queries for each topic for all parameter values, find the best parameter value by cross validation.
    :param dataset_name:
    :param foldername:
    :param validation_filename:
    :return:
    '''
    print("QF_parameters_multiple_validations")
    dataset = dataset_name
    session_dir = "../simulated_sessions/"
    if not os.path.exists(session_dir): os.makedirs(session_dir)
    data_dir = os.path.join(session_dir, dataset)
    if not os.path.exists(data_dir): os.makedirs(data_dir)
    session_ops = Session_operations()
    act_sessions = session_ops.get_actual_sessions(dataset)
    #validation_sets = session_ops.get_validation_set_divisions(act_sessions)
    #validation_sets_2 = session_ops.get_validation_set_divisions(act_sessions)
    #validation_sets = pickle.load(open(os.path.join(data_dir, validation_filename), "rb"))
    if "only_recall" in parameters:
        x_list = [1.0]
    elif "only_precision" in parameters:
        x_list = [0.0]
    elif "alpha" in parameters:
        x_list = [float(i) for i in
                  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]  # alpha #0.0,0.1,0.2,0.3,0.4,0.5,0.6,
    else:
        x_list = [1.0]
    if "beta" in parameters:
        y_list = [float(i) for i in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]  # beta
    else:
        y_list = [1.0]
    parameters_list = itertools.product(x_list, y_list)
    save_dir = os.path.join(data_dir, foldername)  # fit_3_parameters_CCQF_new_r_old_p_noef_100_pts_w_len
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    print(save_dir)
    data_points = []
    best_perf = 0
    topic_descs = read_topic_descs(dataset)
    # topic_rel_docs = read_judgements(dataset)
    topic_sets,topic_set_numbers = session_ops.divide_topic_sessions(act_sessions)
    user_sets,user_set_numbers = session_ops.divide_user_sessions(act_sessions)
    if diff_users:
        topic_sets = user_sets
        topic_set_numbers = user_set_numbers

    best_perf = [0 for i in range(len(topic_sets))]
    best_point = [() for i in range(len(topic_sets))]
    avg_validation_perfs = [[] for i in range(len(topic_sets))]
    for (x, y) in parameters_list:
        alpha = x
        beta = y
        method = 'CCQF_' + str(alpha) + '_' + str(beta) + '_' + 'noef'
        try:
            candidate_queries_topics = pickle.load(open(os.path.join(save_dir, method + "_queries.pk"), 'rb'))
            for topic_num in candidate_queries_topics:
                candidate_queries_topics[topic_num] = [c for c in
                                                       candidate_queries_topics[topic_num]]  # if (len(c[0]) <= 3)
        except:
            print("method not found: ", method)
            candidate_queries_topics = None
        if candidate_queries_topics != None:
            ev = Evaluation_queries(act_sessions, None, candidate_queries_topics)
            # avg_jaccard_similarity_total, avg_jaccard_similarity_2_total,similarity_list_total = ev.query_similarity_evaluation(onlyqueries = True, top_k = 5)
            for idx, topic_set in enumerate(topic_sets):
                if simtype == "f-measure":
                    avg_jaccard_similarity_vali, prec, similarity_list_vali = ev.query_similarity_evaluation(
                        onlyqueries=True, top_k=5, validation_set=True, act_session_queries=topic_set, simtype="prec")

                    avg_jaccard_similarity_vali, recall, similarity_list_vali = ev.query_similarity_evaluation(
                        onlyqueries=True, top_k=5, validation_set=True, act_session_queries=topic_set, simtype="recall")

                    avg_jaccard_similarity_2_vali = float(2.0 * prec * recall) / float(prec + recall)

                else:
                    avg_jaccard_similarity_vali, avg_jaccard_similarity_2_vali, similarity_list_vali = ev.query_similarity_evaluation(
                        onlyqueries=True, top_k=5, validation_set=True, act_session_queries=topic_set, simtype=simtype)

                # avg_jaccard_similarity_2_test_1, avg_jaccard_similarity_2_test_1,similarity_list_test_1 = ev.query_similarity_evaluation(onlyqueries = True, top_k = 1, validation_set = True, act_session_queries = vali_set_test)
                # avg_jaccard_similarity_2_test_10, avg_jaccard_similarity_2_test_10,similarity_list_test_10 = ev.query_similarity_evaluation(onlyqueries = True, top_k = 10, validation_set = True, act_session_queries = vali_set_test)
                avg_validation_perfs[idx] += [(alpha, avg_jaccard_similarity_2_vali)]

                if (avg_jaccard_similarity_2_vali > best_perf[idx]):
                    best_point[idx] = ((alpha, beta), avg_jaccard_similarity_2_vali)
                    best_perf[idx] = avg_jaccard_similarity_2_vali
    sys.stdout = open(os.path.join(save_dir, outfilename), "w")
    print(dataset_name)
    '''
    avg_test_performance = []
    avg_best_parameter = [0,0,0] 
    avg_best_parameter_2 = [0,0,0]

    for i in range(len(validation_sets)):
        print('BEST PARAMETER: ' + str(best_point[i][0][0]) + ' ' + str(best_point[i][0][1]) + ' ' + str(best_point[i][0][2]) + ' ' + str(best_point[i][1]) + ' ' + str(best_point[i][2]) +'\n')
        avg_test_performance += [best_point[i][2]]
    for j in range(3):
        avg_best_parameter[j] = sum([best_point[i][0][j] for i in range(len(validation_sets))])/float(len(validation_sets))

    avg_best_parameter_2[0] = float(sum([best_point[i][0][0] for i in range(len(validation_sets))]))/float(len(validation_sets))
    avg_best_parameter_2[1] = float(sum([(1-best_point[i][0][0])*best_point[i][0][1] for i in range(len(validation_sets))]))/float(len(validation_sets))
    avg_best_parameter_2[2] = float(sum([(1-best_point[i][0][0])*(1-best_point[i][0][1]) for i in range(len(validation_sets))]))/float(len(validation_sets))

    avg_test_performance = float(sum(avg_test_performance))/float(len(avg_test_performance))
    print ('AVG TEST PERFORMANCE: ', avg_test_performance)
    print ("AVG BEST PARAMETER: ", avg_best_parameter)
    print ("AVG BEST PARAMETER: ", avg_best_parameter_2)
    '''
    avg_best_parameter = [0, 0]
    avg_best_parameter_2 = [0, 0]

    for i in range(len(topic_sets)):
        print('TOPIC NUMBER ' + str(topic_set_numbers[i]))
        if not diff_users:
            print("TOPIC DESC: ", topic_descs[topic_set_numbers[i]])
        if best_perf[i] == 0:
            print ("NO BEST PARAMETER")
        else:
            print('BEST PARAMETER: ' + str(best_point[i][0][0]) + ' ' + str(best_point[i][0][1]) + ' ' + str(best_point[i][1]))
            print("AVG VALIDATION PERFS: ", avg_validation_perfs[i])

    #for i in range(len(topic_sets)):
    #    print ("TOPIC NUMBER: ", topic_set_numbers[i])
    #    print("AVG VALIDATION PERFS: ", avg_validation_perfs[i])
    '''
    for j in range(2):
        avg_best_parameter[j] = sum([best_point[i][0][j] for i in range(len(topic_sets))]) / float(
            len(topic_sets))

    avg_best_parameter_2[0] = float(sum([best_point[i][0][0] for i in range(len(topic_sets))])) / float(
        len(topic_sets))
    avg_best_parameter_2[1] = float(sum([best_point[i][0][1] for i in range(len(topic_sets))])) / float(
        len(topic_sets))
    # avg_best_parameter_2[2] = float(sum([(1 - best_point[i][0][0]) * (1 - best_point[i][0][1]) for i in range(len(validation_sets))])) / float(len(validation_sets))

    print("AVG BEST PARAMETER: ", avg_best_parameter)
    print("AVG BEST PARAMETER: ", avg_best_parameter_2)
    '''



def fit_QF_parameters_multiple_validations(dataset_name, foldername, validation_filename, parameters = ["alpha","beta"], simtype = "jsim"):
    '''
    Given the generated queries for each topic for all parameter values, find the best parameter value by cross validation.
    :param dataset_name:
    :param foldername:
    :param validation_filename:
    :return:
    '''
    print ("QF_parameters_multiple_validations")
    dataset = dataset_name
    session_dir = "../simulated_sessions/" 
    if not os.path.exists(session_dir): os.makedirs(session_dir)
    data_dir = os.path.join(session_dir, dataset)
    if not os.path.exists(data_dir): os.makedirs(data_dir)
    session_ops = Session_operations()
    act_sessions = session_ops.get_actual_sessions(dataset)
    #validation_sets = session_ops.get_validation_set_divisions(act_sessions)
    #validation_sets_2 = session_ops.get_validation_set_divisions(act_sessions)
    validation_sets = pickle.load(open(os.path.join(data_dir, validation_filename), "rb"))
    if "only_recall" in parameters:
        x_list = [1.0]
    elif "only_precision" in parameters:
        x_list = [0.0]
    elif "alpha" in parameters:
        x_list = [float(i) for i in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]] #alpha #0.0,0.1,0.2,0.3,0.4,0.5,0.6,
    else:
        x_list = [1.0]
    if "beta" in parameters:
        y_list = [float(i) for i in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]] #beta
    else:
        y_list = [1.0]
    #x_list = [float(i) for i in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]]  #0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0
    #y_list = [float(i) for i in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]]
    #x_list = [float(0.0)]
    #y_list = [float(0.0)]
    #a3_list = [float(i)*0.5 for i in [0.5,1]]
    parameters_list = itertools.product(x_list,y_list)
    bigram_scoring_parameter = 2
    #bigram_scoring_parameter = int(foldername.split("fit_3_parameters_")[1][0])
    #save_dir = os.path.join(data_dir, 'fit_3_parameters_' + str(bigram_scoring_parameter) + '_noef_100_pts_2')
    save_dir = os.path.join(data_dir, foldername) #fit_3_parameters_CCQF_new_r_old_p_noef_100_pts_w_len
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    print (save_dir)
    data_points = []
    best_perf = 0
    topic_descs = read_topic_descs(dataset)
    #topic_rel_docs = read_judgements(dataset)
    best_perf = [0 for i in range(len(validation_sets))]
    best_point = [() for i in range(len(validation_sets))]
    #pickle.dump(validation_sets, open(os.path.join(data_dir, "validation_sets_413_fold.pk"), "wb"))
    #pickle.dump(validation_sets_2, open(os.path.join(data_dir, "validation_sets_2_413_fold.pk"), "wb"))
    avg_validation_perfs = [[],[],[],[]]
    avg_vali_test_perfs = [[],[],[],[]]
    for (x,y) in parameters_list: 
        '''
        a1 = x
        a2 = y*(1.0-x)
        a3 = (1.0-y)*(1.0-x)
        '''
        alpha = x
        beta = y
        #a3 = (1.0-y)
        
        #method = 'CCQF_' + str(a1) + '_' + str(a2) + '_' + str(a3) + '_' + str(bigram_scoring_parameter) + '_' + 'noef'
        method = 'CCQF_' + str(alpha) + '_' + str(beta) + '_' + 'noef'
        try:
            candidate_queries_topics = pickle.load(open(os.path.join(save_dir,method+"_queries.pk"), 'rb'))
            for topic_num in candidate_queries_topics:
                candidate_queries_topics[topic_num] = [c for c in candidate_queries_topics[topic_num]] #if (len(c[0]) <= 3)
        except:
            print ("method not found: ", method)
            candidate_queries_topics = None
        if candidate_queries_topics != None:
            ev = Evaluation_queries(act_sessions, None, candidate_queries_topics)
            #avg_jaccard_similarity_total, avg_jaccard_similarity_2_total,similarity_list_total = ev.query_similarity_evaluation(onlyqueries = True, top_k = 5)
            for idx,(vali_set,vali_set_test) in enumerate(validation_sets):
                query_lens = []
                for topic_num in candidate_queries_topics:
                    query_lens += [len(query[0]) for query in candidate_queries_topics[topic_num][:5]]
                avg_query_len = float(sum(query_lens))/float(len(query_lens))
                query_lens = []
                for topic_num in vali_set_test:
                    query_lens += [len(query) for query in vali_set_test[topic_num]]
                avg_query_len_act = float(sum(query_lens))/float(len(query_lens))
                if simtype == "f-measure":
                    avg_jaccard_similarity_vali, prec, similarity_list_vali = ev.query_similarity_evaluation(onlyqueries=True, top_k=5, validation_set=True, act_session_queries=vali_set, simtype="prec")
                    avg_jaccard_similarity_test, prec_test, similarity_list_test_prec = ev.query_similarity_evaluation(onlyqueries=True, top_k=5, validation_set=True, act_session_queries=vali_set_test, simtype="prec")
                    avg_jaccard_similarity_vali, recall, similarity_list_vali = ev.query_similarity_evaluation(onlyqueries=True, top_k=5, validation_set=True, act_session_queries=vali_set, simtype="recall")
                    avg_jaccard_similarity_test, recall_test, similarity_list_test_rec = ev.query_similarity_evaluation(onlyqueries=True, top_k=5, validation_set=True, act_session_queries=vali_set_test, simtype="recall")
                    avg_jaccard_similarity_2_vali = float(2.0*prec*recall)/float(prec+recall)
                    avg_jaccard_similarity_2_test = float(2.0*prec_test*recall_test)/float(prec_test+recall_test)
                    similarity_list_test = {}
                    for topic_num in similarity_list_test_rec:
                        similarity_list_test[topic_num] = []
                        for rec_i in range(len(similarity_list_test_rec[topic_num])):
                            try:
                                similarity_list_test[topic_num] += [float(2.0*similarity_list_test_rec[topic_num][rec_i]*similarity_list_test_prec[topic_num][rec_i])/float(similarity_list_test_rec[topic_num][rec_i]+similarity_list_test_prec[topic_num][rec_i])]
                            except ZeroDivisionError:
                                similarity_list_test[topic_num] += [0]
                else:
                    avg_jaccard_similarity_vali, avg_jaccard_similarity_2_vali,similarity_list_vali = ev.query_similarity_evaluation(onlyqueries = True, top_k = 5, validation_set = True, act_session_queries = vali_set, simtype = simtype)
                    avg_jaccard_similarity_test, avg_jaccard_similarity_2_test,similarity_list_test = ev.query_similarity_evaluation(onlyqueries = True, top_k = 5, validation_set = True, act_session_queries = vali_set_test, simtype = simtype)
                #avg_jaccard_similarity_2_test_1, avg_jaccard_similarity_2_test_1,similarity_list_test_1 = ev.query_similarity_evaluation(onlyqueries = True, top_k = 1, validation_set = True, act_session_queries = vali_set_test)
                #avg_jaccard_similarity_2_test_10, avg_jaccard_similarity_2_test_10,similarity_list_test_10 = ev.query_similarity_evaluation(onlyqueries = True, top_k = 10, validation_set = True, act_session_queries = vali_set_test)
                avg_validation_perfs[idx] += [(alpha, avg_jaccard_similarity_2_vali)]
                avg_vali_test_perfs[idx] += [(alpha, avg_jaccard_similarity_2_test)]

                if (avg_jaccard_similarity_2_vali > best_perf[idx]):
                    best_point[idx] = ((alpha, beta), avg_jaccard_similarity_2_vali, avg_jaccard_similarity_2_test, similarity_list_test, avg_query_len, avg_query_len_act)
                    best_perf[idx] = avg_jaccard_similarity_2_vali
    print (dataset_name)
    '''
    avg_test_performance = []
    avg_best_parameter = [0,0,0] 
    avg_best_parameter_2 = [0,0,0]

    for i in range(len(validation_sets)):
        print('BEST PARAMETER: ' + str(best_point[i][0][0]) + ' ' + str(best_point[i][0][1]) + ' ' + str(best_point[i][0][2]) + ' ' + str(best_point[i][1]) + ' ' + str(best_point[i][2]) +'\n')
        avg_test_performance += [best_point[i][2]]
    for j in range(3):
        avg_best_parameter[j] = sum([best_point[i][0][j] for i in range(len(validation_sets))])/float(len(validation_sets))
    
    avg_best_parameter_2[0] = float(sum([best_point[i][0][0] for i in range(len(validation_sets))]))/float(len(validation_sets))
    avg_best_parameter_2[1] = float(sum([(1-best_point[i][0][0])*best_point[i][0][1] for i in range(len(validation_sets))]))/float(len(validation_sets))
    avg_best_parameter_2[2] = float(sum([(1-best_point[i][0][0])*(1-best_point[i][0][1]) for i in range(len(validation_sets))]))/float(len(validation_sets))

    avg_test_performance = float(sum(avg_test_performance))/float(len(avg_test_performance))
    print ('AVG TEST PERFORMANCE: ', avg_test_performance)
    print ("AVG BEST PARAMETER: ", avg_best_parameter)
    print ("AVG BEST PARAMETER: ", avg_best_parameter_2)
    '''
    avg_test_performance = []
    avg_best_parameter = [0, 0]
    avg_best_parameter_2 = [0, 0]

    for i in range(len(validation_sets)):
        print('BEST PARAMETER: ' + str(best_point[i][0][0]) + ' ' + str(best_point[i][0][1]) + ' ' + str(best_point[i][1]) + ' ' + str(best_point[i][2]) + '\n')
        avg_test_performance += [best_point[i][2]]
    for i in range(len(validation_sets)):
        print ("AVG VALIDATION PERFS: ", avg_validation_perfs[i])
    for i in range(len(validation_sets)):
        print ("AVG TEST PERFS: ", avg_vali_test_perfs[i])
    for j in range(2):
        avg_best_parameter[j] = sum([best_point[i][0][j] for i in range(len(validation_sets))]) / float(
            len(validation_sets))

    avg_best_parameter_2[0] = float(sum([best_point[i][0][0] for i in range(len(validation_sets))])) / float(
        len(validation_sets))
    avg_best_parameter_2[1] = float(sum([best_point[i][0][1] for i in range(len(validation_sets))])) / float(
        len(validation_sets))
    #avg_best_parameter_2[2] = float(sum([(1 - best_point[i][0][0]) * (1 - best_point[i][0][1]) for i in range(len(validation_sets))])) / float(len(validation_sets))

    avg_test_performance = float(sum(avg_test_performance)) / float(len(avg_test_performance))
    print('AVG TEST PERFORMANCE: ', avg_test_performance)
    print("AVG BEST PARAMETER: ", avg_best_parameter)
    print("AVG BEST PARAMETER: ", avg_best_parameter_2)
    print("AVG QUERY LEN: ", best_point[0][4])
    similarities_lists = {}
    avg_query_len_act = float(best_point[0][5]+best_point[1][5]+best_point[2][5]+best_point[3][5])/float(4)
    print ("ACTUAL QUERY LENGTH AVERAGE: ", avg_query_len_act)
    for i in range(len(validation_sets)):
        for topic_num in best_point[i][3]:
            try:
                there = similarities_lists[topic_num]
                similarities_lists[topic_num] = [similarities_lists[topic_num][idx]+[best_point[i][3][topic_num][idx]] for idx in range(len(best_point[i][3][topic_num]))]
            except KeyError:
                similarities_lists[topic_num] = [[] for idx in range(5)]
                similarities_lists[topic_num] = [similarities_lists[topic_num][idx]+[best_point[i][3][topic_num][idx]] for idx in range(len(best_point[i][3][topic_num]))]
    similarities_lists_list = []
    topics_nums = [int(x) for x in list(similarities_lists.keys())]
    topics_nums.sort()
    topics_nums = [str(x) for x in topics_nums]
    print (len(topics_nums))
    for topic_num in topics_nums: 
        similarities_lists[topic_num]  = [float(sum(similarities_lists[topic_num][idx]))/float(len(similarities_lists[topic_num][idx])) for idx in range(5)]
        #print ("TOPIC NUM: ", topic_num)
        #print (similarities_lists[topic_num])
        similarities_lists_list += similarities_lists[topic_num]
    print (len((similarities_lists_list)))
    print (similarities_lists_list)
    return avg_test_performance, similarities_lists_list, avg_query_len

def QS3_evaluation(dataset_name, filename):
    '''
    OLD method.
    Given the initial simulated queries for all topics, it computesjaccard similarity between the queries and the actual queries.
    :param dataset_name:
    :param filename:
    :return:
    '''
    print ("QF_parameters_multiple_validations")
    dataset = dataset_name
    session_dir = "../simulated_sessions/" 
    if not os.path.exists(session_dir): os.makedirs(session_dir)
    data_dir = os.path.join(session_dir, dataset)
    if not os.path.exists(data_dir): os.makedirs(data_dir)
    session_ops = Session_operations()
    act_sessions = session_ops.get_actual_sessions(dataset)
    candidate_queries_topics = pickle.load(open(os.path.join(data_dir, filename), 'rb'))
    for topic_num in candidate_queries_topics:
        candidate_queries_topics[topic_num] = [c for c in candidate_queries_topics[topic_num]]
    ev = Evaluation_queries(act_sessions, None, candidate_queries_topics)
    avg_jaccard_similarity_total, avg_jaccard_similarity_2_total,similarity_list_total = ev.query_similarity_evaluation(onlyqueries = True, top_k = 5)
    print ("Avg jaccard similarity: ", avg_jaccard_similarity_2_total)
    similarities_lists_list = []
    topics_nums = [int(x) for x in list(similarity_list_total.keys())]
    topics_nums.sort()
    topics_nums = [str(x) for x in topics_nums]
    print (len(topics_nums))
    for topic_num in topics_nums: 
        similarities_lists_list += similarity_list_total[topic_num] 
    print (len((similarities_lists_list)))
    return similarities_lists_list
   


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Simulating sessions")
    parser.add_argument('-simtype', type=str, default="jsim", help="evaluation measure")
    parser.add_argument('-dataset', type=str, default="Session_track_2012", help="evaluation measure")

    '''
    similarities_lists2012_1 = QS3_evaluation("Session_track_2012", "QS3_queries.pk")
    similarities_lists2013_1 = QS3_evaluation("Session_track_2013", "QS3_queries.pk")
    similarities_lists2014_1 = QS3_evaluation("Session_track_2014", "QS3_queries.pk")
    print ("similarity2 = ", similarities_lists2012_1)
    print ("similarity4 = ", similarities_lists2013_1)
    print ("similarity6 = ", similarities_lists2014_1)
    '''
    '''
    efforts = [6]
    for k in efforts:
        find_QF_parameters_new("Session_track_2012", ["Cr_R12g", "Cp_add"], parameters=["alpha"], effort=k)
        find_QF_parameters_new("Session_track_2012", ["Cr_R12g", "Dp_add"], parameters=["alpha"], effort=k)
        find_QF_parameters_new("Session_track_2012", ["Dr_R12g", "Cp_add"], parameters=["alpha"], effort=k)
        find_QF_parameters_new("Session_track_2012", ["Dr_R12g", "Dp_add"], parameters=["alpha"], effort=k)
        find_QF_parameters_new("Session_track_2013", ["Cr_R12g", "Cp_add"], parameters=["alpha"], effort=k)
        find_QF_parameters_new("Session_track_2013", ["Cr_R12g", "Dp_add"], parameters=["alpha"], effort=k)
        find_QF_parameters_new("Session_track_2013", ["Dr_R12g", "Cp_add"], parameters=["alpha"], effort=k)
        find_QF_parameters_new("Session_track_2013", ["Dr_R12g", "Dp_add"], parameters=["alpha"], effort=k)
        find_QF_parameters_new("Session_track_2014", ["Cr_R12g", "Cp_add"], parameters=["alpha"], effort=k)
        find_QF_parameters_new("Session_track_2014", ["Cr_R12g", "Dp_add"], parameters=["alpha"], effort=k)
        find_QF_parameters_new("Session_track_2014", ["Dr_R12g", "Cp_add"], parameters=["alpha"], effort=k)
        find_QF_parameters_new("Session_track_2014", ["Dr_R12g", "Dp_add"], parameters=["alpha"], effort=k)

        
        find_QF_parameters_new("Session_track_2012", ["Cr_R1", "Dp_wd"], parameters=["alpha"], effort = k)
        find_QF_parameters_new("Session_track_2012", ["Dr_R1_wd", "Dp_wd"], parameters=["alpha"], effort = k)
        find_QF_parameters_new("Session_track_2012", ["Dr_R1_wd", "Cp"], parameters=["alpha"], effort = k)
        find_QF_parameters_new("Session_track_2012", ["Dr_R1", "Dp_wd"], parameters=["alpha"], effort = k)
        find_QF_parameters_new("Session_track_2012", ["Dr_R1_wd", "Dp"], parameters=["alpha"], effort = k)
        find_QF_parameters_new("Session_track_2012", ["Cr_R12g", "Cp"], parameters=["alpha"], effort = k)
        find_QF_parameters_new("Session_track_2012", ["Cr_R1", "Dp"], parameters=["alpha"], effort = k)
        find_QF_parameters_new("Session_track_2012", ["Dr_R12g", "Dp"], parameters=["alpha"], effort = k)
        find_QF_parameters_new("Session_track_2012", ["Dr_R1", "Cp"], parameters=["alpha"], effort = k)
        find_QF_parameters_new("Session_track_2012", ["Cr_R1", "Dr_R1_Dp_wd"], parameters=["alpha"], effort = k)
        find_QF_parameters_new("Session_track_2012", ["Cr_R12g", "Dr_R1_Dp"], parameters = ["alpha"], effort = k)
        
    '''


    args = parser.parse_args()
    simtype = args.simtype
    dataset = args.dataset
    efforts = [6]
    methods = ["Dr_R12g_Cp_add", "Dr_R12g_Cp_add"]

    #methods = ["Cr_R1_Dp_wd", "Dr_R1_wd_Dp_wd", "Dr_R1_wd_Cp", "Dr_R1_Dp_wd", "Dr_R1_wd_Dp", "Cr_R12g_Cp", "Cr_R1_Dp", "Dr_R12g_Dp", "Dr_R1_Cp", "Cr_R1_Dr_R1_Dp_wd", "Cr_R12g_Dr_R1_Dp", "Cr_R12g_Cp_add", "Cr_R12g_Dp_add", "Dr_R12g_Cp_add", "Dr_R12g_Dp_add", "Cr", "Dr", "Cp", "Dp"]
    method_perfs = {}
    for method in methods:
        method_perfs[method] = {}
    k = 6
    fit_QF_parameters_diff_topics_users(dataset,"fit_3_parameters_PRE_Dr_R12g_Cp_add_ef" + str(k) + "_100_pts_w_len",outfilename = "Different_topics_best_params.txt",
                                                                                                           parameters=[
                                                                                                               "alpha"],
                                                                                                           simtype=simtype)
    fit_QF_parameters_diff_topics_users(dataset,"fit_3_parameters_PRE_Dr_R12g_Cp_add_ef" + str(k) + "_100_pts_w_len",outfilename = "Different_users_best_params.txt",
                                                                                                           parameters=[
                                                                                                               "alpha"],
                                                                                                           simtype=simtype, diff_users = True)

    '''
    for k in efforts:
        #method_perfs[methods[11]][k],_,method_perfs[methods[11]][0] = fit_QF_parameters_multiple_validations(dataset, "fit_3_parameters_PRE_Cr_R12g_Cp_add_ef" + str(k) + "_100_pts_w_len",
        #                                       "validation_sets_413_fold.pk", parameters=["alpha"], simtype=simtype)
        #method_perfs[methods[12]][k],_,method_perfs[methods[12]][0] = fit_QF_parameters_multiple_validations(dataset, "fit_3_parameters_PRE_Cr_R12g_Dp_add_ef" + str(k) + "_100_pts_w_len",
        #                                       "validation_sets_413_fold.pk", parameters=["alpha"], simtype=simtype)
        method_perfs[methods[13]][k],_,method_perfs[methods[13]][0] = fit_QF_parameters_multiple_validations(dataset, "fit_3_parameters_PRE_Dr_R12g_Cp_add_ef" + str(k) + "_100_pts_w_len",
                                               "validation_sets_413_fold.pk", parameters=["alpha"], simtype=simtype)
        method_perfs[methods[14]][k],_,method_perfs[methods[14]][0] = fit_QF_parameters_multiple_validations(dataset, "fit_3_parameters_PRE_Dr_R12g_Dp_add_ef" + str(k) + "_100_pts_w_len",
                                               "validation_sets_413_fold.pk", parameters=["alpha"], simtype=simtype)

        method_perfs[methods[15]][k], _, method_perfs[methods[11]][0] = fit_QF_parameters_multiple_validations(dataset,
                                                                                                               "fit_3_parameters_PRE_Cr_R12g_Cp_add_ef" + str(
                                                                                                                   k) + "_100_pts_w_len",
                                                                                                               "validation_sets_413_fold.pk",
                                                                                                               parameters=[
                                                                                                                   "only_recall"],
                                                                                                               simtype=simtype)
        
        method_perfs[methods[16]][k], _, method_perfs[methods[13]][0] = fit_QF_parameters_multiple_validations(dataset,
                                                                                                               "fit_3_parameters_PRE_Dr_R12g_Cp_add_ef" + str(
                                                                                                                   k) + "_100_pts_w_len",
                                                                                                               "validation_sets_413_fold.pk",
                                                                                                               parameters=[
                                                                                                                   "only_recall"],
                                                                                                               simtype=simtype)
        method_perfs[methods[17]][k], _, method_perfs[methods[13]][0] = fit_QF_parameters_multiple_validations(dataset,
                                                                                                               "fit_3_parameters_PRE_Dr_R12g_Cp_add_ef" + str(
                                                                                                                   k) + "_100_pts_w_len",
                                                                                                               "validation_sets_413_fold.pk",
                                                                                                               parameters=[
                                                                                                                   "only_precision"],
                                                                                                               simtype=simtype)

        method_perfs[methods[18]][k], _, method_perfs[methods[14]][0] = fit_QF_parameters_multiple_validations(dataset,
                                                                                                               "fit_3_parameters_PRE_Dr_R12g_Dp_add_ef" + str(
                                                                                                                   k) + "_100_pts_w_len",
                                                                                                               "validation_sets_413_fold.pk",
                                                                                                               parameters=[
                                                                                                                   "only_precision"],
                                                                                                               simtype=simtype)
        
        
        method_perfs[methods[0]][k],_,method_perfs[methods[0]][0] = fit_QF_parameters_multiple_validations(dataset, "fit_3_parameters_PRE_Cr_R1_Dp_wd_ef" + str(k) + "_100_pts_w_len",
                                               "validation_sets_413_fold.pk", parameters=["alpha"], simtype=simtype)
        method_perfs[methods[1]][k],_,method_perfs[methods[1]][0] = fit_QF_parameters_multiple_validations(dataset, "fit_3_parameters_PRE_Dr_R1_wd_Dp_wd_ef" + str(k) + "_100_pts_w_len",
                                               "validation_sets_413_fold.pk", parameters=["alpha"], simtype=simtype)
        method_perfs[methods[2]][k],_,method_perfs[methods[2]][0] = fit_QF_parameters_multiple_validations(dataset, "fit_3_parameters_PRE_Dr_R1_wd_Cp_ef" + str(k) + "_100_pts_w_len",
                                               "validation_sets_413_fold.pk", parameters=["alpha"], simtype=simtype)
        method_perfs[methods[3]][k],_,method_perfs[methods[3]][0]  = fit_QF_parameters_multiple_validations(dataset, "fit_3_parameters_PRE_Dr_R1_Dp_wd_ef" + str(k) + "_100_pts_w_len",
                                               "validation_sets_413_fold.pk", parameters=["alpha"], simtype=simtype)
        method_perfs[methods[4]][k],_,method_perfs[methods[4]][0]  = fit_QF_parameters_multiple_validations(dataset, "fit_3_parameters_PRE_Dr_R1_wd_Dp_ef" + str(k) + "_100_pts_w_len",
                                               "validation_sets_413_fold.pk", parameters=["alpha"], simtype=simtype)
        method_perfs[methods[5]][k],_,method_perfs[methods[5]][0]  = fit_QF_parameters_multiple_validations(dataset, "fit_3_parameters_PRE_Cr_R12g_Cp_ef" + str(k) + "_100_pts_w_len",
                                               "validation_sets_413_fold.pk", parameters=["alpha"], simtype=simtype)
        method_perfs[methods[6]][k],_,method_perfs[methods[6]][0]  = fit_QF_parameters_multiple_validations(dataset, "fit_3_parameters_PRE_Cr_R1_Dp_ef" + str(k) + "_100_pts_w_len",
                                               "validation_sets_413_fold.pk", parameters=["alpha"], simtype=simtype)
        method_perfs[methods[7]][k],_,method_perfs[methods[7]][0]  = fit_QF_parameters_multiple_validations(dataset, "fit_3_parameters_PRE_Dr_R12g_Dp_ef" + str(k) + "_100_pts_w_len",
                                               "validation_sets_413_fold.pk", parameters=["alpha"], simtype=simtype)
        method_perfs[methods[8]][k],_,method_perfs[methods[8]][0]  = fit_QF_parameters_multiple_validations(dataset, "fit_3_parameters_PRE_Dr_R1_Cp_ef" + str(k) + "_100_pts_w_len",
                                               "validation_sets_413_fold.pk", parameters=["alpha"], simtype=simtype)
        method_perfs[methods[9]][k],_,method_perfs[methods[9]][0]  = fit_QF_parameters_multiple_validations(dataset, "fit_3_parameters_PRE_Cr_R1_Dr_R1_Dp_wd_ef" + str(k) + "_100_pts_w_len",
                                               "validation_sets_413_fold.pk", parameters=["alpha"], simtype=simtype)
        method_perfs[methods[10]][k],_,method_perfs[methods[10]][0]  = fit_QF_parameters_multiple_validations(dataset, "fit_3_parameters_PRE_Cr_R12g_Dr_R1_Dp_ef" + str(k) + "_100_pts_w_len",
                                             "validation_sets_413_fold.pk", parameters=["alpha"], simtype=simtype)
        
    qs3plus_perf,_,_ = fit_QF_parameters_multiple_validations(dataset,
                                                             "QS3plus_new",
                                                             "validation_sets_413_fold.pk",
                                                             parameters=[], simtype=simtype)
    
    for method in method_perfs:
        print (method, sorted(method_perfs[method].items(), key=lambda x:x[0]))
    print("QS3_plus", qs3plus_perf)
    '''


    '''
    fit_QF_parameters_multiple_validations("Session_track_2012",
                                           "fit_3_parameters_PRE_Cr_R1_Dr_R1_Dp_wd_noef_100_pts_w_len",
                                           "validation_sets_413_fold.pk", parameters=["alpha"])
    fit_QF_parameters_multiple_validations("Session_track_2012", "fit_3_parameters_PRE_Cr_R12g_Dr_R1_Dp_noef_100_pts_w_len",
                                           "validation_sets_413_fold.pk", parameters=["alpha"])
    '''
    '''
    fit_QF_parameters_multiple_validations("Session_track_2012", "fit_3_parameters_PRE_Cr_R12g_Cp_noef_100_pts_w_len", "validation_sets_413_fold.pk")
    fit_QF_parameters_multiple_validations("Session_track_2012",
                                           "fit_3_parameters_PRE_Cr_R12g_Dr_Dp_noef_100_pts_w_len",
                                           "validation_sets_413_fold.pk")
    fit_QF_parameters_multiple_validations("Session_track_2012",
                                           "fit_3_parameters_PRE_Cr_R12a_Cp_noef_100_pts_w_len",
                                           "validation_sets_413_fold.pk")
    fit_QF_parameters_multiple_validations("Session_track_2012",
                                           "fit_3_parameters_PRE_Dr_R12a_Dp_noef_100_pts_w_len",
                                           "validation_sets_413_fold.pk")
    fit_QF_parameters_multiple_validations("Session_track_2012",
                                           "fit_3_parameters_PRE_Dr_R12g_Dp_noef_100_pts_w_len",
                                           "validation_sets_413_fold.pk")
    '''
    #similarities_lists2012_1 = fit_QF_parameters_multiple_validations("Session_track_2012", "fit_3_parameters_CCQF_old_r_old_p_noef_100_pts_w_len", "validation_sets_413_fold.pk")
    #similarities_lists2013_1 = fit_QF_parameters_multiple_validations("Session_track_2013", "fit_3_parameters_CCQF_old_r_old_p_noef_100_pts_w_len", "validation_sets_413_fold.pk")
    #similarities_lists2014_1 = fit_QF_parameters_multiple_validations("Session_track_2014", "fit_3_parameters_CCQF_old_r_old_p_noef_100_pts_w_len", "validation_sets_413_fold.pk")
    #print ("similarity2 = ", similarities_lists2012_1)
    #print ("similarity4 = ", similarities_lists2013_1)
    #print ("similarity6 = ", similarities_lists2014_1)

    #fit_QF_parameters_multiple_validations("Session_track_2012", "fit_3_parameters_CCQF_new_r_new_p_noef_100_pts_wo_len", "validation_sets_2_413_fold.pk")
    #fit_QF_parameters_multiple_validations("Session_track_2013", "fit_3_parameters_CCQF_new_r_new_p_noef_100_pts_wo_len", "validation_sets_2_413_fold.pk")
    #fit_QF_parameters_multiple_validations("Session_track_2014", "fit_3_parameters_CCQF_new_r_new_p_noef_100_pts_wo_len", "validation_sets_2_413_fold.pk")

    '''
    fit_QF_parameters_multiple_validations("Session_track_2012", "fit_3_parameters_2_noef_new_r12c_100_pts_2", "validation_sets_2_413_fold.pk")
    fit_QF_parameters_multiple_validations("Session_track_2013", "fit_3_parameters_2_noef_new_r12c_100_pts_2", "validation_sets_2_413_fold.pk")
    fit_QF_parameters_multiple_validations("Session_track_2014", "fit_3_parameters_2_noef_new_r12c_100_pts_2", "validation_sets_2_413_fold.pk")
    
    
    fit_QF_parameters_multiple_validations("Session_track_2012", "fit_3_parameters_CCQF_new_r_old_p_noef_100_pts_wo_len", "validation_sets_2_413_fold.pk")
    fit_QF_parameters_multiple_validations("Session_track_2013", "fit_3_parameters_CCQF_new_r_old_p_noef_100_pts_wo_len", "validation_sets_2_413_fold.pk")
    fit_QF_parameters_multiple_validations("Session_track_2014", "fit_3_parameters_CCQF_new_r_old_p_noef_100_pts_wo_len", "validation_sets_2_413_fold.pk")
    
    fit_QF_parameters_multiple_validations("Session_track_2012", "fit_3_parameters_CCQF_old_r_old_p_noef_100_pts_wo_len", "validation_sets_413_fold.pk")
    fit_QF_parameters_multiple_validations("Session_track_2013", "fit_3_parameters_CCQF_old_r_old_p_noef_100_pts_wo_len", "validation_sets_413_fold.pk")
    fit_QF_parameters_multiple_validations("Session_track_2014", "fit_3_parameters_CCQF_old_r_old_p_noef_100_pts_wo_len", "validation_sets_413_fold.pk")

    fit_QF_parameters_multiple_validations("Session_track_2012", "fit_3_parameters_CCQF_heuristic_r_old_p_noef_100_pts_wo_len", "validation_sets_413_fold.pk")
    fit_QF_parameters_multiple_validations("Session_track_2013", "fit_3_parameters_CCQF_heuristic_r_old_p_noef_100_pts_wo_len", "validation_sets_413_fold.pk")
    fit_QF_parameters_multiple_validations("Session_track_2014", "fit_3_parameters_CCQF_heuristic_r_old_p_noef_100_pts_wo_len", "validation_sets_413_fold.pk")

    fit_QF_parameters_multiple_validations("Session_track_2012", "fit_3_parameters_CCQF_old_r_old_p_noef_100_pts_wo_len", "validation_sets_2_413_fold.pk")
    fit_QF_parameters_multiple_validations("Session_track_2013", "fit_3_parameters_CCQF_old_r_old_p_noef_100_pts_wo_len", "validation_sets_2_413_fold.pk")
    fit_QF_parameters_multiple_validations("Session_track_2014", "fit_3_parameters_CCQF_old_r_old_p_noef_100_pts_wo_len", "validation_sets_2_413_fold.pk")

    fit_QF_parameters_multiple_validations("Session_track_2012", "fit_3_parameters_CCQF_heuristic_r_old_p_noef_100_pts_wo_len", "validation_sets_2_413_fold.pk")
    fit_QF_parameters_multiple_validations("Session_track_2013", "fit_3_parameters_CCQF_heuristic_r_old_p_noef_100_pts_wo_len", "validation_sets_2_413_fold.pk")
    fit_QF_parameters_multiple_validations("Session_track_2014", "fit_3_parameters_CCQF_heuristic_r_old_p_noef_100_pts_wo_len", "validation_sets_2_413_fold.pk")
    
    fit_QF_parameters_multiple_validations("Session_track_2012", "fit_3_parameters_CCQF_new_rpnoef_100_pts_wo_len", "validation_sets_413_fold.pk")
    fit_QF_parameters_multiple_validations("Session_track_2013", "fit_3_parameters_CCQF_new_rpnoef_100_pts_wo_len", "validation_sets_413_fold.pk")
    fit_QF_parameters_multiple_validations("Session_track_2014", "fit_3_parameters_CCQF_new_rpnoef_100_pts_wo_len", "validation_sets_413_fold.pk")

    fit_QF_parameters_multiple_validations("Session_track_2012", "fit_3_parameters_CCQF_new_rpnoef_100_pts_wo_len", "validation_sets_2_413_fold.pk")
    fit_QF_parameters_multiple_validations("Session_track_2013", "fit_3_parameters_CCQF_new_rpnoef_100_pts_wo_len", "validation_sets_2_413_fold.pk")
    fit_QF_parameters_multiple_validations("Session_track_2014", "fit_3_parameters_CCQF_new_rpnoef_100_pts_wo_len", "validation_sets_2_413_fold.pk")
    
    fit_QF_parameters_multiple_validations("Session_track_2012", "fit_3_parameters_CCQF_old_r_p3_noef_100_pts_w_len", "validation_sets_413_fold.pk")
    fit_QF_parameters_multiple_validations("Session_track_2013", "fit_3_parameters_CCQF_old_r_p3_noef_100_pts_w_len", "validation_sets_413_fold.pk")
    fit_QF_parameters_multiple_validations("Session_track_2014", "fit_3_parameters_CCQF_old_r_p3_noef_100_pts_w_len", "validation_sets_413_fold.pk")
    
    fit_QF_parameters_multiple_validations("Session_track_2012", "fit_3_parameters_CCQF_old_r_p3_noef_100_pts_w_len", "validation_sets_2_413_fold.pk")
    fit_QF_parameters_multiple_validations("Session_track_2013", "fit_3_parameters_CCQF_old_r_p3_noef_100_pts_w_len", "validation_sets_2_413_fold.pk")
    fit_QF_parameters_multiple_validations("Session_track_2014", "fit_3_parameters_CCQF_old_r_p3_noef_100_pts_w_len", "validation_sets_2_413_fold.pk")
    '''

