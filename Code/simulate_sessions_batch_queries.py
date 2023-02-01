import os
import time
import pickle
import argparse
from utils import *
from User_model_new import *
#from indri_rankers import *
from simulate_sessions import Session, Interaction
from evaluation_methods import Evaluation_queries,Session_operations
stopwords_file = "../lemur-stopwords.txt"
old_stdout_target = sys.stdout
cb9_corpus_trec_ids = None
#start_time = time.time()
#cb9_corpus_trec_ids = read_cb9_catb_ids()
#print ("TIME TAKEN: ", time.time()-start_time)
def make_runquery_file(queries, filename):
    '''
    this makes the indri query in indri query format given a list of queries
    :param queries:
    :param filename:
    :return:
    '''
    #queries = ['prime factor']*50
    with open(filename, 'w') as outfile: 
        outfile.write('<parameters>\n')
        queryids = [int(i) for i in queries.keys()]
        queryids.sort()
        queryids = [str(i) for i in queryids]
        for queryid in queryids:
            q = queries[queryid]
            outfile.write("<query>\n")    
            outfile.write("<type>indri</type>\n")
            outfile.write("<number>" + str(queryid) + "</number>\n")
            outfile.write("<text>")
            outfile.write("#combine(" + q + ")")    
            outfile.write("</text>\n")
            outfile.write("</query>\n")
        outfile.write('<index>../../ClueWeb09_catB_index_2</index>\n')
        outfile.write('<printSnippets>true</printSnippets>\n')    
        outfile.write('<trecFormat>true</trecFormat>')
        outfile.write('<count>200</count>') 
        outfile.write('</parameters>\n')

def make_runquery_file2(queries, filename, index_path, snippets = True, count =200):
    '''
    this makes a complex indri query acc. to used in Session track datasets
    :param queries:
    :param filename:
    :param snippets:
    :return:
    '''
    #queries = ['prime factor']*50
    with open(filename, 'w') as outfile: 
        outfile.write('<parameters>\n')
        queryids = [int(i) for i in queries.keys()]
        queryids.sort()
        queryids = [str(i) for i in queryids]
        for queryid in queryids:
            q = queries[queryid]
            outfile.write("<query>\n")    
            outfile.write("<type>indri</type>\n")
            outfile.write("<number>" + str(queryid) + "</number>\n")
            outfile.write("<text>#less(spam -130)")
            outfile.write("#weight( ")
            outfile.write("1 #combine(" + q + ")\n")

            outfile.write("1 #weight(1 #combine(")
            for w in q.split():
                outfile.write(w + ".title ")
            outfile.write(")\n")
            outfile.write("1 #uw(" + q + ").title)")    

            outfile.write("50 #weight(1 #combine(")
            for w in q.split():
                outfile.write(w + ".inlink ")
            outfile.write(")\n")
            outfile.write("1 #uw(" + q + ").inlink)")    

            outfile.write("0.5 #weight(1 #combine(")
            for w in q.split():
                outfile.write(w + ".url ")
            outfile.write(")\n")
            outfile.write("1 #uw(" + q + ").url)")    

            outfile.write(")\n")
            outfile.write("</text>\n")
            outfile.write("</query>\n")
        outfile.write('<index>' + index_path + '</index>\n')
        if snippets:
            outfile.write('<printSnippets>True</printSnippets>\n')    
        else:
            outfile.write('<printSnippets>False</printSnippets>\n')           
        outfile.write('<trecFormat>true</trecFormat>')
        outfile.write('<count>'+str(count)+'</count>')
        outfile.write('</parameters>\n')

def make_runquery_file3(queries, filename, snippets = True):
    #queries = ['prime factor']*50
    with open(filename, 'w') as outfile: 
        outfile.write('<parameters>\n')
        queryids = [int(i) for i in queries.keys()]
        queryids.sort()
        queryids = [str(i) for i in queryids]
        for queryid in queryids:
            q = queries[queryid]
            outfile.write("<query>\n")    
            outfile.write("<type>indri</type>\n")
            outfile.write("<number>" + str(queryid) + "</number>\n")
            outfile.write("<text>#less(spam -130)")
            #outfile.write("#weight( ")
            outfile.write("#combine(" + q + ")\n")
            '''
            outfile.write("1 #weight(1 #combine(")
            for w in q.split():
                outfile.write(w + ".title ")
            outfile.write(")\n")
            outfile.write("1 #uw(" + q + ").title)")    
            outfile.write("50 #weight(1 #combine(")
            for w in q.split():
                outfile.write(w + ".inlink ")
            outfile.write(")\n")
            outfile.write("1 #uw(" + q + ").inlink)")    
            outfile.write("0.5 #weight(1 #combine(")
            for w in q.split():
                outfile.write(w + ".url ")
            outfile.write(")\n")
            outfile.write("1 #uw(" + q + ").url)")    
            outfile.write(")\n")
            '''
            outfile.write("</text>\n")
            outfile.write("</query>\n")
        outfile.write('<index>../../ClueWeb09_catB_index_2</index>\n')
        if snippets:
            outfile.write('<printSnippets>True</printSnippets>\n')    
        else:
            outfile.write('<printSnippets>False</printSnippets>\n')           
        outfile.write('<trecFormat>true</trecFormat>')
        outfile.write('<count>200</count>') 
        outfile.write('</parameters>\n')

def format_batch_results_trec_format(result_file):
    '''
    format indri results into results for the model: [id, score, snippet]
    :param result_file:
    :return:
    '''
    with open(result_file, 'rb') as infile:
        results = {}
        for line in infile:
            line = line.decode("utf-8", errors = 'ignore')
            if ("clueweb09-en" in line) or ("clueweb12-" in line):
                #print (line)
                queryid,_,docid,rank,score,_ = line.strip().split()
                try:
                    results[queryid] += [{"docid":docid,"score":score,"snippet":""}]
                except:
                    results[queryid] = [{"docid":docid,"score":score,"snippet":""}]
                previous_queryid = queryid
            else:
                results[previous_queryid][-1]["snippet"] += line.strip()
    all_formatted_results = {}
    for queryid in results:
        formatted_results = []
        for result in results[queryid]:
            formatted_result={} 
            formatted_result["docid"] = result["docid"]
            formatted_result["title"] = ""
            formatted_result["snippet"] = result["snippet"]
            formatted_result["full_text_lm"] = ""
            formatted_results += [formatted_result]
        all_formatted_results[queryid] = formatted_results
    return all_formatted_results

def sample_testing():
    queries = {}
    for i in range(50):
        queries[str(i)] = 'prime factor'
    #make_runquery_file(queries)
    start_time = time.time()
    os.system("../../indri-5.14/runquery/IndriRunQuery sample_query.txt")
    print ('TIME TAKEN: ', time.time()-start_time)
    #all_formatted_results = format_batch_results_trec_format('sample_output.txt')
#print (all_formatted_results.keys())
#print (all_formatted_results['0'])

def actual_queries_performance_new(dataset, index_path, start, stop):
    '''
    Getting actual queries from session track sessions and running the queries with the constructed indri index and evaluate the resultant ranked list, also evaluate the earch result present in the session.
    :return:
    '''
    print ("actual_queries_performance")
    #dataset = "Session_track_2012"
    session_dir = "../simulated_sessions/" +dataset
    if not os.path.exists(session_dir): os.makedirs(session_dir)
    method = "Actual_queries_" + dataset.lower()
    save_dir = os.path.join(session_dir, method)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    session_ops = Session_operations()
    act_sessions = session_ops.get_actual_sessions(dataset)
    #sys.stdout = open(os.path.join(save_dir, method + str(start) + '_big_idx_out.txt'), 'w')
    topic_rel_docs = read_judgements(dataset)
    queries_list_act_topics = session_ops.get_all_queries(act_sessions, dopreprocess = True)
    queries_list_act_topics_set = {}
    for topic_num in queries_list_act_topics:
        queries_list_act_topics_set[topic_num] = list(set([" ".join(query) for query in queries_list_act_topics[topic_num]]))
    query_results = {}
    result_file = method + "_retrieval_results.pk"
    act_formatted_results = {}
    #if os.path.exists(os.path.join(save_dir, result_file)): act_formatted_results = pickle.load(open(os.path.join(save_dir, result_file) , "rb"))

    sorted_topic_nums = sorted(queries_list_act_topics, key = lambda x: int(x))

    for topic_num in sorted_topic_nums[start:stop]:
        print ("TOPIC NUM: ", topic_num)
        for idx, query in enumerate(queries_list_act_topics_set[topic_num]):
            print (query)
    '''
    for topic_num in sorted_topic_nums[start:stop]:
        try:
            there = topic_rel_docs[topic_num]
        except KeyError:
            continue
        print ("NUMBER OF QUERIES: ", len(queries_list_act_topics_set[topic_num]))
        act_formatted_results[topic_num] = {}
        actual_queries = {}
        for idx,query in enumerate(queries_list_act_topics_set[topic_num]):
            actual_queries[str(idx)] = query
        inputname = os.path.join(save_dir, method + str(start) + "_queries_input")
        outputname = os.path.join(save_dir, method + str(start) + "_results_output")
        make_runquery_file2(actual_queries, inputname, index_path, snippets = False)
        start_time = time.time()
        os.system("../../indri-5.14/runquery/IndriRunQuery " + inputname + "> " + outputname)
        print ('TIME TAKEN: ', time.time()-start_time)
        all_act_formatted_results = format_batch_results_trec_format(outputname)
        for idx,query in enumerate(queries_list_act_topics_set[topic_num]):
            try:
                act_formatted_results[topic_num][query] = all_act_formatted_results[str(idx)]
            except:
                print ("MISSED QUERIES: ", topic_num, query)
                act_formatted_results[topic_num][query] = []
        pickle.dump(act_formatted_results, open(os.path.join(save_dir, result_file), "wb"))
    pickle.dump(act_formatted_results, open(os.path.join(save_dir, result_file) , "wb"))

    act_formatted_results = pickle.load(open(os.path.join(save_dir, result_file) , "rb"))
    ev = Evaluation_queries(act_sessions, None, None, dataset = dataset)
    topic_ndcgs = {}
    for topic_num in act_formatted_results:
        try:
            there = topic_rel_docs[topic_num]
        except KeyError:
            continue
        act_ndcgs_2 = []
        act_ndcgs50_2 = []
        for query in queries_list_act_topics[topic_num]:
            act_results_2 = act_formatted_results[topic_num][" ".join(query)]
            act_ndcgs_2 += [ev.NDCG_eval(act_results_2, topic_num, 10, cb9_corpus_trec_ids)]
            act_ndcgs50_2 += [ev.NDCG_eval(act_results_2, topic_num, 50, cb9_corpus_trec_ids)]

        act_ndcgs = ev.QF_NDCG_act_eval(topic_num, 10, cb9_corpus_trec_ids)
        act_ndcgs50 = ev.QF_NDCG_act_eval(topic_num, 50, cb9_corpus_trec_ids)
        topic_ndcgs[topic_num] = [float(sum(act_ndcgs))/float(len(act_ndcgs)), float(sum(act_ndcgs_2))/float(len(act_ndcgs_2)), float(sum(act_ndcgs50))/float(len(act_ndcgs50)), float(sum(act_ndcgs50_2))/float(len(act_ndcgs50_2)), act_ndcgs, act_ndcgs_2]
    if len(topic_ndcgs)!=0:
        act_ndcgs = sum([topic_ndcgs[topic_num][0] for topic_num in topic_ndcgs])/len(list(topic_ndcgs.keys()))
        act_ndcgs_2 = sum([topic_ndcgs[topic_num][1] for topic_num in topic_ndcgs])/len(list(topic_ndcgs.keys()))
        act_ndcgs50 = sum([topic_ndcgs[topic_num][2] for topic_num in topic_ndcgs])/len(list(topic_ndcgs.keys()))
        act_ndcgs50_2 = sum([topic_ndcgs[topic_num][3] for topic_num in topic_ndcgs])/len(list(topic_ndcgs.keys()))
        outputfilename = method + "_NDCG_scores.txt"
        with open(os.path.join(save_dir, outputfilename), "w") as outfile:
            outfile.write("ACT NDCG10: " + str(act_ndcgs) + "\n")
            outfile.write("ACT NDCG10 2: " + str(act_ndcgs_2) + "\n")
            outfile.write("ACT NDCG50: " + str(act_ndcgs50) + "\n")
            outfile.write("ACT NDCG50 2: " + str(act_ndcgs50_2) + "\n")
    '''
def actual_queries_performance():
    '''
    Getting actual queries from session track sessions and running the queries with the constructed indri index and evaluate the resultant ranked list, also evaluate the earch result present in the session.
    :return:
    '''
    print ("actual_queries_performance")
    dataset = "Session_track_2012"
    session_dir = "../simulated_sessions/" 
    if not os.path.exists(session_dir): os.makedirs(session_dir)
    save_dir = os.path.join(session_dir, "Session_track_2012")
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    session_ops = Session_operations()
    act_sessions = session_ops.get_actual_sessions(dataset)
    sys.stdout = open(os.path.join(save_dir, "Actual_queries_" + '_big_idx_out.txt'), 'w')
    topic_rel_docs = read_judgements(dataset)
    queries_list_act_topics = session_ops.get_all_queries(act_sessions, dopreprocess = True)
    queries_list_act_topics_set = {}
    for topic_num in queries_list_act_topics:
        queries_list_act_topics_set[topic_num] = list(set([" ".join(query) for query in queries_list_act_topics[topic_num]]))
    query_results = {}
    act_formatted_results = {}
    method = "Actual_queries_"
    '''
    for topic_num in queries_list_act_topics:
        act_formatted_results[topic_num] = {}
        actual_queries = {}
        for idx,query in enumerate(queries_list_act_topics_set[topic_num]):
            actual_queries[str(idx)] = query
        inputname = os.path.join(save_dir, method + "queries_input")
        outputname = os.path.join(save_dir, method + "results_output")
        make_runquery_file3(actual_queries, inputname, snippets = False)
        start_time = time.time()
        os.system("../../indri-5.14/runquery/IndriRunQuery " + inputname + "> " + outputname)
        print ('TIME TAKEN: ', time.time()-start_time)
        all_act_formatted_results = format_batch_results_trec_format(outputname)
        for idx,query in enumerate(queries_list_act_topics_set[topic_num]):
            try:
                act_formatted_results[topic_num][query] = all_act_formatted_results[str(idx)]
            except:
                print ("MISSED QUERIES: ", topic_num, query)
                act_formatted_results[topic_num][query] = []
    pickle.dump(act_formatted_results, open(os.path.join(save_dir, "actual_queries_retrieval_results_2.pk") , "wb") )
    '''
    act_formatted_results = pickle.load(open(os.path.join(save_dir, "actual_queries_retrieval_results_2.pk") , "rb") )
    ev = Evaluation_queries(act_sessions, None, None, dataset = dataset)
    topic_ndcgs = {}
    for topic_num in topic_rel_docs:
        act_ndcgs_2 = []
        act_ndcgs50_2 = []
        for query in queries_list_act_topics[topic_num]:
            act_results_2 = act_formatted_results[topic_num][" ".join(query)]
            act_ndcgs_2 += [ev.NDCG_eval(act_results_2, topic_num, 10, cb9_corpus_trec_ids)]
            act_ndcgs50_2 += [ev.NDCG_eval(act_results_2, topic_num, 50, cb9_corpus_trec_ids)]
        act_ndcgs = ev.QF_NDCG_act_eval(topic_num, 10, cb9_corpus_trec_ids)
        act_ndcgs50 = ev.QF_NDCG_act_eval(topic_num, 50, cb9_corpus_trec_ids)
        topic_ndcgs[topic_num] = [float(sum(act_ndcgs))/float(len(act_ndcgs)), float(sum(act_ndcgs_2))/float(len(act_ndcgs_2)), float(sum(act_ndcgs50))/float(len(act_ndcgs50)), float(sum(act_ndcgs50_2))/float(len(act_ndcgs50_2)), act_ndcgs, act_ndcgs_2]
    act_ndcgs = sum([topic_ndcgs[topic_num][0] for topic_num in topic_ndcgs])/len(list(topic_ndcgs.keys()))
    act_ndcgs_2 = sum([topic_ndcgs[topic_num][1] for topic_num in topic_ndcgs])/len(list(topic_ndcgs.keys()))
    act_ndcgs50 = sum([topic_ndcgs[topic_num][2] for topic_num in topic_ndcgs])/len(list(topic_ndcgs.keys()))
    act_ndcgs50_2 = sum([topic_ndcgs[topic_num][3] for topic_num in topic_ndcgs])/len(list(topic_ndcgs.keys()))
    print ("ACT NDCG: ", act_ndcgs)
    print ("ACT NDCG 2: ", act_ndcgs_2)
    print ("ACT NDCG50: ", act_ndcgs50)
    print ("ACT NDCG50 2: ", act_ndcgs50_2)

def test_query_ndcg_performance():
    '''
    Doing Adhoc evaluation
    Getting intial queries and run the top 5 candidate queries using the index, evaluate search results NDCG and compare with NDCG of actual query search results
    :return:
    '''
    print ("DOING test_query_ndcg_performance   ")
    sample_index = pyndri.Index('../../ClueWeb09_catB_sample_index')
    token2id, id2token, id2df = sample_index.get_dictionary()
    id2tf = sample_index.get_term_frequencies()
    doc_collection_lm_dist = {}
    for tokenid in id2tf:
        doc_collection_lm_dist[id2token[tokenid]] = id2tf[tokenid]
    total_noof_words = float(sum(doc_collection_lm_dist.values()))
    doc_collection_lm_dist = {token: float(doc_collection_lm_dist[token])/float(total_noof_words) for token in doc_collection_lm_dist}
    doc_collection_lm_binary_dist = {}
    for tokenid in id2df:
        doc_collection_lm_binary_dist[id2token[tokenid]] = id2df[tokenid]
    total_noof_docs = float(max(doc_collection_lm_dist.values()))
    doc_collection_lm_binary_dist = {token: float(doc_collection_lm_binary_dist[token])/float(total_noof_docs) for token in doc_collection_lm_binary_dist}
    
    dataset = "Session_track_2012"
    session_dir = "../simulated_sessions/" 
    if not os.path.exists(session_dir): os.makedirs(session_dir)
    save_dir = os.path.join(session_dir, "Session_track_2012")
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    #method = 'CCQF_' + str(a1) + '_' + str(a2) + '_' + str(a3) + '_' + str(bigram_scoring_parameter) + '_' + 'noef'
    do_reformulation = False
    parameters = []
    parameters += [do_reformulation]
    bigram_scoring_parameter = 2
    #a1,a2,a3 = 0.125,0.645,0.23
    a1,a2,a3 = 0.1,0.765,0.135
    #a1,a2,a3 = 0.1,0.585,0.315
    r1,r2  = 0.95,0.95
    method = 'CCQF_' + str(a1) + '_' + str(a2) + '_' + str(a3) + '_' + str(bigram_scoring_parameter) + '_noef_query_perf_'
    #method = 'QS3plus_query_perf'
    print ('METHOD: ', method)
    sys.stdout = open(os.path.join(save_dir, method + '_big_idx_out.txt'), 'w')
    topic_descs = read_topic_descs(dataset)
    topic_rel_docs = read_judgements(dataset)
    (topic_bigram_ct, topic_unigram_ct, topic_bigram_prob) = read_bigram_topic_lm(dataset)
    Reformulation_comp =  Reformulation_component_multinomial(parameters = (r1,r2), collection_lm = doc_collection_lm_dist, print_info = True, type2_update= False, no_query_filter = False, nowsu = True)
    Formulation_comp = Query_formulation_CCQF(parameters = (a1,a2,a3,bigram_scoring_parameter,6,math.log(0.000000001), 1), bernoulli = False, collection_lm = doc_collection_lm_dist, topic_bi_probs = (topic_bigram_ct, topic_unigram_ct, topic_bigram_prob), topic_descs = topic_descs, recall_combine = False)
    #Formulation_comp = Query_formulation_QS3plus(collection_lm = doc_collection_lm_dist, total_num_words = total_noof_words)
    #Reformulation_comp = Reformulate_no_reformulation()
    result_judgement_component = Result_judgement_qrels(collection_lm = doc_collection_lm_dist, document_rels_all_topics = topic_rel_docs)
    user_models = {} 
    candidate_queries_topics = {}
    queries ={}
    topic_query_ids = {}
    for topic_num in topic_descs:
        sim_session = Session(topic_num = topic_num)
        user_models[topic_num] = {}
        user_models[topic_num]["user_model"] = User_model(parameters, topic_num, topic_rel_docs[topic_num], dataset, Formulation_comp, Reformulation_comp, result_judgement_component)
        candidate_queries_topics[topic_num] = user_models[topic_num]["user_model"].candidate_queries[:]
        topic_query_ids[topic_num] = []
        for i in range(5):
            queryid = str(topic_num) + str(i)
            topic_query_ids[topic_num] += [queryid]
            queries[queryid] = " ".join(candidate_queries_topics[topic_num][i][0])
    all_keys = list(queries.keys())
    times = int(len(all_keys)/50) + 1
    
    all_formatted_results = {}
    print ("TIMES: ", times)
    for i in range(times):
        new_queries = {}
        for queryid in all_keys[i*50:(i*50)+50]:
            new_queries[queryid] = queries[queryid]
        inputname = os.path.join(save_dir, method + "queries_input")
        outputname = os.path.join(save_dir, method + "results_output")
        make_runquery_file3(new_queries, inputname, snippets = False)
        start_time = time.time()
        os.system("../../indri-5.14/runquery/IndriRunQuery " + inputname + "> " + outputname)
        print ('TIME TAKEN: ', time.time()-start_time)
        all_formatted_results_i = format_batch_results_trec_format(outputname)
        for queryid in all_formatted_results_i:
            all_formatted_results[queryid] = all_formatted_results_i[queryid]
    pickle.dump(all_formatted_results, open(os.path.join(save_dir, method + "_results.pk") , "wb"))
    #all_formatted_results = pickle.load(open(os.path.join(save_dir, method + "_results.pk") , "rb"))
    session_ops = Session_operations()
    act_sessions = session_ops.get_actual_sessions(dataset)
    queries_list_act_topics = session_ops.get_all_queries(act_sessions, dopreprocess = True)
    queries_list_act_topics_set = {}
    for topic_num in queries_list_act_topics:
        queries_list_act_topics_set[topic_num] = list(set([" ".join(query) for query in queries_list_act_topics[topic_num]]))
    '''
    query_results = {}
    act_formatted_results = {}
    for topic_num in queries_list_act_topics:
        act_formatted_results[topic_num] = {}
        actual_queries = {}
        for idx,query in enumerate(queries_list_act_topics_set[topic_num]):
            actual_queries[str(idx)] = query
        inputname = os.path.join(save_dir, method + "queries_input")
        outputname = os.path.join(save_dir, method + "results_output")
        make_runquery_file2(actual_queries, inputname, snippets = False)
        start_time = time.time()
        os.system("../../indri-5.14/runquery/IndriRunQuery " + inputname + "> " + outputname)
        print ('TIME TAKEN: ', time.time()-start_time)
        all_act_formatted_results = format_batch_results_trec_format(outputname)
        for idx,query in enumerate(queries_list_act_topics_set[topic_num]):
            try:
                act_formatted_results[topic_num][query] = all_act_formatted_results[str(idx)]
            except:
                print ("MISSED QUERIES: ", topic_num, query)
                act_formatted_results[topic_num][query] = []
    '''
    #pickle.dump(act_formatted_results, open(os.path.join(save_dir, "actual_queries_retrieval_results.pk") , "wb") )
    act_formatted_results = pickle.load(open(os.path.join(save_dir, "actual_queries_retrieval_results_2.pk") , "rb") )
    ev = Evaluation_queries(act_sessions, None, None, dataset = dataset)
    topic_ndcgs = {}
    for topic_num in topic_rel_docs:
        ndcgs = []
        ndcgs50 = []
        for i in range(5):
            queryid = str(topic_num) + str(i)
            results = all_formatted_results[queryid]
            ndcgs += [ev.NDCG_eval(results, topic_num, 10, cb9_corpus_trec_ids)]
            ndcgs50 += [ev.NDCG_eval(results, topic_num, 50, cb9_corpus_trec_ids)]
        act_ndcgs_2 = []
        act_ndcgs50_2 = []
        for query in queries_list_act_topics[topic_num]:
            act_results_2 = act_formatted_results[topic_num][" ".join(query)]
            act_ndcgs_2 += [ev.NDCG_eval(act_results_2, topic_num, 10, cb9_corpus_trec_ids)]
            act_ndcgs50_2 += [ev.NDCG_eval(act_results_2, topic_num, 50, cb9_corpus_trec_ids)]
        act_ndcgs = ev.QF_NDCG_act_eval(topic_num, 10, cb9_corpus_trec_ids)
        act_ndcgs50 = ev.QF_NDCG_act_eval(topic_num, 50, cb9_corpus_trec_ids)
        topic_ndcgs[topic_num] = [float(sum(ndcgs))/float(len(ndcgs)), float(sum(act_ndcgs))/float(len(act_ndcgs)), float(sum(act_ndcgs_2))/float(len(act_ndcgs_2)), float(sum(ndcgs50))/float(len(ndcgs50)), float(sum(act_ndcgs50))/float(len(act_ndcgs50)), float(sum(act_ndcgs50_2))/float(len(act_ndcgs50_2)), ndcgs, act_ndcgs, act_ndcgs_2]
    act_ndcgs = sum([topic_ndcgs[topic_num][1] for topic_num in topic_ndcgs])/len(list(topic_ndcgs.keys()))
    act_ndcgs_2 = sum([topic_ndcgs[topic_num][2] for topic_num in topic_ndcgs])/len(list(topic_ndcgs.keys()))
    sim_ndcgs = sum([topic_ndcgs[topic_num][0] for topic_num in topic_ndcgs])/len(list(topic_ndcgs.keys()))
    sim_ndcgs50 = sum([topic_ndcgs[topic_num][3] for topic_num in topic_ndcgs])/len(list(topic_ndcgs.keys()))
    act_ndcgs50 = sum([topic_ndcgs[topic_num][4] for topic_num in topic_ndcgs])/len(list(topic_ndcgs.keys()))
    act_ndcgs50_2 = sum([topic_ndcgs[topic_num][5] for topic_num in topic_ndcgs])/len(list(topic_ndcgs.keys()))
    pickle.dump(topic_ndcgs, open(os.path.join(save_dir, method + "_topic_ndcgs.pk") , "wb"))
    print ("SIM NDCG: ", sim_ndcgs)
    print ("ACT NDCG: ", act_ndcgs)
    print ("ACT NDCG 2: ", act_ndcgs_2)
    print ("SIM NDCG50: ", sim_ndcgs50)
    print ("ACT NDCG50: ", act_ndcgs50)
    print ("ACT NDCG50 2: ", act_ndcgs50_2)
    

def query_similarity_controlled_q1(p1,p2, dataset):
    '''
    BETTER USEFUL
    To evaluate Query reformulation
    Get all reformulated pairs from the Session track sessions, for a q1, use its results and clicks to simulate the next query,
    Compare the simulated reformulation queries with actual reformulation query q2 - interms of jaccard similarity.
    :param p1: reformulation parameter p1
    :param p2: reformulation parameter p2
    :return:
    '''
    print ("DOING query_reformulation_controlled_q1")
    sample_index = pyndri.Index('../../ClueWeb09_catB_sample_index')
    token2id, id2token, id2df = sample_index.get_dictionary()
    id2tf = sample_index.get_term_frequencies()
    doc_collection_lm_dist = {}
    for tokenid in id2tf:
        doc_collection_lm_dist[id2token[tokenid]] = id2tf[tokenid]
    total_noof_words = float(sum(doc_collection_lm_dist.values()))
    doc_collection_lm_dist = {token: float(doc_collection_lm_dist[token])/float(total_noof_words) for token in doc_collection_lm_dist}
    doc_collection_lm_binary_dist = {}
    for tokenid in id2df:
        doc_collection_lm_binary_dist[id2token[tokenid]] = id2df[tokenid]
    total_noof_docs = float(max(doc_collection_lm_dist.values()))
    doc_collection_lm_binary_dist = {token: float(doc_collection_lm_binary_dist[token])/float(total_noof_docs) for token in doc_collection_lm_binary_dist}
    
    dataset = dataset
    session_dir = "../simulated_sessions/" 
    if not os.path.exists(session_dir): os.makedirs(session_dir)
    save_dir = os.path.join(session_dir, dataset)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    #method = 'CCQF_' + str(a1) + '_' + str(a2) + '_' + str(a3) + '_' + str(bigram_scoring_parameter) + '_' + 'noef'
    bigram_scoring_parameter = 2
    #a1,a2,a3 = 0.125,0.645,0.23
    a1,a2,a3 = 0.1,0.765,0.135
    r1 = p1
    r2 = p2
    effort = 6
    w_len = 1
    method = 'CCQF_' + str(a1) + '_' + str(a2) + '_' + str(a3) + '_' + str(bigram_scoring_parameter) + '_noef_cont_reform' + str(r1) + '_' + str(r2)
    print ('METHOD: ', method)
    sys.stdout = open(os.path.join(save_dir, method + '_big_idx_out_qs.txt'), 'w')
    topic_descs = read_topic_descs(dataset)
    topic_rel_docs = read_judgements(dataset)
    (topic_bigram_ct, topic_unigram_ct, topic_bigram_prob) = read_bigram_topic_lm(dataset)
    Reformulation_comp =  Reformulation_component_multinomial(parameters = (r1,r2), collection_lm = doc_collection_lm_dist, print_info = True, type2_update= False, no_query_filter = False, nowsu = True)
    Formulation_comp = Query_formulation_CCQF(parameters = (a1,a2,a3,bigram_scoring_parameter, effort,math.log(0.000000001), w_len), bernoulli = False, collection_lm = doc_collection_lm_dist, topic_bi_probs = (topic_bigram_ct, topic_unigram_ct, topic_bigram_prob), topic_descs = topic_descs)
    #Formulation_comp = Query_formulation_QS3plus(collection_lm = doc_collection_lm_dist, total_num_words = total_noof_words)
    #Reformulation_comp = Reformulate_no_reformulation()
    #result_judgement_component = Result_judgement_qrels(collection_lm = doc_collection_lm_dist, document_rels_all_topics = topic_rel_docs)
    result_judgement_component = Result_judgement_clicks(collection_lm = doc_collection_lm_dist, document_rels_all_topics = topic_rel_docs)
    Reformulation_comp.result_judgement = result_judgement_component
    Reformulation_comp.query_formulation = Formulation_comp
        
    session_ops = Session_operations()
    act_sessions = session_ops.get_actual_sessions(dataset)
    act_session_topics = session_ops.get_session_topics(act_sessions)
    reformulated_query_info = {}
    for topic_num in act_session_topics:
        reformulated_query_info[topic_num] = []
        for session in act_session_topics[topic_num]:
            for idx,inte in enumerate(session.interactions):
                if (inte.type == "reformulate") and (idx!=0):
                    q1 = preprocess(" ".join(session.interactions[idx-1].query), lemmatizing = True).split()
                    q2 = preprocess(" ".join(inte.query), lemmatizing = True).split()
                    if (" ".join(q1) != " ".join(q2)):
                        print ('coming here', q1,q2, session.interactions[idx-1].clicks, inte.clicks)
                        reformulated_query_info[topic_num] += [(q1, session.interactions[idx-1].results, session.interactions[idx-1].clicks, q2, inte.results, inte.clicks)]

    all_pairs = 0
    for topic_num in reformulated_query_info:
        all_pairs += len(reformulated_query_info[topic_num])
    print ("ALL PAIRS: ",  all_pairs)
    topic_ndcgs = {}
    ev = Evaluation_queries(act_sessions, None, None, dataset = dataset)
    all_reformulated_queries = {}
    if (os.path.isfile(os.path.join(save_dir, method + "_sim_reform_queries.pk"))):
        all_reformulated_queries = pickle.load(open(os.path.join(save_dir, method + "_sim_reform_queries.pk") , "rb"))
    for topic_num in reformulated_query_info:
        topic_desc = topic_descs[topic_num]
        topic_desc = preprocess(topic_desc, stopwords_file, lemmatizing = True)
        topic_IN = language_model_m(topic_desc)
        Formulation_comp.topic_INs[topic_num] = topic_IN
        if reformulated_query_info[topic_num] == []:
            continue
        if (not os.path.isfile(os.path.join(save_dir, method + "_sim_reform_queries.pk"))):
            reformulated_queries = {}
            actual_reform_queries = {}    
            for idx,info in enumerate(reformulated_query_info[topic_num]):
                (q1, previous_results, previous_clicks, q2, reform_results, reform_clicks) = info
                previous_queries = {}
                s = q1[:]
                s.sort()
                previous_queries[" ".join(s)] = 1
                candidate_queries, updated_topic_IN, updated_precision_lm = Reformulation_comp.reformulate_query(q1, {}, reformulated_query_info[topic_num][idx][1], reformulated_query_info[topic_num][idx][2], topic_num, topic_IN, None, previous_queries)
                queryid = str(idx)
                reformulated_queries[queryid] = " ".join(candidate_queries[0][0])
                actual_reform_queries[queryid] = " ".join(q2)
            all_reformulated_queries[topic_num] = reformulated_queries
        jaccard_similarities = []
        if topic_num in topic_rel_docs:
            for idx,info in enumerate(reformulated_query_info[topic_num]):
                (q1, previous_results, previous_clicks, q2, reform_results, reform_clicks) = info
                jaccard_similarities += [ev.jaccard_similarity([q2], [all_reformulated_queries[topic_num][str(idx)].split()])[0]]
                #reformulated_query_info[topic_num][idx] = (q1, previous_results, previous_clicks, q2, reform_results, reform_clicks, all_formatted_act_results[str(idx)])
        topic_ndcgs[topic_num] = [float(sum(jaccard_similarities))/float(len(jaccard_similarities)), jaccard_similarities]
    similarities = sum([topic_ndcgs[topic_num][0] for topic_num in topic_ndcgs])/len(list(topic_ndcgs.keys()))
    #pickle.dump(reformulated_query_info, open(os.path.join(save_dir, "reformulated_query_info.pk") , "wb"))
    if (not os.path.isfile(os.path.join(save_dir, method + "_sim_reform_queries.pk"))):
        pickle.dump(all_reformulated_queries, open(os.path.join(save_dir, method + "_sim_reform_queries.pk") , "wb"))
    
    #pickle.dump(topic_ndcgs, open(os.path.join(save_dir, method + "_topic_ndcgs.pk") , "wb"))
    for topic_num in topic_ndcgs:
        print ("TOPIC NUM: ", topic_num)
        print ("SIMILARITIES: ", topic_ndcgs[topic_num][-1])
    print ("QUERY SIMILARITIES: ", similarities)
    sys.stdout = old_stdout_target
    return similarities

def query_reformulation_controlled_q1(p1, p2):
    '''
    To evaluate Query reformulation
    Get all reformulated pairs from the Session track sessions, for a q1, use its results and clicks to simulate the next query,
    compare this simulated reformulation query with actual reformulation query q2 - interms of NDCG of search results, jaccard similarity.
    :param p1: reformulation parameter p1
    :param p2: reformulation parameter p2
    :return:
    '''
    print ("DOING query_reformulation_controlled_q1")
    sample_index = pyndri.Index('../../ClueWeb09_catB_sample_index')
    token2id, id2token, id2df = sample_index.get_dictionary()
    id2tf = sample_index.get_term_frequencies()
    doc_collection_lm_dist = {}
    for tokenid in id2tf:
        doc_collection_lm_dist[id2token[tokenid]] = id2tf[tokenid]
    total_noof_words = float(sum(doc_collection_lm_dist.values()))
    doc_collection_lm_dist = {token: float(doc_collection_lm_dist[token])/float(total_noof_words) for token in doc_collection_lm_dist}
    doc_collection_lm_binary_dist = {}
    for tokenid in id2df:
        doc_collection_lm_binary_dist[id2token[tokenid]] = id2df[tokenid]
    total_noof_docs = float(max(doc_collection_lm_dist.values()))
    doc_collection_lm_binary_dist = {token: float(doc_collection_lm_binary_dist[token])/float(total_noof_docs) for token in doc_collection_lm_binary_dist}
    
    dataset = "Session_track_2012"
    session_dir = "../simulated_sessions/" 
    if not os.path.exists(session_dir): os.makedirs(session_dir)
    save_dir = os.path.join(session_dir, "Session_track_2012")
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    #method = 'CCQF_' + str(a1) + '_' + str(a2) + '_' + str(a3) + '_' + str(bigram_scoring_parameter) + '_' + 'noef'
    bigram_scoring_parameter = 2
    #a1,a2,a3 = 0.125,0.645,0.23
    a1,a2,a3 = 0.1,0.765,0.135
    r1 = p1
    r2 = p2
    effort = 6
    w_len = 1
    method = 'CCQF_' + str(a1) + '_' + str(a2) + '_' + str(a3) + '_' + str(bigram_scoring_parameter) + '_noef_cont_reform_' + str(r1) + '_' + str(r2)
    print ('METHOD: ', method)
    sys.stdout = open(os.path.join(save_dir, method + '_big_idx_out_2.txt'), 'w')
    topic_descs = read_topic_descs(dataset)
    topic_rel_docs = read_judgements(dataset)
    (topic_bigram_ct, topic_unigram_ct, topic_bigram_prob) = read_bigram_topic_lm(dataset)
    Reformulation_comp =  Reformulation_component_multinomial(parameters = (r1,r2), collection_lm = doc_collection_lm_dist, print_info = True, type2_update= False, no_query_filter = False, nowsu = True)
    Formulation_comp = Query_formulation_CCQF(parameters = (a1,a2,a3,bigram_scoring_parameter, effort,math.log(0.000000001), w_len), bernoulli = False, collection_lm = doc_collection_lm_dist, topic_bi_probs = (topic_bigram_ct, topic_unigram_ct, topic_bigram_prob), topic_descs = topic_descs)
    #Formulation_comp = Query_formulation_QS3plus(collection_lm = doc_collection_lm_dist, total_num_words = total_noof_words)
    #Reformulation_comp = Reformulate_no_reformulation()
    #result_judgement_component = Result_judgement_qrels(collection_lm = doc_collection_lm_dist, document_rels_all_topics = topic_rel_docs)
    result_judgement_component = Result_judgement_clicks(collection_lm = doc_collection_lm_dist, document_rels_all_topics = topic_rel_docs)
    Reformulation_comp.result_judgement = result_judgement_component
    Reformulation_comp.query_formulation = Formulation_comp
        
    session_ops = Session_operations()
    act_sessions = session_ops.get_actual_sessions(dataset)
    act_session_topics = session_ops.get_session_topics(act_sessions)
    reformulated_query_info_2 = {}
    for topic_num in act_session_topics:
        reformulated_query_info_2[topic_num] = []
        for session in act_session_topics[topic_num]:
            for idx,inte in enumerate(session.interactions):
                if (inte.type == "reformulate") and (idx!=0):
                    q1 = preprocess(" ".join(session.interactions[idx-1].query), lemmatizing = True).split()
                    q2 = preprocess(" ".join(inte.query), lemmatizing = True).split()
                    if (" ".join(q1) != " ".join(q2)):
                        print ('coming here', q1,q2, session.interactions[idx-1].clicks, inte.clicks)
                        reformulated_query_info_2[topic_num] += [(q1, session.interactions[idx-1].results, session.interactions[idx-1].clicks, q2, inte.results, inte.clicks)]

    reformulated_query_info = pickle.load(open(os.path.join(save_dir, "reformulated_query_info_2.pk") , "rb"))
    
    all_pairs = 0
    for topic_num in reformulated_query_info:
        all_pairs += len(reformulated_query_info[topic_num])
    print ("ALL PAIRS: ",  all_pairs)
    topic_ndcgs = {}
    ev = Evaluation_queries(act_sessions, None, None, dataset = dataset)
    simulated_query_results = {}
    all_reformulated_queries = {}
    if os.path.isfile(os.path.join(save_dir, method + "_sim_q_results_2.pk")):
        simulated_query_results = pickle.load(open(os.path.join(save_dir, method + "_sim_q_results.pk"), "rb")) 
    if (os.path.isfile(os.path.join(save_dir, method + "_sim_reform_queries.pk"))):
        all_reformulated_queries = pickle.load(open(os.path.join(save_dir, method + "_sim_reform_queries.pk") , "rb"))
    for topic_num in reformulated_query_info:
        topic_desc = topic_descs[topic_num]
        topic_desc = preprocess(topic_desc, stopwords_file, lemmatizing = True)
        topic_IN = language_model_m(topic_desc)
        Formulation_comp.topic_INs[topic_num] = topic_IN
        if reformulated_query_info[topic_num] == []:
            continue
        if (not os.path.isfile(os.path.join(save_dir, method + "_sim_reform_queries.pk"))):
            reformulated_queries = {}
            actual_reform_queries = {}    
            for idx,info in enumerate(reformulated_query_info[topic_num]):
                (q1, previous_results, previous_clicks, q2, reform_results, reform_clicks, _) = info
                previous_queries = {}
                s = q1[:]
                s.sort()
                previous_queries[" ".join(s)] = 1
                candidate_queries, updated_topic_IN, updated_precision_lm = Reformulation_comp.reformulate_query(q1, {}, reformulated_query_info_2[topic_num][idx][1], reformulated_query_info_2[topic_num][idx][2], topic_num, topic_IN, None, previous_queries)
                queryid = str(idx)
                reformulated_queries[queryid] = " ".join(candidate_queries[0][0])
                actual_reform_queries[queryid] = " ".join(q2)
            all_reformulated_queries[topic_num] = reformulated_queries
        if (not os.path.isfile(os.path.join(save_dir, method + "_sim_q_results_2.pk"))):
            inputname = os.path.join(save_dir, method + "queries_input")
            outputname = os.path.join(save_dir, method + "results_output")
            make_runquery_file3(all_reformulated_queries[topic_num], inputname, snippets = False)
            start_time = time.time()
            os.system("../../indri-5.14/runquery/IndriRunQuery " + inputname + "> " + outputname)
            print ('TIME TAKEN: ', time.time()-start_time)
            all_formatted_results = format_batch_results_trec_format(outputname)
            simulated_query_results[topic_num] = all_formatted_results
            
        '''
        inputname = os.path.join(save_dir, method + "queries_input")
        outputname = os.path.join(save_dir, method + "results_output")        
        make_runquery_file2(actual_reform_queries, inputname, snippets = False)
        start_time = time.time()
        os.system("../../indri-5.14/runquery/IndriRunQuery " + inputname + "> " + outputname)
        print ('TIME TAKEN: ', time.time()-start_time)
        all_formatted_act_results = format_batch_results_trec_format(outputname)
        '''
        act_ndcgs = []
        sim_ndcgs = []
        act_ndcgs_2 = []
        act_ndcgs50 = []
        sim_ndcgs50 = []
        act_ndcgs50_2 = []
        jaccard_similarities = []
        if topic_num in topic_rel_docs:
            for idx,info in enumerate(reformulated_query_info[topic_num]):
                (q1, previous_results, previous_clicks, q2, reform_results, reform_clicks, all_formatted_act_results) = info
                sim_ndcgs += [ev.NDCG_eval(simulated_query_results[topic_num][str(idx)], topic_num, 10, cb9_corpus_trec_ids)] 
                sim_ndcgs50 += [ev.NDCG_eval(simulated_query_results[topic_num][str(idx)], topic_num, 50, cb9_corpus_trec_ids)] 
                act_ndcgs += [ev.NDCG_eval(reform_results, topic_num, 10, cb9_corpus_trec_ids)]
                act_ndcgs50 += [ev.NDCG_eval(reform_results, topic_num, 50, cb9_corpus_trec_ids)]
                try:
                    #act_ndcgs_2 += [ev.NDCG_eval(all_formatted_act_results[str(idx)], topic_num, 10)]
                    act_ndcgs_2 += [ev.NDCG_eval(all_formatted_act_results, topic_num, 10, cb9_corpus_trec_ids)]
                    act_ndcgs50_2 += [ev.NDCG_eval(all_formatted_act_results, topic_num, 50, cb9_corpus_trec_ids)]
                except KeyError:
                    print ('MISSED QUERIES: ', topic_num, q2)
                    act_ndcgs_2 += [0]
                jaccard_similarities += [ev.jaccard_similarity([q2], [all_reformulated_queries[topic_num][str(idx)].split()])[0]]
                #reformulated_query_info[topic_num][idx] = (q1, previous_results, previous_clicks, q2, reform_results, reform_clicks, all_formatted_act_results[str(idx)])
        topic_ndcgs[topic_num] = [float(sum(sim_ndcgs))/float(len(sim_ndcgs)),float(sum(act_ndcgs))/float(len(act_ndcgs)), float(sum(act_ndcgs_2))/float(len(act_ndcgs_2)), float(sum(jaccard_similarities))/float(len(jaccard_similarities)), float(sum(sim_ndcgs50))/float(len(sim_ndcgs50)),float(sum(act_ndcgs50))/float(len(act_ndcgs50)), float(sum(act_ndcgs50_2))/float(len(act_ndcgs50_2)), sim_ndcgs, act_ndcgs, act_ndcgs_2, jaccard_similarities]
    sim_ndcgs = sum([topic_ndcgs[topic_num][0] for topic_num in topic_ndcgs])/len(list(topic_ndcgs.keys()))
    act_ndcgs = sum([topic_ndcgs[topic_num][1] for topic_num in topic_ndcgs])/len(list(topic_ndcgs.keys()))
    act_ndcgs_2 = sum([topic_ndcgs[topic_num][2] for topic_num in topic_ndcgs])/len(list(topic_ndcgs.keys()))
    sim_ndcgs50 = sum([topic_ndcgs[topic_num][4] for topic_num in topic_ndcgs])/len(list(topic_ndcgs.keys()))
    act_ndcgs50 = sum([topic_ndcgs[topic_num][5] for topic_num in topic_ndcgs])/len(list(topic_ndcgs.keys()))
    act_ndcgs50_2 = sum([topic_ndcgs[topic_num][6] for topic_num in topic_ndcgs])/len(list(topic_ndcgs.keys()))
    similarities = sum([topic_ndcgs[topic_num][3] for topic_num in topic_ndcgs])/len(list(topic_ndcgs.keys()))
    #pickle.dump(reformulated_query_info, open(os.path.join(save_dir, "reformulated_query_info.pk") , "wb"))
    if (not os.path.isfile(os.path.join(save_dir, method + "_sim_q_results_2.pk"))):
        pickle.dump(simulated_query_results, open(os.path.join(save_dir, method + "_sim_q_results.pk") , "wb"))
    if (not os.path.isfile(os.path.join(save_dir, method + "_sim_reform_queries.pk"))):
        pickle.dump(all_reformulated_queries, open(os.path.join(save_dir, method + "_sim_reform_queries.pk") , "wb"))
    
    #pickle.dump(topic_ndcgs, open(os.path.join(save_dir, method + "_topic_ndcgs.pk") , "wb"))
    for topic_num in topic_ndcgs:
        print ("TOPIC NUM: ", topic_num)
        print ("SIM NDCG: ", topic_ndcgs[topic_num][-4])
        print ("ACT NDCG: ", topic_ndcgs[topic_num][-3])
        print ("ACT NDCG 2: ", topic_ndcgs[topic_num][-2])
        print ("SIMILARITIES: ", topic_ndcgs[topic_num][-1])
    print ("ACT NDCG: ", act_ndcgs)
    print ("SIM NDCG: ", sim_ndcgs)
    print ("ACT NDCG 2: ", act_ndcgs_2)
    print ("ACT NDCG50: ", act_ndcgs50)
    print ("SIM NDCG50: ", sim_ndcgs50)
    print ("ACT NDCG50 2: ", act_ndcgs50_2)
    print ("QUERY SIMILARITIES: ", similarities)
    sys.stdout = old_stdout_target
    return (act_ndcgs, sim_ndcgs, act_ndcgs_2, act_ndcgs50, sim_ndcgs50, act_ndcgs50_2, similarities)


def actual_reform_queries_performance():
    '''
    Getting all reformulated query pairs from Session track sessions, if  (q1,q2) is a pair runnning all q2s with the index, computing NDCG of the search results from index, search results from the sessions data
    :return:
    '''
    print ("DOING actual reform queries performance")
    dataset = "Session_track_2012"
    session_dir = "../simulated_sessions/" 
    if not os.path.exists(session_dir): os.makedirs(session_dir)
    save_dir = os.path.join(session_dir, "Session_track_2012")
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    sys.stdout = open(os.path.join(save_dir, 'act_reform_q_results_2.txt'), 'w')
    topic_descs = read_topic_descs(dataset)
    topic_rel_docs = read_judgements(dataset)
    session_ops = Session_operations()
    act_sessions = session_ops.get_actual_sessions(dataset)
    act_session_topics = session_ops.get_session_topics(act_sessions)
    reformulated_query_info = {}
    for topic_num in act_session_topics:
        reformulated_query_info[topic_num] = []
        for session in act_session_topics[topic_num]:
            for idx,inte in enumerate(session.interactions):
                if (inte.type == "reformulate") and (idx!=0):
                    q1 = preprocess(" ".join(session.interactions[idx-1].query), lemmatizing = True).split()
                    q2 = preprocess(" ".join(inte.query), lemmatizing = True).split()
                    if (" ".join(q1) != " ".join(q2)):
                        print ('coming here', q1,q2, session.interactions[idx-1].clicks, inte.clicks)
                        reformulated_query_info[topic_num] += [(q1, session.interactions[idx-1].results, session.interactions[idx-1].clicks, q2, inte.results, inte.clicks)]

    #reformulated_query_info = pickle.load(open(os.path.join(save_dir, "reformulated_query_info.pk") , "rb"))
    
    all_pairs = 0
    for topic_num in reformulated_query_info:
        all_pairs += len(reformulated_query_info[topic_num])
    print ("ALL PAIRS: ",  all_pairs)
    topic_ndcgs = {}

    ev = Evaluation_queries(act_sessions, None, None, dataset = dataset)
    for topic_num in reformulated_query_info:
        actual_reform_queries = {}
        if reformulated_query_info[topic_num] == []:
            continue    
        for idx,info in enumerate(reformulated_query_info[topic_num]):
            (q1, previous_results, previous_clicks, q2, reform_results, reform_clicks) = info
            queryid = str(idx)
            actual_reform_queries[queryid] = " ".join(q2)
        inputname = os.path.join(save_dir, "act_reform_q_results" + "queries_input")
        outputname = os.path.join(save_dir, "act_reform_q_results" + "results_output")        
        make_runquery_file3(actual_reform_queries, inputname, snippets = False)
        start_time = time.time()
        os.system("../../indri-5.14/runquery/IndriRunQuery " + inputname + "> " + outputname)
        print ('TIME TAKEN: ', time.time()-start_time)
        all_formatted_act_results = format_batch_results_trec_format(outputname)

        act_ndcgs = []
        act_ndcgs_2 = []
        act_ndcgs50 = []
        act_ndcgs50_2 = []
        if topic_num in topic_rel_docs:
            for idx,info in enumerate(reformulated_query_info[topic_num]):
                (q1, previous_results, previous_clicks, q2, reform_results, reform_clicks) = info
                act_ndcgs += [ev.NDCG_eval(reform_results, topic_num, 10, cb9_corpus_trec_ids)]
                act_ndcgs50 += [ev.NDCG_eval(reform_results, topic_num, 50, cb9_corpus_trec_ids)]
                try:
                    act_ndcgs_2 += [ev.NDCG_eval(all_formatted_act_results[str(idx)], topic_num, 10)]
                    act_ndcgs50_2 += [ev.NDCG_eval(all_formatted_act_results[str(idx)], topic_num, 50)]
                    #act_ndcgs_2 += [ev.NDCG_eval(all_formatted_act_results[str(idx)], topic_num, 10, cb9_corpus_trec_ids)]
                    #act_ndcgs50_2 += [ev.NDCG_eval(all_formatted_act_results[str(idx)], topic_num, 50, cb9_corpus_trec_ids)]
                except KeyError:
                    print ('MISSED QUERIES: ', topic_num, q2)
                    act_ndcgs_2 += [0]
                reformulated_query_info[topic_num][idx] = (q1, previous_results, previous_clicks, q2, reform_results, reform_clicks, all_formatted_act_results[str(idx)])
        print ("TOPIC NUM: ", topic_num)
        print ("ACT NDCG  ", float(sum(act_ndcgs))/float(len(act_ndcgs)), act_ndcgs)
        print ("ACT NDCG50  ", float(sum(act_ndcgs50))/float(len(act_ndcgs50)))
        print ("ACT NDCG 2 ", float(sum(act_ndcgs_2))/float(len(act_ndcgs_2)), act_ndcgs_2)
        print ("ACT NDCG50 2 ", float(sum(act_ndcgs50_2))/float(len(act_ndcgs50_2)))
        topic_ndcgs[topic_num] = [float(sum(act_ndcgs))/float(len(act_ndcgs)), float(sum(act_ndcgs_2))/float(len(act_ndcgs_2)),float(sum(act_ndcgs50))/float(len(act_ndcgs50)), float(sum(act_ndcgs50_2))/float(len(act_ndcgs50_2)), act_ndcgs, act_ndcgs_2]
    pickle.dump(reformulated_query_info, open(os.path.join(save_dir, "reformulated_query_info_3.pk") , "wb"))
    act_ndcgs = sum([topic_ndcgs[topic_num][0] for topic_num in topic_ndcgs])/len(list(topic_ndcgs.keys()))
    act_ndcgs_2 = sum([topic_ndcgs[topic_num][1] for topic_num in topic_ndcgs])/len(list(topic_ndcgs.keys()))
    act_ndcgs50 = sum([topic_ndcgs[topic_num][2] for topic_num in topic_ndcgs])/len(list(topic_ndcgs.keys()))
    act_ndcgs50_2 = sum([topic_ndcgs[topic_num][3] for topic_num in topic_ndcgs])/len(list(topic_ndcgs.keys()))
    for topic_num in topic_ndcgs:
        print ("TOPIC NUM: ", topic_num)
        print ("ACT NDCG: ", topic_ndcgs[topic_num][-2])
        print ("ACT NDCG 2: ", topic_ndcgs[topic_num][-1])
    print ("ACT NDCG: ", act_ndcgs)
    print ("ACT NDCG 2: ", act_ndcgs_2)
    print ("ACT NDCG50: ", act_ndcgs50)
    print ("ACT NDCG50 2: ", act_ndcgs50_2)
    sys.stdout = old_stdout_target
    return


def simulate_batch_queries_1_reformulate(dataset, alpha_parameters, index_path, setting):
    '''
    LATEST METHOD TO SIMULATE SESSIONS
    Generate simulated sessions - initial query and reformulation and run queries of each iteration together using the index.
    :final : dumping simulated sessions and candidate queries
    '''
    # reference model
    '''
    sample_index = pyndri.Index('../../ClueWeb09_catB_sample_index')
    token2id, id2token, id2df = sample_index.get_dictionary()
    id2tf = sample_index.get_term_frequencies()
    doc_collection_lm_dist = {}
    for tokenid in id2tf:
        doc_collection_lm_dist[id2token[tokenid]] = id2tf[tokenid]
    total_noof_words = float(sum(doc_collection_lm_dist.values()))
    doc_collection_lm_dist = {token: float(doc_collection_lm_dist[token]) / float(total_noof_words) for token in
                              doc_collection_lm_dist}
    doc_collection_lm_binary_dist = {}
    for tokenid in id2df:
        doc_collection_lm_binary_dist[id2token[tokenid]] = id2df[tokenid]
    total_noof_docs = float(max(doc_collection_lm_dist.values()))
    doc_collection_lm_binary_dist = {token: float(doc_collection_lm_binary_dist[token]) / float(total_noof_docs) for
                                     token in doc_collection_lm_binary_dist}
    '''
    doc_collection_frequencies = {}
    with open("../unigram_freq.csv", "r") as infile:
        i = 0
        for line in infile:
            if i == 0:
                i = i + 1
                continue
            doc_collection_frequencies[line.strip().split(",")[0]] = float(line.strip().split(",")[1])
    total_noof_words = float(sum(doc_collection_frequencies.values()))
    #print(len(doc_collection_frequencies), doc_collection_frequencies["the"])
    doc_collection_lm_dist = {token: float(doc_collection_frequencies[token]) / float(total_noof_words) for token in
                              doc_collection_frequencies}
    #print(len(doc_collection_lm_dist), doc_collection_lm_dist["the"], min(list(doc_collection_lm_dist.values())))

    '''
    (doc_collection_lm, doc_collection_lm_binary) = pickle.load(open("../TREC_Robust_data/all_doc_language_model.pk", "rb"))
    total_noof_words = sum(doc_collection_lm.values())
    doc_collection_lm_dist = {term:float(doc_collection_lm[term])/float(total_noof_words) for term in doc_collection_lm}
    '''

    session_dir = "../simulated_sessions/"
    if not os.path.exists(session_dir): os.makedirs(session_dir)
    data_dir = os.path.join(session_dir, dataset)
    if not os.path.exists(data_dir): os.makedirs(data_dir)
    # method = 'CCQF_' + str(a1) + '_' + str(a2) + '_' + str(a3) + '_' + str(bigram_scoring_parameter) + '_' + 'noef'
    do_reformulation = True
    # a1,a2,a3 = 0.125,0.645,0.23
    #r1,r2 = 0.9, 0.9
    #a1, a2, a3 =  alpha_parameters[0], alpha_parameters[1], alpha_parameters[2]#BEST PARAMETERS
    alpha, beta = alpha_parameters[0], alpha_parameters[1]
    effort = 6
    min_threshold = math.log(0.0000001)
    w_len = 1
    setting_string = "_".join(setting)
    #setting = "old_r_old_p"
    if w_len == 1:
        save_dir = os.path.join(data_dir, 'fit_3_parameters_PRE_' + setting_string + '_ef6_100_pts_w_len')
    else:
        save_dir = os.path.join(data_dir, 'fit_3_parameters_PRE_' + setting_string + '_noef_100_pts_wo_len')
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    print("SAVE DIR: ", save_dir)
    method = 'CCQF_' + str(alpha) + '_' + str(beta) + '_ef6_basic_reform_lbda_0.5'
    #method = 'CCQF_' + str(a1) + '_' + str(a2) + '_' + str(a3) + '_noef_nowsu_type2_update'
    print('METHOD: ', method)
    topic_descs = read_topic_descs(dataset)
    topic_rel_docs = read_judgements(dataset)
    (topic_bigram_ct, topic_unigram_ct, topic_bigram_prob) = read_bigram_topic_lm(dataset)
    '''
    Reformulation_comp = Reformulation_component_multinomial(parameters=(r1, r2), collection_lm=doc_collection_lm_dist,
                                                             print_info=False, type2_update=True, no_query_filter=False,
                                                             nowsu=True)
    Formulation_comp = Query_formulation_CCQF_new(parameters=(a1, a2, a3, effort, min_threshold, w_len),
                                                  collection_lm=doc_collection_lm_dist,
                                                  topic_bi_probs=(topic_bigram_ct, topic_unigram_ct, topic_bigram_prob),
                                                  topic_descs=topic_descs, print_info=False, setting=setting)
    '''
    Reformulation_comp = Reformulation_component_multinomial(collection_lm=doc_collection_frequencies, topic_descs = topic_descs,
                                                             print_info=False, type2_update=True, no_query_filter=False,
                                                             nowsu=True, collection_num_words = total_noof_words)
    Formulation_comp = Query_formulation_PRE(parameters = (alpha, beta, effort, min_threshold, w_len), collection_lm = doc_collection_lm_dist, topic_bi_probs = (topic_bigram_ct, topic_unigram_ct, topic_bigram_prob), topic_descs = topic_descs, print_info = False, setting = setting)

    # Formulation_comp = Query_formulation_QS3plus(collection_lm = doc_collection_lm_dist, total_num_words = total_noof_words)
    result_judgement_component = Result_judgement_qrels(collection_lm=doc_collection_lm_dist,
                                                        document_rels_all_topics=topic_rel_docs)
    do_reformulation = True
    user_model_parameters = []
    user_model_parameters += [do_reformulation]
    topic_desc_rel_docs = set(topic_descs.keys()).intersection(set(topic_rel_docs.keys()))
    topic_num_nums = [int(s) for s in topic_desc_rel_docs]
    topic_num_nums.sort()
    topic_num_nums = [str(s) for s in topic_num_nums]
    print(topic_num_nums)

    original_stdout = sys.stdout
    sys.stdout = open(os.path.join(save_dir, method + '_big_idx_out.txt'), 'w')
    user_models = {}
    candidate_queries_topics = {}
    queries = {}
    for topic_num in topic_num_nums:
        sim_session = Session(topic_num=topic_num)
        user_models[topic_num] = {}
        user_models[topic_num]["user_model"] = User_model(user_model_parameters, topic_num, topic_rel_docs[topic_num], dataset,
                                                          Formulation_comp, Reformulation_comp,
                                                          result_judgement_component)
        candidate_queries_topics[topic_num] = user_models[topic_num]["user_model"].candidate_queries[:]
        query = user_models[topic_num]["user_model"].first_interaction()
        # user_models[topic_num]["sessions"] = [sim_session]
        user_models[topic_num]["sessions"] = [[sim_session, 1, None, 0, query]]
        user_models[topic_num]["end"] = False
        queries[topic_num] = query
    inputname = os.path.join(save_dir, method + "_queries_input")
    outputname = os.path.join(save_dir, method + "_results_output")
    make_runquery_file2(queries, inputname, index_path, count =50)
    start_time = time.time()
    os.system("../../indri-5.14/runquery/IndriRunQuery " + inputname + "> " + outputname)
    print('TIME TAKEN: ', time.time() - start_time)
    all_formatted_results = format_batch_results_trec_format(outputname)
    
    pickle.dump(all_formatted_results, open(os.path.join(save_dir, method + "_first_query_results.p"), "wb"))
    pickle.dump(user_models, open(os.path.join(save_dir, method + "_first_query_user_models.p"), "wb"))

    sys.stdout = open(os.path.join(save_dir, method + '_big_idx_out_reform_queries.txt'), 'w')
    user_models = pickle.load(open(os.path.join(save_dir, method + "_first_query_user_models.p"), "rb"))
    all_formatted_results = pickle.load(open(os.path.join(save_dir, method + "_first_query_results.p"), "rb"))
    #r1_list = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    #r2_list = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    r1_list = [0.5]
    r2_list = [0.5]
    parameters_list = itertools.product(r1_list, r2_list)
    sample = True

    for iter_idx,(r1, r2) in enumerate(parameters_list):
        method_r1_r2 = method + '_' + str(r1) + '_' + str(r2)
        start_m_time = time.time()
        next_candidate_queries_topics = {}
        for topic_num in topic_num_nums:
            if topic_num in all_formatted_results:
                #print('Topic number: ', topic_num, topic_descs[topic_num])
                formatted_results = all_formatted_results[topic_num][:20]
                start_time = time.time()
                #current_user_model = copy.deepcopy(user_models[topic_num]["user_model"])
                print('TIME TAKEN: ', time.time() - start_time)
                current_user_model = user_models[topic_num]["user_model"]
                #current_user_model.query_reformulation_component.beta = r1
                #current_user_model.query_reformulation_component.gamma = r2
                #if sample:
                #    print (current_user_model.query_reformulation_component.beta, current_user_model.query_reformulation_component.gamma, user_models[topic_num]["user_model"].query_reformulation_component.beta,  user_models[topic_num]["user_model"].query_reformulation_component.gamma )
                #    sample = False
                (next_action_code, next_query, clicked_results, next_candidate_queries) = current_user_model.next_interaction(formatted_results)
                next_candidate_queries = next_candidate_queries
                #print('TIME TAKEN for Reformulation topic: ', time.time() - start_m_time)
                next_candidate_queries_topics[topic_num] = next_candidate_queries[:200]

        print('TIME TAKEN for Reformulation all topics: ', time.time() - start_m_time)

        pickle.dump(next_candidate_queries_topics, open(os.path.join(save_dir, method_r1_r2 + "_reform_queries.pk"), "wb"))
    sys.stdout = original_stdout
def simulate_batch_queries(dataset):
    '''
    Generate simulated sessions - initial query and reformulation and run queries of each iteration together using the index.
    :final : dumping simulated sessions and candidate queries
    '''
    #reference model
    sample_index = pyndri.Index('../../ClueWeb09_catB_sample_index')
    token2id, id2token, id2df = sample_index.get_dictionary()
    id2tf = sample_index.get_term_frequencies()
    doc_collection_lm_dist = {}
    for tokenid in id2tf:
        doc_collection_lm_dist[id2token[tokenid]] = id2tf[tokenid]
    total_noof_words = float(sum(doc_collection_lm_dist.values()))
    doc_collection_lm_dist = {token: float(doc_collection_lm_dist[token])/float(total_noof_words) for token in doc_collection_lm_dist}
    doc_collection_lm_binary_dist = {}
    for tokenid in id2df:
        doc_collection_lm_binary_dist[id2token[tokenid]] = id2df[tokenid]
    total_noof_docs = float(max(doc_collection_lm_dist.values()))
    doc_collection_lm_binary_dist = {token: float(doc_collection_lm_binary_dist[token])/float(total_noof_docs) for token in doc_collection_lm_binary_dist}
    '''
    (doc_collection_lm, doc_collection_lm_binary) = pickle.load(open("../TREC_Robust_data/all_doc_language_model.pk", "rb"))
    total_noof_words = sum(doc_collection_lm.values())
    doc_collection_lm_dist = {term:float(doc_collection_lm[term])/float(total_noof_words) for term in doc_collection_lm}
    '''

    session_dir = "../simulated_sessions/" 
    if not os.path.exists(session_dir): os.makedirs(session_dir)
    save_dir = os.path.join(session_dir, dataset)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    #method = 'CCQF_' + str(a1) + '_' + str(a2) + '_' + str(a3) + '_' + str(bigram_scoring_parameter) + '_' + 'noef'
    do_reformulation = True
    parameters = []
    parameters += [do_reformulation]
    bigram_scoring_parameter = 2
    #a1,a2,a3 = 0.125,0.645,0.23
    a1,a2,a3 = 0.1,0.765,0.135
    r1,r2  = 0.95,0.95
    effort = 6
    setting = ""
    w_len = 1
    method = 'CCQF_' + str(a1) + '_' + str(a2) + '_' + str(a3) + '_' + str(bigram_scoring_parameter) + '_noef_nowsu_' + str(r1) + '_' + str(r2)
    print ('METHOD: ', method)
    sys.stdout = open(os.path.join(save_dir, method + '_big_idx_out.txt'), 'w')
    topic_descs = read_topic_descs(dataset)
    topic_rel_docs = read_judgements(dataset)
    (topic_bigram_ct, topic_unigram_ct, topic_bigram_prob) = read_bigram_topic_lm(dataset)
    Reformulation_comp =  Reformulation_component_multinomial(parameters = (r1,r2), collection_lm = doc_collection_lm_dist, print_info = True, type2_update= False, no_query_filter = False, nowsu = True)
    Formulation_comp = Query_formulation_CCQF(parameters = (a1,a2,a3,bigram_scoring_parameter,effort,math.log(0.000000001)), bernoulli = False, collection_lm = doc_collection_lm_dist, topic_bi_probs = (topic_bigram_ct, topic_unigram_ct, topic_bigram_prob), topic_descs = topic_descs)
    #Formulation_comp = Query_formulation_CCQF_new(parameters=(a1, a2, a3, -1, math.log(0.0000001)), collection_lm=doc_collection_lm_dist)

    #Formulation_comp = Query_formulation_QS3plus(collection_lm = doc_collection_lm_dist, total_num_words = total_noof_words)
    #Reformulation_comp = Reformulate_no_reformulation()
    result_judgement_component = Result_judgement_qrels(collection_lm = doc_collection_lm_dist, document_rels_all_topics = topic_rel_docs)
    topic_num_nums = [int(s) for s in topic_rel_docs.keys()]
    topic_num_nums.sort()
    topic_num_nums = [str(s) for s in topic_num_nums]
    print (topic_num_nums)
        
    user_models = {} 
    candidate_queries_topics = {}
    queries ={}
    for topic_num in topic_num_nums:
        sim_session = Session(topic_num = topic_num)
        user_models[topic_num] = {}
        user_models[topic_num]["user_model"] = User_model(parameters, topic_num, topic_rel_docs[topic_num], dataset, Formulation_comp, Reformulation_comp, result_judgement_component)
        candidate_queries_topics[topic_num] = user_models[topic_num]["user_model"].candidate_queries[:]
        query = user_models[topic_num]["user_model"].first_interaction()
        #user_models[topic_num]["sessions"] = [sim_session]
        user_models[topic_num]["sessions"] = [[sim_session, 1, None, 0, query]]
        user_models[topic_num]["end"] = False
        queries[topic_num] = query
    inputname = os.path.join(save_dir, method + "queries_input")
    outputname = os.path.join(save_dir, method + "results_output")
    make_runquery_file2(queries, inputname)
    start_time = time.time()
    os.system("../../indri-5.14/runquery/IndriRunQuery " + inputname + "> " + outputname)
    print ('TIME TAKEN: ', time.time()-start_time)
    all_formatted_results = format_batch_results_trec_format(outputname)

    for inte_idx in range(1,11):
        print ("Interation number: ", inte_idx)
        queries = {}
        start_time = time.time()
        for topic_num in topic_num_nums:
            print ("Interation number: ", inte_idx)
            print ('Topic number: ', topic_num, topic_descs[topic_num])
            if (user_models[topic_num]["end"]):
                continue
            sim_session = user_models[topic_num]["sessions"][0][0]
            if (user_models[topic_num]["sessions"][0][1] == 1):
                try:
                    formatted_results = all_formatted_results[topic_num]
                except KeyError:
                    formatted_results = []
                user_models[topic_num]["sessions"][0][2] = formatted_results
                user_models[topic_num]["sessions"][0][3] = 0
            i = user_models[topic_num]["sessions"][0][3]
            formatted_results = user_models[topic_num]["sessions"][0][2][i:i+10]
            start_m_time = time.time()
            (next_action_code,next_query,clicked_results) = user_models[topic_num]["user_model"].next_interaction(formatted_results)
            print ('TIME TAKEN for Reformulation topic: ', time.time()-start_m_time)
            sim_session.add_sim_interaction(user_models[topic_num]["sessions"][0][4].split() , formatted_results, clicked_results, user_models[topic_num]["sessions"][0][1])
            print ("next_action_code:{} , next_query:{}".format(next_action_code,next_query))
            if (next_action_code == 2):
                user_models[topic_num]["end"] = True
                #print ("Ending session")
            elif (next_action_code==1):
                user_models[topic_num]["sessions"][0][1] = 1
                query = next_query
                queries[topic_num] = query
                user_models[topic_num]["sessions"][0][4] = query
            elif (next_action_code == 0):
                user_models[topic_num]["sessions"][0][1] = 0
                user_models[topic_num]["sessions"][0][3] += 10 
        print ('TIME TAKEN for Reformulation all topics: ', time.time()-start_time)
        make_runquery_file2(queries, inputname)
        start_time = time.time()
        os.system("../../indri-5.14/runquery/IndriRunQuery " + inputname + "> " + outputname)
        print ('TIME TAKEN: ', time.time()-start_time)
        all_formatted_results = format_batch_results_trec_format(outputname)

    simulated_sessions = []
    for topic_num in user_models:
        sim_session = user_models[topic_num]["sessions"][0][0]
        simulated_sessions += [sim_session]
    pickle.dump(simulated_sessions, open(os.path.join(save_dir, method + "_big_idx_sessions.pk"), "wb"))
    pickle.dump(candidate_queries_topics, open(os.path.join(save_dir, method + "_big_idx_queries.pk"), "wb"))


def fit_RQF_parameters_multiple_validations(dataset_name, foldername, alpha_paramters, simtype="jsim"):
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
    # validation_sets = session_ops.get_validation_set_divisions(act_sessions)
    # validation_sets_2 = session_ops.get_validation_set_divisions(act_sessions)
    validation_filename = "validation_sets_413_fold.pk"
    validation_sets = pickle.load(open(os.path.join(data_dir, validation_filename), "rb"))
    # r1_list = [float(i) for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]]  # 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0
    # r2_list = [float(i) for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]]
    r1_list = [0.5]
    r2_list = [0.5]
    parameters_list = itertools.product(r1_list, r2_list)
    save_dir = os.path.join(data_dir, foldername)  # fit_3_parameters_CCQF_new_r_old_p_noef_100_pts_w_len
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    print(save_dir)
    topic_descs = read_topic_descs(dataset)
    # topic_rel_docs = read_judgements(dataset)
    # pickle.dump(validation_sets, open(os.path.join(data_dir, "validation_sets_413_fold.pk"), "wb"))
    # pickle.dump(validation_sets_2, open(os.path.join(data_dir, "validation_sets_2_413_fold.pk"), "wb"))
    alpha = alpha_paramters[0]
    beta = alpha_paramters[1]
    for (r1, r2) in parameters_list:
        method = 'CCQF_' + str(alpha) + '_' + str(beta) + '_ef6_basic_reform_lbda_0.5' + "_" + str(r1) + "_" + str(r2)
        try:
            candidate_queries_topics = pickle.load(
                open(os.path.join(save_dir, method + "_reform_queries.pk"), 'rb')) # if (len(c[0]) <= 3)
        except:
            print("method not found: ", method)
            candidate_queries_topics = None
        if candidate_queries_topics != None:
            ev = Evaluation_queries(act_sessions, None, candidate_queries_topics)
            if simtype == "f-measure":
                avg_jaccard_similarity_total, precision_score, similarity_list_total = ev.query_similarity_evaluation(onlyqueries=True, top_k=5, simtype="prec")
                avg_jaccard_similarity_total, recall_score, similarity_list_total = ev.query_similarity_evaluation(onlyqueries=True, top_k=5, simtype = "recall")
                f_measure_score = float(2.0*precision_score*recall_score)/float(precision_score+recall_score)
            else:
                avg_jaccard_similarity_total, avg_jaccard_similarity_2_total, similarity_list_total = ev.query_similarity_evaluation(onlyqueries=True, top_k=5, simtype = simtype)
        print(dataset_name)
        try:
            if simtype != "f-measure": print('AVG TEST PERFORMANCE TOTAL : ', avg_jaccard_similarity_2_total)
            if simtype == "f-measure": print('F-measure AVG TEST PERFORMANCE TOTAL : ', f_measure_score)
        except:
            pass
        avg_test_perfs = []
        if candidate_queries_topics != None:
            if simtype == "f-measure": avg_test_f_measure = []
            for idx, (vali_set, vali_set_test) in enumerate(validation_sets):
                if simtype == "f-measure":
                    _, precision_score, _ = ev.query_similarity_evaluation(onlyqueries=True, top_k=5, validation_set=True, act_session_queries=vali_set_test, simtype = "prec")
                    _, recall_score, _ = ev.query_similarity_evaluation(onlyqueries=True, top_k=5, validation_set=True, act_session_queries=vali_set_test, simtype = "recall")
                    f_measure_score = float(2.0 * precision_score * recall_score) / float(precision_score + recall_score)
                    avg_test_f_measure += [f_measure_score]
                else:
                    avg_jaccard_similarity_vali, avg_jaccard_similarity_2_vali, similarity_list_vali = ev.query_similarity_evaluation(
                        onlyqueries=True, top_k=5, validation_set=True, act_session_queries=vali_set, simtype = simtype)
                    avg_jaccard_similarity_test, avg_jaccard_similarity_2_test, similarity_list_test = ev.query_similarity_evaluation(
                        onlyqueries=True, top_k=5, validation_set=True, act_session_queries=vali_set_test, simtype = simtype)
                    avg_test_perfs += [avg_jaccard_similarity_2_test]

            if simtype != "f-measure": print ("AVG TEST PERFORMANCE with vali tests : ", float(sum(avg_test_perfs))/float(len(avg_test_perfs)))
            if simtype == "f-measure": print ("Avg f-measure with vali tests : ", float(sum(avg_test_f_measure))/float(len(avg_test_f_measure)))
    with open(os.path.join(save_dir, "reformulated queries" + "_" + str(alpha) + '_' + str(beta) + ".txt"), "w") as outfile:
        for topic_num in candidate_queries_topics:
            outfile.write("TOPIC NUM: " + str(topic_num) +  topic_descs[topic_num] + "\n")
            outfile.write("CANDIDATE queries\n")
            #print (candidate_queries_topics[topic_num])
            for c in candidate_queries_topics[topic_num]:
                outfile.write(str(c[0]) + "\n")
    '''
    for i in range(len(validation_sets)):
        print('BEST PARAMETER: ' + str(best_point[i][0][0]) + ' ' + str(best_point[i][0][1]) + ' ' + str(
            best_point[i][0][2]) + ' ' + str(best_point[i][0][3]) + ' ' +str(best_point[i][0][4]) + ' ' +str(best_point[i][1]) + ' ' + str(best_point[i][2]) + '\n')
        avg_test_performance += [best_point[i][2]]
    for j in range(5):
        avg_best_parameter[j] = sum([best_point[i][0][j] for i in range(len(validation_sets))]) / float(
            len(validation_sets))

    avg_best_parameter_2[0] = float(sum([best_point[i][0][0] for i in range(len(validation_sets))])) / float(
        len(validation_sets))
    avg_best_parameter_2[1] = float(
        sum([(1 - best_point[i][0][0]) * best_point[i][0][1] for i in range(len(validation_sets))])) / float(
        len(validation_sets))
    avg_best_parameter_2[2] = float(
        sum([(1 - best_point[i][0][0]) * (1 - best_point[i][0][1]) for i in range(len(validation_sets))])) / float(
        len(validation_sets))
    avg_best_parameter_2[3] = avg_best_parameter[3]
    avg_best_parameter_2[4] = avg_best_parameter[4]

    avg_test_performance = float(sum(avg_test_performance)) / float(len(avg_test_performance))
    print('AVG TEST PERFORMANCE: ', avg_test_performance)
    print("AVG BEST PARAMETER: ", avg_best_parameter)
    print("AVG BEST PARAMETER: ", avg_best_parameter_2)

    similarities_lists = {}

    for i in range(len(validation_sets)):
        for topic_num in best_point[i][3]:
            try:
                there = similarities_lists[topic_num]
                similarities_lists[topic_num] = [similarities_lists[topic_num][idx] + [best_point[i][3][topic_num][idx]]
                                                 for idx in range(len(best_point[i][3][topic_num]))]
            except KeyError:
                similarities_lists[topic_num] = [[] for idx in range(5)]
                similarities_lists[topic_num] = [similarities_lists[topic_num][idx] + [best_point[i][3][topic_num][idx]]
                                                 for idx in range(len(best_point[i][3][topic_num]))]
    similarities_lists_list = []
    topics_nums = [int(x) for x in list(similarities_lists.keys())]
    topics_nums.sort()
    topics_nums = [str(x) for x in topics_nums]
    print(len(topics_nums))
    for topic_num in topics_nums:
        similarities_lists[topic_num] = [
            float(sum(similarities_lists[topic_num][idx])) / float(len(similarities_lists[topic_num][idx])) for idx in
            range(5)]
        similarities_lists_list += similarities_lists[topic_num]
    print(len((similarities_lists_list)))
    return similarities_lists_list
    '''


def fit_RQF_parameters_multiple_validations_old(dataset_name, foldername, validation_filename, validation_sets_best_params):
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
    # validation_sets = session_ops.get_validation_set_divisions(act_sessions)
    # validation_sets_2 = session_ops.get_validation_set_divisions(act_sessions)
    validation_sets = pickle.load(open(os.path.join(data_dir, validation_filename), "rb"))
    #r1_list = [float(i) for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]]  # 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0
    #r2_list = [float(i) for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]]
    r1_list = [0.5]
    r2_list = [0.5]
    # x_list = [float(0.0)]
    #y_list = [float(0.0)]
    # a3_list = [float(i)*0.5 for i in [0.5,1]]
    parameters_list = itertools.product(r1_list, r2_list)
    bigram_scoring_parameter = 2
    # bigram_scoring_parameter = int(foldername.split("fit_3_parameters_")[1][0])
    # save_dir = os.path.join(data_dir, 'fit_3_parameters_' + str(bigram_scoring_parameter) + '_noef_100_pts_2')
    save_dir = os.path.join(data_dir, foldername)  # fit_3_parameters_CCQF_new_r_old_p_noef_100_pts_w_len
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    print(save_dir)
    data_points = []
    best_perf = 0
    topic_descs = read_topic_descs(dataset)
    # topic_rel_docs = read_judgements(dataset)
    best_perf = [0 for i in range(len(validation_sets))]
    best_point = [() for i in range(len(validation_sets))]
    # pickle.dump(validation_sets, open(os.path.join(data_dir, "validation_sets_413_fold.pk"), "wb"))
    # pickle.dump(validation_sets_2, open(os.path.join(data_dir, "validation_sets_2_413_fold.pk"), "wb"))
    print (validation_sets_best_params)
    for (r1, r2) in parameters_list:
        for idx, (vali_set, vali_set_test) in enumerate(validation_sets):
            a1,a2,a3 = validation_sets_best_params[idx]
            # method = 'CCQF_' + str(a1) + '_' + str(a2) + '_' + str(a3) + '_' + str(bigram_scoring_parameter) + '_' + 'noef'
            method = 'CCQF_' + str(a1) + '_' + str(a2) + '_' + str(a3) + "_noef_nowsu_type2_update" + "_" + str(r1) + "_" + str(r2)
            try:
                candidate_queries_topics = pickle.load(open(os.path.join(save_dir, method + "_reform_queries.pk"), 'rb'))
                for topic_num in candidate_queries_topics:
                    candidate_queries_topics[topic_num] = [c for c in candidate_queries_topics[topic_num]]  # if (len(c[0]) <= 3)
            except:
                print("method not found: ", method)
                candidate_queries_topics = None
            if candidate_queries_topics != None:
                ev = Evaluation_queries(act_sessions, None, candidate_queries_topics)
                avg_jaccard_similarity_total, avg_jaccard_similarity_2_total, similarity_list_total = ev.query_similarity_evaluation(
                    onlyqueries=True, top_k=5)
                avg_jaccard_similarity_vali, avg_jaccard_similarity_2_vali, similarity_list_vali = ev.query_similarity_evaluation(
                    onlyqueries=True, top_k=5, validation_set=True, act_session_queries=vali_set)
                avg_jaccard_similarity_test, avg_jaccard_similarity_2_test, similarity_list_test = ev.query_similarity_evaluation(
                    onlyqueries=True, top_k=5, validation_set=True, act_session_queries=vali_set_test)
                # avg_jaccard_similarity_2_test_1, avg_jaccard_similarity_2_test_1,similarity_list_test_1 = ev.query_similarity_evaluation(onlyqueries = True, top_k = 1, validation_set = True, act_session_queries = vali_set_test)
                # avg_jaccard_similarity_2_test_10, avg_jaccard_similarity_2_test_10,similarity_list_test_10 = ev.query_similarity_evaluation(onlyqueries = True, top_k = 10, validation_set = True, act_session_queries = vali_set_test)
                if (avg_jaccard_similarity_2_vali > best_perf[idx]):
                    best_point[idx] = ((a1, a2, a3,r1,r2), avg_jaccard_similarity_2_vali, avg_jaccard_similarity_2_test, similarity_list_test)
                    best_perf[idx] = avg_jaccard_similarity_2_vali

    print(dataset_name)
    try:
        print('AVG TEST PERFORMANCE TOTAL : ', avg_jaccard_similarity_2_total)
    except:
        pass
    avg_test_performance = []
    avg_best_parameter = [0, 0, 0, 0, 0]
    avg_best_parameter_2 = [0, 0, 0, 0, 0]
    '''
    for i in range(len(validation_sets)):
        print('BEST PARAMETER: ' + str(best_point[i][0][0]) + ' ' + str(best_point[i][0][1]) + ' ' + str(
            best_point[i][0][2]) + ' ' + str(best_point[i][0][3]) + ' ' +str(best_point[i][0][4]) + ' ' +str(best_point[i][1]) + ' ' + str(best_point[i][2]) + '\n')
        avg_test_performance += [best_point[i][2]]
    for j in range(5):
        avg_best_parameter[j] = sum([best_point[i][0][j] for i in range(len(validation_sets))]) / float(
            len(validation_sets))

    avg_best_parameter_2[0] = float(sum([best_point[i][0][0] for i in range(len(validation_sets))])) / float(
        len(validation_sets))
    avg_best_parameter_2[1] = float(
        sum([(1 - best_point[i][0][0]) * best_point[i][0][1] for i in range(len(validation_sets))])) / float(
        len(validation_sets))
    avg_best_parameter_2[2] = float(
        sum([(1 - best_point[i][0][0]) * (1 - best_point[i][0][1]) for i in range(len(validation_sets))])) / float(
        len(validation_sets))
    avg_best_parameter_2[3] = avg_best_parameter[3]
    avg_best_parameter_2[4] = avg_best_parameter[4]

    avg_test_performance = float(sum(avg_test_performance)) / float(len(avg_test_performance))
    print('AVG TEST PERFORMANCE: ', avg_test_performance)
    print("AVG BEST PARAMETER: ", avg_best_parameter)
    print("AVG BEST PARAMETER: ", avg_best_parameter_2)
    
    similarities_lists = {}

    for i in range(len(validation_sets)):
        for topic_num in best_point[i][3]:
            try:
                there = similarities_lists[topic_num]
                similarities_lists[topic_num] = [similarities_lists[topic_num][idx] + [best_point[i][3][topic_num][idx]]
                                                 for idx in range(len(best_point[i][3][topic_num]))]
            except KeyError:
                similarities_lists[topic_num] = [[] for idx in range(5)]
                similarities_lists[topic_num] = [similarities_lists[topic_num][idx] + [best_point[i][3][topic_num][idx]]
                                                 for idx in range(len(best_point[i][3][topic_num]))]
    similarities_lists_list = []
    topics_nums = [int(x) for x in list(similarities_lists.keys())]
    topics_nums.sort()
    topics_nums = [str(x) for x in topics_nums]
    print(len(topics_nums))
    for topic_num in topics_nums:
        similarities_lists[topic_num] = [
            float(sum(similarities_lists[topic_num][idx])) / float(len(similarities_lists[topic_num][idx])) for idx in
            range(5)]
        similarities_lists_list += similarities_lists[topic_num]
    print(len((similarities_lists_list)))
    return similarities_lists_list
    '''


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Simulating sessions")
    parser.add_argument('-simtype', type=str, default="jsim", help="evaluation measure")
    parser.add_argument('-dataset', type=str, default="Session_track_2012", help="evaluation measure")
    args = parser.parse_args()
    '''
    r1s = [0.5,0.6,0.7,0.8,0.9,0.95,1.0]
    best_r1 = 0.5
    r2 = 0.95
    best_similarity = 0
    for r1 in r1s:
        similarities = query_similarity_controlled_q1(r1,r2, "Session_track_2012")
        if similarities > best_similarity:
            best_r1 = r1
            best_similarity = similarities
    print ("BEST R1: ", best_r1)
    '''
    '''
    #r2s = [0.6,0.7,0.8,0.9,0.95,0.99,1.0]
    r2s = [0.7]
    best_r2 = 0.7
    r1 = 0.7
    best_similarity = 0
    for r2 in r2s:
        (act_ndcgs, sim_ndcgs, act_ndcgs_2, act_ndcgs50, sim_ndcgs50, act_ndcgs50_2, similarities) = query_reformulation_controlled_q1(r1,r2)
        if similarities > best_similarity:
            best_r2 = r2
            best_similarity = similarities
    print ("BEST R2: ", best_r2)
    '''
    '''
    r2s = [0.999, 0.9999, 0.99995, 0.99999, 0.999999,1.0]
    best_r2 = 0.7
    r1 = 0.7
    best_similarity = 0
    for r2 in r2s:
        similarities = query_similarity_controlled_q1(r1,r2, "Session_track_2012")
        if similarities > best_similarity:
            best_r2 = r2
            best_similarity = similarities
    print ("BEST R2: ", best_r2)
    '''
    #actual_queries_performance()
    
    #simulate_batch_queries()
    #sample_testing()
    #test_query_ndcg_performance()

    clueweb09_index = '/srv/local/work/sahitil2/cb09-catb_index'
    clueweb12_index = '/srv/local/work/sahitil2/clueweb12_index'

    '''
    simulate_batch_queries_1_reformulate("Session_track_2014", (0.7,1.0), clueweb12_index, ["Dr_R12g", "Cp_add"])
    simulate_batch_queries_1_reformulate("Session_track_2014", (0.1,1.0), clueweb12_index, ["Dr_R12g", "Dp_add"])
    simulate_batch_queries_1_reformulate("Session_track_2014", (0.125,1.0), clueweb12_index, ["Cr_R12g", "Cp_add"])
    simulate_batch_queries_1_reformulate("Session_track_2014", (0.1,1.0), clueweb12_index, ["Cr_R12g", "Dp_add"])
    simulate_batch_queries_1_reformulate("Session_track_2013", (0.85, 1.0), clueweb12_index, ["Dr_R12g", "Cp_add"])
    simulate_batch_queries_1_reformulate("Session_track_2013", (0.1, 1.0), clueweb12_index, ["Dr_R12g", "Dp_add"])
    simulate_batch_queries_1_reformulate("Session_track_2013", (0.475, 1.0), clueweb12_index, ["Cr_R12g", "Cp_add"])
    simulate_batch_queries_1_reformulate("Session_track_2013", (0.1, 1.0), clueweb12_index, ["Cr_R12g", "Dp_add"])
    simulate_batch_queries_1_reformulate("Session_track_2012", (0.125, 1.0), clueweb09_index, ["Dr_R12g", "Cp_add"])
    simulate_batch_queries_1_reformulate("Session_track_2012", (0.1, 1.0), clueweb09_index, ["Dr_R12g", "Dp_add"])
    simulate_batch_queries_1_reformulate("Session_track_2012", (0.05, 1.0), clueweb09_index, ["Cr_R12g", "Cp_add"])
    simulate_batch_queries_1_reformulate("Session_track_2012", (0.1, 1.0), clueweb09_index, ["Cr_R12g", "Dp_add"])
    '''
    '''
    simtype = args.simtype
    fit_RQF_parameters_multiple_validations("Session_track_2013",
                                            "fit_3_parameters_PRE_Cr_R12g_Cp_add_ef6_100_pts_w_len",(0.475,1.0), simtype = simtype)
    fit_RQF_parameters_multiple_validations("Session_track_2014",
                                            "fit_3_parameters_PRE_Dr_R12g_Cp_add_ef6_100_pts_w_len",(0.7,1.0), simtype = simtype)
    fit_RQF_parameters_multiple_validations("Session_track_2014",
                                            "fit_3_parameters_PRE_Dr_R12g_Dp_add_ef6_100_pts_w_len", (0.1, 1.0),
                                            simtype=simtype)
    fit_RQF_parameters_multiple_validations("Session_track_2014",
                                            "fit_3_parameters_PRE_Cr_R12g_Cp_add_ef6_100_pts_w_len", (0.125, 1.0),
                                            simtype=simtype)
    fit_RQF_parameters_multiple_validations("Session_track_2014",
                                            "fit_3_parameters_PRE_Cr_R12g_Dp_add_ef6_100_pts_w_len", (0.1, 1.0),
                                            simtype=simtype)
    fit_RQF_parameters_multiple_validations("Session_track_2013",
                                            "fit_3_parameters_PRE_Dr_R12g_Cp_add_ef6_100_pts_w_len",(0.85,1.0), simtype = simtype)
    fit_RQF_parameters_multiple_validations("Session_track_2013",
                                            "fit_3_parameters_PRE_Dr_R12g_Dp_add_ef6_100_pts_w_len", (0.1, 1.0),
                                            simtype=simtype)
    fit_RQF_parameters_multiple_validations("Session_track_2013",
                                            "fit_3_parameters_PRE_Cr_R12g_Dp_add_ef6_100_pts_w_len", (0.1, 1.0),
                                            simtype=simtype)
    fit_RQF_parameters_multiple_validations("Session_track_2012",
                                            "fit_3_parameters_PRE_Dr_R12g_Cp_add_ef6_100_pts_w_len",(0.125,1.0), simtype = simtype)
    fit_RQF_parameters_multiple_validations("Session_track_2012",
                                            "fit_3_parameters_PRE_Dr_R12g_Dp_add_ef6_100_pts_w_len", (0.1, 1.0),
                                            simtype=simtype)
    fit_RQF_parameters_multiple_validations("Session_track_2012",
                                            "fit_3_parameters_PRE_Cr_R12g_Cp_add_ef6_100_pts_w_len", (0.05, 1.0),
                                            simtype=simtype)
    fit_RQF_parameters_multiple_validations("Session_track_2012",
                                            "fit_3_parameters_PRE_Cr_R12g_Dp_add_ef6_100_pts_w_len", (0.1, 1.0),
                                            simtype=simtype)
    '''

    #simulate_batch_queries_1_reformulate("Session_track_2012", (0.1,0.5,0.5), clueweb09_index)
    #simulate_batch_queries_1_reformulate("Session_track_2012", (0.0, 0.9, 0.1), clueweb09_index)
    #simulate_batch_queries_1_reformulate("Session_track_2012", (0.1, 1.0, 0.0), clueweb09_index)

    #simulate_batch_queries_1_reformulate("Session_track_2013", (1.0,0.0,1.0), clueweb12_index)
    #simulate_batch_queries_1_reformulate("Session_track_2013", (0.7,0.0,1.0), clueweb12_index)

    #simulate_batch_queries_1_reformulate("Session_track_2013", (0.0, 0.0, 1.0), clueweb12_index)

    #simulate_batch_queries_1_reformulate("Session_track_2014", (0.1,1.0,0.0), clueweb12_index)

    #simulate_batch_queries_1_reformulate("Session_track_2013", (0.1 0.0 1.0), clueweb12_index, "com_r_new_p")

    #simulate_batch_queries_1_reformulate("Session_track_2014", (0.1, 1.0, 0.0), clueweb12_index, "old_r_old_p")

    #simulate_batch_queries_1_reformulate("Session_track_2014", (0.1,0.9,0.1), clueweb12_index, "old_r_new_p")
    #simulate_batch_queries_1_reformulate("Session_track_2014", (0.1,1.0, 0.0), clueweb12_index, "old_r_new_p")


    #simulate_batch_queries_1_reformulate("Session_track_2014", (0.2,0.9,0.1), clueweb12_index, "old_r_new_p")

    #simulate_batch_queries_1_reformulate("Session_track_2014", (0.1, 0.9, 0.1), clueweb12_index, "old_r_p3")
    #simulate_batch_queries_1_reformulate("Session_track_2014", (0.2, 0.9, 0.1), clueweb12_index, "old_r_p3")

    #simulate_batch_queries_1_reformulate("Session_track_2014", (0.0, 0.0, 1.0), clueweb12_index, "old_r_old_p")


    #simulate_batch_queries_1_reformulate("Session_track_2014", (0.0, 0.0, 1.0), clueweb12_index, "old_r_old_p")

    #simulate_batch_queries_1_reformulate("Session_track_2013", (0.5,0.7,0.3), clueweb12_index, "old_r_old_p")
    #simulate_batch_queries_1_reformulate("Session_track_2013", (0.6,0.5,0.5), clueweb12_index, "old_r_old_p")
    #simulate_batch_queries_1_reformulate("Session_track_2013", (0.3,0.6,0.4), clueweb12_index, "old_r_old_p")
    #simulate_batch_queries_1_reformulate("Session_track_2013", (0.3,0.9,0.1), clueweb12_index, "old_r_old_p")
    #simulate_batch_queries_1_reformulate("Session_track_2013", (0.0,1.0,0.0), clueweb12_index, "old_r_old_p")

    '''
    fit_RQF_parameters_multiple_validations("Session_track_2013", "fit_3_parameters_CCQF_old_r_old_p_noef_100_pts_w_len", "validation_sets_413_fold.pk", [(1.0,0.0,1.0), (1.0,0.0,1.0), (1.0,0.0,1.0), (1.0,0.0,1.0)])


    fit_RQF_parameters_multiple_validations("Session_track_2013",
                                            "fit_3_parameters_CCQF_old_r_old_p_noef_100_pts_w_len",
                                            "validation_sets_413_fold.pk",
                                            [(0.0,0.0,1.0), (0.0,0.0,1.0), (0.0,0.0,1.0), (0.0,0.0,1.0)])
    '''
    '''
    print ("=================R12g============")
    fit_RQF_parameters_multiple_validations("Session_track_2013",
                                            "fit_3_parameters_CCQF_old_r_old_p_noef_100_pts_w_len",
                                            "validation_sets_413_fold.pk",
                                            [(0.5,0.0,1.0), (0.5,0.0,1.0), (0.5,0.0,1.0), (0.7,0.0,1.0)])

    print("=================P2============")
    fit_RQF_parameters_multiple_validations("Session_track_2013",
                                            "fit_3_parameters_CCQF_old_r_old_p_noef_100_pts_w_len",
                                            "validation_sets_413_fold.pk",
                                            [(0.0,1.0,0.0), (0.0,1.0,0.0), (0.0,1.0,0.0), (0.0,1.0,0.0)])

    print("=================R12g_P2============")
    fit_RQF_parameters_multiple_validations("Session_track_2013",
                                            "fit_3_parameters_CCQF_old_r_old_p_noef_100_pts_w_len",
                                            "validation_sets_413_fold.pk",
                                            [(0.5,0.7,0.3), (0.6,0.5,0.5), (0.3,0.6,0.4), (0.3,0.9,0.1)])
    '''
    #actual_queries_performance_new("Session_track_2012", clueweb09_index, 10, 50)

    #actual_queries_performance_new("Session_track_2013", clueweb12_index, 20, 70)

    actual_queries_performance_new("Session_track_2014", clueweb12_index, 0, 62)
