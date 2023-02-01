from utils import *
import math
###TODO:
#1.vocabulary expand using the word concept graph
#2.vocabulary expands based on similar words through word embedding

#all_doc_index = pickle.load(open("../TREC_Robust_data/all_doc_index_lemmatized.pk", "rb") )
#doc_keys_list = all_doc_index[1]
#all_doc_index = all_doc_index[0]
#(doc_collection_lm, doc_collection_lm_binary) = pickle.load(open("../TREC_Robust_data/all_doc_language_model.pk", "rb"))
#total_num_words = sum(doc_collection_lm.values())
#doc_collection_lm_dist = {term:float(doc_collection_lm[term])/float(total_num_words) for term in doc_collection_lm}
#total_num_words_binary = sum(doc_collection_lm_binary.values())
#doc_collection_lm_binary_dist = {term:float(doc_collection_lm_binary[term])/float(total_num_words_binary) for term in doc_collection_lm_binary}
#candidate_queries = pickle.load(open("../simulated_sessions/final_results/candidate_queries_basic_recall_precision.pk", "rb"))
#target_documents_details = pickle.load(open("../TREC_Robust_data/topic_rel_doc_details.pk","rb"))
#target_document_vectors = pickle.load(open("../TREC_Robust_data/target_doc_topic_vectors.pk", 'rb'))

#stopwords_file = "../Session_track_2014/clueweb_snippet/lemur-stopwords.txt"
stopwords_file = "../lemur-stopwords.txt"

with open('../supervised_models/baseline_session_lms/topic_mle_training_session_nums_5.json', 'r') as outfile:
    topic_mle = json.load(outfile)
with open('../supervised_models/baseline_session_lms/p_t_w_training_session_nums_5.json', 'r') as outfile:
    p_t_w = json.load(outfile)
with open('../supervised_models/baseline_session_lms/p_t_s_w_training_session_nums_5.json', 'r') as outfile:
    p_t_s_w = json.load(outfile)

def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm==0:
        return normalize_vector
    else:
        return vector/norm

class User_model():
    def __init__(self, parameters = None, topic_num = None, target_click_items = None, dataset = None, query_formulation_component = None, query_reformulation_component = None, result_judgement_component = None):
        self.parameters = parameters
        self.topic_IN = None
        self.collection_IN = None
        self.current_query = None
        self.target_clueweb_ids = target_click_items.copy() #dict(zip(target_click_items, range(len(target_click_items)))) 
        self.session_length = None
        #self.previous_queries = []
        #self.topic_proportion, self.topic_word_distribution = topic_dist_inputs[0], topic_dist_inputs[1]
        self.previous_queries = {}
        self.precision_lm = None
        #self.candidate_queries = list(filter(lambda l: len(l[1])>=100, self.candidate_queries))
        self.topic_num = topic_num
        self.clicks_log_times = 0
        if (query_formulation_component.intl_cand_qurs_all_tpcs) == None:
            query_formulation_component.query_formulation_all_topics(dataset)
        self.query_formulation_component = query_formulation_component 
        #if(result_judgement_component == None):
        #    result_judgement_component = Result_judgement_qrels(target_document_vectors = target_document_vectors, document_rels = target_documents_details)
        self.result_judgement_component = result_judgement_component
        self.candidate_queries = query_formulation_component.get_initial_queries(topic_num)
        self.topic_IN = query_formulation_component.get_topic_IN(topic_num)
        self.query_reformulation_component = query_reformulation_component
        self.query_reformulation_component.result_judgement = self.result_judgement_component
        self.query_reformulation_component.query_formulation = self.query_formulation_component
        self.do_reformulation = parameters[0]
        self.topic_desc = query_formulation_component.get_topic_desc(topic_num)
        '''
        documents_rels = target_documents_details[self.topic_num][3]
        self.target_topic_vectors_2 = []
        documents_rels += [("topic_desc", 3)] 
        for docid,doc_rel in documents_rels:
            if (docid == "topic_desc"):
                self.target_topic_vectors_2 += [target_document_vectors[docid+"_" + str(self.topic_num)]]
            else:
                self.target_topic_vectors_2 += [target_document_vectors[docid]]  
        '''
        '''
        total_content = ""
        for docid in robust_doc_content:
            total_content = total_content + " " + robust_doc_content[docid]
        #print ("total_content done")
        robust_data_collection_IN = Counter(total_content.split())
        total_num_words = sum(robust_data_collection_IN.values())
        robust_data_collection_IN = {x:float(robust_data_collection_IN[x])/float(total_num_words) for x in robust_data_collection_IN}
        self.robust_data_collection_IN = robust_data_collection_IN
        '''
        #self.doc_collection = doc_collection
    #making topic IN    


    def query_formulation_candidate_query(self):
        candidate_queries_1 = self.candidate_queries
        self.candidate_queries = self.candidate_queries[1:]
        Q = candidate_queries_1[0][0]
        print ("Query: ", Q)
        self.current_query = Q
        #Q = postprocess_query(" ".join(Q), self.topic_desc)
        return " ".join(Q)
    
    def query_reformulation(self, results, clicks):
        print ("REFORMULATING QUERY...")
        candidate_queries, updated_topic_IN, updated_precision_lm = self.query_reformulation_component.reformulate_query(self.current_query, self.candidate_queries, results, clicks, self.topic_num, self.topic_IN, self.precision_lm, self.previous_queries)
        #self.topic_IN = updated_topic_IN
        #self.precision_lm = updated_precision_lm
        #self.candidate_queries = candidate_queries
        return candidate_queries


    def continue_or_reformulate1(self, result_product_ids, clicked_products):
        if sum([x[1] for x in clicked_products]) == 0:
            #self.previous_queries += [(self.current_query,r_q)
            current_query = self.current_query[:]
            current_query.sort()
            self.previous_queries[" ".join(current_query)] = 1
            return True
        else:
            return True

    def click_method(self, results):
        #only clicks on target items, clicks are already given here. #user will not click on same result agian
        print ("CLICKING...")
        print ("=========CLICKING INFO=========")
        clicks = []
        for result in results:
            print ('Docid: ' + result['docid'] + ' Snippet: ' + result["snippet"])
            if result["docid"] in self.target_clueweb_ids:
                clicks += [(result["docid"], 1)]
                del(self.target_clueweb_ids[result["docid"]])
                print ('Clicking: 1')
            else:
                clicks += [(result["docid"], 0)]
                print ('Clicking: 0')
        return clicks
    
    def end_session(self, clicked_items):
        self.session_length += 1
        if (len(self.target_clueweb_ids) == 0) or (sum([x[1] for x in clicked_items])>0):
            return True
        else:
            return False
    def end_session_2(self, clicked_items):
        self.session_length += 1
        if ((sum([x[1] for x in clicked_items])>0)):
            self.clicks_log_times += 1
        if (self.clicks_log_times>=3):
            print ("ENDING...")
            return True
        else:
            if (len(self.target_clueweb_ids) ==0):
                print ("ENDING...")
                return True
            return False

    def first_interaction(self):
        self.session_length = 0
        Q = self.query_formulation_candidate_query()
        return (Q)

    def next_interaction(self, result_list):
        clicked_items = self.click_method(result_list)
        #self.update_doc_collection(result_list)
        end = self.end_session_2(clicked_items)
        if (not end):
            reformulate = self.continue_or_reformulate1(result_list, clicked_items)
            if (reformulate):
                if (not self.do_reformulation):
                    Q = self.query_formulation_candidate_query()
                else:
                    candidate_queries = self.query_reformulation(result_list, clicked_items)
                    Q = candidate_queries[0][0]
                    Q = " ".join(Q)
                    #Q = self.query_formulation_candidate_query()
                return (1, Q, clicked_items, candidate_queries)
            else:
                return (0, [], clicked_items, [])
        else:
            return (2,[],clicked_items, [])

class Result_judgement_qrels():
    def __init__(self, parameters = None, collection_lm = None, document_rels_all_topics = None):
        self.parameters = parameters
        self.documents_rels = document_rels_all_topics
        self.doc_collection_lm_dist = collection_lm
        return 

    def evaluate_results(self, results, clicks, topic_num):
        documents_rels = self.documents_rels[topic_num]
        results_relevance = []
        for result in results:
            try:
                there = documents_rels[result['docid']]
                relevance = documents_rels[result['docid']]
                if relevance <= 0:
                    result_relevance = 0
                else: 
                    #result_relevance = float(relevance)/float(4)
                    if relevance >= 1:
                        result_relevance = 1
            except:
                result_relevance = 0
            results_relevance += [result_relevance]
        return results_relevance

class Result_judgement_clicks():
    def __init__(self, parameters = None, collection_lm = None, document_rels_all_topics = None):
        self.parameters = parameters
        self.documents_rels = document_rels_all_topics
        self.doc_collection_lm_dist = collection_lm
        return 

    def evaluate_results(self, results, clicks, topic_num):
        documents_rels = self.documents_rels[topic_num]
        results_relevance = []
        clicks_dict = {}
        for c in clicks:
            clicks_dict[c[0]] = 1
        for idx,result in enumerate(results):
            try:
                there = clicks_dict[result["docid"]]
                result_relevance = 1
            except KeyError: 
                result_relevance = 0
            results_relevance += [result_relevance]
        return results_relevance

class Result_judgement_topics():
    def __init__(self, parameters = None, target_document_vectors = None, document_rels_all_topics = None):
        self.parameters = parameters
        self.target_document_vectors = target_document_vectors
        self.documents_rels = documents_rels
        return 

    def evaluate_results(self, results, topic_num):
        target_topic_vectors_2 = []
        #documents_rels = self.document_rels[topic_num][3]
        documents_rels = [("topic_desc", 3)] 
        target_topic_vectors_2 += [self.target_document_vectors["topic_desc"+"_" + str(topic_num)]]
        print ("DOCUMENTS RELS: ", len(documents_rels), len(self.target_topic_vectors_2))
        print ('DOCUMENTS RELS:', documents_rels)
        results_relevance = []
        lda,comm_dict = target_documents_details["all_topics"][1][0],target_documents_details["all_topics"][1][1]
        #print (len(target_topic_vectors_2))
        #documents_rels = documents_rels[:30]
        print (len(target_topic_vectors_2), len(documents_rels))
        for result in results:
            weighted_sum_num = 0
            weighted_sum_den = 0
            a = lda[comm_dict.doc2bow(result["content"].split())]
            topic_dist_vector = [0]*100
            for v in a:
                topic_dist_vector[v[0]] = v[1]
            for idx,(docid,doc_rel) in enumerate(documents_rels): 
                weighted_sum_num += doc_rel*np.dot(normalize_vector(topic_dist_vector), normalize_vector(target_topic_vectors_2[idx]))
                #weighted_sum_num += doc_rel*np.dot(topic_dist_vector, target_topic_vectors_2[idx])
                weighted_sum_den += documents_rels[idx][1]
            #cos_sim = sum([vector[i]*topic_dist_vector[i] for i in range(len(topic_dist_vector))])
            #weighted_sum_num += documents_rels[idx][1]*cos_sim
            result_relevance = float(weighted_sum_num)/float(weighted_sum_den)
            #normalization of result relevance
            results_relevance += [result_relevance]
        return results_relevance

class Reformulation_component_binary():
    def __init__(self, parameters = (0.9,0.9), query_formulation_component = None, result_judgement_component = None, collection_lm = None, print_info = False):
        self.query_formulation = query_formulation_component
        self.beta = parameters[0]
        self.gamma = parameters[1]
        self.result_judgement = result_judgement_component
        self.doc_collection_lm_dist = collection_lm
        self.print_info = print_info

    def updating_scores(self, query, results, clicks, topic_num, topic_lm, precision_lm, beta, gamma): 
        #print ("COMING UPDATING SCORES")
        results_relevance = self.result_judgement.evaluate_results(results, clicks, topic_num)
        results_word_lm = {}
        results_word_not_lm = {}
        all_result_words = {}
        for word in query:
            results_word_lm[word] = 0
            results_word_not_lm[word] = 0
        for idx,result in enumerate(results):
            result_content = result["snippet"]
            if "full_text_lm" in result:
                if clicks[idx][1] == 1:
                    result_content += " " + result["full_text_lm"]
            result_content = result_content.replace('...', ' ')
            result_content = preprocess(result_content, stopwords_file, lemmatizing = True)
            result_lm = Counter(result_content.split())
            #print (results_relevance[idx])
            for word in result_lm:
                try:
                    results_word_lm[word] += (1*results_relevance[idx]) 
                    results_word_not_lm[word] += ((1-results_relevance[idx])*(1))
                    #print ("coming here")
                except KeyError:
                    results_word_lm[word] = (1*results_relevance[idx]) 
                    results_word_not_lm[word] = ((1-results_relevance[idx])*(1))
                all_result_words[word] = 1  

        total_rel_results =  float(sum(results_relevance))
        if (total_rel_results != 0):
            results_word_lm = {word:float(results_word_lm[word])/float(total_rel_results) for word in results_word_lm} 
        total_irr_results = float(sum([(1-x) for x in results_relevance]))
        results_not_topic_IN = {}
        if (total_irr_results != 0):
            results_word_not_lm = {word:float(results_word_not_lm[word])/float(total_irr_results) for word in results_word_not_lm}
            results_not_topic_IN =  results_word_not_lm
        results_not_topic_IN = {word:results_not_topic_IN[word] for word in results_not_topic_IN if results_not_topic_IN[word]!=0}

        updated_topic_IN= {}
        if (total_rel_results == 0):
            updated_topic_IN = topic_lm
        else:
            updated_topic_IN = {word:(beta)*float(topic_lm[word]) for word in topic_lm}
            query_dict = dict(zip(query, range(len(query))))
            for word in results_word_lm:
                if results_word_lm[word] == 0:
                    try:
                        there = query_dict[word]
                    except KeyError:
                        try:
                            updated_topic_IN[word] += (1-beta)*topic_lm[word]
                        except KeyError:
                            pass
                else:
                    try:
                        updated_topic_IN[word] += (1-beta)*results_word_lm[word]
                    except KeyError:
                        updated_topic_IN[word] = (1-beta)*results_word_lm[word]

        updated_topic_IN = {word:updated_topic_IN[word] for word in updated_topic_IN if updated_topic_IN[word]!=0}
        if (precision_lm == None):
            precision_lm = {}
            for word in topic_lm:
                precision_lm[word] = {}
                try:
                    precision_lm[word]["num"] = (topic_lm[word]*0.02)
                    precision_lm[word]["den"] = (topic_lm[word]*0.02)+(self.doc_collection_lm_dist[word]*0.98)
                except:
                    precision_lm[word]["num"] = topic_lm[word]
                    precision_lm[word]["den"] = 0
            precision_lm = precision_lm
            precision_lm = precision_lm
        #Type1 update
        updated_precision_lm = self.update_EOR_lm(gamma, query, precision_lm, updated_topic_IN, results_not_topic_IN)
        #print ("WORDS TO UPDATE SCORE: ", words_to_update_score)
        if self.print_info:
            print ('=======REFORMULATION INFO BERNOULLI: =========')
            print ("Query: ", query)
            print ('Results: ')
            for idx,result in enumerate(results):
                print ('Docid: ' + result['docid'] + ' Snippet: ' + result["snippet"])
                print ('Relevance: ', results_relevance[idx])
            words_set = [(word,updated_topic_IN[word]) for word in updated_topic_IN]
            words_set = sorted(words_set,key=lambda l:l[1],reverse=True)[:30]
            for (word,score) in words_set:
                try:
                    results_word_lm_score = results_word_lm[word]
                except:
                    results_word_lm_score = 'NA'
                try:
                    results_not_topic_IN_score = results_not_topic_IN[word]
                except:
                    results_not_topic_IN_score = 'NA'
                try:
                    updated_precision_lm_score = updated_precision_lm[word]
                except:
                    updated_precision_lm_score = 'NA'                
                print ('word: {} results_word_lm: {}, results_not_topic_IN: {}, updated_topic_IN: {} updated_precision_lm: {}'.format(word, results_word_lm_score, results_not_topic_IN_score, updated_topic_IN[word], updated_precision_lm_score))               

        return updated_topic_IN,results_not_topic_IN,results_word_lm, updated_precision_lm

    def update_EOR_lm(self, gamma, query, precision_lm, updated_topic_IN, results_not_topic_IN):
        updated_precision_lm = {}
        for word in updated_topic_IN:
            try:
                there = precision_lm[word]
                updated_precision_lm[word] = {}
                #if precision_lm[word]["den"] == 0 
                updated_precision_lm[word]["num"] = precision_lm[word]["num"]/precision_lm[word]["den"]
                updated_precision_lm[word]["den"] = 1
            except KeyError:
                updated_precision_lm[word] = {}
                updated_precision_lm[word]["num"] = 0
                updated_precision_lm[word]["den"] = 1
            try:
                there = results_word_lm[word]
                try:
                    result_precision = (results_word_lm[word])/(results_word_lm[word]+results_not_topic_IN[word])
                except KeyError:
                    result_precision = (results_word_lm[word])/(results_word_lm[word]+0.00000001)
            except KeyError:
                if word in query:
                    result_precision = 0.00000001
                else:
                    result_precision = updated_precision_lm[word]["num"]
            
            updated_precision_lm[word]["num"]  = gamma*updated_precision_lm[word]["num"] + ((1-gamma)*(result_precision))
            updated_precision_lm[word]["den"] = 1
        
        return updated_precision_lm

    def update_EOR_lm_variation(self, precision_lm, updated_topic_IN, results_not_topic_IN):
        updated_precision_lm = {}
        for word in updated_topic_IN:
            try:
                there = precision_lm[word]
                updated_precision_lm[word] = {}
                updated_precision_lm[word]["num"] = precision_lm[word]["num"]
                updated_precision_lm[word]["den"] = precision_lm[word]["den"]
            except:
                updated_precision_lm[word] = {}
                updated_precision_lm[word]["num"] = 0
                updated_precision_lm[word]["den"] = 0
            updated_precision_lm[word]["num"] = updated_topic_IN[word]
            try:
                updated_precision_lm[word]["den"] = gamma*updated_precision_lm[word]["den"]+(1-gamma)*(updated_topic_IN[word] + results_not_topic_IN[word])
            except:
                updated_precision_lm[word]["den"] = gamma*updated_precision_lm[word]["den"]+(1-gamma)*(updated_topic_IN[word] + 0)
        return updated_precision_lm

    def reformulate_query(self, query, candidate_queries, results, clicks, topic_num, topic_IN, precision_lm, previous_queries):
        updated_topic_IN,updated_not_topic_IN,results_word_lm, updated_precision_lm = self.updating_scores(query, results, clicks, topic_num, topic_IN, precision_lm, self.beta, self.gamma)
        query_scores = self.query_formulation.reformulation_method(updated_topic_IN, updated_precision_lm, topic_num)
        new_candidates = sorted(query_scores, key=lambda l :l[1], reverse=True)
        first_query = new_candidates[0][0].copy()
        first_query.sort()
        while (" ".join(first_query) in previous_queries):
            new_candidates = new_candidates[1:]
            first_query = new_candidates[0][0].copy()
            first_query.sort()
        candidate_queries = new_candidates
        return candidate_queries,updated_topic_IN,updated_precision_lm

class Reformulation_component_multinomial():
    def __init__(self, parameters = (0.9,0.9), query_formulation_component = None, result_judgement_component = None, collection_lm = None, topic_descs = None, print_info = False, type2_update = False, no_query_filter = False, nowsu = False, collection_num_words = None):
        self.query_formulation = query_formulation_component
        self.beta = parameters[0]
        self.gamma = parameters[1]
        self.result_judgement = result_judgement_component
        #self.doc_collection_lm_dist = collection_lm
        self.print_info = print_info
        self.type2_update = type2_update
        self.no_query_filter = no_query_filter
        self.topic_descs = topic_descs
        self.nowsu = nowsu
        self.collection_num_words = collection_num_words
        self.doc_collection_frequencies = collection_lm
    def updating_scores_old(self, query, results, clicks, topic_num, topic_lm, precision_lm, beta, gamma):
        #print ("COMING UPDATING SCORES")
        '''
        content = ""
        for result in results:
            snippet_content = result["snippet"].replace('...', ' ')
            content += " " + snippet_content
        for idx,click in enumerate(clicks):
            if click[1] == 1:
                if "full_text_lm" in results[idx]:
                    content += " " + results[idx]["full_text_lm"]
        content = preprocess(content, stopwords_file, lemmatizing = True)
        results_lm = Counter(content.split())
        total_words = sum(results_lm.values())
        results_lm = {word:float(results_lm[word])/float(total_words) for word in results_lm}
        '''
        results_relevance = self.result_judgement.evaluate_results(results, clicks, topic_num)
        results_word_lm = {}
        results_word_not_lm = {}
        #words_to_update_score = list(results_lm.keys()) + list(topic_lm.keys())
        for word in query:
            results_word_lm[word] = 0
            results_word_not_lm[word] = 0
        for idx,result in enumerate(results):
            result_content = result["snippet"]
            if "full_text_lm" in result:
                if clicks[idx][1] == 1:
                    result_content += " " + result["full_text_lm"]
            result_content = result_content.replace('...', ' ')
            result_content = preprocess(result_content, stopwords_file, lemmatizing = True)
            result_lm = Counter(result_content.split())
            result_lm_length = sum(result_lm.values())
            #print (results_relevance[idx])
            for word in result_lm:
                try:
                    results_word_lm[word] += (result_lm[word]*results_relevance[idx]) 
                    results_word_not_lm[word] += ((1-results_relevance[idx])*(result_lm[word]))
                    #print ("coming here")
                except KeyError:
                    results_word_lm[word] = (result_lm[word]*results_relevance[idx]) 
                    results_word_not_lm[word] = ((1-results_relevance[idx])*(result_lm[word]))  
        total_word_score = sum(results_word_lm.values())
        updated_topic_IN= {}
        if (total_word_score == 0):
            updated_topic_IN = topic_lm
        else:
            results_word_lm = {word:float(results_word_lm[word])/float(total_word_score) for word in results_word_lm} 
            updated_topic_IN = {word:(1-beta)*float(results_word_lm[word]) for word in results_word_lm}
            for word in topic_lm:
                try:
                    updated_topic_IN[word] += beta*topic_lm[word]
                except KeyError:
                    updated_topic_IN[word] = beta*topic_lm[word]
        updated_topic_IN = {word:updated_topic_IN[word] for word in updated_topic_IN if updated_topic_IN[word]!=0}
        total_word_not_score = sum(results_word_not_lm.values())
        results_not_topic_IN = {}
        if (total_word_not_score != 0):
            results_word_not_lm = {word:float(results_word_not_lm[word])/float(total_word_not_score) for word in results_word_not_lm}
            results_not_topic_IN =  results_word_not_lm
        results_not_topic_IN = {word:results_not_topic_IN[word] for word in results_not_topic_IN if results_not_topic_IN[word]!=0}
        if (precision_lm == None):
            if self.type2_update:
                #Type2 update
                precision_lm = {}
                for word in topic_lm:
                    precision_lm[word] = {}
                    try:
                        precision_lm[word]["num"] = (topic_lm[word]*0.02)
                        precision_lm[word]["den"] = self.doc_collection_lm_dist[word]*0.98
                    except:
                        precision_lm[word]["num"] = topic_lm[word]*0.02
                        precision_lm[word]["den"] = 0.00000001
            else:
                precision_lm = {}
                for word in topic_lm:
                    precision_lm[word] = {}
                    try:
                        precision_lm[word]["num"] = (topic_lm[word]*0.02)
                        precision_lm[word]["den"] = (topic_lm[word]*0.02)+(self.doc_collection_lm_dist[word]*0.98)
                    except:
                        precision_lm[word]["num"] = topic_lm[word]*0.02
                        precision_lm[word]["den"] = (topic_lm[word]*0.02)
        #Type1 update
        if self.type2_update:
            updated_precision_lm = self.update_EOR_lm_variation(gamma, query, precision_lm, updated_topic_IN, results_not_topic_IN,results_word_lm)
        else:
            updated_precision_lm = self.update_EOR_lm_new(gamma, query, precision_lm, updated_topic_IN, results_not_topic_IN,results_word_lm)
        
        #print ("WORDS TO UPDATE SCORE: ", words_to_update_score)
        if self.print_info:
            print ('=======REFORMULATION INFO MULTINOMIAL: =========')
            print ("Query: ", query)
            print ('Results: ')
            for idx,result in enumerate(results):
                print ('Docid: ' + result['docid'] + ' Snippet: ' + result["snippet"])
                print ('Relevance: ', results_relevance[idx])
            words_set = [(word,updated_topic_IN[word]) for word in updated_topic_IN]
            words_set = sorted(words_set,key=lambda l:l[1],reverse=True)[:30]
            for (word,score) in words_set:
                try:
                    results_word_lm_score = results_word_lm[word]
                except:
                    results_word_lm_score = 'NA'
                try:
                    results_not_topic_IN_score = results_not_topic_IN[word]
                except:
                    results_not_topic_IN_score = 'NA'
                try:
                    updated_precision_lm_score = updated_precision_lm[word]
                except:
                    updated_precision_lm_score = 'NA'                
                print ('word: {} results_word_lm: {}, results_not_topic_IN: {}, updated_topic_IN: {} updated_precision_lm: {}'.format(word, results_word_lm_score, results_not_topic_IN_score, updated_topic_IN[word], updated_precision_lm_score))               

        return updated_topic_IN,results_not_topic_IN,results_word_lm, updated_precision_lm

    def updating_scores(self, query, results, clicks, topic_num, topic_lm, precision_lm, beta, gamma):
        topic_desc = self.topic_descs[topic_num]
        topic_desc = preprocess(topic_desc, stopwords_file, lemmatizing=True)
        results_relevance = self.result_judgement.evaluate_results(results, clicks, topic_num)
        results_word_lm = {}
        results_word_not_lm = {}
        # words_to_update_score = list(results_lm.keys()) + list(topic_lm.keys())

        topic_dist = dict(Counter(topic_desc.split()))
        for word in query:
            results_word_lm[word] = 0
            results_word_not_lm[word] = 0
        for idx, result in enumerate(results):
            result_content = result["snippet"]
            if "full_text_lm" in result:
                if clicks[idx][1] == 1:
                    result_content += " " + result["full_text_lm"]
            result_content = result_content.replace('...', ' ')
            result_content = preprocess(result_content, stopwords_file, lemmatizing=True)
            result_lm = Counter(result_content.split())
            result_lm_length = sum(result_lm.values())
            # print (results_relevance[idx])
            for word in result_lm:
                try:
                    results_word_lm[word] += (result_lm[word] * results_relevance[idx])
                    results_word_not_lm[word] += ((1 - results_relevance[idx]) * (result_lm[word]))
                    # print ("coming here")
                except KeyError:
                    results_word_lm[word] = (result_lm[word] * results_relevance[idx])
                    results_word_not_lm[word] = ((1 - results_relevance[idx]) * (result_lm[word]))
        results_word_lm = {word: results_word_lm[word] for word in results_word_lm if results_word_lm[word]!=0}
        results_word_not_lm = {word: results_word_not_lm[word] for word in results_word_not_lm if results_word_not_lm[word]!=0}
        lbda = 0.5
        total_word_frequencies = (lbda*float(sum(topic_dist.values()))) + float(1.0-lbda)*float(sum(results_word_lm.values()))
        total_topic_words = list(results_word_lm.keys()) + list(topic_dist.keys())
        total_topic_words = list(set(total_topic_words))
        updated_topic_lm = {word:0 for word in total_topic_words}
        for word in total_topic_words:
            try:
                there = topic_dist[word]
                updated_topic_lm[word] += lbda*float(topic_dist[word])/float(total_word_frequencies)
            except KeyError:
                pass
            try:
                there = results_word_lm[word]
                updated_topic_lm[word] += float(1.0-lbda)*float(results_word_lm[word])/float(total_word_frequencies)
            except KeyError:
                pass
        lbda = 0.5
        total_collection_words = list(self.doc_collection_frequencies.keys()) + list(results_word_not_lm.keys())
        total_collection_words = list(set(total_collection_words))
        total_collection_num_words = lbda*float(self.collection_num_words) + float(1.0-lbda)*float(sum(results_word_not_lm.values()))
        updated_collection_lm = {word:0 for word in total_collection_words}
        for word in total_collection_words:
            try:
                there = self.doc_collection_frequencies[word]
                updated_collection_lm[word] += lbda *float(self.doc_collection_frequencies[word]) / float(total_collection_num_words)
            except KeyError:
                pass
            try:
                there = results_word_not_lm[word]
                updated_collection_lm[word] += float(1.0 - lbda)*float(results_word_not_lm[word]) / float(total_collection_num_words)
            except KeyError:
                pass

        return updated_topic_lm, results_word_not_lm, results_word_lm, updated_collection_lm
        #return updated_topic_IN, results_not_topic_IN, results_word_lm, updated_precision_lm

    def update_EOR_lm_new(self, gamma, query, precision_lm, updated_topic_IN, results_not_topic_IN,results_word_lm):
        updated_precision_lm = {}
        for word in updated_topic_IN:
            try:
                there = precision_lm[word]
                updated_precision_lm[word] = {}
                #if precision_lm[word]["den"] == 0 
                updated_precision_lm[word]["num"] = precision_lm[word]["num"]/precision_lm[word]["den"]
                updated_precision_lm[word]["den"] = 1
            except KeyError:
                updated_precision_lm[word] = {}
                updated_precision_lm[word]["num"] = 0.00000001
                updated_precision_lm[word]["den"] = 1
            if (word in query) or (self.nowsu):
                try:
                    there = results_word_lm[word]
                    if results_word_lm[word] != 0:
                        try:
                            there = results_not_topic_IN[word]
                            result_precision = (results_word_lm[word])/(results_word_lm[word]+results_not_topic_IN[word])
                        except KeyError:
                            result_precision = (results_word_lm[word])/(results_word_lm[word]+0)
                    else:
                        result_precision = 0.00000001
                except KeyError:
                    result_precision = 0.00000001
            else:
                try:
                    there = results_word_lm[word]
                    if results_word_lm[word] != 0:
                        try:
                            there = results_not_topic_IN[word]
                            result_precision = (results_word_lm[word])/(results_word_lm[word]+results_not_topic_IN[word])
                        except KeyError:
                            result_precision = (results_word_lm[word])/(results_word_lm[word]+0)
                    else:
                        result_precision = updated_precision_lm[word]["num"]     
                except KeyError:
                    result_precision = updated_precision_lm[word]["num"]
            
            updated_precision_lm[word]["num"]  = gamma*updated_precision_lm[word]["num"] + ((1-gamma)*(result_precision))
            updated_precision_lm[word]["den"] = 1
        return updated_precision_lm

    def update_EOR_lm(self, gamma, query, precision_lm, updated_topic_IN, results_not_topic_IN,results_word_lm):
        #Update only at word level
        updated_precision_lm = {}
        for word in updated_topic_IN:
            try:
                there = precision_lm[word]
                updated_precision_lm[word] = {}
                #if precision_lm[word]["den"] == 0 
                updated_precision_lm[word]["num"] = precision_lm[word]["num"]/precision_lm[word]["den"]
                updated_precision_lm[word]["den"] = 1
            except KeyError:
                updated_precision_lm[word] = {}
                updated_precision_lm[word]["num"] = 0
                updated_precision_lm[word]["den"] = 1
            try:
                there = results_word_lm[word]
                try:
                    result_precision = (results_word_lm[word])/(results_word_lm[word]+results_not_topic_IN[word])
                except KeyError:
                    result_precision = (results_word_lm[word])/(results_word_lm[word]+0)
            except KeyError:
                if word in query:
                    result_precision = 0.00000001
                else:
                    result_precision = updated_precision_lm[word]["num"]
            
            updated_precision_lm[word]["num"]  = gamma*updated_precision_lm[word]["num"] + ((1-gamma)*(result_precision))
            updated_precision_lm[word]["den"] = 1
        return updated_precision_lm

    def update_EOR_lm_variation(self, gamma, query, precision_lm, updated_topic_IN, results_not_topic_IN,results_word_lm):
        updated_precision_lm = {}
        for word in updated_topic_IN:
            try:
                there = precision_lm[word]
                updated_precision_lm[word] = {}
                updated_precision_lm[word]["num"] = precision_lm[word]["num"]
                updated_precision_lm[word]["den"] = precision_lm[word]["den"]
            except KeyError:
                updated_precision_lm[word] = {}
                updated_precision_lm[word]["num"] = 0.00000001
                updated_precision_lm[word]["den"] = 0.00000001
            if (word in query) or (self.nowsu):
                updated_precision_lm[word]["num"] = updated_topic_IN[word]*0.02
                try:
                    score = gamma*updated_precision_lm[word]["den"] + ((1-gamma)*results_not_topic_IN[word]*0.98)
                except:
                    score = gamma*updated_precision_lm[word]["den"] + ((1-gamma)*0*0.98)
                updated_precision_lm[word]["den"] = score
            else:
                try:
                    there = results_word_lm[word]
                    if results_word_lm[word] != 0:
                        updated_precision_lm[word]["num"] = updated_topic_IN[word]*0.02
                        try:
                            score = gamma*updated_precision_lm[word]["den"] + ((1-gamma)*results_not_topic_IN[word]*0.98)
                        except:
                            score = gamma*updated_precision_lm[word]["den"] + ((1-gamma)*0*0.98)
                        updated_precision_lm[word]["den"] = score    
                except KeyError:
                    pass
            
        return updated_precision_lm

    def reformulate_query_old(self, query, candidate_queries, results, clicks, topic_num, topic_IN, precision_lm, previous_queries):
        topic_IN = topic_IN.copy()
        updated_topic_IN,updated_not_topic_IN,results_word_lm, updated_precision_lm = self.updating_scores(query, results, clicks, topic_num, topic_IN, precision_lm, self.beta, self.gamma)
        if self.type2_update:
            new_updates_precision_lm = copy.deepcopy(updated_precision_lm)
            for word in new_updates_precision_lm:
                new_updates_precision_lm[word]["den"] =  updated_precision_lm[word]["num"] + updated_precision_lm[word]["den"]
        else:
            new_updates_precision_lm = updated_precision_lm

        query_scores = self.query_formulation.reformulation_method(updated_topic_IN, new_updates_precision_lm, topic_num)
        new_candidates = sorted(query_scores, key=lambda l :l[1], reverse=True)
        if (not self.no_query_filter):
            first_query = new_candidates[0][0].copy()
            first_query.sort()
            while (" ".join(first_query) in previous_queries):
                new_candidates = new_candidates[1:]
                first_query = new_candidates[0][0].copy()
                first_query.sort()
        candidate_queries = new_candidates
        return candidate_queries,updated_topic_IN,updated_precision_lm

    def reformulate_query(self, query, candidate_queries, results, clicks, topic_num, topic_IN, precision_lm, previous_queries):
        topic_IN = topic_IN.copy()
        updated_topic_IN,results_word_not_lm,results_word_lm, updated_collection_lm = self.updating_scores(query, results, clicks, topic_num, topic_IN, precision_lm, self.beta, self.gamma)
        #new_updates_collection_lm = copy.deepcopy(updated_collection_lm)
        query_scores = self.query_formulation.reformulation_method(updated_topic_IN, updated_collection_lm, topic_num)
        new_candidates = sorted(query_scores, key=lambda l :l[1], reverse=True)
        if (not self.no_query_filter):
            first_query = new_candidates[0][0].copy()
            first_query.sort()
            while (" ".join(first_query) in previous_queries):
                new_candidates = new_candidates[1:]
                first_query = new_candidates[0][0].copy()
                first_query.sort()
        candidate_queries = new_candidates
        return candidate_queries,updated_topic_IN,updated_collection_lm

class Reformulation_component_multinomial_for_CCQD_new():
    def __init__(self, parameters = (0.9,0.9), query_formulation_component = None, result_judgement_component = None, collection_lm = None, print_info = False):
        self.query_formulation = query_formulation_component
        self.beta = parameters[0]
        self.gamma = parameters[1]
        self.result_judgement = result_judgement_component
        self.doc_collection_lm_dist = collection_lm
        self.print_info = print_info

    def updating_scores(self, query, results, clicks, topic_num, topic_lm, precision_lm, beta, gamma): 
        #print ("COMING UPDATING SCORES")
        results_relevance = self.result_judgement.evaluate_results(results, topic_num)
        results_word_lm = {}
        results_word_not_lm = {}
        all_words = {}
        for word in query:
            results_word_lm[word] = 0
            results_word_not_lm[word] = 0
        for idx,result in enumerate(results):
            result_content = result["snippet"]
            if "full_text_lm" in result:
                if clicks[idx][1] == 1:
                    result_content += " " + result["full_text_lm"]
            result_content = result_content.replace('...', ' ')
            result_content = preprocess(result_content, stopwords_file, lemmatizing = True)
            result_lm = Counter(result_content.split())
            result_lm_length = sum(result_lm.values())
            #print (results_relevance[idx])
            for word in result_lm:
                all_words[word] = 1
                try:
                    results_word_lm[word] += (result_lm[word]*results_relevance[idx]) 
                    results_word_not_lm[word] += ((1-results_relevance[idx])*(result_lm[word]))
                    #print ("coming here")
                except KeyError:
                    results_word_lm[word] = (result_lm[word]*results_relevance[idx]) 
                    results_word_not_lm[word] = ((1-results_relevance[idx])*(result_lm[word]))  
                
        all_words = list(set(list(all_words.keys()) + list(topic_lm.keys())))
        total_word_score = sum(results_word_lm.values())
        if total_word_score != 0:
            results_word_lm = {word:float(results_word_lm[word])/float(total_word_score) for word in results_word_lm} 
            
        updated_topic_IN= {}
        if total_word_score == 0:
             updated_topic_IN = topic_lm
        else:
            for word in all_words:
                updated_topic_IN[word] = 0
                try:
                    updated_topic_IN[word] += beta*topic_lm[word]
                except KeyError:
                    updated_topic_IN[word] = 0
                try:
                    updated_topic_IN[word] += (1-beta)*float(results_word_lm[word])
                except KeyError:
                    updated_topic_IN[word] = 0

        updated_topic_IN = {word:updated_topic_IN[word] for word in updated_topic_IN if updated_topic_IN[word]!=0}
        total_word_not_score = sum(results_word_not_lm.values())
        results_not_topic_IN = {}
        if (total_word_not_score != 0):
            results_word_not_lm = {word:float(results_word_not_lm[word])/float(total_word_not_score) for word in results_word_not_lm}
            results_not_topic_IN =  results_word_not_lm
        results_not_topic_IN = {word:results_not_topic_IN[word] for word in results_not_topic_IN if results_not_topic_IN[word]!=0}

        if (precision_lm == None):
            precision_lm = {}
            for word in topic_lm:
                precision_lm[word] = {}
                try:
                    precision_lm[word]["notT"] = self.doc_collection_lm_dist[word]
                except KeyError:
                    precision_lm[word]["notT"] = 0.00000001
        #Type1 update
        updated_precision_lm = self.update_EOR_lm(gamma, query, precision_lm, updated_topic_IN, results_not_topic_IN,results_word_lm)
        #Type2 update
        #updated_precision_lm = self.update_EOR_lm_diff(gamma, query, precision_lm, updated_topic_IN, results_not_topic_IN,results_word_lm)

        #print ("WORDS TO UPDATE SCORE: ", words_to_update_score)
        if self.print_info:
            print ('=======REFORMULATION INFO MULTINOMIAL: =========')
            print ('Results: ')
            for idx,result in enumerate(results):
                print ('Docid: ' + result['docid'] + ' Snippet: ' + result["snippet"])
                print ('Relevance: ', results_relevance[idx])
            words_set = [(word,updated_topic_IN[word]) for word in updated_topic_IN]
            words_set = sorted(words_set,key=lambda l:l[1],reverse=True)[:30]
            for (word,score) in words_set:
                try:
                    results_word_lm_score = results_word_lm[word]
                except:
                    results_word_lm_score = 'NA'
                try:
                    results_not_topic_IN_score = results_not_topic_IN[word]
                except:
                    results_not_topic_IN_score = 'NA'
                try:
                    updated_precision_lm_score = updated_precision_lm[word]
                except:
                    updated_precision_lm_score = 'NA'                
                print ('word: {} results_word_lm: {}, results_not_topic_IN: {}, updated_topic_IN: {} updated_precision_lm: {}'.format(word, results_word_lm_score, results_not_topic_IN_score, updated_topic_IN[word], updated_precision_lm_score))               

        return updated_topic_IN,results_not_topic_IN,results_word_lm, updated_precision_lm


    def update_EOR_lm(self, gamma, query, precision_lm, updated_topic_IN, results_not_topic_IN,results_word_lm):
        total_word_not_score = sum(results_not_topic_IN.values())
        updated_precision_lm = {}
        for word in all_words:
            score = 0
            if total_word_not_score!=0:
                try:
                    there = results_not_topic_IN[word]
                    try:
                        there = precision_lm[word]["notT"]
                        score = gamma*precision_lm[word]["notT"]+(1-gamma)*results_not_topic_IN[word]
                    except KeyError:
                        score = gamma*0+(1-gamma)*results_not_topic_IN[word]
                except KeyError:
                    try:
                        there = precision_lm[word]["notT"]
                        score = gamma*precision_lm[word]["notT"]
                    except KeyError:
                        pass
            else:
                try:
                    there = precision_lm[word]["notT"]
                    score = precision_lm[word]["notT"]
                except KeyError:
                    pass
            if score != 0 :
                updated_precision_lm[word]["notT"] = score
        return updated_precision_lm

    def update_EOR_lm_diff(self, gamma, query, result_precision_lm, updated_topic_IN, results_not_topic_IN,results_word_lm):
        if result_precision_lm == None:
            for word in results_word_lm:
                if results_word_lm[word]!=0:
                    result_precision_lm[word]["T"] = results_word_lm[word]
            for word in results_not_topic_IN:
                if results_not_topic_IN[word]!=0:
                    result_precision_lm[word]["notT"] = results_not_topic_IN[word]
            updated_result_precision_lm = result_precision_lm
        else:
            updated_result_precision_lm = {}
            total_rel_scores = sum(results_word_lm.values())
            total_not_rel_scores = sum(results_not_topic_IN.values())
            word_list = list(set(results_word_lm.keys() + result_precision_lm.keys() + results_not_topic_IN.keys()))
            for word in word_list:
                updated_result_precision_lm[word] = {}
                if total_rel_scores != 0:
                    updated_result_precision_lm[word]["T"] = 0
                    try:
                        updated_result_precision_lm[word]["T"] += 0.5*results_word_lm[word] 
                    except:
                        pass                        
                    try:
                        updated_result_precision_lm[word]["T"] += 0.5*result_precision_lm[word]["T"]
                    except:
                        pass
                else:
                    updated_result_precision_lm[word]["T"] = result_precision_lm[word]["notT"]    
                if total_not_rel_scores != 0:
                    updated_result_precision_lm[word]["notT"] = 0
                    try:
                        updated_result_precision_lm[word]["notT"] += 0.5*results_not_topic_IN[word] 
                    except:
                        pass                        
                    try:
                        updated_result_precision_lm[word]["notT"] += 0.5*result_precision_lm[word]["notT"]
                    except:
                        pass
                else:
                    updated_result_precision_lm[word]["notT"] = result_precision_lm[word]["notT"]
             
        return updated_result_precision_lm


    def reformulate_query(self, query, candidate_queries, results, clicks, topic_num, topic_IN, precision_lm, previous_queries):
        updated_topic_IN,updated_not_topic_IN,results_word_lm, updated_precision_lm = self.updating_scores(query, results, clicks, topic_num, topic_IN, precision_lm, self.beta, self.gamma)
        query_scores = self.query_formulation.reformulation_method(updated_topic_IN, updated_precision_lm, topic_num)
        new_candidates = sorted(query_scores, key=lambda l :l[1], reverse=True)
        first_query = new_candidates[0][0].copy()
        first_query.sort()
        while (" ".join(first_query) in previous_queries):
            new_candidates = new_candidates[1:]
            first_query = new_candidates[0][0].copy()
            first_query.sort()
        candidate_queries = new_candidates
        return candidate_queries,updated_topic_IN,updated_precision_lm

class Query_formulation_CCQF():
    def __init__(self, parameters = (0.6,0.2,0.6,1,-1,math.log(0.0001),1), bernoulli = False, query_reformulation_component = None, collection_lm = None, print_info = False, topic_bi_probs = None, topic_descs = None, topic_INs = None, recall_combine = False):
        self.query_reformulation = query_reformulation_component
        self.a1 = parameters[0]
        self.a2 = parameters[1]
        self.a3 = parameters[2]
        self.bigram_scrng_parameter = parameters[3]
        self.effort_constraint = parameters[4]
        if self.effort_constraint == -1:
            self.min_threshold = parameters[5] #math.log(0.0001) 
        else:
            self.min_threshold = math.log(0.000000000001)
        self.min_effort = None
        self.w_len = parameters[6]
            #self.max_threshold = parameters[5][1] #-1 
        self.recall_combine = recall_combine   
        self.intl_cand_qurs_all_tpcs = None
        self.doc_collection_lm_dist = collection_lm
        self.topic_INs = {}
        self.topic_bigram_ct = None
        self.topic_unigram_ct = None
        self.topic_bigram_prob = None
        self.topic_descs = None
        self.print_info = print_info
        self.bernoulli = bernoulli
        if topic_bi_probs != None:
            self.topic_bigram_ct = topic_bi_probs[0]
            self.topic_unigram_ct = topic_bi_probs[1]
            self.topic_bigram_prob = topic_bi_probs[2]
        if topic_descs != None:
            self.topic_descs = topic_descs
        if topic_INs != None:
            self.topic_INs = topic_INs    
        return

    def query_formulation_all_topics(self, dataset):
        if self.topic_bigram_ct == None:
            self.topic_bigram_ct, self.topic_unigram_ct, self.topic_bigram_prob = read_bigram_topic_lm(dataset)
        if self.topic_descs == None:
            self.topic_descs = read_topic_descs(dataset)
        topic_word_scores = {}
        a1 = self.a1
        a2 = self.a2
        a3 = self.a3
        for topic_num in self.topic_descs:
            print ("TOPIC NUM ", topic_num, self.topic_descs[topic_num])
            if topic_num in self.topic_INs:
                topic_IN = self.topic_INs[topic_num]
            else:
                topic_desc = self.topic_descs[topic_num]
                topic_desc = preprocess(topic_desc, stopwords_file, lemmatizing = True)
                if self.bernoulli:
                    topic_IN = language_model_b(topic_desc)
                else:
                    topic_IN = language_model_m(topic_desc)    
                self.topic_INs[topic_num] = topic_IN 
            words_set = list(topic_IN.keys())
            word_list = []
            '''
            for word in words_set:
                feature0 = math.log(topic_IN[word])                    
                try:
                    feature2 = math.log(float(topic_IN[word]*0.02)/float((topic_IN[word]*0.02)+(0.98*self.doc_collection_lm_dist[word])))  
                except:
                    feature2 = math.log(0.00000001)
                word_list += [(word, feature0, feature2)]
            '''
            for word in words_set:
                feature0 = math.log(topic_IN[word])                    
                try:
                    feature2 = float(topic_IN[word]*0.02)/float((topic_IN[word]*0.02)+(0.98*self.doc_collection_lm_dist[word])) 
                except:
                    feature2 = 0.00000001
                word_list += [(word, feature0, feature2)]
            candidate_queries = self.query_generation(word_list, topic_num)
            print ("CANDIDATE QUERIES: ")
            for query in candidate_queries[:10]:
                print(query)        
            topic_word_scores[topic_num] = candidate_queries
        self.intl_cand_qurs_all_tpcs = topic_word_scores
        return topic_word_scores

    def get_initial_queries(self, topic_num):
        return self.intl_cand_qurs_all_tpcs[topic_num]

    def get_topic_IN(self, topic_num):
        return self.topic_INs[topic_num]

    def get_topic_desc(self, topic_num):
        return self.topic_descs[topic_num]

    def reformulation_method(self, updated_topic_IN, precision_lm, topic_num):
        words_set = [(word,updated_topic_IN[word]) for word in updated_topic_IN]
        words_set = sorted(words_set,key=lambda l:l[1],reverse=True)[:30]
        words_set = [word[0] for word in words_set]
        word_scores = []
        for word in words_set:
            feature0 = math.log(updated_topic_IN[word])
            if (precision_lm[word]["den"] == 0) or (precision_lm[word]["num"] == 0):
                feature2 = 0.00000001
            else:
                feature2 = float(precision_lm[word]["num"])/float(precision_lm[word]["den"])
            word_scores += [(word, feature0, feature2)]
        word_scores = sorted(word_scores, key= lambda l: l[2], reverse=True)
        candidate_queries = self.query_generation(word_scores, topic_num)
        print ("CANDIDATE QUERIES: ")
        for query in candidate_queries[:10]:
            print(query)  
        #print ("CANDIDATE QUERIES: ", candidate_queries[:10])        
        return candidate_queries

    def query_generation(self, word_list, topic_num):
        a1 = self.a1
        a2 = self.a2
        a3 = self.a3
        possible_queries = []
        candidate_queries = {}
        current_list_dict = {}
        for word in word_list:
            query_dict = {}
            query_dict[word[0]] = 1
            max_query = [[word[0]], word[1], word[2], self.query_bigram_score([word[0]],topic_num, self.topic_INs[topic_num]), query_dict]
            possible_queries += [max_query]
            #if self.print_info:
            print ('Possible queries: ', max_query)
            s2 = word[2]
            if (s2*1 >= self.min_threshold) or (self.effort_constraint != -1): #and (s2<self.max_threshold):
                current_list_dict[max_query[0][0]] = 1
                candidate_queries[max_query[0][0]] = [max_query[0], self.combine_r_p(max_query[1], max_query[2], max_query[3], len(max_query[0]))]
        if (self.effort_constraint == -1) or (self.effort_constraint >= 2):
            possible_queries_bigrams = []
            for idx,query in enumerate(possible_queries):
                for word in word_list:
                    try:
                        there = query[4][word[0]] 
                    except:                        
                        s1 = ((query[1]*len(query[0]))+word[1])/(len(query[0])+1)
                        s2 = ((query[2]*len(query[0]))+word[2])/(len(query[0])+1)
                        s3 = self.query_bigram_score(query[0]+[word[0]], topic_num, self.topic_INs[topic_num])
                        new_query_dict = query[4].copy() 
                        new_query_dict[word[0]] = 1
                        max_query = [query[0]+[word[0]], s1, s2, s3, new_query_dict] 
                        possible_queries_bigrams += [max_query]
                        if self.print_info:
                            print ('Possible queries: ', max_query)
                        if (max_query[2]*len(max_query[0]) >= self.min_threshold) or (self.effort_constraint != -1):  #(s2*len(max_query[0]) >= self.min_threshold): #and (s2<self.max_threshold):
                            s = max_query[0].copy()
                            s.sort()
                            score = self.combine_r_p(max_query[1], max_query[2], max_query[3], len(max_query[0]))
                            try:
                                there = current_list_dict[" ".join(s)]
                                if score > candidate_queries[" ".join(s)][1]:
                                    candidate_queries[" ".join(s)]= [max_query[0], score]
                            except:
                                candidate_queries[" ".join(s)]= [max_query[0], score]
                                current_list_dict[" ".join(s)] = 1

            possible_queries = possible_queries_bigrams
            word_list = sorted(word_list, key= lambda l: l[2], reverse=True)
            #candidate_queries = sorted(candidate_queries, key = lambda l:l[1], reverse = True)
            #print (len(word_list)) 
            if self.effort_constraint != -1:
                num_iters = self.effort_constraint - 2
            else:
                num_iters =  len(word_list)-2    
            for i in range(num_iters):
                for idx,query in enumerate(possible_queries):
                    max_score = -1
                    max_query = None
                    for iter_idx,word in enumerate(word_list):
                        try:
                            there = query[4][word[0]] 
                        except:                        
                            s1 = ((query[1]*len(query[0]))+word[1])/(len(query[0])+1)
                            s2 = ((query[2]*len(query[0]))+word[2])/(len(query[0])+1)
                            s3 = self.query_bigram_score(query[0]+[word[0]], topic_num, self.topic_INs[topic_num])
                            score = self.combine_r_p(s1, s2, s3, (len(query[0])+1))
                            if (score > max_score) or (iter_idx == 0):
                                max_score = score
                                new_query_dict = query[4].copy() 
                                new_query_dict[word[0]] = 1
                                max_query = [query[0]+[word[0]], s1, s2, s3, new_query_dict] 
                                #max_query_list = 
                    if max_query!=None:
                        possible_queries[idx] = max_query
                        if self.print_info:
                            print ('Possible queries: ', max_query)
                    if (max_query!=None):
                        if (max_query[2]*len(max_query[0]) >= self.min_threshold) or (self.effort_constraint != -1): #and (max_query[2]<self.max_threshold):
                            s = max_query[0].copy()
                            s.sort()
                            score = self.combine_r_p(max_query[1], max_query[2], max_query[3], len(max_query[0]))
                            try:
                                there = current_list_dict[" ".join(s)]
                                #print (candidate_queries[" ".join(s)][1])
                                if score > candidate_queries[" ".join(s)][1]:
                                    #print ('coming here')
                                    candidate_queries[" ".join(s)]= [max_query[0], score]
                            except:
                                candidate_queries[" ".join(s)]= [max_query[0], score]
                                current_list_dict[" ".join(s)] = 1
            #candidate_queries = sorted(candidate_queries, key = lambda l:l[1], reverse = True)
        candidate_queries = sorted(candidate_queries.items(), key = lambda l:l[1][1], reverse = True)
        if self.min_effort != None:
            candidate_queries = [(c[1][0],c[1][1]) for c in candidate_queries if len(c[1][0])>=self.min_effort] 
        else:
            candidate_queries = [(c[1][0],c[1][1]) for c in candidate_queries]
        return candidate_queries

    def combine_r_p(self, s1, s2, s3, query_len):
        if (self.recall_combine == False):        
            try:
                s2 = math.log(s2)
            except:
                s2 = math.log(0.00000001)
            return (self.a1*s1)+(self.a2*s2)+(self.a3*s3)
        else:
            try:
                s2 = math.log(s2)
            except:
                s2 = math.log(0.00000001)
            s1 = math.exp(s1*query_len)
            s3 = math.exp(s3*query_len)
            s1 = self.a1*s3 + (1-self.a1)*s1
            if self.w_len == 1:
                return ((1-self.a2)*(float(1)/float(query_len))*math.log(s1))+(self.a2*s2)
            else:
                return ((1-self.a2)*math.log(s1))+(self.a2*s2)
                
    def query_bigram_score(self, query, topic_num, topic_IN):
        if (self.bigram_scrng_parameter == 1):
            num_phrases = 0
            for idx in range(len(query)-1):
                word1 = query[idx]
                word2 = query[idx+1]
                if (word1+" "+word2) in self.topic_bigram_ct[topic_num]:
                    #num_phrases += self.topic_bigram_ct[topic_num][word1+" "+word2]
                    num_phrases += float(self.topic_bigram_ct[topic_num][word1+" "+word2])/float(max(list(self.topic_bigram_ct[topic_num].values())))
            if len(query) == 1:
                prob = float(1)/float(math.exp(-0)+1)
            else:
                prob = float(1)/float(math.exp(-float(num_phrases)/float(len(query)-1))+1)
            if prob == 0:
                return math.log(0.00000001)
            else:
                return math.log(prob)
        elif (self.bigram_scrng_parameter == 2):
            prob = 0
            try:
                prob += math.log(topic_IN[query[0]])
            except KeyError:
                prob += math.log(0.00000001)
            #d = float(len(list(self.topic_bigram_prob[topic_num].keys())))
            mu = float(4)
            words_d = len(list(self.topic_unigram_ct[topic_num].keys()))
            for idx in range(len(query)-1):
                word1 = query[idx]
                word2 = query[idx+1]
                try:
                    d = self.topic_unigram_ct[topic_num][word1]
                except KeyError:
                    d = 0
                if (word1+" "+word2) in self.topic_bigram_prob[topic_num]:
                    prob += math.log((self.topic_bigram_prob[topic_num][word1+" "+word2]*(float(d)/float(d+mu)))+((float(1)/float(words_d))*(float(mu)/float(d+mu))))
                else:
                    prob += math.log((float(1)/float(words_d))*(float(mu)/float(d+mu)))
            prob = float(prob)/float(len(query))
            return prob

class Query_formulation_PRE():
    def __init__(self, parameters = (0.6,0.2,-1,math.log(0.0001),1), bernoulli = False, query_reformulation_component = None, collection_lm = None, print_info = False, topic_bi_probs = None, topic_descs = None, topic_INs = None, setting = None):
        '''
        class for the Query formulation process - can simulate all instantiations of PRE framework
        :param parameters: parameters of recall and precision a1, a2, a3, effort min threshold, effort min threshold, length normalization
        :param bernoulli: whether the Recall, precision should be bernoulli or multinomila probabilities
        :param query_reformulation_component:
        :param collection_lm: collection language model
        :param print_info:
        :param topic_bi_probs: bigram counts and probabilities in sessiont rack topics
        :param topic_descs:
        :param topic_INs:
        :param setting: to specify the combination of recall and precision method - "old_r_new_p"
        '''

        self.query_reformulation = query_reformulation_component
        self.alpha = parameters[0]
        self.beta = parameters[1]
        #self.a1 = parameters[0]
        #self.a2 = parameters[1]
        #self.a3 = parameters[2]
        self.effort_constraint = parameters[2]
        '''
        if self.effort_constraint == -1:
            self.min_threshold = parameters[3] #math.log(0.0001)
        else:
            self.min_threshold = math.log(0.000000000001)
        '''
        self.min_effort = None
            #self.max_threshold = parameters[5][1] #-1
        self.w_len = parameters[4]
        self.Cr = False
        self.Dr = False
        self.Dp = False
        self.Cp = False
        self.r_setting = setting[0]
        self.p_setting = setting[1]

        '''
        self.old_r_new_p = False
        self.old_r_old_p = False
        self.new_r_old_p = False
        self.heuristic_r_new_p = False
        self.new_r_new_p = False
        self.heuristic_r_old_p = False
        self.com_r_new_p = False
        self.old_r_p3 = False
        self.heucom_r_old_p = False
        self.com_r_p3 = False
        self.com_r_old_p = False
        self.new_r_p3 = False
        if setting == 'old_r_new_p':
            self.old_r_new_p = True
        elif setting == 'old_r_old_p':
            self.old_r_old_p = True
        elif setting == 'new_r_old_p':
            self.new_r_old_p = True
        elif setting == "heuristic_r_new_p":
            self.heuristic_r_new_p = True
        elif setting == "heuristic_r_old_p":
            self.heuristic_r_old_p = True
        elif setting == "com_r_new_p":
            self.com_r_new_p = True
        elif setting == "old_r_p3":
            self.old_r_p3 = True
        elif setting == "heucom_r_old_p":
            self.heucom_r_old_p = True
        elif setting == "new_r_new_p":
            self.new_r_new_p = True
        elif setting == "com_r_p3":
            self.com_r_p3 = True
        elif setting == "new_r_p3":
            self.new_r_p3 = True
        elif setting == "com_r_old_p":
            self.com_r_old_p = True
        '''
        self.intl_cand_qurs_all_tpcs = None
        self.doc_collection_lm_dist = collection_lm
        self.topic_INs = {}
        self.topic_bigram_ct = None
        self.topic_unigram_ct = None
        self.topic_bigram_prob = None
        self.topic_descs = None
        self.print_info = print_info
        self.bernoulli = bernoulli
        '''
        self.specific_doc_collection_lm_dist = None
        if self.specific_doc_collection_lm_dist != None:
            self.doc_collection_lm_dist = {word: self.doc_collection_lm_dist[word] * alpha for word in
                                           self.doc_collection_lm_dist}
            for word in self.specific_doc_collection_lm_dist:
                try:
                    self.doc_collection_lm_dist[word] += (1 - alpha) * self.specific_doc_collection_lm_dist[word]
                except KeyError:
                    self.doc_collection_lm_dist[word] = (1 - alpha) * self.specific_doc_collection_lm_dist[word]
        '''
        if topic_bi_probs != None:
            self.topic_bigram_ct = topic_bi_probs[0]
            self.topic_unigram_ct = topic_bi_probs[1]
            self.topic_bigram_prob = topic_bi_probs[2]
        if topic_descs != None:
            self.topic_descs = topic_descs
        if topic_INs != None:
            self.topic_INs = topic_INs
        '''
        if (self.old_r_new_p == True) or (self.old_r_old_p == True) or (self.heuristic_r_old_p == True) or (
                self.heuristic_r_new_p == True) or (self.old_r_p3):
            x = self.a1
            y = self.a2
            self.a1 = x
            self.a2 = (1.0 - x) * y
            self.a3 = (1.0 - x) * (1.0 - y)
        '''
        return

    def query_formulation_all_topics(self, dataset):
        '''
        makes/uses topic lm, bg lm
        computes feature[0] - recall of a word, feature[2] - precision of a word.
        :param dataset:
        :return: initial queries
        '''
        if self.topic_bigram_ct == None:
            self.topic_bigram_ct, self.topic_unigram_ct, self.topic_bigram_prob = read_bigram_topic_lm(dataset)
        if self.topic_descs == None:
            self.topic_descs = read_topic_descs(dataset)
        topic_word_scores = {}
        for topic_num in self.topic_descs:
            print("TOPIC NUM ", topic_num, self.topic_descs[topic_num])
            if topic_num in self.topic_INs:
                topic_IN = self.topic_INs[topic_num]
            else:
                topic_desc = self.topic_descs[topic_num]
                topic_desc = preprocess(topic_desc, stopwords_file, lemmatizing=True)
                if self.bernoulli:
                    topic_IN = language_model_b(topic_desc)
                else:
                    topic_IN = language_model_m(topic_desc)
                self.topic_INs[topic_num] = topic_IN

            words_set = list(topic_IN.keys())
            word_list = []
            for word in words_set:
                feature0 = [topic_IN[word], topic_IN[word]]
                try:
                    feature2 = [self.doc_collection_lm_dist[word], self.doc_collection_lm_dist[word]]
                except KeyError:
                    feature2 = [0.00000001,0.00000001]
                word_list += [(word, feature0, feature2)]
            candidate_queries = self.query_generation(word_list, topic_num)
            print("CANDIDATE QUERIES: ")
            for query in candidate_queries[:10]:
                print(query)
            topic_word_scores[topic_num] = candidate_queries
        self.intl_cand_qurs_all_tpcs = topic_word_scores
        return topic_word_scores

    def get_initial_queries(self, topic_num):
        return self.intl_cand_qurs_all_tpcs[topic_num]

    def get_topic_IN(self, topic_num):
        return self.topic_INs[topic_num]

    def get_topic_desc(self, topic_num):
        return self.topic_descs[topic_num]

    def reformulation_method(self, updated_topic_IN, updated_collection_lm, topic_num):
        words_set = [(word, updated_topic_IN[word]) for word in updated_topic_IN]
        words_set = sorted(words_set, key=lambda l: l[1], reverse=True)[:30]
        words_set = [word[0] for word in words_set]
        word_list = []
        for word in words_set:
            feature0 = [updated_topic_IN[word], updated_topic_IN[word]]
            try:
                feature2 = [updated_collection_lm[word], updated_collection_lm[word]]
            except KeyError:
                feature2 = [0.00000001, 0.00000001]
            word_list += [(word, feature0, feature2)]
        print ("SUM OF ALL COLLECTION WORD PROBABILITIES: ", sum(updated_collection_lm.values()))
        candidate_queries = self.query_generation(word_list, topic_num)
        print("CANDIDATE QUERIES: ")
        for query in candidate_queries[:10]:
            print(query)
        return candidate_queries


    def query_generation(self, word_list, topic_num):
        '''
        Using the single word recall and precision, queries are generated and their scores are combined (combine_r_p function)
        query  generation: all unigrams and bigrams are candidate queries, for each bigram extend with best word to get candidate trigrams and so on..
        recall and precision are computed in various ways through diff. methods.
        Deciphering methods:
        old_r: R12g
        new_r: R3
        heuristic_r: R_heu
        com_r: R12a
        heucomr: R_heu combined arithmetically
        old_p: average word precision
        new_p: whole query precision
        self.a1: bigram unigram combination one.
        p_3: the p(Q|T) and P(Q|C) are first length normalized and then precision(p(T|Q)) is computed, so p_3 don't need extra length normalization.

        :param word_list: words to be used for making queries
        :param topic_num:
        :return: (queries, scores)
        '''
        possible_queries = []
        candidate_queries = {}
        for word in word_list:
            query_dict = {}
            query_dict[word[0]] = 1
            max_query = [[word[0]], word[1] + word[1], word[2], query_dict, [[word[1][0]], [word[1][1]]], [word[2][0]]]
            possible_queries += [max_query]
            if self.print_info:
                print('Possible queries: ', max_query)
            candidate_queries[max_query[0][0]] = [max_query[0], self.combine_r_p_new(max_query[1], max_query[2], max_query[4], max_query[5], 1, self.r_setting, self.p_setting)]
        words_d = len(list(self.topic_unigram_ct[topic_num].keys()))
        if (self.effort_constraint == -1) or (self.effort_constraint >= 2):
            possible_queries_bigrams = []
            for idx, query in enumerate(possible_queries):
                for word in word_list:
                    try:
                        there = query[3][word[0]]
                    except KeyError:
                        feature0, probability_list = self.cr_dr_uni_bigram_probability_2(query[1], query[0], word, topic_num, words_d, query[4])
                        #feature0 += self.disjunctive_r_uni_bigram_probability(query[1], query[0], word, topic_num, words_d)
                        recall_unibi_prob_list = probability_list
                        feature2 = [0, 0]
                        feature2[0] = query[2][0] * word[2][0]
                        feature2[1] = query[2][1] + word[2][1]
                        precision_list = query[5] + [word[2][0]]
                        '''
                        if self.new_r_old_p or self.old_r_old_p or (
                                self.heuristic_r_old_p == True) or self.heucom_r_old_p:
                            feature2[2] = query[2][2] + word[2][2]
                        else:
                            if self.old_r_p3 or self.com_r_p3:
                                feature2[2] = self.new_p_3(feature2[0], feature2[1], 2)
                            else:
                                feature2[2] = float(feature2[0] * 0.02) / float(feature2[0] * 0.02 + feature2[1] * 0.98)
                        '''
                        new_query_dict = query[3].copy()
                        new_query_dict[word[0]] = 1
                        max_query = [query[0] + [word[0]], feature0, feature2, new_query_dict, recall_unibi_prob_list, precision_list]
                        possible_queries_bigrams += [max_query]
                        if self.print_info:
                            print('Possible queries: ', max_query)

                        s = max_query[0].copy()
                        s.sort()
                        score = self.combine_r_p_new(max_query[1], max_query[2], max_query[4], max_query[5], 2, self.r_setting, self.p_setting)
                        try:
                            there = candidate_queries[" ".join(s)]
                            if score > candidate_queries[" ".join(s)][1]:
                                candidate_queries[" ".join(s)] = [max_query[0], score]
                        except:
                            candidate_queries[" ".join(s)] = [max_query[0], score]

            possible_queries = possible_queries_bigrams
            # candidate_queries = sorted(candidate_queries, key = lambda l:l[1], reverse = True)
            # print (len(word_list))
            if self.effort_constraint != -1:
                num_iters = self.effort_constraint - 2
            else:
                num_iters = len(word_list) - 2
            for i in range(num_iters):
                for idx, query in enumerate(possible_queries):
                    max_score = -1
                    max_query = None
                    for iter_idx, word in enumerate(word_list):
                        try:
                            there = query[3][word[0]]
                        except:
                            # recall computations part
                            feature0, probability_list = self.cr_dr_uni_bigram_probability_2(query[1], query[0], word, topic_num, words_d, query[4])
                            #feature0 += self.disjunctive_r_uni_bigram_probability(query[1][2:4], query[0], word, topic_num, words_d)
                            recall_unibi_prob_list = probability_list
                            feature2 = [0, 0]
                            feature2[0] = query[2][0] * query[2][0]
                            feature2[1] = query[2][1] + word[2][1]
                            precision_list = query[5] + [word[2][1]]
                            score = self.combine_r_p_new(feature0, feature2, recall_unibi_prob_list, precision_list, i + 3, self.r_setting, self.p_setting)
                            if (score > max_score) or (iter_idx == 0):
                                max_score = score
                                new_query_dict = query[3].copy()
                                new_query_dict[word[0]] = 1
                                max_query = [query[0] + [word[0]], feature0, feature2, new_query_dict, recall_unibi_prob_list, precision_list]
                                # max_query_list =
                    if max_query != None:
                        possible_queries[idx] = max_query
                        if self.print_info:
                            print('Possible queries: ', max_query)
                    if (max_query != None):
                        s = max_query[0].copy()
                        s.sort()
                        score = self.combine_r_p_new(max_query[1], max_query[2], max_query[4], max_query[5], i + 3, self.r_setting, self.p_setting)
                        try:
                            there = candidate_queries[" ".join(s)]
                            if score > candidate_queries[" ".join(s)][1]:
                                candidate_queries[" ".join(s)] = [max_query[0], score]
                        except:
                            candidate_queries[" ".join(s)] = [max_query[0], score]
            # candidate_queries = sorted(candidate_queries, key = lambda l:l[1], reverse = True)
        # candidate_queries = {query: [candidate_queries[query][0],float(candidate_queries[query][1])/float(len(candidate_queries[query][0]))] for query in candidate_queries}
        candidate_queries = sorted(candidate_queries.items(), key=lambda l: l[1][1], reverse=True)
        if self.min_effort != None:
            candidate_queries = [(c[1][0], c[1][1]) for c in candidate_queries if len(c[1][0]) >= self.min_effort]
        else:
            candidate_queries = [(c[1][0], c[1][1]) for c in candidate_queries]
        return candidate_queries

    def uni_bigram_probability(self, probability, query, word, topic_num, words_d):
        mu = float(4)
        try:
            d = self.topic_unigram_ct[topic_num][query[-1]]
        except KeyError:
            d = 0
        try:
            bigram_prob = (self.topic_bigram_prob[topic_num][query[-1] + " " + word[0]] * (
                        float(d) / float(d + mu))) + ((float(1) / float(words_d)) * (float(mu) / float(d + mu)))
        except KeyError:
            bigram_prob = (float(1) / float(words_d)) * (float(mu) / float(d + mu))
        '''
        try:
            bigram_prob = self.topic_bigram_prob[topic_num][query[-1] + " " + word[0]]
        except KeyError:
            bigram_prob = 0.00000001
        '''
        unigram_prob = word[1]
        if unigram_prob == 0:
            unigram_prob = 0.00000001
        if bigram_prob == 0:
            bigram_prob = 0.00000001
        if self.print_info:
            print('Uni Bigram probability Query: ', query, word[0], bigram_prob, unigram_prob)
        new_probability = probability * (self.a1 * bigram_prob + (1 - self.a1) * unigram_prob)
        return new_probability

    def cr_dr_uni_bigram_probability_2(self, probability, query, word, topic_num, words_d, recall_list):
        mu = float(4)
        try:
            d = self.topic_unigram_ct[topic_num][query[-1]]
        except KeyError:
            d = 0
        try:
            bigram_prob = (self.topic_bigram_prob[topic_num][query[-1] + " " + word[0]] * (
                        float(d) / float(d + mu))) + ((float(1) / float(words_d)) * (float(mu) / float(d + mu)))
        except KeyError:
            bigram_prob = (float(1) / float(words_d)) * (float(mu) / float(d + mu))
        unigram_prob = word[1][0]
        if self.print_info:
            print('Uni Bigram probability Query: ', query, word[0], bigram_prob, unigram_prob)
        if unigram_prob == 0: unigram_prob = 0.00000001
        if bigram_prob == 0: bigram_prob = 0.00000001
        # print (probability[0], probability[1], unigram_prob, bigram_prob)
        new_probability = [probability[0] * unigram_prob, probability[1] * bigram_prob, probability[2] + unigram_prob, probability[3] + bigram_prob]
        new_probability_list = [ recall_list[0] + [unigram_prob], recall_list[1] + [bigram_prob]]
        # print (probability[0],probability[1], probability)
        return new_probability,new_probability_list
    '''
    def disjunctive_r_uni_bigram_probability(self, probability, query, word, topic_num, words_d):
        mu = float(4)
        try:
            d = self.topic_unigram_ct[topic_num][query[-1]]
        except KeyError:
            d = 0
        try:
            bigram_prob = (self.topic_bigram_prob[topic_num][query[-1] + " " + word[0]] * (
                    float(d) / float(d + mu))) + ((float(1) / float(words_d)) * (float(mu) / float(d + mu)))
        except KeyError:
            bigram_prob = (float(1) / float(words_d)) * (float(mu) / float(d + mu))
        unigram_prob = word[1][0]
        if self.print_info:
            print('Uni Bigram probability Query: ', query, word[0], bigram_prob, unigram_prob)
        if unigram_prob == 0: unigram_prob = 0.00000001
        if bigram_prob == 0: bigram_prob = 0.00000001
        # print (probability[0], probability[1], unigram_prob, bigram_prob)
        new_probability = [probability[2] + unigram_prob, probability[3] + bigram_prob]
        # print (probability[0],probability[1], probability)
        return new_probability
    '''
    def heuristic_bigram_probability(self, probability, query, word, topic_num, words_d):
        num_phrases = 0
        query = query + [word[0]]
        for idx in range(len(query) - 1):
            word1 = query[idx]
            word2 = query[idx + 1]
            if (word1 + " " + word2) in self.topic_bigram_ct[topic_num]:
                num_phrases += float(self.topic_bigram_ct[topic_num][word1 + " " + word2]) / float(
                    max(list(self.topic_bigram_ct[topic_num].values())))
                # num_phrases += (self.topic_bigram_prob[topic_num][word1+" "+word2])
                # num_phrases += float(self.topic_bigram_ct[topic_num][word1+" "+word2])
        if len(query) == 1:
            prob = float(1) / float(math.exp(-0) + 1)
        else:
            if self.w_len == 1:
                prob = float(1) / float(math.exp(-float(num_phrases) / float(len(query) - 1)) + 1)
            elif self.w_len == 0:
                prob = float(1) / float(math.exp(-float(num_phrases)) + 1)

        unigram_prob = word[1][0]
        if self.print_info:
            print('Uni Bigram probability Query: ', query, bigram_prob, unigram_prob)
        if unigram_prob == 0:
            unigram_prob = 0.00000001

        new_probability = [probability[0] * unigram_prob, prob]
        return new_probability

    def new_p_3(self, feature0, feature1, query_len):
        if self.w_len == 1:
            feature0 = math.pow(feature0, float(1) / float(query_len))
            feature1 = math.pow(feature1, float(1) / float(query_len))
        elif self.w_len == 0:
            feature0 = feature0
            feature1 = feature1
        return float(feature0 * 0.02) / float(feature0 * 0.02 + feature1 * 0.98)

    def combine_r_p_new(self, feature0, feature2, recall_list, precision_list, query_len, r_setting, p_setting):
        '''
        Deciphering parameters:
        self.a1: is the inside parameter between unigram and bigram or it is one of the outside parameter for recall: a1, a3 are parameters for unigram and bigram respectively
        self.a2: parameter for precision.  (a1+a3 will be (1-a2))
        self.a3: is either (1-a2) when a1 is inside parameter, or it is one of the outside parameters for recall:: a1, a3 are parameters for unigram and bigram respectively
        ONLY new_r_old_p, new_r_new_p it is reverse: a2 is for recall, a3 is for precision, a2=1-a3. a1 is inside parameter for recall.
        :param feature0:
        :param feature2:
        :param query_len:
        :return:
        '''
        if r_setting == "Cr_R12g" and p_setting == "Cp_add":
            length_exponent = (float(1) / float(query_len))
            recall_score = (feature0[0] ** (self.beta)) * (feature0[1] ** (1 - self.beta))
            precision_score = feature2[0]
            if self.w_len == 1:
                recall_score = (float(recall_score)) ** (length_exponent)
                precision_score = float(precision_score) ** (length_exponent)
            # precision_score = float(1.0)/float(1.0+precision_score)
            precision_score = (1-precision_score)
            precision_score = (precision_score) ** float(1 - self.alpha)
            recall_score = float(recall_score) ** float(self.alpha)
            ratio = float(recall_score * precision_score)
            return ratio
        if r_setting == "Dr_R12g" and p_setting == "Dp_add":
            length_exponent = (float(1) / float(query_len))
            recall_score = (feature0[2] ** (self.beta)) * (feature0[3] ** (1 - self.beta))
            precision_score = feature2[1]
            if self.w_len == 1:
                recall_score = (float(recall_score)) / (float(query_len))
                precision_score = float(precision_score) / (float(query_len))
            # precision_score = float(1.0)/float(1.0+precision_score)
            precision_score = (1-precision_score)
            precision_score = (precision_score) ** float(1 - self.alpha)
            recall_score = float(recall_score) ** float(self.alpha)
            ratio = float(recall_score * precision_score)
            return ratio
        if r_setting == "Cr_R12g" and p_setting == "Dp_add":
            length_exponent = (float(1) / float(query_len))
            recall_score = (feature0[0] ** (self.beta)) * (feature0[1] ** (1 - self.beta))
            precision_score = feature2[1]
            if self.w_len == 1:
                recall_score = (float(recall_score)) ** (length_exponent)
                precision_score = float(precision_score) / (float(query_len))
            # precision_score = float(1.0)/float(1.0+precision_score)
            precision_score = (1-precision_score)
            precision_score = (precision_score) ** float(1 - self.alpha)
            recall_score = float(recall_score) ** float(self.alpha)
            ratio = float(recall_score * precision_score)
            return ratio
        if r_setting == "Dr_R12g" and p_setting == "Cp_add":
            length_exponent = (float(1) / float(query_len))
            recall_score = (feature0[2] ** (self.beta)) * (feature0[3] ** (1 - self.beta))
            precision_score = feature2[0]
            if self.w_len == 1:
                recall_score = (float(recall_score)) / (float(query_len))
                precision_score = float(precision_score)**(length_exponent)
            # precision_score = float(1.0)/float(1.0+precision_score)
            #print (precision_score)
            precision_score = (1-precision_score)
            precision_score = (precision_score) ** float(1 - self.alpha)
            recall_score = float(recall_score) ** float(self.alpha)
            ratio = float(recall_score * precision_score)
            return ratio
        if r_setting == "Cr_R12g" and p_setting == "Cp":
            length_exponent = (float(1) / float(query_len))
            recall_score =  (feature0[0]**(self.beta))*(feature0[1]**(1-self.beta))
            precision_score = feature2[0]
            if self.w_len == 1:
                recall_score = (float(recall_score))** (length_exponent)
                precision_score  =  float(precision_score) ** (length_exponent)
            #precision_score = float(1.0)/float(1.0+precision_score)
            precision_score = (precision_score) ** (1 - self.alpha)
            recall_score = recall_score ** (self.alpha)
            ratio = float(recall_score)/float(precision_score)
            #ratio = float(ratio) / float(ratio+1.0)
            return ratio
        if r_setting == "Cr_R12a" and p_setting == "Cp":
            length_exponent = (float(1) / float(query_len))
            recall_score = (feature0[0]*(self.beta)) + (feature0[1]*(1 - self.beta))
            precision_score = (feature2[0])**(1-self.alpha)
            recall_score = recall_score ** (self.alpha)
            if self.w_len == 1:
                ratio = (float(recall_score) / float(precision_score)) ** (length_exponent)
            else:
                ratio = (float(recall_score) / float(precision_score))
            #ratio = float(ratio) / float(ratio + 1.0)
            return ratio
        if r_setting == "Dr_R12g" and p_setting == "Dp":
            length_exponent = (float(1) / float(query_len))
            recall_score = (feature0[2]**(self.beta))* (feature0[3]**(1-self.beta))
            if self.w_len == 1:
                recall_score = float(recall_score)/float(query_len)
                precision_score = float(feature2[1])/float(query_len)
            else:
                recall_score = float(recall_score)
                precision_score = float(feature2[1])
            recall_score = recall_score ** (self.alpha)
            precision_score = precision_score** (1-self.alpha)
            ratio = (float(recall_score) / float(precision_score))
            #ratio = float(ratio) / float(ratio + 1.0)
            return ratio
        if r_setting == "Dr_R12a" and p_setting == "Dp":
            length_exponent = (float(1) / float(query_len))
            recall_score = (feature0[2]*(self.beta)) + (feature0[3]*(1 - self.beta))
            if self.w_len == 1:
                recall_score = float(recall_score) / float(query_len)
                precision_score = float(feature2[1]) / float(query_len)
            else:
                recall_score = float(recall_score)
                precision_score = float(feature2[1])
            recall_score = recall_score ** (self.alpha)
            precision_score = precision_score ** (1 - self.alpha)
            ratio = (float(recall_score) / float(precision_score))
            #ratio = float(ratio) / float(ratio + 1.0)
            return ratio
        if r_setting == "Cr_R12g" and p_setting == "Dr_R1_Dp":
            length_exponent = (float(1) / float(query_len))
            recall_score = (feature0[0] ** (self.beta)) * (feature0[1] ** (1 - self.beta))
            if self.w_len == 1:
                Dr_score = float(feature0[2]) / float(query_len)
                Dp_score = float(feature2[1]) / float(query_len)
            else:
                Dr_score = float(feature0[2])
                Dp_score = float(feature2[1])
            precision_score = float(Dr_score** (float(self.alpha)/float(2.0)))/float(Dp_score** (1-self.alpha))
            recall_score = recall_score ** (float(self.alpha)/float(2.0))

            if self.w_len == 1:
                score = (recall_score**(length_exponent))*precision_score
            else:
                score = recall_score * precision_score

            #ratio = float(ratio) / float(ratio + 1.0)
            return score
        if r_setting == "Cr_R1" and p_setting == "Dr_R1_Dp_wd":
            length_exponent = (float(1) / float(query_len))
            recall_score = feature0[0]
            precision = sum([(recall_list[0][idx]**(self.alpha/float(2.0)))/(precision_list[idx] ** (1 - self.alpha)) for idx in range(len(precision_list))])
            #precision = [float(p)/float(p+1.0) for p in precision]
            if self.w_len == 1:
                recall_score = float(recall_score)**float(length_exponent)
                precision_score = float(precision) / float(query_len)
            recall_score = recall_score**(self.alpha/float(2.0))
            #recall_score = (recall_score)/(recall_score+1.0)
            score = recall_score * precision_score
            #ratio = float(ratio) / float(ratio + 1.0)
            return score
        if r_setting == "Cr_R1" and p_setting == "Dp":
            length_exponent = (float(1) / float(query_len))
            recall_score = (feature0[0])
            if self.w_len == 1:
                recall_score = float(recall_score)**float(length_exponent)
                precision_score = float(feature2[1]) / float(query_len)
            else:
                recall_score = float(recall_score)
                precision_score = float(feature2[1])
            recall_score = recall_score ** (self.alpha)
            precision_score = precision_score ** (1 - self.alpha)
            ratio = (float(recall_score) / float(precision_score))
            #ratio = float(ratio) / float(ratio + 1.0)
            return ratio
        if r_setting == "Cr_R1" and p_setting == "Dp_wd":
            length_exponent = (float(1) / float(query_len))
            recall_score = (feature0[0])
            precision = sum([(float(1.0)/float(p))**(1 - self.alpha) for p in precision_list])
            #precision = sum([float(p)/float(p+1.0) for p in precision])
            if self.w_len == 1:
                recall_score = float(recall_score)**float(length_exponent)
                precision_score = float(precision) / float(query_len)
            else:
                recall_score = float(recall_score)
                precision_score = float(precision)
            recall_score = recall_score ** (self.alpha)
            #recall_score = (recall_score) / (recall_score + 1.0)
            ratio = (float(recall_score)*float(precision_score))
            #ratio = float(ratio) / float(ratio + 1.0)
            return ratio
        if r_setting == "Dr_R1_wd" and p_setting == "Dp_wd":
            length_exponent = (float(1) / float(query_len))
            total_score = sum([(recall_list[0][idx]**(self.alpha))/(precision_list[idx] ** (1 - self.alpha)) for idx in range(len(precision_list))])
            #total_score = sum([float(t)/float(t+1.0) for t in total_score])
            if self.w_len == 1:
                total_score = float(total_score) / float(query_len)
            return total_score
        if r_setting == "Dr_R1_wd" and p_setting == "Cp":
            length_exponent = (float(1) / float(query_len))
            recall_score = sum([(recall_list[0][idx]**(self.alpha)) for idx in range(len(precision_list))])
            precision_score = (feature2[0])
            if self.w_len == 1:
                recall_score = float(recall_score) / float(query_len)
                precision_score = float(precision_score)**(length_exponent)
            precision_score = (precision_score) ** (1 - self.alpha)
            ratio = (float(recall_score) / float(precision_score))
            return ratio
        if r_setting == "Dr_R1" and p_setting == "Cp":
            length_exponent = (float(1) / float(query_len))
            recall_score = feature0[2]
            precision_score = (feature2[0])
            if self.w_len == 1:
                recall_score = float(recall_score) / float(query_len)
                precision_score = float(precision_score)**(length_exponent)
            recall_score = recall_score**(self.alpha)
            precision_score = (precision_score) ** (1 - self.alpha)
            ratio = (float(recall_score) / float(precision_score))
            return ratio

        if r_setting == "Dr_R1" and p_setting == "Dp_wd":
            length_exponent = (float(1) / float(query_len))
            recall_score = feature0[2]
            precision = sum([(float(1.0) / float(p)) ** (1 - self.alpha) for p in precision_list])
            #precision = sum([float(p) / float(p + 1.0) for p in precision])
            if self.w_len == 1:
                recall_score = float(recall_score) / float(query_len)
                precision_score = float(precision) / float(query_len)
            total_score = (recall_score**(self.alpha))*precision_score
            return total_score

        if r_setting == "Dr_R1_wd" and p_setting == "Dp":
            length_exponent = (float(1) / float(query_len))
            recall_score = sum([(recall_list[0][idx] ** (self.alpha)) for idx in range(len(precision_list))])
            precision_score = feature2[1]
            if self.w_len == 1:
                recall_score = float(recall_score) / float(query_len)
                precision_score = float(precision_score) / float(query_len)
            precision_score = precision_score**(1-self.alpha)
            return (float(recall_score)/float(precision_score))

    def combine_r_p(self, feature0, feature2, query_len):
        '''
        Deciphering parameters:
        self.a1: is the inside parameter between unigram and bigram or it is one of the outside parameter for recall: a1, a3 are parameters for unigram and bigram respectively
        self.a2: parameter for precision.  (a1+a3 will be (1-a2))
        self.a3: is either (1-a2) when a1 is inside parameter, or it is one of the outside parameters for recall:: a1, a3 are parameters for unigram and bigram respectively
        ONLY new_r_old_p, new_r_new_p it is reverse: a2 is for recall, a3 is for precision, a2=1-a3. a1 is inside parameter for recall.
        :param feature0:
        :param feature2:
        :param query_len:
        :return:
        '''
        if self.w_len == 1:
            if (self.heuristic_r_old_p == True) or self.heucom_r_old_p:
                feature2 = float(feature2) / float(query_len)
                if self.heuristic_r_old_p:
                    score1 = ((float(1) / float(query_len)) * self.a1 * math.log(feature0[0])) + (
                                self.a3 * math.log(feature0[1]))
                    return score1 + self.a2 * math.log(feature2)
                elif self.heucom_r_old_p:
                    score1 = (float(1) / float(query_len)) * (
                                self.a3 * math.log(self.a1 * feature0[0] + (1 - self.a1) * feature0[1]))
                    return score1 + self.a2 * math.log(feature2)
            elif self.old_r_old_p or self.new_r_old_p:
                feature2 = float(feature2) / float(query_len)
                if self.old_r_old_p or (self.heuristic_r_old_p == True):
                    if self.print_info:
                        print("COMBINE R P :", feature0[0], feature0[1], feature2)
                    score1 = (float(1) / float(query_len)) * (
                                self.a1 * math.log(feature0[0]) + self.a3 * math.log(feature0[1]))
                    return score1 + self.a2 * math.log(feature2)
                elif self.new_r_old_p:
                    score1 = (float(1) / float(query_len)) * (self.a2 * math.log(feature0))
                    return score1 + self.a3 * math.log(feature2)
            elif self.old_r_p3:
                score1 = (float(1) / float(query_len)) * (
                            self.a1 * math.log(feature0[0]) + self.a3 * math.log(feature0[1])) + self.a2 * math.log(
                    feature2)
                return score1
            else:
                if self.heuristic_r_new_p:
                    if self.print_info:
                        print("COMBINE R P :", feature0[0], feature0[1], feature2)
                    score1 = ((float(1) / float(query_len)) * (
                                self.a1 * math.log(feature0[0]) + self.a2 * math.log(feature2))) + (
                                         self.a3 * math.log(feature0[1]))
                    return score1
                elif self.old_r_new_p == True:
                    if self.print_info:
                        print("COMBINE R P :", feature0[0], feature0[1], feature2)
                    score1 = (float(1) / float(query_len)) * (
                                self.a1 * math.log(feature0[0]) + self.a3 * math.log(feature0[1]) + self.a2 * math.log(
                            feature2))
                    return score1
                elif self.com_r_new_p:
                    score1 = (float(1) / float(query_len)) * (self.a3 * math.log(
                        self.a1 * feature0[0] + (1 - self.a1) * feature0[1]) + self.a2 * math.log(feature2))
                    return score1
                elif self.new_r_new_p:
                    return (float(1) / float(query_len)) * (self.a2 * math.log(feature0) + self.a3 * math.log(feature2))
                elif self.com_r_p3:
                    score1 = (float(1) / float(query_len)) * (self.a3 * math.log(
                        self.a1 * feature0[0] + (1 - self.a1) * feature0[1])) + self.a2 * math.log(feature2)
                    return score1
                elif self.com_r_old_p:
                    pass
                elif self.new_r_p3:
                    pass
        else:
            if self.old_r_old_p or self.new_r_old_p or (self.heuristic_r_old_p == True):
                feature2 = float(feature2) / float(query_len)
                if self.old_r_old_p or (self.heuristic_r_old_p == True):
                    if self.print_info:
                        print("COMBINE R P :", feature0[0], feature0[1], feature2)
                    score1 = (self.a1 * math.log(feature0[0]) + self.a3 * math.log(feature0[1]))
                    return score1 + self.a2 * math.log(feature2)
                elif self.new_r_old_p:
                    score1 = (self.a2 * math.log(feature0))
                    return score1 + self.a3 * math.log(feature2)
            elif self.heucom_r_old_p:
                feature2 = float(feature2) / float(query_len)
                score1 = (float(1) / float(query_len)) * (
                            self.a3 * math.log(self.a1 * feature0[0] + (1 - self.a1) * feature0[1]))
                return score1 + self.a2 * math.log(feature2)
            elif self.old_r_p3:
                score1 = (self.a1 * math.log(feature0[0]) + self.a3 * math.log(feature0[1]) + self.a2 * math.log(
                    feature2))
                return score1
            else:
                if (self.old_r_new_p == True) or self.heuristic_r_new_p:
                    if self.print_info:
                        print("COMBINE R P :", feature0[0], feature0[1], feature2)
                    score1 = (self.a1 * math.log(feature0[0]) + self.a3 * math.log(feature0[1]) + self.a2 * math.log(
                        feature2))
                    return score1
                elif self.com_r_new_p:
                    score1 = (self.a3 * math.log(
                        self.a1 * feature0[0] + (1 - self.a1) * feature0[1]) + self.a2 * math.log(feature2))
                    return score1
                elif self.new_r_new_p:
                    return (self.a2 * math.log(feature0) + self.a3 * math.log(feature2))
                elif self.com_r_p3:
                    score1 = self.a3 * math.log(
                        self.a1 * feature0[0] + (1 - self.a1) * feature0[1]) + self.a2 * math.log(feature2)
                    return score1
                elif self.com_r_old_p:
                    pass
                elif self.new_r_p3:
                    pass

    def final_precision_update(self, feature2, gamma, previous_query):
        return gamma * (feature2[2]) + (1 - gamma) * feature2[5]


class Query_formulation_CCQF_new():
    def __init__(self, parameters = (0.6,0.2,0.6,-1,math.log(0.0001),1), bernoulli = False, query_reformulation_component = None, collection_lm = None, print_info = False, topic_bi_probs = None, topic_descs = None, topic_INs = None, setting = None):
        '''
        class for the Query formulation process - can simulate all instantiations of PRE framework
        :param parameters: parameters of recall and precision a1, a2, a3, effort min threshold, effort min threshold, length normalization
        :param bernoulli: whether the Recall, precision should be bernoulli or multinomila probabilities
        :param query_reformulation_component:
        :param collection_lm: collection language model
        :param print_info:
        :param topic_bi_probs: bigram counts and probabilities in sessiont rack topics
        :param topic_descs:
        :param topic_INs:
        :param setting: to specify the combination of recall and precision method - "old_r_new_p"
        '''

        self.query_reformulation = query_reformulation_component
        self.a1 = parameters[0]
        self.a2 = parameters[1]
        self.a3 = parameters[2]
        self.effort_constraint = parameters[3]
        if self.effort_constraint == -1:
            self.min_threshold = parameters[4] #math.log(0.0001) 
        else:
            self.min_threshold = math.log(0.000000000001)
        self.min_effort = None
            #self.max_threshold = parameters[5][1] #-1  
        self.w_len = parameters[5]
        self.old_r_new_p = False  
        self.old_r_old_p = False
        self.new_r_old_p = False
        self.heuristic_r_new_p = False
        self.new_r_new_p = False
        self.heuristic_r_old_p = False
        self.com_r_new_p = False
        self.old_r_p3 = False
        self.heucom_r_old_p = False
        self.com_r_p3 = False
        self.com_r_old_p = False
        self.new_r_p3 = False
        if setting == 'old_r_new_p':
            self.old_r_new_p = True  
        elif setting == 'old_r_old_p':
            self.old_r_old_p = True
        elif setting == 'new_r_old_p':
            self.new_r_old_p = True
        elif setting == "heuristic_r_new_p":
            self.heuristic_r_new_p = True
        elif setting == "heuristic_r_old_p":
            self.heuristic_r_old_p = True
        elif setting == "com_r_new_p":
            self.com_r_new_p = True
        elif setting == "old_r_p3":
            self.old_r_p3 = True
        elif setting == "heucom_r_old_p":
            self.heucom_r_old_p = True
        elif setting == "new_r_new_p":
            self.new_r_new_p = True
        elif setting == "com_r_p3":
            self.com_r_p3 = True
        elif setting == "new_r_p3":
            self.new_r_p3 = True
        elif setting == "com_r_old_p":
            self.com_r_old_p = True

        self.intl_cand_qurs_all_tpcs = None
        self.doc_collection_lm_dist = collection_lm
        self.topic_INs = {}
        self.topic_bigram_ct = None
        self.topic_unigram_ct = None
        self.topic_bigram_prob = None
        self.topic_descs = None
        self.print_info = print_info
        self.bernoulli = bernoulli
        self.specific_doc_collection_lm_dist = None
        if self.specific_doc_collection_lm_dist != None:
            self.doc_collection_lm_dist = {word:self.doc_collection_lm_dist[word]*alpha for word in self.doc_collection_lm_dist}
            for word in self.specific_doc_collection_lm_dist:
                try:
                    self.doc_collection_lm_dist[word] += (1-alpha)*self.specific_doc_collection_lm_dist[word]
                except KeyError:
                    self.doc_collection_lm_dist[word] = (1-alpha)*self.specific_doc_collection_lm_dist[word]
        if topic_bi_probs != None:
            self.topic_bigram_ct = topic_bi_probs[0]
            self.topic_unigram_ct = topic_bi_probs[1]
            self.topic_bigram_prob = topic_bi_probs[2]
        if topic_descs != None:
            self.topic_descs = topic_descs
        if topic_INs != None:
            self.topic_INs = topic_INs
        if (self.old_r_new_p == True) or (self.old_r_old_p == True) or (self.heuristic_r_old_p==True) or (self.heuristic_r_new_p==True) or (self.old_r_p3):
            x = self.a1
            y = self.a2
            self.a1 = x 
            self.a2 = (1.0-x)*y
            self.a3 = (1.0-x)*(1.0-y)
        
        return

    def query_formulation_all_topics(self, dataset):
        '''
        makes/uses topic lm, bg lm
        computes feature[0] - recall of a word, feature[2] - precision of a word.
        :param dataset:
        :return: initial queries
        '''
        if self.topic_bigram_ct == None:
            self.topic_bigram_ct, self.topic_unigram_ct, self.topic_bigram_prob = read_bigram_topic_lm(dataset)
        if self.topic_descs == None:
            self.topic_descs = read_topic_descs(dataset)
        topic_word_scores = {}
        for topic_num in self.topic_descs:
            print ("TOPIC NUM ", topic_num, self.topic_descs[topic_num])
            if topic_num in self.topic_INs:
                topic_IN = self.topic_INs[topic_num]
            else:
                topic_desc = self.topic_descs[topic_num]
                topic_desc = preprocess(topic_desc, stopwords_file, lemmatizing = True)
                if self.bernoulli:
                    topic_IN = language_model_b(topic_desc)
                else:
                    topic_IN = language_model_m(topic_desc)    
                self.topic_INs[topic_num] = topic_IN 

            words_set = list(topic_IN.keys())
            word_list = []
            for word in words_set: 
                if (self.heuristic_r_new_p == True) or (self.heuristic_r_old_p == True) or self.heucom_r_old_p:
                    feature0 = [topic_IN[word], 0.5]
                elif (self.old_r_new_p == True) or (self.old_r_old_p == True) or self.com_r_new_p or self.old_r_p3 or self.com_r_p3:
                    feature0 = [topic_IN[word], topic_IN[word]]
                else:
                    feature0 = topic_IN[word]
                feature2 = [0,0,0]
                if (self.old_r_new_p == True) or self.com_r_new_p or (self.old_r_old_p == True) or (self.heuristic_r_new_p == True) or (self.heuristic_r_old_p == True) or self.old_r_p3 or self.heucom_r_old_p or self.com_r_p3:
                    feature2[0] = feature0[0]
                else:
                    feature2[0] = feature0
                try:
                    feature2[1] = self.doc_collection_lm_dist[word]
                    feature2[2] = float(feature2[0]*0.02)/float(feature2[0]*0.02+feature2[1]*0.98)
                except KeyError:
                    feature2[1] = 0.00000001
                    feature2[2] = float(feature2[0]*0.02)/float(feature2[0]*0.02+feature2[1]*0.98)
                word_list += [(word, feature0, feature2)]
            candidate_queries = self.query_generation(word_list, topic_num)
            print ("CANDIDATE QUERIES: ")
            for query in candidate_queries[:10]:
                print(query)        
            topic_word_scores[topic_num] = candidate_queries
        self.intl_cand_qurs_all_tpcs = topic_word_scores
        return topic_word_scores

    def get_initial_queries(self, topic_num):
        return self.intl_cand_qurs_all_tpcs[topic_num]

    def get_topic_IN(self, topic_num):
        return self.topic_INs[topic_num]

    def get_topic_desc(self, topic_num):
        return self.topic_descs[topic_num]

    def reformulation_method(self, updated_topic_IN, precision_lm, topic_num):
        words_set = [(word,updated_topic_IN[word]) for word in updated_topic_IN]
        words_set = sorted(words_set,key=lambda l:l[1],reverse=True)[:30]
        words_set = [word[0] for word in words_set]
        word_scores = []
        for word in words_set:
            if (self.heuristic_r_new_p == True) or (self.heuristic_r_old_p == True) or self.heucom_r_old_p:
                feature0 = [updated_topic_IN[word], 0.5]
            elif (self.old_r_new_p == True) or (self.old_r_old_p == True) or self.com_r_new_p:
                feature0 = [updated_topic_IN[word],updated_topic_IN[word]]
            else:   
                feature0 =  updated_topic_IN[word]
            feature2 = [0,0,0]
            if (self.old_r_new_p == True) or (self.old_r_old_p == True) or (self.heuristic_r_new_p) or (self.heuristic_r_old_p == True): 
                feature2[0] = feature0[0]
            else:
                feature2[0] = feature0
            if self.new_r_old_p or self.old_r_old_p or (self.heuristic_r_old_p == True):
                if (precision_lm[word]["den"] == 0) or (precision_lm[word]["num"] == 0):
                    feature2[2] = 0.00000001
                else:
                    feature2[2] = float(precision_lm[word]["num"])/float(precision_lm[word]["den"])
            '''
            try:
                there = precision_lm[word]["notT"]
                if (precision_lm[word]["notT"] == 0):
                    feature2[1] = 0.00000001
                else:
                    feature2[1] = precision_lm[word]["notT"]
            except:
               feature2[1] = 0.00000001
            feature2[2] = float(feature2[0]*0.02)/float(feature2[0]*0.02+feature2[1]*0.98)
            '''
            word_scores += [(word, feature0, feature2)]
        candidate_queries = self.query_generation(word_scores, topic_num)
        print ("CANDIDATE QUERIES: ")
        for query in candidate_queries[:10]:
            print(query)  
        return candidate_queries

    def query_generation(self, word_list, topic_num):
        '''
        Using the single word recall and precision, queries are generated and their scores are combined (combine_r_p function)
        query  generation: all unigrams and bigrams are candidate queries, for each bigram extend with best word to get candidate trigrams and so on..
        recall and precision are computed in various ways through diff. methods.
        Deciphering methods:
        old_r: R12g
        new_r: R3
        heuristic_r: R_heu
        com_r: R12a
        heucomr: R_heu combined arithmetically
        old_p: average word precision
        new_p: whole query precision
        self.a1: bigram unigram combination one.
        p_3: the p(Q|T) and P(Q|C) are first length normalized and then precision(p(T|Q)) is computed, so p_3 don't need extra length normalization.

        :param word_list: words to be used for making queries
        :param topic_num:
        :return: (queries, scores)
        '''
        possible_queries = []
        candidate_queries = {}
        for word in word_list:
            query_dict = {}
            query_dict[word[0]] = 1
            max_query = [[word[0]], word[1], word[2], query_dict]
            possible_queries += [max_query]
            #if self.print_info:
            print ('Possible queries: ', max_query)
            candidate_queries[max_query[0][0]] = [max_query[0], self.combine_r_p(max_query[1], max_query[2][2], 1)]
        words_d = len(list(self.topic_unigram_ct[topic_num].keys()))
        if (self.effort_constraint == -1) or (self.effort_constraint >= 2):
            possible_queries_bigrams = []
            for idx,query in enumerate(possible_queries):
                for word in word_list:
                    try:
                        there = query[3][word[0]] 
                    except KeyError:
                        if (self.heuristic_r_new_p == True) or (self.heuristic_r_old_p == True) or self.heucom_r_old_p:
                            feature0 = self.heuristic_bigram_probability(query[1],query[0], word, topic_num, words_d) 
                        elif (self.old_r_new_p == True) or (self.old_r_old_p == True) or self.com_r_new_p or self.old_r_p3 or self.com_r_p3:
                            feature0 = self.uni_bigram_probability_2(query[1],query[0], word, topic_num, words_d) 
                        else:
                            feature0 = self.uni_bigram_probability(query[1],query[0], word, topic_num, words_d)     
                        feature2 = [0,0,0]
                        if self.old_r_new_p == True or (self.heuristic_r_new_p == True) or self.com_r_new_p or self.old_r_p3 or self.com_r_p3:
                            feature2[0] = feature0[0]
                        elif (self.new_r_new_p == True):
                            feature2[0] = feature0
                        #if query[2][1] == 0.00000001 or word[2][1] == 0.00000001:
                        #    feature2[1] = 0.00000001
                        #else:
                        feature2[1] = query[2][1]*word[2][1]
                        if self.new_r_old_p or self.old_r_old_p or (self.heuristic_r_old_p == True) or self.heucom_r_old_p:
                            feature2[2] = query[2][2]+word[2][2]
                        else:
                            if self.old_r_p3 or self.com_r_p3:
                                feature2[2] = self.new_p_3(feature2[0], feature2[1], 2)
                            else:
                                feature2[2] = float(feature2[0]*0.02)/float(feature2[0]*0.02+feature2[1]*0.98)
                        new_query_dict = query[3].copy() 
                        new_query_dict[word[0]] = 1
                        max_query = [query[0]+[word[0]], feature0, feature2, new_query_dict] 
                        possible_queries_bigrams += [max_query]
                        if self.print_info:
                            print ('Possible queries: ', max_query)
                    
                        s = max_query[0].copy()
                        s.sort()
                        score = self.combine_r_p(max_query[1], max_query[2][2], 2)
                        try:
                            there = candidate_queries[" ".join(s)]
                            if score > candidate_queries[" ".join(s)][1]:
                                candidate_queries[" ".join(s)] = [max_query[0], score]
                        except:
                            candidate_queries[" ".join(s)]= [max_query[0], score]

            possible_queries = possible_queries_bigrams
            #candidate_queries = sorted(candidate_queries, key = lambda l:l[1], reverse = True)
            #print (len(word_list)) 
            if self.effort_constraint != -1:
                num_iters = self.effort_constraint - 2
            else:
                num_iters =  len(word_list)-2    
            for i in range(num_iters):
                for idx,query in enumerate(possible_queries):
                    max_score = -1
                    max_query = None
                    for iter_idx,word in enumerate(word_list):
                        try:
                            there = query[3][word[0]] 
                        except:
                            #recall computations part
                            if (self.heuristic_r_new_p == True) or (self.heuristic_r_old_p == True) or self.heucom_r_old_p:
                                feature0 = self.heuristic_bigram_probability(query[1],query[0], word, topic_num, words_d) 
                            elif (self.old_r_new_p == True) or (self.old_r_old_p == True) or self.com_r_new_p or self.old_r_p3 or self.com_r_p3:
                                feature0 = self.uni_bigram_probability_2(query[1],query[0], word, topic_num, words_d) 
                            else:
                                feature0 = self.uni_bigram_probability(query[1],query[0], word, topic_num, words_d)
                            #precision computations part
                            feature2 = [0,0,0]
                            if (self.old_r_new_p == True) or (self.heuristic_r_new_p == True) or self.com_r_new_p or self.old_r_p3 or self.com_r_p3:
                                feature2[0] = feature0[0]
                            elif (self.new_r_new_p == True):
                                feature2[0] = feature0 
                            #if query[2][1] == 0.00000001 or word[2][1] == 0.00000001:
                            #    feature2[1] = 0.00000001
                            #else:
                            feature2[1] = query[2][1]*word[2][1]
                            if self.new_r_old_p or self.old_r_old_p or (self.heuristic_r_old_p == True) or self.heucom_r_old_p:
                                feature2[2] = query[2][2]+word[2][2]
                            else:
                                if self.old_r_p3 or self.com_r_p3:
                                    feature2[2] = self.new_p_3(feature2[0], feature2[1], i+3)
                                else:
                                    feature2[2] = float(feature2[0]*0.02)/float(feature2[0]*0.02+feature2[1]*0.98)
                            score = self.combine_r_p(feature0, feature2[2], i+3)
                            if (score > max_score) or (iter_idx==0):
                                max_score = score
                                new_query_dict = query[3].copy() 
                                new_query_dict[word[0]] = 1
                                max_query = [query[0]+[word[0]], feature0, feature2, new_query_dict] 
                                #max_query_list = 
                    if max_query!=None:
                        possible_queries[idx] = max_query
                        if self.print_info:
                            print ('Possible queries: ', max_query)
                    if (max_query!=None):
                        s = max_query[0].copy()
                        s.sort()
                        score = self.combine_r_p(max_query[1], max_query[2][2], i+3)
                        try:
                            there = candidate_queries[" ".join(s)]
                            if score > candidate_queries[" ".join(s)][1]:
                                candidate_queries[" ".join(s)]= [max_query[0], score]
                        except:
                            candidate_queries[" ".join(s)]= [max_query[0], score]
            #candidate_queries = sorted(candidate_queries, key = lambda l:l[1], reverse = True)
        #candidate_queries = {query: [candidate_queries[query][0],float(candidate_queries[query][1])/float(len(candidate_queries[query][0]))] for query in candidate_queries}
        candidate_queries = sorted(candidate_queries.items(), key = lambda l:l[1][1], reverse = True)            
        if self.min_effort != None:
            candidate_queries = [(c[1][0],c[1][1]) for c in candidate_queries if len(c[1][0])>=self.min_effort] 
        else:
            candidate_queries = [(c[1][0],c[1][1]) for c in candidate_queries]
        return candidate_queries


    def uni_bigram_probability(self, probability , query, word, topic_num, words_d):
        mu = float(4)
        try:
            d = self.topic_unigram_ct[topic_num][query[-1]] 
        except KeyError:
            d = 0    
        try:
            bigram_prob = (self.topic_bigram_prob[topic_num][query[-1]+" "+word[0]]*(float(d)/float(d+mu)))+((float(1)/float(words_d))*(float(mu)/float(d+mu)))
        except KeyError:
            bigram_prob  = (float(1)/float(words_d))*(float(mu)/float(d+mu))
        '''
        try:
            bigram_prob = self.topic_bigram_prob[topic_num][query[-1] + " " + word[0]]
        except KeyError:
            bigram_prob = 0.00000001
        '''
        unigram_prob = word[1]
        if unigram_prob == 0:
            unigram_prob = 0.00000001
        if bigram_prob == 0:
            bigram_prob = 0.00000001
        if self.print_info:
            print ('Uni Bigram probability Query: ', query, word[0], bigram_prob, unigram_prob)
        new_probability = probability*(self.a1*bigram_prob + (1-self.a1)*unigram_prob)
        return new_probability

    def uni_bigram_probability_2(self, probability, query, word, topic_num, words_d):
        mu = float(4)
        try:
            d = self.topic_unigram_ct[topic_num][query[-1]] 
        except KeyError:
            d = 0   
        try:
            bigram_prob = (self.topic_bigram_prob[topic_num][query[-1]+" "+word[0]]*(float(d)/float(d+mu)))+((float(1)/float(words_d))*(float(mu)/float(d+mu)))
        except KeyError:
            bigram_prob  = (float(1)/float(words_d))*(float(mu)/float(d+mu))
        unigram_prob = word[1][0]
        if self.print_info:
            print ('Uni Bigram probability Query: ', query, word[0], bigram_prob, unigram_prob)
        if unigram_prob == 0:
            unigram_prob = 0.00000001
        if bigram_prob == 0:
            bigram_prob = 0.00000001
        #print (probability[0], probability[1], unigram_prob, bigram_prob)
        new_probability = [probability[0]*unigram_prob, probability[1]*bigram_prob]
        #print (probability[0],probability[1], probability)
        return new_probability

    def heuristic_bigram_probability(self, probability, query, word, topic_num, words_d):
        num_phrases = 0
        query = query + [word[0]]
        for idx in range(len(query)-1):
            word1 = query[idx]
            word2 = query[idx+1]
            if (word1+" "+word2) in self.topic_bigram_ct[topic_num]:
                num_phrases += float(self.topic_bigram_ct[topic_num][word1+" "+word2])/float(max(list(self.topic_bigram_ct[topic_num].values())))
                #num_phrases += (self.topic_bigram_prob[topic_num][word1+" "+word2])
                #num_phrases += float(self.topic_bigram_ct[topic_num][word1+" "+word2])
        if len(query) == 1:
            prob = float(1)/float(math.exp(-0)+1)
        else:
            if self.w_len == 1:
                prob = float(1)/float(math.exp(-float(num_phrases)/float(len(query)-1))+1)
            elif self.w_len == 0:
                prob = float(1)/float(math.exp(-float(num_phrases))+1)
                
        unigram_prob = word[1][0]
        if self.print_info:
            print ('Uni Bigram probability Query: ', query, bigram_prob, unigram_prob)
        if unigram_prob == 0:
            unigram_prob = 0.00000001
        
        new_probability = [probability[0]*unigram_prob, prob]
        return new_probability

    def new_p_3(self,feature0, feature1, query_len):
        if self.w_len == 1:
            feature0 = math.pow(feature0, float(1)/float(query_len))
            feature1 = math.pow(feature1, float(1)/float(query_len))
        elif self.w_len == 0:
            feature0 = feature0
            feature1 = feature1
        return float(feature0*0.02)/float(feature0*0.02+feature1*0.98)

    def combine_r_p(self,feature0, feature2, query_len):
        '''
        Deciphering parameters:
        self.a1: is the inside parameter between unigram and bigram or it is one of the outside parameter for recall: a1, a3 are parameters for unigram and bigram respectively
        self.a2: parameter for precision.  (a1+a3 will be (1-a2))
        self.a3: is either (1-a2) when a1 is inside parameter, or it is one of the outside parameters for recall:: a1, a3 are parameters for unigram and bigram respectively
        ONLY new_r_old_p, new_r_new_p it is reverse: a2 is for recall, a3 is for precision, a2=1-a3. a1 is inside parameter for recall.
        :param feature0:
        :param feature2:
        :param query_len:
        :return:
        '''
        if self.w_len == 1:
            if  (self.heuristic_r_old_p == True) or self.heucom_r_old_p:
                feature2 = float(feature2)/float(query_len)
                if self.heuristic_r_old_p:
                    score1 = ((float(1)/float(query_len))*self.a1*math.log(feature0[0]))+(self.a3*math.log(feature0[1]))
                    return score1+self.a2*math.log(feature2)
                elif self.heucom_r_old_p:
                    score1 = (float(1)/float(query_len))*(self.a3*math.log(self.a1*feature0[0]+(1-self.a1)*feature0[1]))
                    return score1+self.a2*math.log(feature2)
            elif self.old_r_old_p or self.new_r_old_p:
                feature2 = float(feature2)/float(query_len)
                if self.old_r_old_p or (self.heuristic_r_old_p == True):
                    if self.print_info:
                        print ("COMBINE R P :", feature0[0], feature0[1], feature2)
                    score1 = (float(1)/float(query_len))*(self.a1*math.log(feature0[0])+self.a3*math.log(feature0[1]))
                    return score1+self.a2*math.log(feature2)
                elif self.new_r_old_p:
                    score1 = (float(1)/float(query_len))*(self.a2*math.log(feature0))
                    return score1+self.a3*math.log(feature2)
            elif self.old_r_p3:
                score1 = (float(1)/float(query_len))*(self.a1*math.log(feature0[0])+self.a3*math.log(feature0[1]))+self.a2*math.log(feature2)
                return score1
            else:
                if self.heuristic_r_new_p:
                    if self.print_info:
                        print ("COMBINE R P :", feature0[0], feature0[1], feature2)
                    score1 = ((float(1)/float(query_len))*(self.a1*math.log(feature0[0])+self.a2*math.log(feature2)))+(self.a3*math.log(feature0[1]))
                    return score1
                elif self.old_r_new_p == True:
                    if self.print_info:
                        print ("COMBINE R P :", feature0[0], feature0[1], feature2)
                    score1 = (float(1)/float(query_len))*(self.a1*math.log(feature0[0])+self.a3*math.log(feature0[1])+self.a2*math.log(feature2))
                    return score1
                elif self.com_r_new_p:
                    score1 = (float(1)/float(query_len))*(self.a3*math.log(self.a1*feature0[0]+(1-self.a1)*feature0[1])+self.a2*math.log(feature2))
                    return score1
                elif self.new_r_new_p:
                    return (float(1)/float(query_len))*(self.a2*math.log(feature0)+self.a3*math.log(feature2))
                elif self.com_r_p3:
                    score1 = (float(1) / float(query_len)) * (self.a3 * math.log(self.a1 * feature0[0] + (1 - self.a1) * feature0[1])) + self.a2 * math.log(feature2)
                    return score1
                elif self.com_r_old_p:
                    pass
                elif self.new_r_p3:
                    pass
        else:
            if self.old_r_old_p or self.new_r_old_p or (self.heuristic_r_old_p == True):
                feature2 = float(feature2)/float(query_len)
                if self.old_r_old_p or (self.heuristic_r_old_p == True):
                    if self.print_info:
                        print ("COMBINE R P :", feature0[0], feature0[1], feature2)
                    score1 = (self.a1*math.log(feature0[0])+self.a3*math.log(feature0[1]))
                    return score1+self.a2*math.log(feature2)
                elif self.new_r_old_p:
                    score1 = (self.a2*math.log(feature0))
                    return score1+self.a3*math.log(feature2)
            elif self.heucom_r_old_p:
                feature2 = float(feature2)/float(query_len)
                score1 = (float(1)/float(query_len))*(self.a3*math.log(self.a1*feature0[0]+(1-self.a1)*feature0[1]))
                return score1+self.a2*math.log(feature2)
            elif self.old_r_p3:
                score1 = (self.a1*math.log(feature0[0])+self.a3*math.log(feature0[1])+self.a2*math.log(feature2))
                return score1
            else:
                if (self.old_r_new_p == True) or self.heuristic_r_new_p:
                    if self.print_info:
                        print ("COMBINE R P :", feature0[0], feature0[1], feature2)
                    score1 = (self.a1*math.log(feature0[0])+self.a3*math.log(feature0[1])+self.a2*math.log(feature2))
                    return score1
                elif self.com_r_new_p:
                    score1 = (self.a3*math.log(self.a1*feature0[0]+(1-self.a1)*feature0[1])+self.a2*math.log(feature2))
                    return score1
                elif self.new_r_new_p:
                    return (self.a2*math.log(feature0)+self.a3*math.log(feature2))
                elif self.com_r_p3:
                    score1 = self.a3 * math.log(self.a1 * feature0[0] + (1 - self.a1) * feature0[1]) + self.a2 * math.log(feature2)
                    return score1
                elif self.com_r_old_p:
                    pass
                elif self.new_r_p3:
                    pass
    
    def reformulation_method_type2(self, gamma, previous_query, updated_topic_IN, precision_lm, topic_num):
        words_set = [(word,updated_topic_IN[word]) for word in updated_topic_IN]
        words_set = sorted(words_set,key=lambda l:l[1],reverse=True)[:30]
        words_set = [word[0] for word in words_set]
        word_scores = []
        for word in words_set:
            feature0 = updated_topic_IN[word]
            feature2 = [0,0,0,0,0,0]
            feature2[0] = feature0
            try:
                feature2[1] = self.doc_collection_lm_dist[word]
            except:
                feature2[1] = 0.00000001
            feature2[2] = float(feature0*0.02)/float(feature0*0.02+feature2[1]*0.98)
            try:
                feature2[3] = precision_lm[word]["T"]
            except:
                feature2[3] = 0
            try:                
                feature2[4] = precision_lm[word]["notT"]
            except:
                feature2[4] = 0   
            if float(feature2[3]*0.02+feature2[4]*0.98)!=0:
                feature2[5] = float(feature2[3]*0.02)/float(feature2[3]*0.02+feature2[4]*0.98)
            else:
                feature2[5] = 0.00000001
            word_scores += [(word, feature0, feature2)]
        word_scores = sorted(word_scores, key= lambda l: l[2], reverse=True)

        possible_queries = []
        candidate_queries = {}
        for word in word_list:
            query_dict = {}
            query_dict[word[0]] = 1
            max_query = [[word[0]], word[1], word[2], query_dict]
            possible_queries += [max_query]
            #if self.print_info:
            print ('Possible queries: ', max_query)
            candidate_queries[max_query[0][0]] = [max_query[0], self.combine_r_p(max_query[1], self.final_precision_update(max_query[2], gamma, previous_query))]
        
        if (self.effort_constraint == -1) or (self.effort_constraint >= 2):
            possible_queries_bigrams = []
            for idx,query in enumerate(possible_queries):
                for word in word_list:
                    try:
                        there = query[3][word[0]] 
                    except: 
                        feature0 = self.uni_bigram_probability(query[1],query[0], word, topic_num) 
                        feature2 = [0,0,0,0,0,0]
                        feature2[0] = feature0
                        if query[2][1] == 0.00000001 or word[2][1] == 0.00000001:
                            feature2[1] = 0.00000001
                        else:
                            feature2[1] = query[2][1]*word[2][1]
                        feature2[2] = float(feature0*0.02)/float(feature0*0.02+feature2[1]*0.98)
                        feature2[3] = query[2][3]*word[2][3]
                        feature2[4] = query[2][4]*word[2][4]
                        if float(feature2[3]*0.02+feature2[4]*0.98)!=0:
                            feature2[5] = float(feature2[3]*0.02)/float(feature2[3]*0.02+feature2[4]*0.98)
                        else:
                            feature2[5] = 0.00000001
                        new_query_dict = query[3].copy() 
                        new_query_dict[word[0]] = 1
                        max_query = [query[0]+[word[0]], feature0, feature2, new_query_dict] 
                        possible_queries_bigrams += [max_query]
                        if self.print_info:
                            print ('Possible queries: ', max_query)
                    
                        s = max_query[0].copy()
                        s.sort()
                        score = self.combine_r_p(max_query[1], self.final_precision_update(max_query[2], gamma, previous_query))
                        try:
                            there = candidate_queries[" ".join(s)]
                            if score > candidate_queries[" ".join(s)][1]:
                                candidate_queries[" ".join(s)] = [max_query[0], score]
                        except:
                            candidate_queries[" ".join(s)]= [max_query[0], score]

            possible_queries = possible_queries_bigrams
            #candidate_queries = sorted(candidate_queries, key = lambda l:l[1], reverse = True)
            #print (len(word_list)) 
            if self.effort_constraint != -1:
                num_iters = self.effort_constraint - 2
            else:
                num_iters =  len(word_list)-2    
            for i in range(num_iters):
                for idx,query in enumerate(possible_queries):
                    max_score = math.log(0.000000001)
                    max_query = None
                    for word in word_list:
                        try:
                            there = query[3][word[0]] 
                        except:                        
                            feature0 = self.uni_bigram_probability(query[1],query[0], word, topic_num) 
                            feature2 = [0,0,0]
                            feature2[0] = feature0
                            if query[2][1] == 0.00000001 or word[2][1] == 0.00000001:
                                feature2[1] = 0.00000001
                            else:
                                feature2[1] = query[2][1]*word[2][1]
                            feature2[2] = float(feature0*0.02)/float(feature0*0.02+feature2[1]*0.98)
                            feature2[3] = query[2][3]*word[2][3]
                            feature2[4] = query[2][4]*word[2][4]
                            if float(feature2[3]*0.02+feature2[4]*0.98)!=0:
                                feature2[5] = float(feature2[3]*0.02)/float(feature2[3]*0.02+feature2[4]*0.98)
                            else:
                                feature2[5] = 0.00000001
                            score = self.combine_r_p(feature0, self.final_precision_update(feature2, gamma, previous_query))
                            if (score > max_score):
                                max_score = score
                                new_query_dict = query[3].copy() 
                                new_query_dict[word[0]] = 1
                                max_query = [query[0]+[word[0]], feature0, feature2, new_query_dict] 
                                #max_query_list = 
                    if max_query!=None:
                        possible_queries[idx] = max_query
                        if self.print_info:
                            print ('Possible queries: ', max_query)
                    if (max_query!=None):
                        s = max_query[0].copy()
                        s.sort()
                        score = self.combine_r_p(max_query[1], self.final_precision_update(max_query[2], gamma, previous_query))
                        try:
                            there = candidate_queries[" ".join(s)]
                            if score > candidate_queries[" ".join(s)][1]:
                                candidate_queries[" ".join(s)]= [max_query[0], score]
                        except:
                            candidate_queries[" ".join(s)]= [max_query[0], score]
            #candidate_queries = sorted(candidate_queries, key = lambda l:l[1], reverse = True)
        candidate_queries = {query: [candidate_queries[query][0],float(candidate_queries[query][1])/float(len(candidate_queries[query][0]))] for query in candidate_queries}
        candidate_queries = sorted(candidate_queries.items(), key = lambda l:l[1][1], reverse = True)            
        if self.min_effort != None:
            candidate_queries = [(c[1][0],c[1][1]) for c in candidate_queries if len(c[1][0])>=self.min_effort] 
        else:
            candidate_queries = [(c[1][0],c[1][1]) for c in candidate_queries]

        print ("CANDIDATE QUERIES: ")
        for query in candidate_queries[:10]:
            print(query)  
        return candidate_queries

    def final_precision_update(self, feature2, gamma, previous_query):
        return gamma*(feature2[2]) + (1-gamma)*feature2[5]

class Query_formulation_DTC():
    def __init__(self, query_reformulation_component = None):
        self.self.query_reformulation = query_reformulation_component
        self.p_t_s_w = {}
        for word in p_t_s_w:
            self.p_t_s_w[word] = p_t_s_w[word].copy()
        return

    def reformulation_method(self, results=None):
        content = ""
        for result in results:
            content += " " + result["content"]
        content_words = content.split()    
        topic_desc = preprocess(self.topic_desc, lemmatizing = True)
        topic_desc_words = topic_desc.split()
        smoothing_lm = {}
        for word in content_words:
            smoothing_lm[word] = 1
        for word in topic_desc_words:
            smoothing_lm[word] = 1
        num_unique_words = len(smoothing_lm.keys())
        smoothing_lm = {word:float(1)/float(num_unique_words) for word in smoothing_lm}
        content_lm = Counter(content.split())
        for word in content_lm:
            try:
                there = self.p_t_s_w[word]
                self.p_t_s_w[word][self.topic_num] +=  content_lm[word]
                self.p_t_s_w[word]["total"] +=  content_lm[word]
            except KeyError:
                self.p_t_s_w[word] = {}
                self.p_t_s_w[word][self.topic_num] = content_lm[word]
                self.p_t_s_w[word]["total"] = content_lm[word] 
                
        #print ("coming here first")
        [p_l_t, p_w_l_t] = topic_mle[self.topic_num]
        final_mle = {}
        N = 100
        mu = 8
        candidate_queries = []
        p_l_t_items = p_l_t.items()
        #print ("coming here second")
        p_l_t_keys = [p[0] for p in p_l_t_items]
        p_l_t_values = [p[1] for p in p_l_t_items]
        #print ("coming here third")
        for i in range(N):
            chosen_l = np.random.choice(p_l_t_keys, p = p_l_t_values) 
            all_other_words_sum = sum(p_w_l_t[chosen_l].values())
            list(smoothing_lm.keys()) + list(p_w_l_t[chosen_l].keys())
            p_w_l_t_smoothed = {}
            for word in p_w_l_t[chosen_l]:
                try:
                    p_w_l_t_smoothed[word] = float(p_w_l_t[chosen_l][word] + mu*smoothing_lm[word])/float(all_other_words_sum+mu)
                except KeyError:
                    p_w_l_t_smoothed[word] = float(p_w_l_t[chosen_l][word])/float(all_other_words_sum+mu)
            for word in smoothing_lm:
                try:
                    there = p_w_l_t[chosen_l][word]
                except KeyError:
                    p_w_l_t_smoothed[word] = float(mu*smoothing_lm[word])/float(all_other_words_sum+mu)

            p_w_l_t_sorted = sorted(p_w_l_t_smoothed.items(), key = lambda l: l[1], reverse=True)
            candidate_query = []
            #while (len(candidate_query) != chosen_l):
            for (word,score) in p_w_l_t_sorted:
                if int(len(candidate_query)) == int(chosen_l):
                    break
                if(random.random()>=0.5):
                    candidate_query += [word]
                #print (int(len(candidate_query)), int(chosen_l), candidate_query)
            #print ("Chosen l candidate query: ", chosen_l, candidate_query)
            candidate_queries += [candidate_query]
        #print ("Chosen ls: ", [len(query) for query in candidate_queries])
        #print ("coming here fourth")
        #print (candidate_queries[:10])
        candidates_queries_scoring = {}
        T = len(topic_mle.keys())
        W = len(doc_collection_lm.keys())
        candidate_query_scores = []
        candidate_queries_new = []
        #print ("coming here fifth")
        for query in candidate_queries:
            first_query = query.copy()
            first_query.sort()
            if (" ".join(first_query) in self.previous_queries):
                continue
            else:
                query_score = 1
                for word in query:
                    word_score = 1
                    try:
                        word_score = float(p_t_w[word][self.topic_num] + self.p_t_s_w[word][self.topic_num] + mu*(float(1)/float(T)))/float(p_t_w[word]["total"] + self.p_t_s_w[word]["total"] + mu)
                    except KeyError:
                        word_score = float(mu*(float(1)/float(T)))/float(mu)
                    word_score = word_score*float(1)/float(W)
                    query_score *= word_score 
                candidate_query_scores += [query_score]
                candidate_queries_new += [query]
        candidate_queries = candidate_queries_new
        candidate_query_scores = [float(s)/float(sum(candidate_query_scores)) for s in candidate_query_scores]
        #print ("coming here sixth")
        final_query_scores = list(zip(candidate_queries, candidate_query_scores))
        print (final_query_scores[:10])
        i = np.random.choice(range(len(candidate_query_scores)), p = candidate_query_scores) ###SHOULD CHANGE
        Q = final_query_scores[i][0]
        print ("coming here")
        self.current_query = Q
        print ("Query: ", Q)
        return " ".join(Q)

    def query_formulation_with_sessions(self, results = None):
        topic_desc = preprocess(self.topic_desc, lemmatizing = True)
        topic_desc_words = topic_desc.split()
        smoothing_lm = {}
        for word in topic_desc_words:
            smoothing_lm[word] = 1
        num_unique_words = len(smoothing_lm.keys())
        smoothing_lm = {word:float(1)/float(num_unique_words) for word in smoothing_lm}
        
        #print ("coming here first")
        [p_l_t, p_w_l_t] = topic_mle[self.topic_num]
        final_mle = {}
        N = 100
        mu = 8
        candidate_queries = []
        p_l_t_items = p_l_t.items()
        #print ("coming here second")
        p_l_t_keys = [p[0] for p in p_l_t_items]
        p_l_t_values = [p[1] for p in p_l_t_items]
        #print ("coming here third")
        for i in range(N):
            chosen_l = np.random.choice(p_l_t_keys, p = p_l_t_values) 
            all_other_words_sum = sum(p_w_l_t[chosen_l].values())
            list(smoothing_lm.keys()) + list(p_w_l_t[chosen_l].keys())
            p_w_l_t_smoothed = {}
            for word in p_w_l_t[chosen_l]:
                try:
                    p_w_l_t_smoothed[word] = float(p_w_l_t[chosen_l][word] + mu*smoothing_lm[word])/float(all_other_words_sum+mu)
                except KeyError:
                    p_w_l_t_smoothed[word] = float(p_w_l_t[chosen_l][word])/float(all_other_words_sum+mu)
            for word in smoothing_lm:
                try:
                    there = p_w_l_t[chosen_l][word]
                except KeyError:
                    p_w_l_t_smoothed[word] = float(mu*smoothing_lm[word])/float(all_other_words_sum+mu)
            p_w_l_t_sorted = sorted(p_w_l_t_smoothed.items(), key = lambda l: l[1], reverse=True)
            #print ("Ps sorted: ", p_w_l_t_sorted)
            candidate_query = []
            #while (len(candidate_query) != chosen_l):
            for (word,score) in p_w_l_t_sorted:
                if int(len(candidate_query)) == int(chosen_l):
                    break
                if(random.random()>=0.5):
                    candidate_query += [word]
                #print (int(len(candidate_query)), int(chosen_l), candidate_query)
            #print ("Chosen l candidate query: ", chosen_l, candidate_query)
            candidate_queries += [candidate_query]
        #print ("Chosen ls: ", [len(query) for query in candidate_queries])
        #print ("coming here fourth")
        #print (candidate_queries[:10])
        candidates_queries_scoring = {}
        T = len(topic_mle.keys())
        W = len(doc_collection_lm.keys())
        candidate_query_scores = []
        #print ("coming here fifth")
        for query in candidate_queries:
            query_score = 1
            for word in query:
                word_score = 1
                try:
                    word_score = float(p_t_w[word][self.topic_num] + self.p_t_s_w[word][self.topic_num] + mu*(float(1)/float(T)))/float(p_t_w[word]["total"] + self.p_t_s_w[word]["total"] + mu)
                except KeyError:
                    word_score = float(mu*(float(1)/float(T)))/float(mu)
                word_score = word_score*float(1)/float(W)
                query_score *= word_score 
            candidate_query_scores += [query_score]
        candidate_query_scores = [float(s)/float(sum(candidate_query_scores)) for s in candidate_query_scores]
        #print ("coming here sixth")
        final_query_scores = list(zip(candidate_queries, candidate_query_scores))
        self.candidate_queries = final_query_scores
        print (final_query_scores[:10])
        i = np.random.choice(range(len(candidate_query_scores)), p = candidate_query_scores) ###SHOULD CHANGE
        Q = final_query_scores[i][0]
        self.current_query = Q
        print ("coming here")
        print ("Query: ", Q)
        return (" ".join(Q))

class Reformulate_no_reformulation():
    def __init__(self, query_formulation_component = None, result_judgement_component = None, collection_lm = None):
        self.query_formulation = query_formulation_component
        self.result_judgement = result_judgement_component

    def reformulate_query(self, candidate_queries, results, clicks, topic_num, topic_IN, precision_lm, previous_queries):
        updated_topic_IN = topic_IN
        updated_precision_lm = precision_lm
        first_query = candidate_queries[0][0].copy()
        first_query.sort()
        while (" ".join(first_query) in previous_queries):
            candidate_queries = candidate_queries[1:]
            first_query = candidate_queries[0][0].copy()
            first_query.sort()
        return candidate_queries,updated_topic_IN,updated_precision_lm

class Query_formulation_QS3plus():
    def __init__(self, query_reformulation_component = None, collection_lm = None , total_num_words = None):
        self.query_reformulation = query_reformulation_component
        self.d = total_num_words
        self.doc_collection_lm_dist = collection_lm
        self.intl_cand_qurs_all_tpcs = None
        self.topic_INs = {}
        self.topic_descs = None
        return

    def get_candidate_queries():
        return self.query_formulation_QS3plus()


    def get_topic_desc(self, topic_num):
        return self.topic_descs[topic_num]

    def query_formulation_all_topics(self, dataset):
        topic_descs = read_topic_descs(dataset)
        self.topic_descs = topic_descs
        self.intl_cand_qurs_all_tpcs = {}
        for topic_num in topic_descs:
            print ("TOPIC NUM ", topic_num)
            topic_desc = topic_descs[topic_num]
            topic_desc = preprocess(topic_desc, lemmatizing = True)
            topic_desc2 = preprocess(topic_desc, stopwords_file, lemmatizing = True)
            topic_IN = language_model_m(topic_desc2)
            self.topic_INs[topic_num] = topic_IN
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
            #self.collection_IN = self.make_collection_IN(self.topic_IN, self.C_u_1)
            bigrams_scores = []
            for bigram in all_possible_bigrams:
                bigram_score = 0
                try:
                    bigram_score += bigram_topic_IN[bigram]
                except KeyError:
                    pass
                bigrams_scores += [(bigram,bigram_score)]
            bigrams_scores = sorted(bigrams_scores, key = lambda l: l[1], reverse = True)
            all_queries = []
            for (bigram,score) in bigrams_scores:
                query_score = 0
                for word in bigram.split():
                    try:
                        query_score += math.log(topic_IN[word]/self.doc_collection_lm_dist[word])
                    except:
                        query_score += math.log(topic_IN[word]/new_word_probability(self.d))
                for word in topic_IN:
                    try:
                        word_score = math.log(topic_IN[word]/self.doc_collection_lm_dist[word])
                    except:
                        word_score = math.log(topic_IN[word]/new_word_probability(self.d)) 
                    all_queries += [(bigram.split() + [word],query_score + word_score)]
            all_queries = sorted(all_queries, key = lambda l: l[1], reverse=True)
            self.intl_cand_qurs_all_tpcs[topic_num] = all_queries
        return self.intl_cand_qurs_all_tpcs

    def get_initial_queries(self, topic_num):
        return self.intl_cand_qurs_all_tpcs[topic_num]
    
    def get_topic_IN(self, topic_num):
        return self.topic_INs[topic_num]

    def query_formulation(self, topic_num, dataset):
        self.candidate_queries = None
        try:
            topic_desc = self.topic_descs[topic_num]
        except:
            topic_desc = read_topic_descs(dataset)
        topic_desc = preprocess(topic_desc, lemmatizing = True)
        topic_desc2 = preprocess(topic_desc, stopwords_file, lemmatizing = True)
        topic_IN = language_model_m(topic_desc2)
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
        Q = []
        max_length = 4
        #self.collection_IN = self.make_collection_IN(self.topic_IN, self.C_u_1)
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
                    query_score += math.log(topic_IN[word]/self.doc_collection_lm_dist[word])
                except:
                    query_score += math.log(topic_IN[word]/new_word_probability(self.d))
            for word in topic_IN:
                if word not in Q:
                    try:
                        word_score = math.log(topic_IN[word]/self.doc_collection_lm_dist[word])
                    except:
                        word_score = math.log(topic_IN[word]/new_word_probability(self.d)) 
                    all_queries += [(bigram.split() + [word],query_score + word_score)]
        all_queries = sorted(all_queries, key = lambda l: l[1], reverse=True)
        self.candidate_queries = all_queries
        return self.candidate_queries




