from User_model_utils import *
import math
###TODO:
#1.vocabulary expand using the word concept graph
#2.vocabulary expands based on similar words through word embedding

all_doc_index = pickle.load(open("../TREC_Robust_data/all_doc_index_lemmatized.pk", "rb") )
doc_keys_list = all_doc_index[1]
all_doc_index = all_doc_index[0]
(doc_collection_lm, doc_collection_lm_binary) = pickle.load(open("../TREC_Robust_data/all_doc_language_model.pk", "rb"))
total_num_words = sum(doc_collection_lm.values())
doc_collection_lm_dist = {term:float(doc_collection_lm[term])/float(total_num_words) for term in doc_collection_lm}
total_num_words_binary = sum(doc_collection_lm_binary.values())
doc_collection_lm_binary_dist = {term:float(doc_collection_lm_binary[term])/float(total_num_words_binary) for term in doc_collection_lm_binary}
#candidate_queries = pickle.load(open("../simulated_sessions/final_results/candidate_queries_basic_recall_precision.pk", "rb"))
bigram_topic_lm = read_bigram_topic_lm()
target_documents_details = pickle.load(open("../TREC_Robust_data/topic_rel_doc_details.pk","rb"))
target_document_vectors = pickle.load(open("../TREC_Robust_data/target_doc_topic_vectors.pk", 'rb'))
'''
for topic_num in candidate_queries:
	query_scores = []
	alpha = 0.8
	for (query,doc_list, score) in candidate_queries[topic_num]:
		num_phrases = 0
		for idx,word1 in enumerate(query):
			for word2 in query[idx+1:]:
				if (word1+" "+word2) in bigram_topic_lm[topic_num]:
					num_phrases += bigram_topic_lm[topic_num][word1+" "+word2]

				#if (word1 + " " +word2) in keywords:
				#	num_phrases += 1
		if (len(query)>1) and (num_phrases>0):
			p_bigrams = float(num_phrases)/float(len(query)*(len(query)-1))
		else:
			p_bigrams = 0.000001
		score1 = alpha*float(score)/float(3.0) + (1-alpha)*math.log(p_bigrams) 
		score2 = alpha*float(score)/float(3.0) + (1-alpha)*float(1)/float(math.exp(-float(num_phrases)/float(len(query)))+1)
		query_scores += [(query, score, float(1)/float(math.exp(-float(num_phrases)/float(len(query)))+1),score2)]
	query_scores = sorted(query_scores, key =lambda l:l[3],reverse=True)
	candidate_queries[topic_num] = query_scores
'''
candidate_queries = get_greedy_sol_queries_unsupervised(math.log(0.0001), -1)

with open('../supervised_models/baseline_session_lms/topic_mle_training_session_nums_5.json', 'r') as outfile:
    topic_mle = json.load(outfile)
with open('../supervised_models/baseline_session_lms/p_t_w_training_session_nums_5.json', 'r') as outfile:
    p_t_w = json.load(outfile)
with open('../supervised_models/baseline_session_lms/p_t_s_w_training_session_nums_5.json', 'r') as outfile:
    p_t_s_w = json.load(outfile)

def normalize_vector(vector):
	norm = np.linalg.norm(vector)
	if norm==0:
		return vector
	else:
		return vector/norm
class User_model():
	def __init__(self, parameters = None, doc_collection = None, robust_doc_content = None, topic_desc = None, topic_num = None, target_click_items = None, all_doc_bigram_lm = None):
		self.rho = parameters[0] 
		self.max_threshold  = parameters[1]
		self.C_u_1 = {}
		for clueweb_id in doc_collection:
			#if (random.random() <= self.rho):
			self.C_u_1[clueweb_id] = doc_collection[clueweb_id]
		for docid in robust_doc_content:
			#if (random.random() <= self.rho):
			self.C_u_1[docid] = robust_doc_content[docid]
		'''
		self.C_u_2 = {}
		size = int(self.rho*len(doc_collection.keys()))
		selected_ids = random.sample(list(doc_collection.keys()), size)
		for clueweb_id in selected_ids:
			self.C_u_2[clueweb_id] = doc_collection[clueweb_id]
		'''
		self.topic_desc = topic_desc
		self.topic_IN = None
		self.collection_IN = None
		self.current_query = None
		self.target_clueweb_ids = target_click_items #dict(zip(target_click_items, range(len(target_click_items)))) 
		self.session_length = None
		#self.previous_queries = []
		self.all_doc_collection_lm = doc_collection_lm
		self.d = total_num_words
		self.all_doc_bigram_IN = all_doc_bigram_lm
		#self.topic_proportion, self.topic_word_distribution = topic_dist_inputs[0], topic_dist_inputs[1]
		self.previous_queries = {}
		self.word_scores = None
		self.candidate_queries = candidate_queries[topic_num][:]
		#self.candidate_queries = list(filter(lambda l: len(l[1])>=100, self.candidate_queries))
		self.QS3_queries = None
		self.topic_desc_lm = preprocess(topic_desc, "../Session_track_2014/clueweb_snippet/lemur-stopwords.txt", lemmatizing=True)
		self.topic_desc_lm = Counter(self.topic_desc_lm.split())
		self.topic_desc_lm = {word:float(self.topic_desc_lm[word])/float(sum(self.topic_desc_lm.values())) for word in self.topic_desc_lm}
		self.topic_num = topic_num
		self.p_t_s_w = {}
		for word in p_t_s_w:
			self.p_t_s_w[word] = p_t_s_w[word].copy()
		self.clicks_log_times = 0
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
		self.target_topic_vectors_2 = []
		documents_rels = [("topic_desc", 3)] 
		self.target_topic_vectors_2 += [target_document_vectors["topic_desc"+"_" + str(self.topic_num)]]
		
		self.documents_rels = documents_rels
		print ("DOCUMENTS RELS: ", len(documents_rels), len(self.target_topic_vectors_2))
		print ('DOCUMENTS RELS:', documents_rels)
		self.precision_lm  = None
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
	def make_topic_IN(self, topic):
		IN = Counter(topic.split())
		IN = {x:float(IN[x])/float(sum(IN.values())) for x in IN}
		print (topic, IN)
		return IN
	
	def make_collection_IN(self, topic_IN, doc_collection):
		collection_IN = {}
		'''
		for doc in doc_collection:
			total_content = doc_collection[doc]["content"] + " " + doc_collection[doc]["title"]
			words = Counter(total_content.split())
			for word in words:
				try:
					collection_IN[word] += 1
				except:
					collection_IN[word] = 1
		'''
		'''
		total_content = ""
		for docid in doc_collection:
			total_content = total_content + " " + doc_collection[docid]["content"] + " " + doc_collection[docid]["title"]
		#print ("total_content done")
		collection_IN = Counter(total_content.split())
		#print ("collection IN done 1")
		total_num_words = sum(collection_IN.values())
		collection_IN = {x:float(collection_IN[x])/float(total_num_words) for x in collection_IN}
		'''
		collection_IN = self.all_doc_language_model
		#print ("collection IN done 2")
		all_words = list(set(list(topic_IN.keys()) + list(collection_IN.keys())))
		#print ("collection IN done 3")
		uniform_prob = {word:float(1)/float(len(all_words)) for word in all_words}
		#print ("collection IN done 4")
		collection_IN = dirichlet_smoothing(collection_IN, uniform_prob)
		#print ("collection IN done 5")
		return collection_IN
	
	def prob_to_add_word(self, x):
		return (math.exp(x)/(math.exp(x)+1))

	def query_formulation_diff_C_u(self):
		print ("Previous queries: ", self.previous_queries)
		self.topic_IN = self.make_topic_IN(self.topic_desc)
		p_a = len(self.C_u_1)
		Q = []
		R_q_docs = self.C_u_1
		max_length = 4
		while (p_a > self.max_threshold) and (len(Q)<max_length):
			self.collection_IN = self.make_collection_IN(self.topic_IN, R_q_docs) #uncomment if the collection IN is hanging after every word
			max_score = -99999999
			max_word = None
			for word in self.topic_IN:
				if word not in Q:
					word_score = math.log(self.topic_IN[word]/self.collection_IN[word])
					print ("word:{} word_score: {}".format(word, word_score))
					if word_score >  max_score:
						max_score = word_score
						max_word = word
			if (max_word == None):
				break
			new_R_q_docs = {}
			r_q = 0
			for doc in R_q_docs:
				if (max_word in R_q_docs[doc]["content"]) or (max_word in R_q_docs[doc]["title"]):
					new_R_q_docs[doc] = R_q_docs[doc]
					r_q += 1
			r_q += 1
			success = False
			for (prev_query,prev_query_r_q) in self.previous_queries:
				if (set(Q+[max_word]).intersection(set(prev_query)) == set(prev_query)):
					if(prev_query_r_q == 0):
						print (self.topic_IN.keys())
						del(self.topic_IN[max_word])
						success = True
						break
			if(success):
				continue
			ratio = float(r_q)/float(len(self.C_u_1.keys()))
			#p_a = self.prob_to_add_word(ratio)
			p_a = r_q
			print ("ratio: {} p_a: {}", ratio, p_a)
			R_q_docs = new_R_q_docs
			Q = Q + [max_word]
			print ("Query: ", Q)
		self.current_query = Q
		return " ".join(Q)

	def query_formulation_same_C_u(self):
		print ("Previous queries: ", self.previous_queries)
		self.topic_IN = self.make_topic_IN(self.topic_desc)
		p_a = len(self.C_u_1)
		Q = []
		R_q_docs = self.C_u_1
		max_length = 4
		#self.collection_IN = self.make_collection_IN(self.topic_IN, self.C_u_1)
		self.collection_IN = self.all_doc_collection_lm
		word_scores = None
		while (p_a > self.max_threshold) and (len(Q)<max_length):
			print ("WORD SCORES: ", word_scores)
			max_score = -99999999
			max_word = None
			if (word_scores != None):
				if (len(word_scores) != 0):
					max_word = word_scores[0][0]
					max_score = word_scores[0][1]
					word_scores = word_scores[1:]
			else:
				word_scores = []
				for word in self.topic_IN:
					if word not in Q:
						try:
							word_score = math.log(self.topic_IN[word]/self.collection_IN[word])
						except:
							word_score = math.log(self.topic_IN[word]/new_word_probability(self.d))
						word_scores += [(word,word_score)]
						print ("word:{} word_score: {}".format(word, word_score))
						if word_score >  max_score:
							max_score = word_score
							max_word = word
				word_scores = sorted(word_scores, key = lambda l:l[1], reverse=True)
				word_scores = word_scores[1:]
			if (max_word == None):
				if (len(Q) != 0):
					success = False
					for (prev_query,prev_query_r_q) in self.previous_queries:
						if (set(Q).intersection(set(prev_query)) == set(prev_query)):
							self.topic_IN = self.make_topic_IN(self.topic_desc)
							for w in Q:
								del(self.topic_IN[w])
							Q = []
							R_q_docs = self.C_u_1
							word_scores = None
							p_a = self.max_threshold
							success = True
							break
					if(success):
						continue
				break	
			'''			
			new_R_q_docs = {}
			r_q = 0
			for doc in R_q_docs:
				if (max_word in R_q_docs[doc]["content"]) or (max_word in R_q_docs[doc]["title"]):
					new_R_q_docs[doc] = R_q_docs[doc]
					r_q += 1
			r_q += 1
			'''
			#function using index
			new_R_q_docs = {}
			r_q = 0
			try:
				doc_list = all_doc_index[max_word].keys()
				intersection_docs = list(set(R_q_docs.keys()).intersection(set(doc_list)))
				for docid in intersection_docs:
					new_R_q_docs[docid] = R_q_docs[docid]
				r_q = len(intersection_docs)
			except KeyError:
				pass
			r_q += 1
			success = False
			for (prev_query,prev_query_r_q) in self.previous_queries:
				if (set(Q+[max_word]).intersection(set(prev_query)) == set(prev_query)):
					if(prev_query_r_q == 0):
						print (self.topic_IN.keys())
						del(self.topic_IN[max_word])
						success = True
						break
			if(success):
				continue
			ratio = float(r_q)/float(len(self.C_u_1.keys()))
			#p_a = self.prob_to_add_word(ratio)
			p_a = r_q
			print ("ratio: {} p_a: {}", ratio, p_a)
			R_q_docs = new_R_q_docs
			Q = Q + [max_word]
			print ("Query: ", Q)
		self.current_query = Q
		return " ".join(Q)

	def query_formulation_p_a(self):
		print ("Previous queries: ", self.previous_queries)
		self.topic_IN = self.make_topic_IN(self.topic_desc)
		p_a = len(self.C_u_1)
		Q = []
		R_q_docs = self.C_u_1
		max_length = 4
		#self.collection_IN = self.make_collection_IN(self.topic_IN, self.C_u_1)
		self.collection_IN = self.all_doc_collection_lm
		word_scores = None
		while (p_a > self.max_threshold) and (len(Q)<max_length):
			print ("WORD SCORES: ", word_scores)
			max_score = -99999999
			max_word = None
			if (word_scores != None):
				if (len(word_scores) != 0):
					max_word = word_scores[0][0]
					max_score = word_scores[0][1]
					word_scores = word_scores[1:]
			else:
				word_scores = []
				for word in self.topic_IN:
					if word not in Q:
						for idx,topic in enumerate(self.topic_word_distribution):
							try:
								there = topic[word]
								word_topic_idx = idx
								break
							except:
								pass
						try:
							word_score = math.log(self.topic_proportion[word_topic_idx])+math.log(self.topic_IN[word]/self.topic_word_distribution[word_topic_idx][word])
						except:
							word_score = math.log(self.topic_proportion[word_topic_idx])+math.log(self.topic_IN[word]/new_word_probability(self.d))
						word_scores += [(word,word_score)]
						print ("word:{} word_score: {} word_topic_index:{}".format(word, word_score, word_topic_idx))
						if word_score >  max_score:
							max_score = word_score
							max_word = word
				word_scores = sorted(word_scores, key = lambda l:l[1], reverse=True)
				word_scores = word_scores[1:]
			if (max_word == None):
				if (len(Q) != 0):
					success = False
					for (prev_query,prev_query_r_q) in self.previous_queries:
						if (set(Q).intersection(set(prev_query)) == set(prev_query)):
							self.topic_IN = self.make_topic_IN(self.topic_desc)
							for w in Q:
								del(self.topic_IN[w])
							Q = []
							R_q_docs = self.C_u_1
							word_scores = None
							p_a = self.max_threshold
							success = True
							break
					if(success):
						continue
				break	
			#function using index
			new_R_q_docs = {}
			r_q = 0
			try:
				doc_list = all_doc_index[max_word].keys()
				intersection_docs = list(set(R_q_docs.keys()).intersection(set(doc_list)))
				for docid in intersection_docs:
					new_R_q_docs[docid] = R_q_docs[docid]
				r_q = len(intersection_docs)
			except KeyError:
				pass
			r_q += 1
			success = False
			for (prev_query,prev_query_r_q) in self.previous_queries:
				if (set(Q+[max_word]).intersection(set(prev_query)) == set(prev_query)):
					if(prev_query_r_q == 0):
						print (self.topic_IN.keys())
						del(self.topic_IN[max_word])
						success = True
						break
			if(success):
				continue
			ratio = float(r_q)/float(len(self.C_u_1.keys()))
			#p_a = self.prob_to_add_word(ratio)
			p_a = r_q
			print ("ratio: {} p_a: {}", ratio, p_a)
			R_q_docs = new_R_q_docs
			Q = Q + [max_word]
			print ("Query: ", Q)
		self.current_query = Q
		return " ".join(Q)

	def query_formulation_list(self):
		return

	def query_formulation_efficient(self):
		word_scores = self.keyword_word_scores_1(doc_collection_lm, doc_collection_lm_dist, self.topic_desc)
		l = len(word_scores)
		Q = []
		word_index = -1 
		Q = []
		R_q_docs = self.C_u_1.keys()
		query_index_values = ""
		p_a = len(self.C_u_1.keys())
		max_length = 4
		while (p_a > self.max_threshold) and (len(Q)<max_length):
			if (word_index == l-1):
				temp_query = " ".join(Q)
				try:
					there = self.previous_queries[temp_query]
					print ("present in previous queries")
					word_index = 1	
					Q = []
					continue
				except:
					break
			word_index = word_index+1
			r_q = 0
			try:
				doc_list = all_doc_index[word_scores[word_index][0]].keys()
				intersection_docs = list(set(R_q_docs).intersection(set(doc_list)))
				new_R_q_docs = intersection_docs
				r_q = len(new_R_q_docs)
			except KeyError:
				print ("WORD DOES NOT EXIST")
				new_R_q_docs = {}
				pass
			r_q += 1
			success = False
			try:
				temp_query = " ".join(Q+[word_scores[word_index][0]])
				there = self.previous_queries[temp_query]
				print ("present in previous queries")
				continue
			except KeyError:
				pass
			p_a = r_q
			print ("p_a: {}", p_a)
			R_q_docs = new_R_q_docs
			Q += [word_scores[word_index][0]]
			print ("Query: ", Q)
		word_scores_dict = dict(word_scores)
		total_query_score = 0
		for word in Q:
			 total_query_score += word_scores_dict[word]
		total_query_score = float(total_query_score)/float(len(Q))
		print ("QUERY:{} QUERY SCORE: ".format(Q, total_query_score))
		self.current_query = Q
		return " ".join(Q)

	def query_formulation_candidate_query(self):
		candidate_queries_1 = self.candidate_queries
		self.candidate_queries = self.candidate_queries[1:]
		Q = candidate_queries_1[0][0]
		print ("Query: ", Q)
		self.current_query = Q
		return " ".join(Q)

	#def greedy_soln_candidate_query(self):
	#	candidate_queries

	def reformulate_update_scores_efficient(self, results, topic_num, topic_desc_lm, alpha, beta): 
		content = ""
		for result in results:
			content += " " + result["content"]
		results_lm = Counter(content.split())
		total_words = sum(results_lm.values())
		results_lm = {word:float(results_lm[word])/float(total_words) for word in results_lm}
		updated_collection_lm = {word:alpha*doc_collection_lm_dist[word] for word in doc_collection_lm_dist}
		for word in results_lm:
			try:
				updated_collection_lm[word] += (1-alpha)*results_lm[word]
			except:
				updated_collection_lm[word] = (1-alpha)*results_lm[word]

		results_relevance = self.judge_2(results, topic_num)
		results_word_lm = {}
		results_word_not_lm = {}
		words_to_update_score = list(results_lm.keys()) + list(topic_desc_lm.keys())
		results_word_lm = dict(zip(words_to_update_score, range(len(words_to_update_score))))
		results_word_not_lm = dict(zip(words_to_update_score, range(len(words_to_update_score))))
		for idx,result in enumerate(results):
			result_lm = Counter(result["content"].split())
			result_lm_length = sum(result_lm.values())
			print (results_relevance[idx])
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
			print ("WORDS TO UPDATE SCORE: ", words_to_update_score)
			print (total_word_score)
			print (results_relevance)
			updated_topic_IN = topic_desc_lm
		else:
			results_word_lm = {word:float(results_word_lm[word])/float(total_word_score) for word in results_word_lm} 
			updated_topic_IN = {word:(1-beta)*float(results_word_lm[word]) for word in results_word_lm}
			for word in topic_desc_lm:
				updated_topic_IN[word] += beta*topic_desc_lm[word]
		updated_topic_IN = {word:updated_topic_IN[word] for word in updated_topic_IN if updated_topic_IN[word]!=0}
		total_word_not_score = sum(results_word_not_lm.values())
		updated_not_topic_IN = {}
		if (total_word_not_score == 0):
			updated_not_topic_IN = {}
		else:
			results_word_not_lm = {word:float(results_word_not_lm[word])/float(total_word_not_score) for word in results_word_not_lm}
			updated_not_topic_IN =  results_word_not_lm
		updated_not_topic_IN = {word:updated_not_topic_IN[word] for word in updated_not_topic_IN if updated_not_topic_IN[word]!=0}

		return updated_collection_lm,updated_topic_IN,updated_not_topic_IN

	def reformulate_update_scores_efficient_simple(self, results, topic_num, topic_lm, precision_lm, beta, gamma): 
		content = ""
		for result in results:
			content += " " + result["content"]
		results_lm = Counter(content.split())
		total_words = sum(results_lm.values())
		results_lm = {word:float(results_lm[word])/float(total_words) for word in results_lm}

		results_relevance = self.judge_2(results, topic_num)
		results_word_lm = {}
		results_word_not_lm = {}
		words_to_update_score = list(results_lm.keys()) + list(topic_lm.keys())
		results_word_lm = dict(zip(words_to_update_score, range(len(words_to_update_score))))
		results_word_not_lm = dict(zip(words_to_update_score, range(len(words_to_update_score))))
		for idx,result in enumerate(results):
			result_lm = Counter(result["content"].split())
			result_lm_length = sum(result_lm.values())
			print (results_relevance[idx])
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
			print ("WORDS TO UPDATE SCORE: ", words_to_update_score)
			print (total_word_score)
			print (results_relevance)
			updated_topic_IN = topic_lm
		else:
			results_word_lm = {word:float(results_word_lm[word])/float(total_word_score) for word in results_word_lm} 
			updated_topic_IN = {word:(1-beta)*float(results_word_lm[word]) for word in results_word_lm}
			for word in topic_lm:
				updated_topic_IN[word] += beta*topic_lm[word]
		updated_topic_IN = {word:updated_topic_IN[word] for word in updated_topic_IN if updated_topic_IN[word]!=0}
		total_word_not_score = sum(results_word_not_lm.values())
		if (precision_lm == None):
			precision_lm = {}
			for word in topic_lm:
				precision_lm[word] = {}
				try:
					precision_lm[word]["num"] = topic_lm[word]
					precision_lm[word]["den"] = doc_collection_lm_dist[word]
				except:
					precision_lm[word]["num"] = topic_lm[word]
					precision_lm[word]["den"] = 0
			self.precision_lm = precision_lm
			precision_lm = precision_lm
		#Type1 update
		results_not_topic_IN = {}
		if (total_word_not_score == 0):
			print ("WORDS TO UPDATE SCORE: ", words_to_update_score)
			print (total_word_not_score)
			print (results_relevance)
			results_not_topic_IN = {}
		else:
			results_word_not_lm = {word:float(results_word_not_lm[word])/float(total_word_not_score) for word in results_word_not_lm}
			results_not_topic_IN =  results_word_not_lm
		results_not_topic_IN = {word:results_not_topic_IN[word] for word in results_not_topic_IN if results_not_topic_IN[word]!=0}
		'''
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
		'''
		
		updated_precision_lm = {}
		for word in updated_topic_IN:
			try:
				there = precision_lm[word]
				updated_precision_lm[word] = {}
				#if precision_lm[word]["den"] == 0 
				updated_precision_lm[word]["num"] = precision_lm[word]["num"]/precision_lm[word]["den"]
				updated_precision_lm[word]["den"] = 1
			except:
				updated_precision_lm[word] = {}
				updated_precision_lm[word]["num"] = 0
				updated_precision_lm[word]["den"] = 1
			try:
				updated_precision_lm[word]["num"]  = gamma*updated_precision_lm[word]["num"] + ((1-gamma)*(updated_topic_IN[word])/(updated_topic_IN[word]+results_not_topic_IN[word]))
				updated_precision_lm[word]["den"] = 1
			except:
				updated_precision_lm[word]["num"]  = gamma*updated_precision_lm[word]["num"] + ((1-gamma)*(updated_topic_IN[word])/(updated_topic_IN[word]+0))
				updated_precision_lm[word]["den"] = 1
		
		return updated_topic_IN,results_not_topic_IN,results_word_lm, updated_precision_lm


	def judge_2(self, results, topic_num):
		target_topic_vectors_2 = self.target_topic_vectors_2
		results_relevance = []
		lda,comm_dict = target_documents_details["all_topics"][1][0],target_documents_details["all_topics"][1][1]
		#print (len(target_topic_vectors_2))
		#documents_rels = documents_rels[:30]
		print (len(target_topic_vectors_2), len(self.documents_rels))
		for result in results:
			weighted_sum_num = 0
			weighted_sum_den = 0
			a = lda[comm_dict.doc2bow(result["content"].split())]
			topic_dist_vector = [0]*100
			for v in a:
				topic_dist_vector[v[0]] = v[1]
			for idx,(docid,doc_rel) in enumerate(self.documents_rels): 
				weighted_sum_num += doc_rel*np.dot(normalize_vector(topic_dist_vector), normalize_vector(target_topic_vectors_2[idx]))
				#weighted_sum_num += doc_rel*np.dot(topic_dist_vector, target_topic_vectors_2[idx])
				weighted_sum_den += self.documents_rels[idx][1]
			#cos_sim = sum([vector[i]*topic_dist_vector[i] for i in range(len(topic_dist_vector))])
			#weighted_sum_num += documents_rels[idx][1]*cos_sim
			result_relevance = float(weighted_sum_num)/float(weighted_sum_den)
			#normalization of result relevance
			results_relevance += [result_relevance]
		return results_relevance
	def judge(self, results, topic_num):
		results_relevance = []
		[target_doc_lm, target_doc_weighted_lm, documents, documents_rels, topic_lda_comm_dict] = target_documents_details[topic_num]
		lda,comm_dict = target_documents_details["all_topics"][1][0],target_documents_details["all_topics"][1][1]
		texts = [d.split() for d in documents]
		target_corpus = [comm_dict.doc2bow(text) for text in texts]
		target_topic_vectors = lda[target_corpus]
		documents_rels += [("topic_desc", 3)]
		print (len(target_topic_vectors), len(documents_rels), len(documents))
		for result in results:
			weighted_sum_num = 0
			weighted_sum_den = 0
			a = lda[comm_dict.doc2bow(result["content"].split())]
			topic_dist_vector = [0]*100
			for v in a:
				topic_dist_vector[v[0]] = v[1]
			for idx,vect in enumerate(target_topic_vectors):
				vector = [0]*100
				for v in vect:
					vector[v[0]] = v[1]
				print (vector, len(vector))
				print (topic_dist_vector, len(topic_dist_vector))
				cos_sim = sum([vector[i]*topic_dist_vector[i] for i in range(len(topic_dist_vector))])
				weighted_sum_num += documents_rels[idx][1]*cos_sim
				weighted_sum_den += documents_rels[idx][1]
			result_relevance = float(weighted_sum_num)/float(weighted_sum_den)
			#normalization of result relevance
			results_relevance += [result_relevance]
		return results_relevance
	def reformulate_new_candidate_queries(self, results):
		updated_collection_lm,updated_topic_IN,updated_not_topic_IN = self.reformulate_update_scores_efficient(results, self.topic_num, self.topic_desc_lm, 1.0,0.9)
		word_scores = get_reformulated_word_scores_clueweb(updated_collection_lm, updated_topic_IN, updated_not_topic_IN, 0.9, "../supervised_models/log_reg_model_word_type_1_var_session_nums_5.pk")
		#ame candidate_queries
		word_scores_dict = dict(word_scores)
		alpha = 0.8
		query_scores = []
		for query_tuple in self.candidate_queries:
			query = query_tuple[0]
			prev_word_score = query_tuple[1]
			bigram_score = query_tuple[2]
			word_score = float(sum([word_scores_dict[word] for word in query]))/float(len(query))
			new_total_score = alpha*float(word_score)/float(3.0) + (1-alpha)*bigram_score
			query_scores += [(query, word_score, bigram_score, new_total_score)]
		self.candidate_queries = sorted(query_scores, key=lambda l :l[3], reverse=True)
		return

	def reformulate_new_new_greedy_soln_queries(self, results):
		updated_topic_IN,updated_not_topic_IN,results_word_lm, updated_precision_lm = self.reformulate_update_scores_efficient_simple(results, self.topic_num, self.topic_desc_lm, self.precision_lm, 0.9, 0.7)
		self.topic_desc_lm = updated_topic_IN
		self.precision_lm = updated_precision_lm
		query_scores = get_reformulated_greedy_soln_queries(math.log(0.0001), updated_topic_IN, self.precision_lm, updated_not_topic_IN, results_word_lm, doc_collection_lm_dist, -1 , self.topic_num)
		
		new_candidates = sorted(query_scores, key=lambda l :l[1], reverse=True)
		first_query = new_candidates[0][0].copy()
		first_query.sort()
		while (" ".join(first_query) in self.previous_queries):
			new_candidates = new_candidates[1:]
			first_query = new_candidates[0][0].copy()
			first_query.sort()
		self.candidate_queries = new_candidates
		return

	def reformulate_new_candidate_queries_unsup(self, results):
		updated_collection_lm,updated_topic_IN,updated_not_topic_IN = self.reformulate_update_scores_efficient(results, self.topic_num, self.topic_desc_lm, 1.0,0.7)
		word_scores = get_unsup_reformulated_word_scores_clueweb(updated_collection_lm, updated_topic_IN, updated_not_topic_IN, 0.7)
		#ame candidate_queries
		word_scores_dict = dict(word_scores)
		alpha = 0.8
		query_scores = []
		for query_tuple in self.candidate_queries:
			query = query_tuple[0]
			prev_word_score = query_tuple[1]
			bigram_score = query_tuple[2]
			word_score = float(sum([word_scores_dict[word] for word in query]))/float(len(query))
			new_total_score = alpha*float(word_score)/float(3.0) + (1-alpha)*bigram_score
			query_scores += [(query, word_score, bigram_score, new_total_score)]
		self.candidate_queries = sorted(query_scores, key=lambda l :l[3], reverse=True)
		return

	def reformulate_new_new_candidate_queries(self, results):
		updated_collection_lm,updated_topic_IN,updated_not_topic_IN = self.reformulate_update_scores_efficient(results, self.topic_num, self.topic_desc_lm, 1.0, 0.7)
		word_scores = get_reformulated_word_scores_clueweb(updated_collection_lm, updated_topic_IN, updated_not_topic_IN, 0.7)
		#print (len(word_scores)) 
		new_candidates = query_formulation_list(word_scores[:30], doc_keys_list, all_doc_index, 1, 500, self.topic_num)
		#print (len(new candidates))
		new_candidates = self.add_query_phrase_scores(new_candidates, self.topic_num)
		first_query = new_candidates[0][0].copy()
		first_query.sort()
		while (" ".join(first_query) in self.previous_queries):
			new_candidates = new_candidates[1:]
			first_query = new_candidates[0][0].copy()
			first_query.sort()
		self.candidate_queries = new_candidates
		return

	def add_query_phrase_scores(self, candidate_queries,topic_num):
		query_scores = []
		alpha = 0.8
		for (query,doc_list, score) in candidate_queries:
			num_phrases = 0
			for idx,word1 in enumerate(query):
				for word2 in query[idx+1:]:
					if (word1+" "+word2) in bigram_topic_lm[topic_num]:
						num_phrases += bigram_topic_lm[topic_num][word1+" "+word2]

					#if (word1 + " " +word2) in keywords:
					#	num_phrases += 1
			if (len(query)>1) and (num_phrases>0):
				p_bigrams = float(num_phrases)/float(len(query)*(len(query)-1))
			else:
				p_bigrams = 0.000001
			score1 = alpha*score/float(3.0) + (1-alpha)*math.log(p_bigrams) 
			score2 = alpha*score/float(3.0) + (1-alpha)*float(1)/float(math.exp(-float(num_phrases)/float(len(query)))+1)
			query_scores += [(query, score, float(1)/float(math.exp(-float(num_phrases)/float(len(query)))+1),score2)]
		query_scores = sorted(query_scores, key =lambda l:l[3],reverse=True)
		return query_scores

	def keyxword_word_scores_1(self,doc_collection_lm, doc_collection_lm_dist, topic_desc):
		if (self.word_scores == None):
			d = sum(doc_collection_lm.values())
			total_num_words = sum(doc_collection_lm.values())
			topic_IN = get_topic_IN_with_keywords_2(topic_desc)
			word_scores = []
			for word in topic_IN:
				try:
					word_score = math.log(topic_IN[word]/doc_collection_lm_dist[word])
				except:
					word_score = 0 #math.log(topic_IN[word]/new_word_probability(d))
				word_scores += [(word,word_score)]
				print ("word:{} word_score: {}".format(word, word_score))
				try:
					print ("Doc collection word score: ", doc_collection_lm[word])
				except:
					print ("Doc collection word score: Not available")
			word_scores = sorted(word_scores, key = lambda l :l[1], reverse=True)
			self.word_scores = word_scores
			print ("WORD SCORES: ", word_scores)
		return self.word_scores

	
	def keyword_word_scores_2(self,doc_collection_lm, doc_collection_lm_dist, topic_desc):
		feature_weights = [0,1.0]
		if (self.word_scores == None):
			d = sum(doc_collection_lm.values())
			total_num_words = sum(doc_collection_lm.values())
			topic_IN = get_topic_IN_with_keywords(topic_desc)
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
			self.word_scores = word_scores
			print ("WORD SCORES: ", word_scores)
		return self.word_scores

	def basic_word_scores_1(self,doc_collection_lm, doc_collection_lm_dist, topic_desc):
		if (self.word_scores == None):
			d = sum(doc_collection_lm.values())
			total_num_words = sum(doc_collection_lm.values())
			topic_desc = preprocess(topic_desc, "../Session_track_2014/clueweb_snippet/lemur-stopwords.txt", lemmatizing = True)
			topic_IN = self.make_topic_IN(topic_desc)
			word_scores = []
			for word in topic_IN:
				try:
					word_score = math.log(topic_IN[word]/doc_collection_lm_dist[word])
				except:
					word_score = 0 #math.log(topic_IN[word]/new_word_probability(d))
				word_scores += [(word,word_score)]
				print ("word:{} word_score: {}".format(word, word_score))
				try:
					print ("Doc collection word score: ", doc_collection_lm[word])
				except:
					print ("Doc collection word score: Not available")
			word_scores = sorted(word_scores, key = lambda l :l[1], reverse=True)
			self.word_scores = word_scores
			print ("WORD SCORES: ", word_scores)
		return self.word_scores

	def basic_word_scores_2(self,doc_collection_lm, doc_collection_lm_dist, topic_desc):
		feature_weights = [3.0/4.0,1.0/4.0]
		if (self.word_scores == None):
			d = sum(doc_collection_lm.values())
			total_num_words = sum(doc_collection_lm.values())
			topic_desc = preprocess(topic_desc, "../Session_track_2014/clueweb_snippet/lemur-stopwords.txt", lemmatizing = True)
			topic_IN = self.make_topic_IN(topic_desc)
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
			self.word_scores = word_scores
			print ("WORD SCORES: ", word_scores)
		return self.word_scores

	def query_formulation_QS3(self):
		print ("Previous queries: ", self.previous_queries)
		topic_desc = preprocess(self.topic_desc, lemmatizing = True)
		topic_desc2 = preprocess(self.topic_desc, "../Session_track_2014/clueweb_snippet/lemur-stopwords.txt", lemmatizing = True)
		self.topic_IN = self.make_topic_IN(topic_desc2)
		text = dict()
		text[1] = topic_desc
		bigram_topic_IN = get_bigram_word_lm(text)
		all_possible_bigrams = {}
		for word1 in self.topic_IN:
			for word2 in self.topic_IN:
				if(word1!=word2):
					bigram = word1 + " " + word2
					all_possible_bigrams[bigram] = 1
		Q = []
		max_length = 4
		#self.collection_IN = self.make_collection_IN(self.topic_IN, self.C_u_1)
		self.collection_IN = self.all_doc_collection_lm
		max_score = -99999999
		max_bigram = None
		for bigram in all_possible_bigrams:
			bigram_score = 0
			try:
				bigram_score += bigram_topic_IN[bigram]
			except KeyError:
				pass
			if (bigram_score>max_score):
				max_score = bigram_score
				max_bigram = bigram
		Q = max_bigram.split()
		max_score = -99999999
		max_word = None
		while(max_word == None):	
			for word in self.topic_IN:
				if word not in Q:
					try:
						word_score = math.log(self.topic_IN[word]/self.collection_IN[word])
					except:
						word_score = math.log(self.topic_IN[word]/new_word_probability(self.d))
					print ("word:{} word_score: {}".format(word, word_score))
					if word_score >  max_score:
						max_score = word_score
						max_word = word
			if(max_word == None):
				break
			success = False
			try:
				temp_query = " ".join(Q+[max_word])
				there = self.previous_queries[temp_query]
				print ("present in previous queries")
				del(self.topic_IN[max_word])
				success = True
			except KeyError:
				pass
			if(success):
				max_score = -99999999
				max_word = None
		if (max_word != None):
			Q = Q+ [max_word]
		self.current_query = Q
		print ("Query: ", Q)
		return " ".join(Q)

	def query_formulation_QS3plus(self):
		print ("Previous queries: ", self.previous_queries)
		if (self.QS3_queries == None):
			self.candidate_queries = None
			topic_desc = preprocess(self.topic_desc, lemmatizing = True)
			topic_desc2 = preprocess(self.topic_desc, "../Session_track_2014/clueweb_snippet/lemur-stopwords.txt", lemmatizing = True)
			self.topic_IN = self.make_topic_IN(topic_desc2)
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
			self.collection_IN = self.all_doc_collection_lm
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
						query_score += math.log(self.topic_IN[word]/doc_collection_lm_dist[word])
					except:
						query_score += math.log(self.topic_IN[word]/new_word_probability(self.d))
				for word in self.topic_IN:
					if word not in Q:
						try:
							word_score = math.log(self.topic_IN[word]/doc_collection_lm_dist[word])
						except:
							word_score = math.log(self.topic_IN[word]/new_word_probability(self.d)) 
						all_queries += [(bigram.split() + [word],query_score + word_score)]
			all_queries = sorted(all_queries, key = lambda l: l[1], reverse=True)
			self.QS3_queries = all_queries
			self.candidate_queries = all_queries
		candidate_queries = self.QS3_queries[0][0]
		self.QS3_queries = self.QS3_queries[1:]
		Q = candidate_queries
		print ("Query: ", Q)
		self.current_query = Q
		return " ".join(Q)

	def query_likelihood(self, query):
		print ("Previous queries: ", self.previous_queries)
		self.topic_IN = self.make_topic_IN(self.topic_desc)
		R_q_docs = self.C_u_1
		log_likelihood = 0
		while (query!= []):
			self.collection_IN = self.make_collection_IN(self.topic_IN, R_q_docs) 
			max_likelihood = -99999999
			max_word = None
			for word in query:
				word_likelihood = word_score = math.log(self.topic_IN[word]/self.collection_IN[word])
				if word_likelihood >  max_likelihood:
					max_likelihood = word_likelihood
					max_word = word
			log_likelihood += max_likelihood
			new_R_q_docs = {}
			r_q = 0
			for doc in R_q_docs:
				if (max_word in R_q_docs[doc]["content"]) or (max_word in R_q_docs[doc]["title"]):
					new_R_q_docs[doc] = R_q_docs[doc]
					r_q += 1
			r_q += 1
			p_a = self.prob_to_add_word(r_q-self.max_threshold)
			log_likelihood += math.log(p_a)
			print ("max_word: {} p_a: {} log_likelihood: {} ", p_a, max_word, log_likelihood)
			R_q_docs = new_R_q_docs
			query.remove(max_word)
		return log_likelihood

	def no_reformulation_query_sequence(self):
		query_sequence = [' '.join(query[0]) for query in self.candidate_queries]
		return query_sequence

	def click_sequence(results_sequence):
		click_sequence = []
		for results in results_sequence:
			clicked_items = self.click_method(results)
			click_sequence += [clicked_items]
		for idx,clicked_items in enumerate(click_sequence):
			if sum([x[1] for x in clicked_products]) > 0:
				self.clicks_log_times += 1
			if (self.clicks_log_times>=3):
				break
		return click_sequence[:(idx+1)]

	def continue_or_reformulate1(self, result_product_ids, clicked_products):
		#if no clicked product then reformulate
		'''
		r_q = 0
		for doc in self.C_u_1:
			present = 0
			for word in self.current_query:
				if (word in self.C_u_1[doc]["content"]) or (word in self.C_u_1[doc]["title"]):
					present += 1
			if(present == len(self.current_query)):
				r_q += 1
		'''
		'''
		R_q_docs = self.C_u_1
		r_q = 0
		for word in self.current_query:
			try:
				new_R_q_docs = {}
				doc_list = all_doc_index[word].keys()
				intersection_docs = list(set(R_q_docs.keys()).intersection(set(doc_list)))
				for docid in intersection_docs:
					new_R_q_docs[docid] = R_q_docs[docid]
				r_q = len(intersection_docs)
				R_q_docs = new_R_q_docs
			except KeyError:
				r_q = 0
				break
		if (r_q <= self.max_threshold):
			r_q = 0
		'''
		if sum([x[1] for x in clicked_products]) == 0:
			#self.previous_queries += [(self.current_query,r_q)]
			self.current_query.sort()
			self.previous_queries[" ".join(self.current_query)] = 1
			return True
		else:
			return True

	def click_method(self, results):
		#only clicks on target items, clicks are already given here. #user will not click on same result agian
		clicks = []
		for result in results:
			if result["docid"] in self.target_clueweb_ids:
				clicks += [(result["docid"], 1)]
				del(self.target_clueweb_ids[result["docid"]])
			else:
				clicks += [(result["docid"], 0)]
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
			return True
		else:
			if (len(self.target_clueweb_ids) ==0):
				return True
			return False

	def first_interaction(self):
		self.session_length = 0
		Q = self.query_formulation_QS3plus()
		return (Q)

	def next_interaction(self, result_list):
		clicked_items = self.click_method(result_list)
		#self.update_doc_collection(result_list)
		end = self.end_session_2(clicked_items)
		if (not end):
			reformulate = self.continue_or_reformulate1(result_list, clicked_items)
			if (reformulate):
				#self.reformulate_new_new_greedy_soln_queries(result_list)
				Q = self.query_formulation_QS3plus()
				return (1, Q, clicked_items)
			else:
				return (0, [], clicked_items)
		else:
			return (2,[],clicked_items)

	def query_reformulation_with_sessions(self, results=None):
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
				print (int(len(candidate_query)), int(chosen_l), candidate_query)
			print ("Chosen l candidate query: ", chosen_l, candidate_query)
			candidate_queries += [candidate_query]
		print ("Chosen ls: ", [len(query) for query in candidate_queries])
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
			print ("Ps sorted: ", p_w_l_t_sorted)
			candidate_query = []
			#while (len(candidate_query) != chosen_l):
			for (word,score) in p_w_l_t_sorted:
				if int(len(candidate_query)) == int(chosen_l):
					break
				if(random.random()>=0.5):
					candidate_query += [word]
				print (int(len(candidate_query)), int(chosen_l), candidate_query)
			print ("Chosen l candidate query: ", chosen_l, candidate_query)
			candidate_queries += [candidate_query]
		print ("Chosen ls: ", [len(query) for query in candidate_queries])
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







