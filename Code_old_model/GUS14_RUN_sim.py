#QCM METHOD AND SAT CLICKS
from create_dataset import *
from User_model_utils import *
import json
import pickle
(doc_collection_lm, doc_collection_lm_binary) = pickle.load(open("../TREC_Robust_data/all_doc_language_model.pk", "rb"))
total_num_words = sum(doc_collection_lm.values())
doc_collection_lm_dist = {term:float(doc_collection_lm[term])/float(total_num_words) for term in doc_collection_lm}
total_num_words_binary = sum(doc_collection_lm_binary.values())
doc_collection_lm_binary_dist = {term:float(doc_collection_lm_binary[term])/float(total_num_words_binary) for term in doc_collection_lm_binary}
IDX = {}
all_doc_index = pickle.load(open("../TREC_Robust_data/all_doc_index_lemmatized.pk", "rb") )
N = len(all_doc_index[1].keys()) 
all_doc_index = all_doc_index[0]
for word in doc_collection_lm_binary:
	IDX[word] = math.log(float(N)/float(doc_collection_lm_binary[word] + 1))+1
big_corpus = json.load(open("../TREC_Robust_data/each_doc_lm.json", "r"))
big_doc_l = json.load(open("../TREC_Robust_data/each_doc_length.json", "r"))

class GUS14_RUN():
	def __init__(self, QCM_params, w_param):
		self.corpus = big_corpus
		self.stopwords = {}
		self.doc_l = big_doc_l
		self.mu = 5000
		self.parameters = QCM_params
		self.prev_docid_scores = {}
		for docid in self.corpus:
			self.prev_docid_scores[docid] = []
		self.prev_results = []
		self.prev_queries = []
		self.queries = [" "]
		self.w = w_param
		self.session_doc_SAT_clicks = {}
	def query_score_RUN1(self, query):
		i = 0
		docid_scores = {}
		for docid in self.corpus:
			docid_scores[docid] = 0
			for word in query.split():
				try:
					docid_scores[docid] += float(self.corpus[docid][word]+self.mu*doc_collection_lm_dist[word])/float(self.doc_l[docid]+self.mu)
				except KeyError:
					try:
						docid_scores[docid] += float(self.mu*doc_collection_lm_dist[word])/float(self.doc_l[docid]+self.mu)
					except KeyError:
						docid_scores[docid] += float(self.mu)/float(self.doc_l[docid]+self.mu)*float(1)/float(len(all_doc_index.keys()))						
			#if (i%1000 ==0):
			#	print (i)
			i = i+1
		sorted_word_scores = sorted(docid_scores.items(), key=lambda l: l[1], reverse=True)
		return sorted_word_scores
	def QCM(self, query):
		q1 = self.queries[-1].split()
		q2 = query.split()
		theme_words = list(set(q2).intersection(set(q1)))
		removed_words = list(set(q1).difference(set(q2)))
		if (q1 == []):
			added_words = []
		else:
			added_words = list(set(q2).difference(set(q1)))
		d_prev_lm = self.get_best_previous_document(self.prev_results, q1)
		docid_scores = {}
		cumm_docid_scores_1 = {}
		cumm_docid_scores_2 = {}
		i = 0
		for docid in self.corpus:
			docid_scores[docid] = 0
			for word in query.split():
				try:
					docid_scores[docid] += math.log(float(self.corpus[docid][word]+self.mu*doc_collection_lm_dist[word])/float(self.doc_l[docid]+self.mu))
				except KeyError:
					try:
						docid_scores[docid] += math.log(float(self.mu*doc_collection_lm_dist[word])/float(self.doc_l[docid]+self.mu))
					except KeyError:
						docid_scores[docid] += math.log(float(self.mu)/float(self.doc_l[docid]+self.mu))*float(1)/float(len(all_doc_index.keys()))						
			for word in theme_words:
				try:
					word_score = math.log(float(self.corpus[docid][word]+self.mu*doc_collection_lm_dist[word])/float(self.doc_l[docid]+self.mu))
				except KeyError:
					try:
						word_score = math.log(float(self.mu*doc_collection_lm_dist[word])/float(self.doc_l[docid]+self.mu))
					except KeyError:
						word_score = math.log(float(self.mu)/float(self.doc_l[docid]+self.mu))*float(1)/float(len(all_doc_index.keys()))						
				try:
					docid_scores[docid] += self.parameters[0]*(1-d_prev_lm[word])*word_score
				except KeyError:
					docid_scores[docid] += self.parameters[0]*word_score
			for word in removed_words:
				try:
					word_score = math.log(float(self.corpus[docid][word]+self.mu*doc_collection_lm_dist[word])/float(self.doc_l[docid]+self.mu))
				except KeyError:
					try:
						word_score = math.log(float(self.mu*doc_collection_lm_dist[word])/float(self.doc_l[docid]+self.mu))
					except KeyError:
						word_score = math.log(float(self.mu)/float(self.doc_l[docid]+self.mu))*float(1)/float(len(all_doc_index.keys()))						
				try:
					docid_scores[docid] -= self.parameters[3]*(d_prev_lm[word])*word_score
				except KeyError:
					docid_scores[docid] += 0
			for word in added_words:
				try:
					word_score = math.log(float(self.corpus[docid][word]+self.mu*doc_collection_lm[word])/float(self.doc_l[docid]+self.mu))
				except KeyError:
					try:
						word_score = math.log(float(self.mu*doc_collection_lm_dist[word])/float(self.doc_l[docid]+self.mu))
					except KeyError:
						word_score = math.log(float(self.mu)/float(self.doc_l[docid]+self.mu))*float(1)/float(len(all_doc_index.keys()))						
				try:
					docid_scores[docid] -= self.parameters[1]*(d_prev_lm[word])*word_score
				except KeyError:
					try:
						docid_scores[docid] += self.parameters[2]*(IDX[word])*word_score
					except KeyError:
						docid_scores[docid] += self.parameters[2]*(math.log(float(N)/float(1)) + 1)*word_score						
			cumm_docid_scores_1[docid] = docid_scores[docid] + sum([score for score in self.prev_docid_scores[docid]])
			cumm_docid_scores_2[docid] = docid_scores[docid]
			for idx,score in enumerate(self.prev_docid_scores[docid]):
				if(self.prev_queries[idx][1] == 1):
					cumm_docid_scores_2[docid] += self.prev_docid_scores[docid] #better scoring
				else:
					cumm_docid_scores_2[docid] += self.w*self.prev_docid_scores[docid] #better scoring
			self.prev_docid_scores[docid] += [docid_scores[docid]]
			if (i%100000 ==0):
				print (i)
			i = i+1
		cumm_docid_scores_1 = sorted(cumm_docid_scores_1.items(), key=lambda l: l[1], reverse=True)
		cumm_docid_scores_2 = sorted(cumm_docid_scores_2.items(), key=lambda l: l[1], reverse=True)
		return cumm_docid_scores_1,cumm_docid_scores_2
	def get_best_previous_document(self, prev_results, query):
		max_score = 0
		if prev_results[-1] == []:
			return {}
		max_lm = {}
		for result in prev_results[-1]:
			d_prev_lm = Counter(result["content"].split())
			d_length = sum(d_prev_lm.values())
			d_prev_lm = {word:float(d_prev_lm[word])/float(d_length) for word in d_prev_lm}
			probability = 1
			for word in query:
				try:
					probability *= (1-d_prev_lm[word])
				except KeyError:
					probability *= (1)
			probability = 1 - probability
			if(probability > max_score):
				max_score = probability
				max_lm = d_prev_lm
		return max_lm
	def QCM_SAT_clicks(self, query):
		q1 = self.queries[-1].split()
		q2 = query.split()
		theme_words = list(set(q2).intersection(set(q1)))
		removed_words = list(set(q1).difference(set(q2)))
		if (q1 == []):
			added_words = []
		else:
			added_words = list(set(q2).difference(set(q1)))
		d_prev_lm = self.get_best_previous_document(self.prev_results, q1)
		docid_scores = {}
		cumm_docid_scores_1 = {}
		cumm_docid_scores_2 = {}
		i = 0
		for docid in self.corpus:
			docid_scores[docid] = 0
			for word in query.split():
				try:
					docid_scores[docid] += math.log(float(self.corpus[docid][word]+self.mu*doc_collection_lm_dist[word])/float(self.doc_l[docid]+self.mu))
				except KeyError:
					try:
						docid_scores[docid] += math.log(float(self.mu*doc_collection_lm_dist[word])/float(self.doc_l[docid]+self.mu))
					except KeyError:
						docid_scores[docid] += math.log(float(self.mu)/float(self.doc_l[docid]+self.mu))*float(1)/float(len(all_doc_index.keys()))						
			for word in theme_words:
				try:
					word_score = math.log(float(self.corpus[docid][word]+self.mu*doc_collection_lm_dist[word])/float(self.doc_l[docid]+self.mu))
				except KeyError:
					try:
						word_score = math.log(float(self.mu*doc_collection_lm_dist[word])/float(self.doc_l[docid]+self.mu))
					except KeyError:
						word_score = math.log(float(self.mu)/float(self.doc_l[docid]+self.mu))*float(1)/float(len(all_doc_index.keys()))						
				try:
					docid_scores[docid] += self.parameters[0]*(1-d_prev_lm[word])*word_score
				except KeyError:
					docid_scores[docid] += self.parameters[0]*word_score
			for word in removed_words:
				try:
					word_score = math.log(float(self.corpus[docid][word]+self.mu*doc_collection_lm_dist[word])/float(self.doc_l[docid]+self.mu))
				except KeyError:
					try:
						word_score = math.log(float(self.mu*doc_collection_lm_dist[word])/float(self.doc_l[docid]+self.mu))
					except KeyError:
						word_score = math.log(float(self.mu)/float(self.doc_l[docid]+self.mu))*float(1)/float(len(all_doc_index.keys()))						
				try:
					docid_scores[docid] -= self.parameters[3]*(d_prev_lm[word])*word_score
				except KeyError:
					docid_scores[docid] += 0
			for word in added_words:
				try:
					word_score = math.log(float(self.corpus[docid][word]+self.mu*doc_collection_lm[word])/float(self.doc_l[docid]+self.mu))
				except KeyError:
					try:
						word_score = math.log(float(self.mu*doc_collection_lm_dist[word])/float(self.doc_l[docid]+self.mu))
					except KeyError:
						word_score = math.log(float(self.mu)/float(self.doc_l[docid]+self.mu))*float(1)/float(len(all_doc_index.keys()))						
				try:
					docid_scores[docid] -= self.parameters[1]*(d_prev_lm[word])*word_score
				except KeyError:
					try:
						docid_scores[docid] += self.parameters[2]*(IDX[word])*word_score
					except KeyError:
						docid_scores[docid] += self.parameters[2]*(math.log(float(N)/float(1)) + 1)*word_score						
			cumm_docid_scores_1[docid] = docid_scores[docid] + sum([score for score in self.prev_docid_scores[docid]])
			cumm_docid_scores_2[docid] = docid_scores[docid]
			for idx,score in enumerate(self.prev_docid_scores[docid]):
				if(self.prev_queries[idx][1] == 1):
					cumm_docid_scores_2[docid] += self.prev_docid_scores[docid] #better scoring
				else:
					cumm_docid_scores_2[docid] += self.w*self.prev_docid_scores[docid] #better scoring
			self.prev_docid_scores[docid] += [docid_scores[docid]]
			if (i%100000 ==0):
				print (i)
			i = i+1
		total_sat_click_sum = sum([1*self.session_doc_SAT_clicks[s][0] + 2*self.session_doc_SAT_clicks[s][1] for s in self.session_doc_SAT_clicks])
		for docid in cumm_docid_scores_2:
			if docid in self.session_doc_SAT_clicks:
				cumm_docid_scores_2[docid] += float(1*self.session_doc_SAT_clicks[docid][0] + 2*self.session_doc_SAT_clicks[docid][1])/float(total_sat_click_sum)  
				cumm_docid_scores_1[docid] += float(1*self.session_doc_SAT_clicks[docid][0] + 2*self.session_doc_SAT_clicks[docid][1])/float(total_sat_click_sum)  
		cumm_docid_scores_1 = sorted(cumm_docid_scores_1.items(), key=lambda l: l[1], reverse=True)
		cumm_docid_scores_2 = sorted(cumm_docid_scores_2.items(), key=lambda l: l[1], reverse=True)
		return cumm_docid_scores_1,cumm_docid_scores_2

	def query_score_RUN2_R2(self,query):
		cumm_docid_scores_1,cumm_docid_scores_2 = self.QCM(query)
		sorted_word_scores = sorted(cumm_docid_scores_1.items(), key=lambda l: l[1], reverse=True)
		return sorted_word_scores

	def update_with_results(self, results, clicks, query):
		query = " ".join(query)
		self.queries += [preprocess(query, lemmatizing=True)]
		if self.prev_queries == []:
			self.prev_results += [results[:10]]
			if clicks != []:
				for click in clicks:
					click_dict = {}
					click_dict = results[click[1]-1].copy()
					click_dict["click"] = 1 
					self.prev_results[-1] += [click_dict]
				self.prev_queries += [[query, 1]]
			else:
				self.prev_queries += [[query, 0]]
		else:
			if (self.prev_queries[-1][0] == query):
				self.prev_results[-1] += results[:10]
			else:
				self.prev_results += [results[:10]]
			if clicks != []:
				for click in clicks:
					click_dict = {}
					click_dict = results[click[1]-1].copy()
					click_dict["click"] = 1 
					self.prev_results[-1] += [click_dict]
				if (self.prev_queries[-1][0] == query):
					self.prev_queries[-1][1] = 1
				else:
					self.prev_queries += [[query, 1]]	
			else:
				if (self.prev_queries[-1][0] == query):
					self.prev_queries[-1][1] += 0
				else:
					self.prev_queries += [[query, 0]]
		if clicks != []:
			if click[2] >= 30:
				try:
					self.session_doc_SAT_clicks[click[0]][1] += 1
				except KeyError:
					self.session_doc_SAT_clicks[click[0]] = [0,1]				
			elif (click[2]>10 and click[2]<30):
				try:
					self.session_doc_SAT_clicks[click[0]][0] += 1
				except KeyError:
					self.session_doc_SAT_clicks[click[0]] = [1,0]
	def use_session_information(self, session):
		for interaction in session.interactions[:-1]:
			self.update_with_results(interaction.results, interaction.clicks, interaction.query)


topic_descs = read_topic_descs()
topic_rel_docs = read_judgements()
ideal_user_sessions,precise_user_sessions,recall_user_sessions,all_sessions = select_sessions(topic_rel_docs)
robust_data_collection = read_robust_data_collection()
clueweb_snippet_collection,clueweb_snippet_collection_2 = read_clueweb_snippet_data()
print("started reading done")
print ("Num docs: ", len(robust_data_collection))
for docid in clueweb_snippet_collection_2:
	try:
		there = robust_data_collection[docid]
		print ("WEIRD: " , docid)
	except:
		robust_data_collection[docid] = clueweb_snippet_collection_2[docid]
print("started reading done")
i = 0

act_sessions = pickle.load(open("../simulated_sessions/simulated_session_all_topics_word_sup_1_var.pk", "rb"))
NDCG = []
for session in act_sessions:
	try:
		there = topic_rel_docs[session.topic_num]
	except:
		continue
	topic_rel_docs_scores = sorted(topic_rel_docs[session.topic_num].items(), key=lambda l:l[1], reverse=True)
	#print ("IDEAL NDCG@10 at final query: ",compute_NDCG(topic_rel_docs_scores, topic_rel_docs[session.topic_num], 10, list(big_corpus.keys()) ))
	print ("taking time??")
	GUS_ranker = GUS14_RUN([2.2, 1.8, 0.07, 0.4],0.65)
	there = 0
	for (docid,score) in topic_rel_docs_scores:
		if (score > 0):
			try:
				big_corpus[docid]
				there += 1
			except KeyError:
				pass
	print ("Num rel documents there: ", there) 
	#continue
	GUS_ranker.use_session_information(session)
	print ("Session length: ", len(session.interactions))
	if(len(session.interactions) == 1):
		scores1 = GUS_ranker.query_score_RUN1(preprocess(" ".join(session.interactions[-1].query),lemmatizing=True))
		scores2 = scores1
	else:
		scores1,scores2 = GUS_ranker.QCM_SAT_clicks(preprocess(" ".join(session.interactions[-1].query),lemmatizing=True))
		#scores2 = scores1
		#scores1,scores2 = GUS_ranker.QCM(preprocess(" ".join(session.interactions[-1].query),lemmatizing=True))
	ndcg = compute_NDCG(scores1, topic_rel_docs[session.topic_num], 10, list(big_corpus.keys()))
	print ("NDCG@10 at final query: ", ndcg)
	NDCG += [ndcg]
	
print ("Average NDCG: ", float(sum(NDCG))/float(len(NDCG)), float(len(NDCG)))




