#QCM METHOD AND SAT CLICKS
from create_dataset import *
from User_model_utils import *
import json
(doc_collection_lm, doc_collection_lm_binary) = pickle.load(open("../TREC_Robust_data/all_doc_language_model.pk", "rb"))
total_num_words = sum(doc_collection_lm.values())
doc_collection_lm_dist = {term:float(doc_collection_lm[term])/float(total_num_words) for term in doc_collection_lm}
total_num_words_binary = sum(doc_collection_lm_binary.values())
doc_collection_lm_binary_dist = {term:float(doc_collection_lm_binary[term])/float(total_num_words_binary) for term in doc_collection_lm_binary}
IDX = {}
all_doc_index = pickle.load(open("../TREC_Robust_data/all_doc_index_lemmatized.pk", "rb") )
N = len(all_doc_index[1].keys()) 
for word in doc_collection_lm_binary:
	IDX[word] = math.log(float(N)/float(doc_collection_lm_binary[word]))
big_corpus = json.load(open("../TREC_Robust_data/each_doc_lm.json", "r"))
big_doc_l = json.load(open("../TREC_Robust_data/each_doc_length.json", "r"))

class ICTNET():
	def __init__(self, QCM_params, w_param):
		self.corpus = big_corpus
		self.stopwords = {}
		self.doc_l = big_doc_l
		self.mu = 5000
		self.parameters = QCM_params
		self.prev_docid_scores = {}
		for docid in self.corpus:
			self.prev_docid_scores[docid] = []
		self.prev_results = {}
		self.prev_queries = []
		self.w = w_param
		self.session_doc_SAT_clicks = {}
	def QE(self, query):
		word_weight = {}
		self.queries += [query]
		for idx,query in self.queries[::-1]:
			for word in query.split():
				try:
					there = word_weight[word]
				except KeyError:
					word_weight[word] = math.exp(0.05*(len(self.queries)-idx))
		weighted_BM25_score = {}
		for word in word_weight:
			try:
				there = all_doc_index[word]
				for docid in all_doc_index[word]:
						
			except:

	def VDM(self, query):


	def use_session_information(self, session):
		for interaction in session.interactions[:-1]:
			self.queries += [interaction.query]
	