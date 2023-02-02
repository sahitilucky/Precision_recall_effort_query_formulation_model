from create_dataset import *
from User_model_utils import *
from BM25_ranker import BM25_ranker
import random
from collections import Counter
import math
import pickle
import scipy
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

all_doc_index = pickle.load(open("../TREC_Robust_data/all_doc_index_lemmatized.pk", "rb") )
all_doc_index = all_doc_index[0]
(doc_collection_lm, doc_collection_lm_binary) = pickle.load(open("../TREC_Robust_data/all_doc_language_model.pk", "rb"))
total_num_words = sum(doc_collection_lm.values())
doc_collection_lm_dist = {term:float(doc_collection_lm[term])/float(total_num_words) for term in doc_collection_lm}
total_num_words_binary = sum(doc_collection_lm_binary.values())
doc_collection_lm_binary_dist = {term:float(doc_collection_lm_binary[term])/float(total_num_words_binary) for term in doc_collection_lm_binary}
topic_descs = read_topic_descs()
def make_topic_IN(topic):
	IN = Counter(topic.split())
	IN = {x:float(IN[x])/float(sum(IN.values())) for x in IN}
	#print (topic, IN)
	return IN

def new_word_probability(d):
    mu = 8
    return (float(mu)/float(d+mu))*(float(1)/float(d))

def make_topics(all_content):
	for docid in all_content:
		all_content[docid] = preprocess(all_content[docid], "../Session_track_2014/clueweb_snippet/lemur-stopwords.txt", lemmatizing = True)
	return

def make_word_feature_vectors(training_instances, R_q_docs):
	word_features = []
	for topic_number in training_instances:
		topic_desc = topic_descs[topic_number]
		keywords_IN = get_topic_IN_with_keywords(topic_desc)	
		keyword_binary = get_keywords_binary(topic_desc)
		topic_desc = preprocess(topic_desc, "../Session_track_2014/clueweb_snippet/lemur-stopwords.txt", lemmatizing = True)
		topic_IN = make_topic_IN(topic_desc)
#		words_set = list(set(list(training_instances[topic_number].keys()) + list(topic_IN.keys())))
		words_set = list(topic_IN.keys())
		for word in words_set:
			features = []
			try:
				features += [math.log(topic_IN[word])]
			except KeyError:
				features += [math.log(0.00000001)]				#possible if we expand topic_iIN with synonyms
			'''
			try:
				features += [math.log(keywords_IN[word])]
			except KeyError:
				features += [math.log(0.00000001)]	
			'''			
			'''
			try:
				features += [keyword_binary[word]]
			except KeyError:
				features += [0]
			'''
			try:
				features += [math.log(doc_collection_lm_dist[word])]
			except KeyError:
				features += [math.log(0.00000001)]
			try:
				features += [math.log(topic_IN[word]/doc_collection_lm_dist[word])]
			except KeyError:
				features += [math.log(0.00000001)]
			if word in training_instances[topic_number]:
				features += [training_instances[topic_number][word]["rank"]]
				features += [word,topic_number, training_instances[topic_number][word]["sessions"]]
			else:
				features += [0]
				features += [word,topic_number,[]]
			word_features += [features]
	return word_features 

def make_word_feature_vectors_better(training_instances, topic_descs):
	word_features = []
	for topic_number in training_instances:
		topic_desc = topic_descs[topic_number]
		topic_desc = preprocess(topic_desc, "../Session_track_2014/clueweb_snippet/lemur-stopwords.txt", lemmatizing = True)
		topic_IN = make_topic_IN(topic_desc)
#		words_set = list(set(list(training_instances[topic_number].keys()) + list(topic_IN.keys())))
		words_set = list(topic_IN.keys())
		for word in words_set:
			features = []
			try:
				features += [math.log(topic_IN[word])]
			except KeyError:
				features += [math.log(0.00000001)]				#possible if we expand topic_iIN with synonyms
			try:
				features += [math.log(doc_collection_lm_dist[word])]
			except KeyError:
				features += [math.log(0.00000001)] #[math.log(new_word_probability(total_num_words))] #[math.log(0.00000001)]
			try:
				features += [math.log(topic_IN[word]/(doc_collection_lm_dist[word]))]
			except KeyError:
				features += [math.log(0.00000001)] #[math.log(topic_IN[word]/new_word_probability(total_num_words))]
			if word in training_instances[topic_number]:
				features += [training_instances[topic_number][word]["rank"]]
				features += [word,topic_number, training_instances[topic_number][word]["sessions"]]
			else:
				features += [0]
				features += [word,topic_number,[]]
			word_features += [features]
	return word_features 

def make_feature_vectors_prev_intrs(topic_IN,doc_collection_lm_dist):		
	return

def make_query_feature_vectors(training_instances):
	
	return

def get_independent_word_instances(sessions):
	queries_list_act = [(list(filter(lambda l: l.type =="reformulate", session.interactions)),session.topic_num, session.session_num) for session in sessions]
	queries = []
	for interactions,topic_number,session_num in queries_list_act:
		for idx,interaction in enumerate(interactions):
			query = preprocess(" ".join(interaction.query),"../Session_track_2014/clueweb_snippet/lemur-stopwords.txt", lemmatizing = True)
			queries += [(query.split(" "), topic_number, session_num, idx+1)]
	all_words = []
	for query,topic_number,session_num,rank in queries:
		all_words += [(word, topic_number,session_num,rank) for word in query]
	topic_numbers = list(set([word[1] for word in all_words]))
	training_instances = {}

	for topic_number in topic_numbers:
		training_instances[topic_number] = {}

	for (word,topic_number,session_num,rank) in all_words:
		try:
			training_instances[topic_number][word]["rank"] += [float(1.0)/float(rank)] 
			training_instances[topic_number][word]["sessions"] += [session_num] 
		except KeyError:
			training_instances[topic_number][word] = {}
			training_instances[topic_number][word]["rank"] = [float(1.0)/float(rank)] 
			training_instances[topic_number][word]["sessions"] = [session_num] 

	for topic_number in training_instances:
		for word in training_instances[topic_number]:
			training_instances[topic_number][word]["rank"] = float(sum(training_instances[topic_number][word]["rank"]))/float(len(training_instances[topic_number][word]["rank"]))
	return training_instances
def get_independent_query_instances(sessions):
	queries_list_act = [(list(filter(lambda l: l.type =="reformulate", session.interactions)),session.topic_num, session.session_num) for session in sessions]
	queries = []
	for interactions,topic_number,session_num in queries_list_act:
		for idx,interaction in enumerate(interactions):
			query = preprocess(" ".join(interaction.query),"../Session_track_2014/clueweb_snippet/lemur-stopwords.txt", lemmatizing = True)		
			queries += [(query, topic_number, session_num, idx+1)]
	topic_numbers = list(set([query[1] for query in queries]))
	for topic_number in topic_numbers:
		training_instances[topic_number] = {}
	for (query,topic_number,session_num,rank) in all_queries:
		try:
			training_instances[topic_number][query]["rank"] += [float(1.0)/float(rank)] 
			training_instances[topic_number][query]["sessions"] += [session_num] 
		except KeyError:
			training_instances[topic_number][query] = {}
			training_instances[topic_number][query]["rank"] = [float(1.0)/float(rank)] 
			training_instances[topic_number][query]["sessions"] = [session_num] 
						
	for topic_number in training_instances:
		for query in training_instances[topic_number]:
			training_instances[topic_number][query]["rank"] = float(sum(training_instances[topic_number][query]["rank"]))/float(len(training_instances[topic_number][query]["rank"]))
			
def get_query_instances_prev_intrs():
	return

def write_features(features, filename):
	with open(filename, "w") as outfile:
		for f in features:
			for i in f[:-3]:
				outfile.write(str(i)+ " ")
			outfile.write("# ")
			outfile.write(" word " + f[-3] + " topic number " + f[-2] + " session list" + str(f[-1]))
			outfile.write("\n")

def learn_word_model_1(features, model_filename):
	X = [f[:-4] for f in features]
	Y = [f[-4] for f in features]
	comment = [f[-3:] for f in features]
	max_y = max(Y)
	Y = [math.ceil(float(y)*2/float(max_y)) for y in Y]
	#scalar = StandardScaler()
	#scalar.fit(X)
	#X = scalar.transform(X,copy=True)
	clf = LogisticRegression(random_state=0).fit(np.array(X), np.array(Y))
	pickle.dump(clf, open(model_filename, "wb"))
	return clf

def test_word_model_1(features, model_filename):
	clf = pickle.load(open(model_filename, "rb"))
	X = [f[:-4] for f in features]
	Y_o = [f[-4] for f in features]
	comment = [f[-3:] for f in features]
	max_y = max(Y_o)
	Y = [math.ceil(float(y)*2/float(max_y)) for y in Y_o]
	#scalar = StandardScaler()
	#scalar.fit(X)
	#X = scalar.transform(X,copy=True)
	y_predict = clf.predict(np.array(X))
	accuracy = float(sum([int(Y[i]==y_predict[i]) for i in range(len(Y))]))/float(len(Y))
	precision = float(sum([int(Y[i]==y_predict[i] and Y[i]!=0) for i in range(len(Y))]))/float(sum([int(y_predict[i]!=0) for i in range(len(y_predict))]))
	print ("Accuracy: {} %".format(accuracy*100))
	print ("Precision: {} %".format(precision*100))
	y_predict_proba = clf.predict_proba(np.array(X))
	word_scores = [(clf.classes_[0]*score[0]+clf.classes_[1]*score[1]+clf.classes_[2]*score[2]) for score in y_predict_proba]

#	max_y = max(Y_o)
#	Y = [float(y*2)/float(max_y) for y in Y_o]
#	y_predict_proba = clf.predict_proba(np.array(X))
	return (accuracy*100,precision*100)

def predict_word_score_1(features, model_filename):
	clf = pickle.load(open(model_filename, "rb"))
	X = [f[:-4] for f in features]
	Y_o = [f[-4] for f in features]
	comment = [f[-3:] for f in features]
	y_predict_proba = clf.predict_proba(np.array(X))
	word_scores = [float(clf.classes_[0]*score[0]+clf.classes_[1]*score[1]+clf.classes_[2]*score[2]) for score in y_predict_proba]
	return word_scores

def make_train_test_splits():
	topic_rel_docs = read_judgements()
	ideal_user_sessions,precise_user_sessions,recall_user_sessions,all_sessions = select_sessions(topic_rel_docs, read_full=True)
	print ("Num sessions: ",len(all_sessions))
	all_session_topic_nums = {}
	for session in all_sessions:
	    topic_num = session.getElementsByTagName("topic")[0].getAttribute("num")
	    try:
	        all_session_topic_nums[topic_num] += [session] 
	    except KeyError:
	        all_session_topic_nums[topic_num] = [session]
	training_sessions_1 = []
	testing_sessions_1 = []
	training_sessions_2 = []
	testing_sessions_2 = []
	training_sessions_3 = []
	testing_sessions_3 = []
	training_sessions_4 = []
	testing_sessions_4 = []
	training_sessions_5 = []
	testing_sessions_5 = []
	for topic_num in all_session_topic_nums:
		sessions = all_session_topic_nums[topic_num]
		topic_sessions = np.random.permutation(sessions)
		#five_fold_splits
		fold1 = list(topic_sessions[:int(float(len(topic_sessions))/float(5))])
		fold2 = list(topic_sessions[int(float(len(topic_sessions))/float(5)): 2*int(float(len(topic_sessions))/float(5))])
		fold3 = list(topic_sessions[2*int(float(len(topic_sessions))/float(5)): 3*int(float(len(topic_sessions))/float(5))])
		fold4 = list(topic_sessions[3*int(float(len(topic_sessions))/float(5)): 4*int(float(len(topic_sessions))/float(5))])
		fold5 = list(topic_sessions[4*int(float(len(topic_sessions))/float(5)):])
		#print (len(training_sessions_1))
		training_sessions_1 = training_sessions_1 + fold2 + fold3 + fold4 + fold5
		testing_sessions_1 = testing_sessions_1 + fold1
		training_sessions_2 += fold1 + fold3 + fold4 + fold5
		testing_sessions_2 += fold2
		training_sessions_3 += fold1 + fold2 + fold4 + fold5
		testing_sessions_3 += fold3
		training_sessions_4 += fold1 + fold2 + fold3 + fold5
		testing_sessions_4 += fold4
		training_sessions_5 += fold1 + fold2 + fold3 + fold4
		testing_sessions_5 += fold5
	training_session_nums_1 = []
	for session in training_sessions_1:
		training_session_nums_1 += [session.getAttribute("num")]
	
	testing_session_nums_1 = []
	for session in testing_sessions_1:
		testing_session_nums_1 += [session.getAttribute("num")]
	
	training_session_nums_2 = []
	for session in training_sessions_2:
		training_session_nums_2 += [session.getAttribute("num")]
	
	testing_session_nums_2 = []
	for session in testing_sessions_2:
		testing_session_nums_2 += [session.getAttribute("num")]
	
	training_session_nums_3 = []
	for session in training_sessions_3:
		training_session_nums_3 += [session.getAttribute("num")]
	
	testing_session_nums_3 = []
	for session in testing_sessions_3:
		testing_session_nums_3 += [session.getAttribute("num")]
	
	training_session_nums_4 = []
	for session in training_sessions_4:
		training_session_nums_4 += [session.getAttribute("num")]
	
	testing_session_nums_4 = []
	for session in testing_sessions_4:
		testing_session_nums_4 += [session.getAttribute("num")]
	
	training_session_nums_5 = []
	for session in training_sessions_5:
		training_session_nums_5 += [session.getAttribute("num")]
	
	testing_session_nums_5 = []
	for session in testing_sessions_5:
		testing_session_nums_5 += [session.getAttribute("num")]

	pickle.dump(training_session_nums_1, open("../supervised_models/train_test_splits/training_session_nums_1.pk", "wb"))
	pickle.dump(testing_session_nums_1, open("../supervised_models/train_test_splits/testing_session_nums_1.pk", "wb"))
	pickle.dump(training_session_nums_2, open("../supervised_models/train_test_splits/training_session_nums_2.pk", "wb"))
	pickle.dump(testing_session_nums_2, open("../supervised_models/train_test_splits/testing_session_nums_2.pk", "wb"))
	pickle.dump(training_session_nums_3, open("../supervised_models/train_test_splits/training_session_nums_3.pk", "wb"))
	pickle.dump(testing_session_nums_3, open("../supervised_models/train_test_splits/testing_session_nums_3.pk", "wb"))
	pickle.dump(training_session_nums_4, open("../supervised_models/train_test_splits/training_session_nums_4.pk", "wb"))
	pickle.dump(testing_session_nums_4, open("../supervised_models/train_test_splits/testing_session_nums_4.pk", "wb"))
	pickle.dump(training_session_nums_5, open("../supervised_models/train_test_splits/training_session_nums_5.pk", "wb"))
	pickle.dump(testing_session_nums_5, open("../supervised_models/train_test_splits/testing_session_nums_5.pk", "wb"))
	
def main():
	topic_rel_docs = read_judgements()
	ideal_user_sessions,precise_user_sessions,recall_user_sessions,all_sessions = select_sessions(topic_rel_docs, read_full=True)
	print ("Num sessions: ",len(all_sessions))
	'''
	all_sessions = np.random.permutation(all_sessions)
	training_sessions = all_sessions[:float(len(all_sessions))/float(5)]
	testing_sessions = all_sessions[float(len(all_sessions))/float(5):]
	print ("Num training_sessions: ", len(training_sessions))
	'''
	act_sessions = []
	#training_session_nums_1 = pickle.load(open("../supervised_models/train_test_splits/training_session_nums_5.pk", "rb"))
	#session_nums_dict = dict(zip(training_session_nums_1, range(len(training_session_nums_1))))
	for session in all_sessions:
		#if session.getAttribute("num") in session_nums_dict:
		act_session = Session(session)
		act_sessions += [act_session]
	print (" getting word instances")
	training_instances = get_independent_word_instances(act_sessions)
	print (" done word instances")
	robust_doc_content = read_robust_data_collection()
	clueweb_snippet_collection,clueweb_snippet_collection_2 = read_clueweb_snippet_data()
	R_q_docs = list(robust_doc_content.keys()) + list(clueweb_snippet_collection_2.keys())
	all_content = {}
	for docid in robust_doc_content:
		all_content[docid] = robust_doc_content[docid]
	for docid in clueweb_snippet_collection_2:
		all_content[docid] = clueweb_snippet_collection_2[docid]
	#make_topics(all_content)
	print (" getting word features")
	word_features = make_word_feature_vectors(training_instances, R_q_docs)
	print ("  write features")
	write_features(word_features, "../supervised_models/word_features_type_1_var_session_nums_all.txt")
	print (" done writing features")
	print (" learning word model")
	learn_word_model_1(word_features, "../supervised_models/log_reg_model_word_type_1_var_session_nums_all.pk")
	print (" done word model")
	print (" training accuracy")
	accuracy,precision = test_word_model_1(word_features, "../supervised_models/log_reg_model_word_type_1_var_session_nums_all.pk")

	'''
	act_sessions = []
	testing_session_nums_1 = pickle.load(open("../supervised_models/train_test_splits/testing_session_nums_5.pk", "rb"))
	session_nums_dict = dict(zip(testing_session_nums_1, range(len(testing_session_nums_1))))
	for session in all_sessions:
		if session.getAttribute("num") in session_nums_dict:
			act_session = Session(session)
			act_sessions += [act_session]

	print (" getting word instances")
	testing_instances = get_independent_word_instances(act_sessions)
	print (" done word instances")
	#make_topics(all_content)
	print (" getting word features")
	word_features = make_word_feature_vectors(testing_instances, R_q_docs)
	#learn_word_model_1(word_features)
	print (" testing accuracy")
	accuracy2,precision2 = test_word_model_1(word_features, "../supervised_models/log_reg_model_word_type_1_var_session_nums_5.pk")
	with open("../supervised_models/supervised_model_results.txt", 'a+') as outfile:
		outfile.write("log_reg_model_word_type_1 math.log features session_nums_5" + "\n")
		outfile.write("Training Accuracy:{} Precision:{} \n".format(accuracy, precision))
		outfile.write("Testing Accuracy:{} Precision:{} \n".format(accuracy2, precision2))
	#positive_training_instances = get_independent_query_instances(act_sessions)
	#make_query_feature_vectors(positive_training_instances, R_q_docs)
	#learn_query_model_1(query_)
	'''
if __name__ == "__main__":
    main();


