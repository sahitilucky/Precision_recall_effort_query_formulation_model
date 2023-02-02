from create_dataset import *
from User_model_utils import Session
topic_rel_docs = read_judgements()
def create_query_list_clueweb12():
	ideal_user_sessions,precise_user_sessions,recall_user_sessions,all_sessions = select_sessions(topic_rel_docs, read_full= True)
	clueweb_snippet_collection,clueweb_snippet_collection_2 = read_clueweb_snippet_data()
	act_sessions = []
	i = 0
	for session in all_sessions:
		act_session = Session(session, clueweb_snippet_collection_2)
		act_sessions += [act_session]
		if (i%100 == 0):
			print (i)
		i += 1
	all_queries = {}
	for session in act_sessions[:100]:
		queries = [interaction.query for interaction in session.interactions if interaction.type == "reformulate"]
		for query in queries:
			all_queries[" ".join(query)] = session.topic_num
	print ("Num queries: " , len(all_queries))
	queries = list(all_queries.items())
	for i in range(int(float(len(all_queries))/float(50))+1):
		q_list = [query[0] for query in queries[i*50:(i*50)+50]]
		t_list = [topic[1] for topic in queries[i*50:(i*50)+50]]
			
		with open("../Clueweb12/first_100_sessions/act_query_topic_mapping.txt", "a+") as outfile2:
			with open("../Clueweb12/first_100_sessions/query_subsets_first_100s_"+ str(i*50)+ "_"+ str(i*50+50)+ ".txt", "w") as outfile:
				outfile.write("<parameters>" + "\n")
				#outfile.write("<printSnippets>true</printSnippets>" + "\n")
				#outfile.write("<printDocuments>true</printDocuments>" + "\n")
				for idx,q in enumerate(q_list):
					outfile.write("<query>\n")	
					outfile.write("<type>indri</type>\n")
					outfile.write("<number>" + str((i*50)+idx) + "</number>\n")
					outfile.write("<text>\n")
					outfile.write("#combine(" + q + ")\n")	
					outfile.write("</text>\n")
					outfile.write("</query>\n")
					outfile2.write(str((i*50)+idx) + "\t" + q +"\n")
					outfile2.write(t_list[idx] +"\n")
				outfile.write("</parameters>\n")


def create_sim_query_list_clueweb12(pre_lemmatization_words=None):
	simulated_sessions = pickle.load(open("../simulated_sessions/simulated_session_all_topics_word_sup_1_query_unsup_session_nums_1.pk", "rb"))
	all_queries = {}
	for session in simulated_sessions:
		queries = [interaction.query for interaction in session.interactions if interaction.type == "reformulate"]
		for query in queries:
			all_queries[" ".join(query)] = session.topic_num
	print ("Num queries: " , len(all_queries))
	queries = list(all_queries.items())
	for i in range(int(float(len(all_queries))/float(50))+1):
		q_list = [query[0] for query in queries[i*50:(i*50)+50]]
		t_list = [topic[1] for topic in queries[i*50:(i*50)+50]]
			
		with open("../Clueweb12/first_100_sessions/our_uw/our_uw_sim_query_topic_mapping.txt", "a+") as outfile2:
			with open("../Clueweb12/first_100_sessions/our_uw/our_uw_session_nums_1_"+ str(i*50)+ "_"+ str(i*50+50)+ ".txt", "w") as outfile:
				outfile.write("<parameters>" + "\n")
				#outfile.write("<printSnippets>true</printSnippets>" + "\n")
				#outfile.write("<printDocuments>true</printDocuments>" + "\n")
				for idx,q in enumerate(q_list):
					'''
					modified_q = []
					for word in q.split():
						try:
							modified_q += [pre_lemmatization_words[word][0][0]]
						except KeyError:
							modified_q += [word]
					modified_q = " ".join(modified_q)
					'''
					outfile.write("<query>\n")	
					outfile.write("<type>indri</type>\n")
					outfile.write("<number>" + str((i*50)+idx) + "</number>\n")
					outfile.write("<text>\n")
					outfile.write("#combine( #uw(" + q + ") )\n")	
					outfile.write("</text>\n")
					outfile.write("</query>\n")
					outfile2.write(str((i*50)+idx) + "\t" + q +"\n")
					outfile2.write(t_list[idx] +"\n")
				outfile.write("</parameters>\n")

def adhoc_evaluation():
	queries = {}
	with open("../Clueweb12/first_100_sessions/act_query_topic_mapping.txt", "r") as infile2:
		i = 0
		for line in infile2:
			if i%2 == 0:
				queryid = line.strip().split("\t")[0] 
			if i%2 ==1:
				queries[queryid] = line.strip()
			i = i +1
	topic_rel_docs = read_judgements()
	query_predict_docs = {}
	for i in range(6):
		with open("../Clueweb12/first_100_sessions/queries_"+ str(i*50)+ "_"+ str(i*50+50)+ "_docs.txt", "r") as infile:
			for line in infile:
				queryid,_,docid,_,score,_ = line.strip().split()
				try:
					query_predict_docs[queryid] += [(docid,score)]
				except:
					query_predict_docs[queryid] = [(docid,score)]
	avg_ndcg = []
	for queryid in query_predict_docs:
		ndcg = compute_NDCG(query_predict_docs[queryid], topic_rel_docs[queries[queryid]], 10, topic_rel_docs[queries[queryid]].keys())
		avg_ndcg += [ndcg]
	print ("Adhoc Evaluation Avg NDCG@10: ", float(sum(avg_ndcg))/float(len(avg_ndcg)))

def lemmatized_to_words():
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
	for docid in robust_data_collection:
		robust_data_collection[docid] = preprocess_2(robust_data_collection[docid], lemmatizing = True)
		i += 1
		if (i%100000 == 0):
			print (i)
	pre_lemmatization_words = get_pre_lemmatization_words()
	print ("Num: ", len(pre_lemmatization_words))
	for word in pre_lemmatization_words:
		pre_lemmatization_words[word] = sorted(pre_lemmatization_words[word].items(),key=lambda l:l[1], reverse=True)
	return pre_lemmatization_words
#pre_lemmatization_words = lemmatized_to_words()
#create_sim_query_list_clueweb12()
adhoc_evaluation()
#improvement in the queries??
#adhoc evaluation - can be done using the Clueweb12 dataset using the batch query service
#Session creation using batch service - takes time but probably can be done ??
#Session creation benefits - can evaluate reformulate correctly, can evaluation session length similarities correctly
#GUSRUN, ICTNET(??) - can be evaluated over small created dataset only, but still uses the session created by original dataset
#BM25, MLE evaluation on sessions - can be done with smaller datasets and the ranks are then compared.
#Query similarity evaluation - is done, and is not effect by the things above
#session interaction evaluation - can't be done with clueweb services, with gusrun not very important may be. 
#reformulate parameters changing - analysis





