from create_dataset import *
from BM25_ranker import BM25_ranker
import pickle
import json
import time



#language_models = get_unigram_language_model(robust_data_collection)
#pickle.dump(language_models, open("../TREC_Robust_data/all_doc_language_model.pk", "wb"))

'''
bm25_ranker = BM25_ranker(k1 = 1.2, b = 0.75, k3 = 500)
bm25_ranker.make_inverted_index_2(robust_data_collection, "../Session_track_2014/clueweb_snippet/lemur-stopwords.txt")
indeces = (bm25_ranker.index.index , bm25_ranker.dlt)
pickle.dump(indeces, open("../TREC_Robust_data/all_doc_index_lemmatized.pk", "wb"))
'''
'''
stopwords = {}
with open("../Session_track_2014/clueweb_snippet/lemur-stopwords.txt", "r") as infile:
	for line in infile:
		stopwords[line.strip()] = 1

bigram_word_frequencies = {}
i = 0 
for docid in robust_data_collection:
	words = robust_data_collection[docid].split()
	bigram_sequences = [" ".join(words[i:i+2]) for i in range(len(words)-1)]
	for bigram_word in bigram_sequences:
		try:
			there = stopwords[bigram_word.split()[0]]
			there = stopwords[bigram_word.split()[1]]
		except:
			try:
				bigram_word_frequencies[bigram_word] += 1
			except:
				bigram_word_frequencies[bigram_word] = 1
	i += 1
	if (i%1000)==0:
		print (i)

print ("dumping")
pickle.dump(bigram_word_frequencies, open("../TREC_Robust_data/all_doc_bigram_language_model.pk","wb"))
'''

(doc_collection_lm, doc_collection_lm_binary) = pickle.load(open("../TREC_Robust_data/all_doc_language_model.pk", "rb"))
total_num_words = sum(doc_collection_lm.values())
doc_collection_lm_dist = {term:float(doc_collection_lm[term])/float(total_num_words) for term in doc_collection_lm}
total_num_words_binary = sum(doc_collection_lm_binary.values())
doc_collection_lm_binary_dist = {term:float(doc_collection_lm_binary[term])/float(total_num_words_binary) for term in doc_collection_lm_binary}
each_doc_lm = {}
each_doc_length = {}
for docid in robust_data_collection:
	each_doc_lm[docid] = Counter(robust_data_collection[docid].split())
	each_doc_length[docid] = len(robust_data_collection[docid].split())
	
with open('../TREC_Robust_data/each_doc_lm.json', 'w') as outfile:
    json.dump(each_doc_lm, outfile)
with open('../TREC_Robust_data/each_doc_length.json', 'w') as outfile:
	json.dump(each_doc_length, outfile)
'''
start_time = time.time()
with open('../TREC_Robust_data/each_doc_lm.json', 'r') as infile:
    each_doc_lm = json.load(infile)
print ("TIME TAKEN: ", (time.time()-start_time))
'''

#Adhoc evaluation for reformulation: - how many times increased
#Adhoc evaluation for query quality: - how good is the query, we just need real queries, use the trec robust data queries and compare ndcg? 
#If real session are available: query similarity
#Currently query quality is very low with reformulation.

#making candidates and then rescoring is good startegy than not making candidates and adding word carefully and make the query directly
#simply making one query is not better than rescoring the queries
#making candidates with only word scores and then rescoring it based on whole query score.
#REFORMULATION QUERIES: not done better investigationg at this - should vary parameters and check
#query length - some contrained query length better than random query legnth
