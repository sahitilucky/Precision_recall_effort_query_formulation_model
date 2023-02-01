import pyndri
import time
#from utils import *


# Queries the index with 'hello world' and returns the first 1000 results.
def indri_ranker(query, index):
    #start_time = time.time()
    results = index.query(query, results_requested=100, include_snippets = True)
    #print ('TIME TAKEN: ', (time.time() - start_time))
    doc_scores = []
    #print (results[0])
    for (int_document_id, score, snippet) in results:
        ext_document_id, _ = index.document(int_document_id)
        doc_scores += [(ext_document_id, score, snippet)]
    return doc_scores

def indri_qry_language_session_ranker():
    return

def bm25_ranker():
    clueweb_snippet_collection,clueweb_snippet_collection_2 = read_clueweb_snippet_data()
    print ("Num docs: ", len(clueweb_snippet_collection_2))
    i = 0
    for docid in clueweb_snippet_collection_2:
        clueweb_snippet_collection_2[docid] = preprocess(clueweb_snippet_collection_2[docid], "../Session_track_2014/clueweb_snippet/lemur-stopwords.txt", lemmatizing = True) 
        i += 1
        if (i%100000 == 0):
            print (i)
    bm25_ranker = BM25_ranker(k1 = 1.2, b = 0.75, k3 = 500)
    bm25_ranker.make_inverted_index_2(clueweb_snippet_collection_2, "../Session_track_2014/clueweb_snippet/lemur-stopwords.txt")
    return bm25_ranker

if __name__ == "__main__":
    index = pyndri.Index('../../ClueWeb09_catB_sample_index')
    #indri_ranker("US tax code", index)
    '''
    results = indri_ranker("wed india", index)
    print ("wed india")
    for result in results[:10]:
        print (result[0], result[2])
    results = indri_ranker("indian wedding", index)
    print ("wedding india")
    for result in results[:10]:
        print (result[0], result[2])
    '''
    '''
    with open('../Session_track_2012/sample_index_doc_ids.txt', 'w') as outfile:
        for document_id in range(index.document_base(), index.maximum_document()):
            #outfile.write(index.document(document_id))
            #outfile.write(str(document_id) + '\n')
            outfile.write(index.document(document_id)[0] + '\n')
    '''
    index = pyndri.Index('../../ClueWeb09_catB_part_indeces/ClueWeb09_catB_index_en0001')
    results = indri_ranker("indian wedding", index)
    for document_id in range(index.document_base(), index.maximum_document()):
        print (index.document(document_id)[0])
        break
    
