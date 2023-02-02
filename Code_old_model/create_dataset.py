import xml.dom.minidom 
import metapy
import string
from BM25_ranker import BM25_ranker
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from collections import Counter
import pickle
import math
import numpy as np
import json
import time
lemmatizer = WordNetLemmatizer()
lemmatized_words = {}

def getText(nodelist):
    rc = []
    for node in nodelist:
        if node.nodeType == node.TEXT_NODE:
            rc.append(node.data)
    return ''.join(rc)

def longest_common_substring(S,T):
    m = len(S)
    n = len(T)
    counter = [[0]*(n+1) for x in range(m+1)]
    longest = 0
    lcs_set = []
    lcs_indices = []
    for i in range(m):
        for j in range(n):
            if S[i] == T[j]:
                c = counter[i][j] + 1
                counter[i+1][j+1] = c
                if c > longest:
                    lcs_set = []
                    lcs_indices = []
                    longest = c
                    lcs_indices.append((i-c+1,i+1,j-c+1,j+1))
                    lcs_set.append(S[i-c+1:i+1])
                elif c == longest:
                    lcs_indices.append((i-c+1,i+1,j-c+1,j+1))
                    lcs_set.append(S[i-c+1:i+1])
    #print (lcs_set)
    #print (lcs_indices)
    return lcs_set, lcs_indices

def join_the_content(snippets):
    content = snippets[0]    
    for i in range(1,len(snippets)):
        #print ("Snippet %d: %s",i, snippets[i])
        lcs_set,lcs_indices = longest_common_substring(content, snippets[i])
        #print (lcs_set)
        #print (lcs_indices)
        if (lcs_set == []):
            #print ("Empty lcs set")
            #print ("Content:", content)
            #print ("Snippet: ", snippets[i])
            lcs_set = ""
            lcs_indices = (len(content),len(content),0,0)
        else:
            lcs_set,lcs_indices = lcs_set[0],lcs_indices[0]
        joined_text = ''
        if len(content[0:lcs_indices[0]]) > len(snippets[i][0:lcs_indices[2]]):
            joined_text = content[0:lcs_indices[0]]
        else:
            joined_text =  snippets[i][0:lcs_indices[2]]
        joined_text += lcs_set
        if len(content[lcs_indices[1]:]) > len(snippets[i][lcs_indices[3]:]):
            joined_text += content[lcs_indices[1]:]
        else:
            joined_text +=  snippets[i][lcs_indices[3]:]
        content = joined_text
        #print ("joined_content:", content)
    #print ("Final content:" , content)
    return content

def test_BM25_ranker():
    document_content = []
    with open("../Session_Track_2014/clueweb_snippet/clueweb_snippet.dat", "r") as infile:
        for line in infile:
            document_content += [line.strip()]
    idx = metapy.index.make_inverted_index('../Session_Track_2014/clueweb_snippet_data.toml')
    ranker = metapy.index.OkapiBM25(k1 = 1.2, b = 0.75, k3 = 500)
    print (idx.num_docs())
    print (idx.unique_terms())
    #print (idx.term_text(620391))
    #debugging
    top_k = 10
    query = metapy.index.Document()
    query.content('bollywood growth')
    results = ranker.score(idx, query, top_k)
    for result in results:
        print (str(0)+'\t'+ str(result[0])+ '\t' + str(result[1]) +'\n')
        print ("Document content:", document_content[result[0]-1])
        print ('Ranked Document doc id' + str(result[0]) + ' Content: ' + ' Score: ' + str(result[1]))
    return
def test_BM25_ranker_2():
    document_content = []
    with open("../Session_Track_2014/clueweb_snippet/clueweb_snippet.dat", "r") as infile:
        for line in infile:
            document_content += [line.strip()]
    bm25_ranker = BM25_ranker(k1 = 1.2, b = 0.75, k3 = 500)
    bm25_ranker.make_inverted_index("../Session_Track_2014/clueweb_snippet/clueweb_snippet.dat", "../Session_track_2014/clueweb_snippet/lemur-stopwords.txt")
    results = bm25_ranker.score("bollywood government", 10)
    for result in results:
        print (str(0)+'\t'+ str(result[0])+ '\t' + str(result[1]) +'\n')
        print ("Document content:", document_content[result[0]])
        print ('Ranked Document doc id' + str(result[0]) + ' Content: ' + ' Score: ' + str(result[1]))
    return

def create_snippet_dataset():
    session_data = xml.dom.minidom.parse('../Session_Track_2014/sessiontrack2014.xml')
    results = session_data.getElementsByTagName("result")
    document_content = {}
    for result in results:
        clueweb_id = getText(result.getElementsByTagName("clueweb12id")[0].childNodes)
        title = getText(result.getElementsByTagName("title")[0].childNodes)               
        content = getText(result.getElementsByTagName("snippet")[0].childNodes)
        content = (' '.join(content.replace("."," ").split()))
        try:
            document_content[clueweb_id]["content"] +=[content]  
        except:
            try:
                document_content[clueweb_id]["content"] = [content]
            except:            
                document_content[clueweb_id] = {}
                document_content[clueweb_id]["title"] = title
                document_content[clueweb_id]["content"] = [content]
    #print ("Num docs: ", len(document_content.keys()))
    i = 0
    for clueweb_id in document_content:
        i += 1
        if (i%500 == 0):
            print (i)
        #print (document_content[clueweb_id]["content"])
        document_content[clueweb_id]["content"] = join_the_content(document_content[clueweb_id]["content"]) 
        #break
    #making line corpus for metapy
    return document_content
def get_stopwords():
    stopwords = {}
    with open("../Session_track_2014/clueweb_snippet/lemur-stopwords.txt", "r") as infile:
        for line in infile:
            stopwords[line.strip()] = 1
    return stopwords


def preprocess(text,stopword_file = None, lemmatizing= False):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    new_text = text
    if stopword_file!=None:
        stopwords = {}
        with open(stopword_file, "r") as infile:
            for line in infile:
                stopwords[line.strip()] = 1
        new_text = ""
        for word in text.split():
            try:
                stopwords[word]
            except:
                new_text += word + " "
    if (lemmatizing):
        words = new_text.lower().split()
        #lemmatizer = WordNetLemmatizer()
        lemmas = []
        for word in words:
            #if word not in stopwords:
            try:
                lemmas.append(lemmatized_words[word])
            except:
                lemma = lemmatizer.lemmatize(word, pos='v')
                lemma1 = lemmatizer.lemmatize(word, pos='n')
                if (lemma == word) and (lemma1 != word):
                    lemma = lemma1
                lemmas.append(lemma)
                lemmatized_words[word] = lemma
                #print ("coming here", word, lemma)

        new_text = ' '.join(lemmas)
    return new_text

pre_lemmatization_words = {}
def preprocess_2(text,stopword_file = None, lemmatizing= False):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    new_text = text
    if stopword_file!=None:
        stopwords = {}
        with open(stopword_file, "r") as infile:
            for line in infile:
                stopwords[line.strip()] = 1
        new_text = ""
        for word in text.split():
            try:
                stopwords[word]
            except:
                new_text += word + " "
    if (lemmatizing):
        words = new_text.lower().split()
        #lemmatizer = WordNetLemmatizer()
        lemmas = []
        for word in words:
            #if word not in stopwords:
            try:
                lemmas.append(lemmatized_words[word])
                the_lemma = lemmatized_words[word]
                the_word = word
            except:
                lemma = lemmatizer.lemmatize(word, pos='v')
                lemma1 = lemmatizer.lemmatize(word, pos='n')
                if (lemma == word) and (lemma1 != word):
                    lemma = lemma1
                lemmas.append(lemma)
                lemmatized_words[word] = lemma
                the_lemma = lemma
                the_word = word
            try:
                pre_lemmatization_words[the_lemma][the_word] += 1
            except KeyError:
                try:
                    pre_lemmatization_words[the_lemma][the_word] = 1
                except KeyError:
                    pre_lemmatization_words[the_lemma] = {}
                    pre_lemmatization_words[the_lemma][the_word] = 1    
        new_text = ' '.join(lemmas)
    return new_text

def get_pre_lemmatization_words():
    return pre_lemmatization_words

def write_datasets(document_content):
    with open("../Session_Track_2014/clueweb_snippet/clueweb_snippet.dat", "w") as outfile1:
        with open("../Session_Track_2014/clueweb_snippet/clueweb_line_corpus_id_mapping.txt", "w") as outfile2:
            idx = 1
            for clueweb_id in document_content:
                outfile2.write(clueweb_id + "\t" + str(idx) + '\n') 
                content = document_content[clueweb_id]["title"] + " " + document_content[clueweb_id]["content"]
                content = content.translate(str.maketrans('', '', string.punctuation))
                outfile1.write(content + '\n') 
                idx += 1
    with open("../Session_Track_2014/clueweb_snippet_data.txt", "w") as outfile1:
        for clueweb_id in document_content:
            outfile1.write("# " + clueweb_id + '\n')
            outfile1.write(document_content[clueweb_id]["title"] + '\n')
            outfile1.write(document_content[clueweb_id]["content"] + '\n')

def read_robust_data_collection():
    with open("../TREC_Robust_data/Robust_data_corpus.txt", "r") as infile:
        i = 0
        robust_doc_content = {}
        for line in infile:
            if(i%2==0):
                doc_id = line.strip().split(" ")[1]
            else:
                robust_doc_content[doc_id] = " ".join(line.strip().split(" ")[17:])
            i+=1
            #if (i == 20000):
            #    break
    return robust_doc_content

def read_clueweb_snippet_data():
    document_content = {}
    document_content_2 = {}
    with open("../Session_Track_2014/clueweb_snippet_data.txt", "r") as infile:
        line = infile.readline()
        while(line.strip() != ""):
            clueweb_id = line.strip().split("# ")[1]
            document_content[clueweb_id] = {}
            line = infile.readline()
            document_content[clueweb_id]["title"]  = line.strip()
            line = infile.readline()
            document_content[clueweb_id]["content"] = line.strip()
            line = infile.readline()
            document_content_2[clueweb_id] =  document_content[clueweb_id]["title"] + " " + document_content[clueweb_id]["content"]
        return (document_content, document_content_2)

stopwords = get_stopwords()
def get_bigram_word_lm(data_collection):
    bigram_word_frequencies = {}
    i = 0 
    for docid in data_collection:
        words = data_collection[docid].split()
        bigram_sequences = [" ".join(words[i:i+2]) for i in range(len(words)-1)]
        for bigram_word in bigram_sequences:
            try:
                there = stopwords[bigram_word.split()[0]]
                there = stopwords[bigram_word.split()[1]]
            except KeyError:
                try:
                    bigram_word_frequencies[bigram_word] += 1
                except:
                    bigram_word_frequencies[bigram_word] = 1
        i += 1
        if (i%1000)==0:
            print (i)
    return bigram_word_frequencies

def get_unigram_language_model(data_collection):
    word_frequencies = {}
    word_binary_frequencies = {}
    i = 0
    for docid in data_collection:
        words = data_collection[docid].split()
        words_dict = {}
        for word in words:
            words_dict[word] = 1
            try:
                word_frequencies[word] += 1
            except:
                word_frequencies[word] = 1
        for word in words_dict:
            try:
                word_binary_frequencies[word] +=1             
            except:
                word_binary_frequencies[word] = 1
        i += 1
        if (i%1000)==0:
            print (i)
    return (word_frequencies, word_binary_frequencies)        

def compute_NDCG(predict_rel_docs, act_rel_doc_dict, cutoff, corpus_docids):
    dcg = 0
    for i in range(min(cutoff, len(predict_rel_docs))):
        doc_id = predict_rel_docs[i][0]
        if doc_id in act_rel_doc_dict:
            rel_level = act_rel_doc_dict[doc_id]
        else:
            rel_level = 0
        dcg += float(math.pow(2, rel_level) - 1)/float(np.log2(i+2))

    ideal_sorted = {}
    for docid in corpus_docids:
        try:
            ideal_sorted[docid] = act_rel_doc_dict[docid]
        except:
            ideal_sorted[docid] = 0         
    ideal_sorted = sorted(ideal_sorted.items(), key= lambda l:l[1] , reverse=True)

    idcg = 0
    for i in range(min(cutoff, len(ideal_sorted))):
        idcg += float(math.pow(2, ideal_sorted[i][1]) - 1) / float(np.log2(i+2))
    if idcg == 0:
        idcg = 1.0

    return float(dcg)/float(idcg)

def read_bigram_topic_lm(dataset):
    topic_descs = read_topic_descs(datset)
    topic_bigram_lm = {}
    for topic_num in topic_descs:
        topic_bigram_lm[topic_num] = {}
        topic_desc = preprocess(topic_descs[topic_num], "../Session_track_2014/clueweb_snippet/lemur-stopwords.txt",lemmatizing = True)
        topic_desc = topic_desc.split()
        for i in range(len(topic_desc) - 1):
            try:
                topic_bigram_lm[topic_num][topic_desc[i]+" "+topic_desc[i+1]] += 1
            except KeyError:
                topic_bigram_lm[topic_num][topic_desc[i]+" "+topic_desc[i+1]] = 1      
    return topic_bigram_lm

def read_bigram_topic_lm_trec_robust():
    topic_descs = read_trec_robust_topic_descs()
    topic_bigram_lm = {}
    for topic_num in topic_descs:
        topic_bigram_lm[topic_num] = {}
        topic_desc = preprocess(topic_descs[topic_num], "../Session_track_2014/clueweb_snippet/lemur-stopwords.txt",lemmatizing = True)
        topic_desc = topic_desc.split()
        for i in range(len(topic_desc) - 1):
            try:
                topic_bigram_lm[topic_num][topic_desc[i]+" "+topic_desc[i+1]] += 1
            except KeyError:
                topic_bigram_lm[topic_num][topic_desc[i]+" "+topic_desc[i+1]] = 1      
    return topic_bigram_lm
'''
def read_judgements():
    topic_rel_docs = {}
    with open("../Session_Track_2014/judgments.txt", "r") as infile:
        for line in infile:
            topic_num,ignore,doc_id,rel = line.strip().split()
            try:
                there = topic_rel_docs[topic_num]
            except KeyError:
                topic_rel_docs[topic_num] = {}
            if (int(rel) > 0):
                topic_rel_docs[topic_num][doc_id] = int(rel)
    return topic_rel_docs

def read_topic_descs():
    topic_descs = {}
    topics_data = xml.dom.minidom.parse('../Session_Track_2014/topictext-890.xml')
    topics = topics_data.getElementsByTagName("topic")
    for topic in topics:
        topic_num = topic.getAttribute("num")
        topic_desc = getText(topic.getElementsByTagName("desc")[0].childNodes)
        topic_descs[topic_num] = topic_desc
    return topic_descs
'''

def select_sessions(topic_rel_docs, read_full = False):
    session_data = xml.dom.minidom.parse('../Session_Track_2014/sessiontrack2014.xml')
    sessions = session_data.getElementsByTagName("session")
    ideal_user_sessions = []
    precise_user_sessions = []
    recall_user_sessions = []
    all_sessions = []
    for session in sessions:
        topic_num = session.getElementsByTagName("topic")[0].getAttribute("num")
        try:
            rel_docs = topic_rel_docs[topic_num]
        except:
            rel_docs = {}
        interactions = session.getElementsByTagName("interaction")
        result_doc_ids = []
        click_doc_ids = []
        for interaction in interactions:
            results = interaction.getElementsByTagName("result")
            try:
                clicked_items = interaction.getElementsByTagName("click")
                for click in clicked_items:
                    click_doc_ids += [getText(click.getElementsByTagName("docno")[0].childNodes)]
            except:
                pass
            for result in results:
                result_doc_ids += [getText(result.getElementsByTagName("clueweb12id")[0].childNodes)]
            
        precision = 0
        for doc_id in click_doc_ids:
            if doc_id in rel_docs:
                precision += 1
        if click_doc_ids!=[]:
            precision = float(precision)/float(len(click_doc_ids))
        else:
            pass
            #print ("no clicks")
        recall = 0
        recall_total = 0 
        for doc_id in result_doc_ids:
            if doc_id in rel_docs:
                recall_total += 1
                if doc_id in click_doc_ids:
                    recall += 1
        if (recall_total!=0):
            recall = float(recall)/float(recall_total)
        else:
            pass
            #print ("no rel docs in result")
        if (precision == 1) and recall==1:
            ideal_user_sessions += [session]
        if (precision == 1):
            #print ( click_doc_ids)
            precise_user_sessions += [session]
        if (recall == 1):
            recall_user_sessions += [session]
        all_sessions += [session]
        #print (session.getAttribute("num"), precision, recall)
        if (read_full == False):
            if (int(session.getAttribute("num")) > 101):
                break
    return ideal_user_sessions,precise_user_sessions,recall_user_sessions,all_sessions

def target_document_details(doc_collection, dataset):
    topic_descs = read_topic_descs(dataset)
    target_rel_docs = read_judgements(dataset)
    topic_rel_doc_details = {}
    all_documents = []
    for topic_num in target_rel_docs:
        docids = target_rel_docs[topic_num]
        target_doc_lm = {}
        target_doc_weighted_lm = {}
        documents = []
        documents_rels = []
        for docid in docids:
            try:
                content = doc_collection[docid]
                for word in content.split():
                    try:
                        target_doc_lm[word] += 1 
                        target_doc_weighted_lm[word] += docids[docid]*1 
                    except KeyError:
                        target_doc_lm[word] = 1 
                        target_doc_weighted_lm[word] = docids[docid]*1 
                documents += [content]
                documents_rels += [(docid,docids[docid])]
            except KeyError:
                pass
        topic_desc = preprocess(topic_descs[topic_num], "../Session_track_2014/clueweb_snippet/lemur-stopwords.txt" ,lemmatizing = True)
        documents += [topic_desc]
        all_documents += documents
        topic_lda_comm_dict = LDA_topics(documents,topic_num, 10) 
        topic_rel_doc_details[topic_num] = [target_doc_lm, target_doc_weighted_lm, documents, documents_rels, topic_lda_comm_dict]
    all_topics_lda_comm_dict = LDA_topics(all_documents, "all" , 100)
    topic_rel_doc_details["all_topics"] = [ all_documents, all_topics_lda_comm_dict]
    return topic_rel_doc_details

from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
def LDA_topics(documents, topic_num, numtopics):
    texts = [d.split() for d in documents]
    common_dictionary = Dictionary(texts)
    #print ("doing this...")
    print ('doing doc2bow')
    corpus = [common_dictionary.doc2bow(text) for text in texts]
    print ("done this...dng LDA")
    lda = LdaModel(corpus, num_topics = numtopics)
    print ("done LDA..")
    lda.save("../supervised_models/lda_models_trec_robust/trec_robust_" + str(topic_num) + "_ldamodel.model")
    print ("done this...")
    return (lda,common_dictionary)

def make_preprocess_robust_data(dataset):
    topic_descs = read_topic_descs(dataset)
    topic_rel_docs = read_judgements(dataset)
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
    robust_data_collection_wo_stopwords = {}
    for docid in robust_data_collection:
        robust_data_collection_wo_stopwords[docid] = preprocess(robust_data_collection[docid], "../Session_track_2014/clueweb_snippet/lemur-stopwords.txt", lemmatizing = True)
        robust_data_collection[docid] = preprocess(robust_data_collection[docid], lemmatizing = True)
        i += 1
        if (i%100000 == 0):
            print (i)
    with open('../TREC_Robust_data/robust_data_collection_preprocessed.json', 'w') as outfile:
        json.dump(robust_data_collection, outfile)
    with open('../TREC_Robust_data/robust_data_collection_preprocessed_stopwords.json', 'w') as outfile:
        json.dump(robust_data_collection_wo_stopwords, outfile)

def load_preprocess_robust_data():
    with open('../TREC_Robust_data/robust_data_collection_preprocessed.json', 'r') as infile:
        robust_data_collection = json.load(infile)
    with open('../TREC_Robust_data/robust_data_collection_preprocessed_stopwords.json', 'r') as infile:
        robust_data_collection_wo_stopwords = json.load(infile)
    return (robust_data_collection,robust_data_collection_wo_stopwords)

def read_trec_robust_judgements():
    document_id_to_num = {}
    document_num_to_id = {}
    topic_rel_docs = {}
    with open('../Luyu/data/document_id_mapping_gen_data.txt', 'r') as infile:
        for line in infile:
            doc_num,doc_id = line.strip().split(' ')
            document_id_to_num[doc_id] = doc_num
            document_num_to_id[doc_num] = doc_id
    with open('../Luyu/data/qrel', 'r') as infile:
        i = 0
        for line in infile:
            queryid,_,docid,rel = line.strip().split()
            try:
                there = topic_rel_docs[queryid]
            except:
                topic_rel_docs[queryid] = {}
            try:
                if (int(rel) > 0):
                    topic_rel_docs[queryid][document_id_to_num[docid]] = int(rel)
            except KeyError:
                i += 1
                pass
    print ("Num gone: ", i)
    return topic_rel_docs
def read_trec_robust_queries():
    query_details = {}
    text = ''
    with open('../Luyu/data/queriesFull.txt', 'r') as inputfile:
        for line in inputfile:
            if '<top>' in line:
                pass
            elif '<num>' in line:
                queryid = line.strip().split('<num>')[1].split('Number: ')[1]
                query_details[queryid] = []
            elif '<title>' in line:
                text = line.strip().split('<title>')[1]
                query_details[queryid] += [text]
            elif '<desc>' in line:
                text = ''
            elif '<narr>' in line:
                query_details[queryid] += [text]
                text = ''
            elif '</top>' in line:
                query_details[queryid] += [text]
                text = ''
            else:
                text += ' ' + line.strip()
    topic_descs = {}
    for queryid in query_details:
        topic_descs[queryid] = query_details[queryid][0]
    print ("NUM topics: ", len(topic_descs))
    return topic_descs

def read_trec_robust_topic_descs():
    query_details = {}
    text = ''
    with open('../Luyu/data/queriesFull.txt', 'r') as inputfile:
        for line in inputfile:
            if '<top>' in line:
                pass
            elif '<num>' in line:
                queryid = line.strip().split('<num>')[1].split('Number: ')[1]
                query_details[queryid] = []
            elif '<title>' in line:
                text = line.strip().split('<title>')[1]
                query_details[queryid] += [text]
            elif '<desc>' in line:
                text = ''
            elif '<narr>' in line:
                query_details[queryid] += [text]
                text = ''
            elif '</top>' in line:
                query_details[queryid] += [text]
                text = ''
            else:
                text += ' ' + line.strip()
    topic_descs = {}
    for queryid in query_details:
        topic_descs[queryid] = query_details[queryid][1]
    print ("NUM topics: ", len(topic_descs))
    return topic_descs
def trec_robust_target_document_details(doc_collection, dataset):
    topic_descs = read_topic_descs(dataset)
    target_rel_docs = read_judgements(dataset)
    topic_rel_doc_details = {}
    all_documents = {}
    i = 0
    for topic_num in target_rel_docs:
        i = i+1
        docids = target_rel_docs[topic_num]
        target_doc_lm = {}
        target_doc_weighted_lm = {}
        documents = []
        documents_rels = []
        for docid in docids:
            try:
                content = doc_collection[docid]
                for word in content.split():
                    try:
                        target_doc_lm[word] += 1 
                        target_doc_weighted_lm[word] += docids[docid]*1 
                    except KeyError:
                        target_doc_lm[word] = 1 
                        target_doc_weighted_lm[word] = docids[docid]*1 
                documents += [content]
                documents_rels += [(docid,docids[docid])]
            except KeyError:
                pass
        topic_desc = preprocess(topic_descs[topic_num], "../Session_track_2014/clueweb_snippet/lemur-stopwords.txt" ,lemmatizing = True)        
        documents += [topic_desc]
        for idx,(docid,rel) in enumerate(documents_rels):
           all_documents[docid] = documents[idx]
        all_documents["topic_desc_" + str(topic_num)] = documents[-1]
        topic_rel_doc_details[topic_num] = [target_doc_lm, target_doc_weighted_lm, documents, documents_rels]
        print ("TOPIC NUM {} done".format(i))
    return topic_rel_doc_details,all_documents


def make_trec_robust_LDA_model():
    with open('../TREC_Robust_data/robust_data_collection_preprocessed_stopwords.json', 'r') as infile:
        robust_data_collection_wo_stopwords = json.load(infile)
    topic_rel_doc_details,all_documents_subset = trec_robust_target_document_details(robust_data_collection_wo_stopwords)
    all_documents = []
    for docid in robust_data_collection_wo_stopwords:
        all_documents += [robust_data_collection_wo_stopwords[docid]]
    start_time = time.time()
    print ('lda topics')
    all_topics_lda_comm_dict = LDA_topics(all_documents_subset, "all" , 100)
    print ("TIME TAKEN: ",  time.time()-start_time)
    topic_rel_doc_details["all_topics"] = [[],all_topics_lda_comm_dict]

    pickle.dump(topic_rel_doc_details,open("../TREC_Robust_data/trec_robust_topic_rel_doc_details.pk","wb"))

def put_vector_dict(vector_dict, idx):
    try:
        vector_dict[idx]
        return vector_dict[idx]
    except KeyError:
        return 0


def language_model_m(self, topic):
    IN = Counter(topic.split())
    IN = {x:float(IN[x])/float(sum(IN.values())) for x in IN}
    print (topic, IN)
    return IN


def read_topic_descs(dataset):
    if (dataset == "Session_Track_2012") or (dataset == "Session_Track_2013"):
        topic_descs = {}
        topics_data = xml.dom.minidom.parse('../' + dataset + '/topics.xml')
        topics = topics_data.getElementsByTagName("topic")
        for topic in topics:
            topic_num = topic.getAttribute("num")
            topic_desc = getText(topic.getElementsByTagName("desc")[0].childNodes)
            topic_descs[topic_num] = topic_desc
        return topic_descs
    elif (dataset == "Session_Track_2014"):
        topic_descs = {}
        topics_data = xml.dom.minidom.parse('../' + dataset + '/topictext-890.xml')
        topics = topics_data.getElementsByTagName("topic")
        for topic in topics:
            topic_num = topic.getAttribute("num")
            topic_desc = getText(topic.getElementsByTagName("desc")[0].childNodes)
            topic_descs[topic_num] = topic_desc
        return topic_descs

def read_judgements(dataset):
    if dataset == "Session_Track_2012" or dataset == "Session_Track_2013":
        filename = "../" + dataset + "qrels.txt"
    elif dataset == "Session_Track_2014":
        filename = "../" + dataset + "judgments.txt"
    topic_rel_docs = {}
    with open(filename, "r") as infile:
        for line in infile:
            topic_num,ignore,doc_id,rel = line.strip().split()
            try:
                there = topic_rel_docs[topic_num]
            except KeyError:
                topic_rel_docs[topic_num] = {}
            if (int(rel) > 0):
                topic_rel_docs[topic_num][doc_id] = int(rel)
    return topic_rel_docs


def main():
    #document_content = create_snippet_dataset()
    #write_datasets(document_content)    
    #test_BM25_ranker_2()
    #read_bigram_topic_lm()
    '''
    print("started reading")
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
    print ("Num docs: ", len(robust_data_collection))
    i = 0
    for docid in robust_data_collection:
        robust_data_collection[docid] = preprocess(robust_data_collection[docid],"../Session_track_2014/clueweb_snippet/lemur-stopwords.txt" ,lemmatizing = True)
        i += 1
        if (i%100000 == 0):
            print (i)

    topic_rel_doc_details = target_document_details(robust_data_collection)
    pickle.dump(topic_rel_doc_details,open("../TREC_Robust_data/topic_rel_doc_details.pk","wb"))
    #lda_model = LdaModel.load("../supervised_models/ldamodel.model")
    '''
    #make_preprocess_robust_data()
    #with open('../TREC_Robust_data/robust_data_collection_preprocessed_stopwords.json', 'r') as infile:
    #    robust_data_collection_wo_stopwords = json.load(infile)
    print("started reading")
    clueweb_snippet_collection,clueweb_snippet_collection_2 = read_clueweb_snippet_data()
    #print ("started reading done")
    #print ("Num docs: ", len(robust_data_collection))
    print ("Num docs: ", len(clueweb_snippet_collection_2))
    i = 0
    for docid in clueweb_snippet_collection_2:
        clueweb_snippet_collection_2[docid] = preprocess(clueweb_snippet_collection_2[docid],"../Session_track_2014/clueweb_snippet/lemur-stopwords.txt" ,lemmatizing = True)
        i += 1
        if (i%100000 == 0):
            print (i)

    topic_rel_doc_details,all_documents_subset = trec_robust_target_document_details(clueweb_snippet_collection_2)
    print ('lda topics')
    topic_rel_doc_details = pickle.load(open("../TREC_Robust_data/topic_rel_doc_details.pk","rb"))
    (lda,comm_doct) = topic_rel_doc_details["all_topics"][1][0],topic_rel_doc_details["all_topics"][1][1]
    texts = []
    text_ids = []
    for d in all_documents_subset:
        texts += [all_documents_subset[d].split()]
        text_ids += [d]
    print ('doing doc2bow')
    corpus = [comm_doct.doc2bow(text) for text in texts]
    print ("done doc2bow")
    document_vectors = lda[corpus]
    print (document_vectors[0])
    target_document_vectors = {}
    print (len(document_vectors))
    j = 0 
    list_100 = range(100)
    for idx, vect in enumerate(document_vectors):
        vector_dict = dict(vect)
        vector = [0]*100
        #vector = [put_vector_dict(vector_dict, i) for i in list_100]
        for v in vector_dict:
            vector[v] = vector_dict[v]
        target_document_vectors[text_ids[idx]] = vector
        j += 1
        if (j%100 == 0):
            print (j) 
    pickle.dump(target_document_vectors, open("../TREC_Robust_data/target_doc_topic_vectors.pk", 'wb'))

if __name__ == "__main__":
    main();
