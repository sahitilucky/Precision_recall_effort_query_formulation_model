__author__ = 'Sahiti and Nick Hirakawa'


from math import log


class BM25_ranker():
	def __init__(self, k1 = 1.2, b = 0.75, k3 = 500):
		self.corpus = {}
		self.stopwords = {}
		self.index = None
		self.dlt = None 
		self.avg_dl = None
		self.k1 = k1
		self.k2 = k3
		self.b = b
		
	def make_inverted_index(self, corpus_file, stopwords_file = None):
		with open(corpus_file, "r") as infile:
			idx = 0
			for line in infile:
				self.corpus[idx] = line.strip()
				idx +=1
		self.stopwords = {}
		if stopwords_file != None:
			with open(stopwords_file, "r") as infile:
				for line in infile:
					self.stopwords[line.strip()] = 1
		self.index, self.dlt, self.avg_dl = self.build_data_structures(self.corpus, self.stopwords)

	def make_inverted_index_2(self, corpus, stopwords_file = None):
		self.corpus = corpus
		self.stopwords = {}
		if stopwords_file != None:
			with open(stopwords_file, "r") as infile:
				for line in infile:
					self.stopwords[line.strip()] = 1
		self.index, self.dlt, self.avg_dl = self.build_data_structures(self.corpus, self.stopwords)

	def build_data_structures(self, corpus, stopwords):
		idx = InvertedIndex()
		dlt = dict()
		i = 0
		for docid in corpus:

			#build inverted index
			if (i%1000) ==0 :
				print (i)
			length = 0
			for word in corpus[docid].split():
				try:
					there = stopwords[word]
				except:
					idx.add(str(word), docid)
					length += 1
			i = i + 1
			#build document length table
			#length = len(corpus[str(docid)])
			dlt[docid] = length
		avg_dl = float(sum(list(dlt.values())))/float(len(dlt.values()))
		return idx, dlt, avg_dl

	def score(self, query, top_k):
		query = query.split()
		query_result = dict()
		for term in query:
			try:
				doc_dict = self.index[term] # retrieve index entry
				#print (doc_dict)
				for docid, freq in doc_dict.items(): #for each document and its word frequency
					score = self.score_BM25(n=len(doc_dict), f=freq, qf=1, N=len(self.dlt),
									   dl=self.dlt[docid], avdl=self.avg_dl) # calculate score
					#print (score)
					try: 			#this document has already been scored once
						query_result[docid] += score
					except KeyError:
						query_result[docid] = score
			except KeyError:
				pass
		#print (query_result)
		query_result = sorted(query_result.items(), key = lambda l : l[1], reverse= True)
		return query_result
	def score_BM25(self, n, f, qf, N, dl, avdl):
		k1 = self.k1
		k2 = self.k2
		b = self.b
		K = k1 * ((1-b) + b * (float(dl)/float(avdl)) )
		first = log( ( (0.5) / (0.5) ) / ( (n + 0.5) / (N - n + 0.5)) )
		second = ((k1 + 1) * f) / (K + f)
		third = ((k2+1) * qf) / (k2 + qf)
		return first * second * third



class InvertedIndex:

	def __init__(self):
		self.index = dict()

	def __contains__(self, item):
		return item in self.index

	def __getitem__(self, item):
		return self.index[item]

	def add(self, word, docid):
		try:
			self.index[word][docid] += 1
		except KeyError:
			try:
				self.index[word][docid] = 1
			except KeyError:
				d = dict()
				d[docid] = 1
				self.index[word] = d

	#frequency of word in document
	def get_document_frequency(self, word, docid):
		if word in self.index:
			if docid in self.index[word]:
				return self.index[word][docid]
			else:
				raise LookupError('%s not in document %s' % (str(word), str(docid)))
		else:
			raise LookupError('%s not in index' % str(word))

	#frequency of word in index, i.e. number of documents that contain word
	def get_index_frequency(self, word):
		if word in self.index:
			return len(self.index[word])
		else:
			raise LookupError('%s not in index' % word)



