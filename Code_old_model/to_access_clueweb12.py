import json
import metapy
import os
import pytoml
import requests
import unittest
jsonreq = {
'countmax': "100",
'formattype' : "1",
'indextype': 'cataparams'
}
filereq = {
'infile': open('../Clueweb12/first_100_sessions/query_subsets_100_150.txt','r')
}
url = "http://boston.lti.cs.cmu.edu/Services/clueweb12_batch/"
with open('sample_queries.txt','rb') as f:
    response = requests.post(url, files={'infile':f}, data=jsonreq, auth=('udel','blu3h3n'))
print (response)
jdata = response.text
print(jdata)    





