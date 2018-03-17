
import json
import string
import time
# "pip install elasticsearch" for python2.x
# "pip3 install elasticsearch" for python3.x
from elasticsearch import Elasticsearch
import os
import pprint
esURL = "http://76.103.116.89:9200"
esauthargs = ('elastic','blu3tone.dataviz')
es = Elasticsearch(esURL, http_auth=esauthargs, timeout=1000)
esinfo = es.info()

if 'tagline' in esinfo and esinfo['tagline'] == 'You Know, for Search':

    mapping = {k:v for k,v in es.indices.get_mapping(index='_all').iteritems() if not k.startswith('.')}
    
    with open('esMappings.json','w') as fd:
       fd.write(json.dumps(mapping))

