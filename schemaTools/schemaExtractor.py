import json
import pprint
import time
import sys

class Stat(object):
    def __init__(self):
        self.co = 0
        self.type = set()
        self.max_l = 0
        self.min_l = 10000000
        self.max_n = -999999
        self.min_n = 9999999999

    def r(self, v):
        t=type(v)
        self.type.add(t)
        self.co += 1
        if t is unicode:
            l = len(v)
            if l > self.max_l:
                self.max_l = l
            if l < self.min_l:
                self.min_l = l
        elif t in [float, int, long] :
            if v > self.max_n:
                self.max_n=v
            if v < self.min_n:
                self.min_n=v

    def __str__(self):
        return "%8s, (%5s < %2s), types:%s" % (
            self.co, 
            self.min_l if list(self.type)[0] is unicode else self.min_n, 
            self.max_l if list(self.type)[0] is unicode else self.max_n, 
            ','.join((str(t) for t in self.type)) )


def parse(fileList):
    stats = {}
    data = []
    #start = time.time()
    for filename in fileList:
        try:
            with open(filename) as f:
                if filename.endswith('.jsonl'):
                    data.append((json.loads(l) for l in f.readlines()))
                else:
                    data.append(json.loads(f.read()))
        except Exception, e:
            print 'Error with %s %s' % (filename, e)
    #print time.time() - start

    def parseInstToSchema(obj, schema, key='',level=0):
        
        oType = type(obj)
        if oType is dict:
            for k,v in obj.iteritems():
                schema[k] = parseInstToSchema(v, schema.get(k,{}), '%s.%s' % (key, k))
        elif oType is list:
            for i in xrange(len(obj)):
                index = i if type(obj[i]) not in (dict, list) else '_%s_' % level
                k = '%s->%s' % (key, index)
                schema[index] = parseInstToSchema(obj[i], schema.get(index,{}), k, level + 1)
        else:
            s = stats.setdefault(key, Stat())
            if s is not None:
                s.r(obj)
            return oType
        return schema

    mySchema = {}
    for objs in data:
        if type(objs) is not dict:
            for obj in objs:
                try:
                    parseInstToSchema(obj, mySchema, '_')
                except :
                    print 'Error with obj: %s' % obj
        elif type(objs) is dict:
            # This is named data (i.e. has a key)
            keyStats = Stat()
            namedSchema = {}
            for k, obj in objs.iteritems():
                keyStats.r(k)
                parseInstToSchema(obj, namedSchema,'__keyName__')
            # add keyStatus to the stats dict
            stats['__keyName__'] = keyStats
            mySchema['__keyName__'] = namedSchema
            
        else:
            print 'invalid data set, needs to be an array(list) or object(named list) %s...' % str(objs)[:20]

    return mySchema,stats

if __name__ == '__main__':
    
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        print 'usage: <json data file(s)>'
        sys.exit(0)

    schema, stats = parse(files)
    print '**************** Schema ******************'        
    pprint.pprint(schema)

    print '***************** Stats ******************'
    schemaKeys = sorted(stats.keys())
    maxWidth = max((len(s) for s in schemaKeys))
    for k in schemaKeys:
        s=stats[k]
        if s.co:
            print '%-*s: %s' % (maxWidth, k, str(s))

