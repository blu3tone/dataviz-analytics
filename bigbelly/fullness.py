import json, sys
import csv
from glob import glob
from os import path
from collections import OrderedDict
from datetime import datetime
from pytz import timezone

tz=timezone('US/Eastern')
Account = OrderedDict()
BigBellyData = OrderedDict()
bigbellyFolder = "/Users/matthew/dataviz-analytics/bigbelly/data/"

def find_ext(dir, pattern, ext):
    return glob(path.join(dir,"{}.{}".format(pattern, ext)))

def skiplines(csv_data, number):
    for _ in range(number):
        next(csv_data)

def getaccount(csv_data, number):
    for _ in range(number):
        s = next(csv_data).strip('\n')
        if s:
            Account.update(dict([s.split(": ")]))

    dto_naive = datetime.strptime(Account['Generated On'], "%m/%d/%Y")
    Account['timestamp'] = tz.localize(dto_naive).strftime("%Y-%m-%d")

def loadBigBellyAsset(fname):
    with open(fname) as csv_data:
        skiplines(csv_data, 10)
        reader = csv.DictReader(csv_data)
        for row in reader:
            row['geo'] = dict(coordinates=dict(lat=row['Lat'],lon=row['Lng']))
            del row['Lng']
            del row['Lat']
            if row['Serial'] in BigBellyData:
                raise Exception('Duplicate Serial number')
            row['Gallons'] = 0
            BigBellyData[row['Serial']] = row

def addTotalVolume(fname):
    with open(fname) as csv_data:
        skiplines(csv_data, 9)
        reader = csv.DictReader(csv_data)
        for row in reader:
            key = row['Serial']
            if row['Capacity'] == 'STANDARD':
                duplicate = True
                for qual in ('-right','-left'):
                    if key+qual in BigBellyData:
                        key += qual
                        duplicate = False
                        break
                if duplicate:
                    raise Exception('Duplicate Serial number')      
            BigBellyData[key].update(dict([('Gallons', int(row['Gallons']))]))

def addCollectFrequency(fname):
    with open(fname) as csv_data:
        getaccount(csv_data, 10)
        reader = csv.DictReader(csv_data)
        for row in reader:
            key = row['Serial']
            if row['Capacity'].upper() == 'STANDARD':
                if row["Stream Type"].upper() in ("PAPER", "BOTTLES/CANS") and \
                   (key+'-right' in BigBellyData or key+'-left' in BigBellyData):
                    if key+'-left' in BigBellyData:
                        key += '-left'
                    else:
                        key += '-right'
                else:
                    raise Exception('Duplicate Serial number')
            avgDaysToFull = row['Avg. Days to Full']
            if avgDaysToFull == '-':
                    avgDaysToFull = 180
            else:
                n,u = avgDaysToFull.split(' ')
                if u == "hours":
                    avgDaysToFull = round(float(n)/24,1)
                else:
                    avgDaysToFull = float(n)    
            BigBellyData[key].update(dict([('AvgDaysToFull', avgDaysToFull),
                                           ('timestamp',Account['timestamp']),
                                           ('TimePeriod', Account['Time Period'])]))

#for fname in find_ext(bigbellyFolder, "[AHR]*CLEAN", "csv"):
    #if fname.split('/')[-1].startswith('Heatmap Report'):
        #addTotalVolume(fname)
    #elif fname.split('/')[-1].startswith('Account Assets'):
        #loadBigBellyAsset(fname)
    #elif fname.split('/')[-1].startswith('Ready to Collect'):
        #addCollectFrequency(fname)


#with open(bigbellyFolder+"data.jsonl","w") as wfp:
    #for v in BigBellyData.values():
        #print(json.dumps(v), file=wfp)

for fname in find_ext(bigbellyFolder, "[A]*CLEAN", "csv"):
    if fname.split('/')[-1].startswith('Account Assets'):
        loadBigBellyAsset(fname)

fullnesslog = []
for fname in find_ext(bigbellyFolder, "fullnesslog*", "json"):
    with open(fname) as json_data:
        d = json.load(json_data)
        for e in d:
            fullnesslog.append(e)
    

def timestamp2string(tstamp,form='%Y-%m-%d %H:%M:%S'):
    return datetime.fromtimestamp(tstamp).strftime(form)

def buildcomponent(c,u):
    c.update(u)
    return c
_tsl = []
with open(bigbellyFolder+"fullness.jsonl","w") as wfp, open(bigbellyFolder+"fullnessbycomponent.jsonl","w") as wcfp:
    for i in range(0,len(fullnesslog),2):
        _ts = fullnesslog[i]['timeStamp']/1000
        _tsl.append(_ts)
        #print(timestamp2string(_ts))
        for j in range(i+2,len(fullnesslog),2):
            if fullnesslog[j] == fullnesslog[i] and fullnesslog[j+1] == fullnesslog[i+1]:
                print(i,j,fullnesslog[i])
        f = json.loads(fullnesslog[i+1])
        for e in f:
            e.update(fullnesslog[i])
            loc = {"geo": {"coordinates": {"lat": e["latitude"], "lon": e["longitude"]}}}
            e.update(loc)
            print(json.dumps(e), file=wfp)
            keys = ['description', 'geo', 'timeStamp']
            _source = {x:e[x] for x in keys}
            _source['serialNumber'] = e['stations'][0]['serialNumber']
            for c in map(lambda c: buildcomponent(c, _source), e['stations'][0]['components']):
                key = str(c['serialNumber'])
                if c['position'].lower() != 'center':
                    key += "-{}".format(str(c['position']).lower())
                bb = BigBellyData.get(key, None)
                if bb:
                    c['streamType'] = bb['Streams']
                    c['capacity'] = bb['Capacity']
                else:
                    raise Exception('Missing key {}'.format(key))
                for o,n in [('componentType','Model'),('description','Description'),('serialNumber','Serial'),('streamType','Streams'),('capacity','Capacity')]:
                    c[n] = c.pop(o)
                if c['groups']:
                    raise Exception('Non-empty groups {}'.format(c['groups']))
                else:
                    c.pop('groups')
                #if c['labelNumber']:
                    #raise Exception('Non-empty labelNumber {}'.format(c['labelNumber']))
                #else:
                    #c.pop('labelNumber')
                #if c['alertList']:
                    #raise Exception('Non-empty alertList {}'.format(c['alertList']))
                #else:
                    #c.pop('alertList')
                print(json.dumps(c), file=wcfp)
_tsl.sort()
for _ts in _tsl:
    print(timestamp2string(_ts))