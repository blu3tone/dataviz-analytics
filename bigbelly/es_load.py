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

collectionResponseTime = []
for fname in find_ext(bigbellyFolder, "collectionResponseTime*", "jsonl"):
    with open(fname) as json_data:
        d = json.load(json_data)
        for e in d["collectionResponseTime"]:
            collectionResponseTime.append(e)
    


with open(bigbellyFolder+"collectionData.jsonl","w") as wfp:
    for v in collectionResponseTime:
        key = str(v['serial-number']%10000000)
        if v['position'] != 'center':
            key += "-{}".format(v['position'])
        bb = BigBellyData.get(key, None)
        if bb:
            v['geo'] = bb['geo']
            v['Streams'] = bb['Streams']
            v['Fullness Threshold'] = v.pop('fullness-thershold')['descriptionKey'].split('.')[-1]+'%'
            v['Latest Fullness'] = v.pop('latest-fullness')['descriptionKey'].split('.')[-1]+'%'
            v['capacity'] = bb['Capacity']
            v['Reason'] = ' '.join(map(str.capitalize, v.pop('reason').pop('descriptionKey').split('.')[-1].replace('_change','',1).split('_')))
            for o,n in [('componentType','Model'),('description','Description'),('serial-number','Serial'),('capacity','Capacity')]:
                v[n] = v.pop(o)
        else:
            raise Exception('Missing key {}'.format(key))
        print(json.dumps(v), file=wfp)
