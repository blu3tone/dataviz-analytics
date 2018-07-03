import json
import time
from datetime import datetime
from pytz import timezone,utc

APInformation = dict()
if APInformation == dict():
    with open('data/AccessPoints.json') as APfile:    
        APInformation = json.load(APfile)
    
tz=timezone('US/Eastern')
def evtime2DateTimeString(evtime):   
    utc_dt=datetime.utcfromtimestamp(evtime).replace(tzinfo=utc)
    # Display this time in the local timezone like this:
    dt = utc_dt.astimezone(tz)
    return dt.strftime("%a %m/%d/%y %H:%M:%S")    

def time2DayAndHour(evtime):  
    # Returns day,hour since 1 Jan 1970
    # evTime is POSIX UTC.  Returns day and hour referenced to local clock time:   
   
    # Use the datetime package.  Code the Meraki eventTime as a dt object with UTZ timezone
    utc_dt=datetime.utcfromtimestamp(evtime).replace(tzinfo=utc)
    # Convert to local timezone:
    dt = utc_dt.astimezone(tz)
    # Here is the difference in time with UTC
    offs=dt.utcoffset()
    
    # Shift the posix time by the offset and return as days and hours since
    # 00:00 on Jan 1 1970 in the local timezone
    
    hour = int(evtime + offs.seconds)//3600 + offs.days*24 
    day = hour//24
    return day,hour

def dayOfWeek(day):
    return time.localtime(day*24*3600+40000).tm_wday

def day2String(day):
    dt=datetime.utcfromtimestamp(day*24*3600).replace(tzinfo=tz).astimezone(utc) 
    return dt.strftime("%a %m/%d/%y")

class eventLog(object):

    def __init__(self, filenames=[]):

        #filenames = ['allEvents0.json']
        filenames = ['smalltest.json']
        if filenames == []:
            filenames = ['allEvents0.json',
                         'allEvents1.json',
                         'allEvents2.json',
                         'allEvents3.json',
                         'allEvents4.json',
                         'allEvents5.json',
                         'allEvents6.json',
                         'allEvents7.json',
                         'allEvents8.json',
                         'allEvents9.json',
                         'allEvents10.json',
                         'allEvents11.json',
                         'allEvents12.json',
                         'allEvents13.json',
                         'allEvents14.json'
                         ]

        for jsonfilename in filenames:

            path = "data/"+jsonfilename

            with open(path) as json_data:
                d = json.load(json_data)
                fileresultname = path + "l"
                self.writeJsonl(d, fileresultname)

    def writeJsonl(self, d, fileresultname):

        with open(fileresultname, 'w') as outfile:

            for e in d:
                if str(e["ap_id"]) in APInformation:
                    ofd = outfile
                    lat,lon = APInformation[str(e["ap_id"])]['location']
                else:
                    lat = lon = 0.0
                    ofd = sys.stderr
                e['day'], e['hour'] =  time2DayAndHour(e['time_f'])
                e['date_time'] = evtime2DateTimeString(e['time_f'])
                e['dow'] = dayOfWeek(e['day'])
                e['lat'],e['lon'] = lat,lon
                e['ap_name'] = APInformation[str(e["ap_id"])]['name']
                for k,v in e['details'].items(): e[k] = v
                del e['details']
                json.dump(e, ofd, separators=(',', ':'))
                ofd.write("\n")

if __name__ == '__main__':
    eventLog()