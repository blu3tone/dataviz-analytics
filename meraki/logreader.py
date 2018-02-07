
from __future__ import print_function

import json
import string
import operator
import time
from datetime import datetime, timedelta
from pytz import timezone,utc
import itertools
import os
from collections import Counter
import  re
re_digits= re.compile(r'(\d+)')
from APGroupGraph import APNetwork, PrintAPNClientCountDistribution


def embedded_numbers(s):
    pieces=re_digits.split(s)
    pieces[1::2]=list(map(int,pieces[1::2]))
    return pieces

def cmp_strings_with_embedded_numbers(a,b):
    ap=embedded_numbers(a)
    bp=embedded_numbers(b)
    tup=list(zip(ap,bp))
    for x,y in tup:
        if cmp(x,y): return cmp(x,y)
    return cmp(len(ap),len(bp))


#if hasattr(time, 'tzset'):
    ## Windows does not support tzset).
    #os.environ['TZ'] = os.environ['TZ']='US/Eastern'
    #time.tzset()
    #print("Processing event times in the %s time zone" % (",".join(time.tzname)))
#else:
    #print("Timing is UTC")

tz=timezone('US/Eastern')

DoW =['Mon','Tue','Wed','Thu','Fri','Sat','Sun']


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
    
def evtime2String(evtime):   
    utc_dt=datetime.utcfromtimestamp(evtime).replace(tzinfo=utc)
    # Display this time in the local timezone like this:
    dt = utc_dt.astimezone(tz)
    return dt.strftime("%a %m/%d/%y")
 
with open('data/AccessPoints.json') as APfile:    
    APInformation = json.load(APfile)
    APfile.close()
            
with open('data/prefixes.json') as PFXfile:
    ethernetPrefixDict = json.load(PFXfile)

def manufacturer(mac):
    return ethernetPrefixDict.get(mac[:8],"Unknown")
    
WAPGraph= {}       

def subsetClientGraph(clients, start, end):
    G= {}
    
    for edge in list(edgeDict.values()):
        if edge.InTimeWindow(clients,start,end):
            ap1,ap2 = edge.EdgeId
            G.setdefault(ap1,[]).append((ap2,edge))
            G.setdefault(ap2,[]).append((ap1,edge))
    return G  

def printMovementGraph():
    print('''
    The graph shows movement of clients between APs.  Graph format is x:  [y: count, ...]
    where count is the number of movements of clients between APs x and y. 
    
    In this case the count is incremented each time a mac client moves between Access Points,
    so this is indicator of the people traffic between the locations
    ''')
    
    for n1 in sorted(WAPGraph, key=lambda x: x.index):
        print("{:>2}: {}".format(n1.index,
                                 ", ".join([", ".join("{:>2}: {:>5d}".format(n2.index,e.totalCount()) 
                                                for n2,e in sorted(WAPGraph[n1], key=lambda x : x[0].index )) ])))
 
def printClientGraph():
    print('''
    The graph shows movement of clients between APs.  Graph format is x:  [y: count, ...]
    where count is the number of clients that moved between APs x and y

    In this case the count is incremented only the first time the client mac moves between 
    the pair of Access Points, in either direction. This is an indicator of how many unique visitors moved between 
    these locations over the full period of analysis.
    
    ''')
    
    for n1 in sorted(WAPGraph, key=lambda x: x.index):
        print("{:>2}:".format(n1.index), end=' ')
        print(("[%s]") % (", ".join([", ".join("{:>2}: {:>5d}".format(n2.index,e.movementClients()) 
                                                for n2,e in sorted(WAPGraph[n1], key=lambda x : x[0].index )) ]))) 
 
def printCountToClientRatioGraph(): 
    print('''
       The graph provides an indication of regular user movement of clients between APs.  
       Graph format is x:  [y: ratio, ...] where ratio is the count of clients that moved 
       between APs x and y divided by the number of unique mac addresses that made this move. 
       Bigger means the same devices made the transition
       between the pair of APs more often.
       ''') 
    
    
    for n1 in sorted(WAPGraph, key=lambda x: x.index):
        print("{:>2}:".format(n1.index), end=' ')
        print(("[%s]") % (", ".join([", ".join("{:>2}: {:>5.2f}".format(n2.index,e.CountToClientRatio()) 
                                                for n2,e in sorted(WAPGraph[n1], key=lambda x : x[0].index )) ]))) 

edgeDict = {} 
           
def edge(ap1,ap2,client,eventTime):
    edgeId = tuple(sorted([ap1, ap2],key=operator.attrgetter('index')))
    
    if edgeId in edgeDict:
        edge = edgeDict[edgeId]
    else:
        edge = _edge(edgeId)
   
    edge.event(client, eventTime)


class _edge(object):
    
    def __init__(self, EdgeId):
        # Show movement from one WAP to WAP 
        self.edgeId=EdgeId
        edgeDict[EdgeId] = self
        ap1,ap2=EdgeId
        self.points= (ap1.index, ap2.index)
        self.name="({},{})".format(self.points[0],self.points[1])
        ap1.edges.append((ap2,self))
        ap2.edges.append((ap1,self))
        self.count={}
        WAPGraph.setdefault(ap1,[]).append((ap2, self))
        WAPGraph.setdefault(ap2,[]).append((ap1, self))
        
    def __str__(self):
        return "Edge ({},{})".format(self.points[0],self.points[1])
        
    def event(self, client, eventTime):
        self.count.setdefault(client,[]).append(eventTime)
        
    def movementCount(self, clients=[], start=0,end=None):
        if not end:
            if not clients and start==0:
                return sum([len(self.count[client]) for client in self.count])
            end = time.time()
        if not clients:
            clients = list(self.count.keys())
        
        return sum((self.count[client] for client in clients 
                    if (min(itertools.chain(*list(self.count[client].values()))) < end 
                        and max(itertools.chain(*list(self.count[client].values()))) > start)                    
                   ))
 
    def movementClients(self, start=0, end=None):
        if not end:
            if start == 0:
                return len(self.count)
            end = time.time()

        return len([client for client in self.count 
                   if ((start < max(self.count[client].values))
                       and (end > min(self.count[client].values)))])
    
    def CountToClientRatio(self, start=0, end=None):
        count=self.movementCount(clients=[],start=start, end=end)
        clients=self.movementClients(start=start,end=end) 
        return count/clients
        
    def totalCount(self):
        return sum([len(self.count[client]) for client in self.count])
    
    
    def inTimeWindow(self,days=[],hours=[], start=None, end=None):
        if days==[]:
            daySet=set(range(7))
        else:
            daySet=set(days)

        if hours==[]:
            hourSet=set(range(24))
        else:
            hourSet=set(hours)    
    
        if start == None: start = 0
        if end == None: end=time.time()

        daysAndHours = [(day,hour) for day,hour in (time2DayAndHour(eventTime) 
                        for  eventTime in itertools.chain(*self.count.values()) if (start < eventTime < end))] 
        
        weekdaysActive = set([dayOfWeek(day) for day,__ in daysAndHours])
        hoursActive =  set([hour%24 for __,hour in daysAndHours])
    
        return ((hourSet.intersection(set((hour%24 for hour in hoursActive))) != set())
                and (daySet.intersection(set((dayOfWeek(day)) for day in weekdaysActive)) != set() ))
    
    
    
    
APDict = {}    

def wap(event):
    apID = event["ap_id"]
    
    if not apID in APDict:
        APDict[apID] = _wap(event)
    
    return APDict[apID]

    
    
class _wap(object):
    instances =[]  # Keep track of instances
    
    
    def __init__(self,event):
        self.ap_id = event["ap_id"]

        if not str(self.ap_id) in APInformation:
            print("Key Error - AP ID %s missing from APInformation json file" % (str(self.ap_id)))
            self.name="Unknown"
            self.location=None 
        else:
            self.name = APInformation[str(self.ap_id)]['name']
            lat,lng  = tuple(APInformation[str(self.ap_id)]['location'])
            self.location = (lat,lng)   # swap lat and long to give x=longitude and y=latitude 
        self.days = {}
        self.hours = {}
        self.edges=[]
        
     
        self.coords=[0,0]        
        # Number the APs sequentially from 0 to n-1
        # Use wap.instances to lookup the AP using its index
        
        self.index = len(_wap.instances) 
        _wap.instances.append(self)
        
    def __str__(self):
        return str(self.ap_id)
        
    @property
    def Capacity(self):
        return len(self.edges)
        
    def appendEvent(self, event, client):
        
        eventTime = float(event["time_f"])
        day,hour = time2DayAndHour(eventTime)

        self.days.setdefault(day,set()).add(client)
       
        if (event["el_type"] == "disassociation"): 
            duration = float(event["details"]["duration"])
            durationHours = max([int(float(duration)//3600),2])
            for hr in range(hour,durationHours+1):
                self.hours.setdefault(hr,set()).add(client)
        else:    
            self.hours.setdefault(hour,set()).add(client)

    def printClientCountByHour(self):

        print("\n{:>4d}: {}".format(self.index, self.name))
        print("{:>18}: {:>6s} {}".format(self.ap_id, "Hour:",
                            " ".join(("{:>3d}".format(int(hour%24)) for hour in range(24)))
                                            ))

        for day in sorted(self.days.keys()):
        
            print("{:>18s}: {:>6d}".format(day2String(day), 
                                            len(self.days.get(day,[]))), end= ' ')
            
            for hour in range(day*24, (day+1)*24):
                print ("{:>3d}".format(len(self.hours.get(hour,[]))), end= ' ')
            print()
                  
                      
        
    def printClientCountByWeekDay(self):
        weekdayTotals = [0]*7
        
        for day in (list(self.days)):
            dayofweek = time.localtime(day*24*3600+40000).tm_wday
            weekdayTotals[dayofweek] += len(self.days[day])
                                            
        for wDay in range (7):
            print(("%4s: %6d") % (DoW[wDay], weekdayTotals[wDay]))
    
    def networkEdges(self, start=0, end= None):
        if end == None:
            end = time.time()
        
        return [(nap,e) for nap,e in self.edges 
                for client in e.count 
               if (min(itertools.chain(*list(e.count[client].values()))) < end 
                   and max(itertools.chain(*list(e.count[client].values()))) > start)]
       
       
    def inTimeWindow(self,days=[],hours=[], start=None, end=None):
        if days==[]:
            daySet=set(range(7))
        else:
            daySet=set(days)

        if hours==[]:
            hourSet=set(range(24))
        else:
            hourSet=set(hours)    
    
        if start == None: start = 0
        if end == None: end = time.time()
        
        startDay, startHour = time2DayAndHour(start)
        endDay, endHour = time2DayAndHour(end)
    
        return (hourSet.intersection((hour % 24) for hour in self.hours.keys()
                                      if startHour <= hour <= endHour) 
                and daySet.intersection(dayOfWeek(day)  for day in self.days.keys()
                                      if startDay <= day <= endDay))



'''
The event log is scanned in reverse time order.  
Disassociation events are thus seen before their corresponding association event

1.  A disassociation

The expected association time is the eventtime of the disassociation 
event minus the "duration" reported in the "details" section


If not already expecting an Association, save the ExpectedAssociationTime and duration 

If already expecting an Association, based on the "duration" data in the new disassociation message
calculate a new ExpectedAssociation Time.  If the new
ExpectedAssociation time is less than the saved one, replace the ExpectedAssociationTime and increase the
"duration" so that it spans the time from the original Disassociation message to the newly referenced association time


2.  An association

If the eventtime of the association event equals or is less than 
the ExpectedAssociationTime, then the two events define the start and end 
of a connection.  Add duration to the connection time and clear the ExpectedAssociation state

else do nothing

When processing a new disassociation message:

# If there is already an expected association, we replace 
# the associated duration with the union of their durations:
# 
#  Case 1   Fix <  0  (don't accumulate Duration 2)
#
#            ------------Time Increasing ------>
#         <------------File Read direction ---------           
#
#          AS1        AS2         DIS2        DIS1
#
#           ^          ^             ^         ^
#           |          |             |         |
#           |           -Duration 2 -          |
#           |                                  |
#            -----------Duration 1 ------------
#
#  Duration 2 overlaps Duration 1 and is ignored.
#                
#  Case 2   Fix >  0  (include the non overlapping part of Duration2)
#
#         <------------File Read direction ---------           
#
#          AS2        AS1         DIS2        DIS1    
#
#           ^          ^             ^         ^
#           |          |             |         |
#            ----------+--Duration 2-          |
#                      |                       |
#                      |                       |
#                       -------Duration 1 -----
#
#           |---fix----|  
#
#  The amount to fix is determined from the difference between the 
#  times of the anticipated Association events
#
#  Case 3   Missing association event
#
#            ------------Time Increasing ------>
#         <------------File Read direction ---------           
#
#          AS2         DIS2        (AS1)      DIS1
#                                 missing  
#           ^          ^             ^         ^
#           |          |             |         |
#            Duration 2               Duration 1 
#
# A second Disassociation event is read without first reading the association for an earlier event. 
# Process the pending association event, then proceed with the next as a new event.

'''

ClientDict = {}

def client(event):
    clientMac = event["cli_mac"]
    # Add MAC address to the dictionary if we haven't seen it before
    if not clientMac in ClientDict:
        ClientDict[clientMac] = _client(event)
    
    return ClientDict[clientMac]
    
        
class _client(object):
    
    def __init__(self, event):
        self.mac = event["cli_mac"]
        self.name = event["cli_name"]
        if self.name == None:
            self.name='unknown'
  
        self.duration ={}
        self.days = set()
        self.hours = set()
        self.aps = set()
        
        # For all APs
        self.clientConnectTime = 0      # Time connected to any AP but no double counting
        self.AnyAPExpectedAssociationTime = 0
        self.AllAPAssociatedDuration = 0
        
        # Indexed by AP
        self.totalConnectTime={}
        self.ExpectedAssociationTime = {}
        self.associatedDuration = {}
        
        # Mobility
        # A transition to a new AP is counted when the client
        # disassociates from one AP and Associates with another 
        # within some time interval, say 24 hours.
        
        self.activeAssociations = set()
        self.mostRecentAssociation = None
        
        
    def logEvent(self, eventtime, assocTime, duration, mac, ap):
        date=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(assocTime))
        print("{:>20},{:>17.3f},{:>16},{:>10.2f},{:>15.2f},{:>22}".format(mac, assocTime, ap, duration, eventtime, date), file=eventfp)
        
    def appendEvent(self, event, ap):
        
        def ProcessAnyAPAssociation():
            if (self.AnyAPExpectedAssociationTime > 0
                and (eventTime - self.AnyAPExpectedAssociationTime) < 0.5):
                
                self.logEvent(eventTime, 
                              self.AnyAPExpectedAssociationTime,   
                              self.AllAPAssociatedDuration, 
                              self.mac, 
                              "0")

                self.clientConnectTime += self.AllAPAssociatedDuration
                self.AnyAPExpectedAssociationTime=0
                self.AllAPAssociatedDuration=0
                
        def ProcessAPAssociation():
            if (ap in self.ExpectedAssociationTime 
                and eventTime - self.ExpectedAssociationTime[ap] < 0.5):
                lap = ap
                et = self.ExpectedAssociationTime.pop(lap)
                duration = self.associatedDuration.pop(lap,0)

                self.logEvent(eventTime, 
                              et, 
                              duration, 
                              self.mac, 
                              ap.ap_id)    

                self.duration.setdefault(day,{})
                
                
                if not lap in self.duration[day]:
                    self.duration[day][lap]=duration
                else:
                    self.duration[day][lap]+=duration
    
                if not lap in self.totalConnectTime:
                    self.totalConnectTime[lap] = duration
                else: 
                    self.totalConnectTime[lap] += duration
                    
                # Scanning is back in time, so paradoxically we remove the ap
                # from the active associations list when we process an 
                # association event
                
                self.activeAssociations.discard(lap)
                    
            
        eventTime = event["time_f"]
        day, hour = time2DayAndHour(eventTime)
       
        #self.events.setdefault(day,[]).append(event)
        self.days.add(day)
        self.hours.add(hour)
        self.aps.add(ap)

        if (event["el_type"] == "association"): 
            ProcessAnyAPAssociation()
            ProcessAPAssociation()
            # Mark the days and hours client was connected 
            #self.days.update(set(range(day, day + int(duration//24//3600) + 1)))
            #self.hours.update(set(range(hour, hour + int(duration//3600) + 1)))
       
        elif (event["el_type"] == "disassociation"): 
            duration = float(event["details"]["duration"])  
      
            # Create an edge to all aps in the Active association list
            for nap in self.activeAssociations:
                edge(ap,nap, self, eventTime)
            else: # or an edge to the most recent ap visited if it was visited within 12 hours ago
                if self.mostRecentAssociation:
                    nap, t = self.mostRecentAssociation
                    if (ap != nap) and (t - eventTime < 12*3600):
                        edge(ap,nap, self, eventTime)
                
            
            if duration < 2*3600:
                #duration = min(duration,2*3600)
                
                # Events in the log file are read in reverse time order.  Thus
                # the disassociation event is read before the matching association event.
                #  
            
                expectedAssTime = eventTime - duration
        
                # Two types of running total:
                # 1 Disassociations and associations with any AP
                # 2 Disassociations and associations on a per AP basis
                
                
                if (self.AnyAPExpectedAssociationTime < 1 or 
                    self.AllAPAssociatedDuration < 1) :
                    fix=0
                    self.AnyAPExpectedAssociationTime = expectedAssTime
                    self.AllAPAssociatedDuration = duration
                else:
                    if eventTime < self.AnyAPExpectedAssociationTime - 1:
                        # Expected association should have happened a second ago 
                        # Assume that event got lost, and process it now
                        
                        ProcessAnyAPAssociation()
                        fix=0
                        self.AnyAPExpectedAssociationTime = expectedAssTime
                        self.AllAPAssociatedDuration = duration                        
                    else:    
                        # We have a new disassociation.  
                        
                        fix = self.AnyAPExpectedAssociationTime - expectedAssTime
                        assert fix < duration + 1, ("Fix {} can't be bigger than duration {}".format(fix, duration))
                        if fix > 0:
                            self.AnyAPExpectedAssociationTime = expectedAssTime
                            self.AllAPAssociatedDuration += fix
                        
                if not ap in self.ExpectedAssociationTime:
                    fixap=0
                    self.ExpectedAssociationTime[ap] = expectedAssTime
                    self.associatedDuration[ap] = duration
                else:
                    if eventTime < self.ExpectedAssociationTime[ap] - 1:
                        # The expected association should have already been processed
                        # So process it now, and simply queue the new disassociation
                        ProcessAPAssociation()
                        fixap=0
                        self.ExpectedAssociationTime[ap] = expectedAssTime
                        self.associatedDuration[ap] = duration                        
                    else:
                        fixap = self.ExpectedAssociationTime[ap] - expectedAssTime
                        
                        assert fixap < duration + 1,  ("Fixap {} can't be bigger than duration {}".format(fixap, duration))
                        if fixap > 30*24*3600:  #  10000000:
                            print("Month long duration %d on mac %s at time %f" %(fixap, self.mac, eventTime))
                            
                            
                        if fixap > 0:
                            self.ExpectedAssociationTime[ap] = expectedAssTime
                            self.associatedDuration[ap] += fixap 
                   
                starthour = hour - int(duration//3600)     # hours since jan 1 1970  
                startday = day - int(duration//3600//24)
    
                # Mark the days and hours client is connected 
                self.days.update(set(range(startday, day + 1)))
                self.hours.update(set(range(starthour, hour + 1)))                

        self.mostRecentAssociation=(ap,eventTime)

    def ClientRange(self):
        # The count of APs connected to by this client for each calendar day
        return [(day, len(self.duration[day])) for day in list(self.duration.keys())]


    def ClientDuration(self): 
        return self.clientConnectTime + self.AllAPAssociatedDuration    
        
    def ClientAPDuration(self): 
        return sum (self.totalConnectTime.values()) + sum(self.associatedDuration.values())

    def ClientAPDurationLeftovers(self): 
        return sum(self.associatedDuration.values())

    def ClientDays(self):
        return len(self.days)

    def ClientHours(self):
        return len(self.hours)
    
    def APNetwork(self):
        apn = APNetwork(self.aps, self)
        return apn
        

class eventLog(object):

    def __init__(self, filenames=[]): 
                 
        
        filenames= ['allEvents4.json']
        
        if filenames ==[]:
            filenames=[ 'allEvents0.json', 
                        'allEvents1.json', 
                        'allEvents2.json', 
                        'allEvents3.json', 
                        'allEvents4.json']
            
       
        global eventfp 
        eventfp = open('allEvents.csv','w')  

        self.eventsProcessed = 0
   
        for jsonfilename in filenames:
            filename = 'data/'+jsonfilename
            print("Getting event data from file %s " % (filename))

            with open(filename) as json_data:
                d = json.load(json_data)
      
            self.LoadEvents(d) 
            
        eventfp.close()
        
    
    def LoadEvents(self,d):
        for event in d:
            eventTime = event["time_f"]
            date = time.strftime("%Y-%m-%d", time.localtime(eventTime))
            
            a = wap(event)
            c = client(event)
          
            c.appendEvent(event, a)
            a.appendEvent(event, c)

            self.eventsProcessed+=1
            
        print(("%d events loaded" % (self.eventsProcessed)))
        
        
    def subGraph(self, days=[], hours=[]):
        apList = [ap for ap in WAPGraph if ap.inTimeWindow(days,hours)]
        edgeTuples = [(ap,edge) for ap,edge in itertools.chain(*WAPGraph.values()) 
                    if edge.inTimeWindow(days,hours) ]
   
        sg={}
   
        # Get all nodes that are active during the window
        
        # add all nodes that were visited, even if the client didn't go anywhere else
        for ap in apList:
            sg.setdefault(ap,[])
        
        
        # and add all nodes and edges that were visited by moving clients
        for ap,edge in edgeTuples:
            sg.setdefault(ap,[])
            
            ap1,ap2 = edge.edgeId
            
            sg.setdefault(ap1,[]).append((ap2,edge))
            sg.setdefault(ap2,[]).append((ap1,edge))
  
        return sg
    

    def printAccessPointUsage(self):   

        print(("\n\nCount of clients connected to each of the %d Access Points" % (len(APDict))))
        
        for apID in sorted(APDict.keys()):
            a = APDict[apID]
            print("AP %d: (%s)" % (apID, a.name)) 
            a.printClientCountByDay()
  
        print("\n Access by Day of Week")  
  
        for apID in sorted(APDict.keys()):
            a = APDict[apID]
            print("AP %d: (%s)" % (apID, a.name)) 
            a.printClientCountByWeekDay()  
        
    def printAccessPointUsageColumns(self):   
        print(("\n\nCount of clients connected to each of the %d Access Points" % (len(APDict))))
        minday = min(min(ap.days) for ap in _wap.instances)
        maxday = max(max(ap.days) for ap in _wap.instances)
        
        print("{:>18}{}".format("Access Point:", ''.join(("{:>5d}".format(id)) for id in range(len(_wap.instances))))) 
        
        for day in range(minday,maxday+1):
            print("{:>18}{}".format(day2String(day), ''.join(["{:>5d}".format(len(ap.days.get(day,[]))) for ap in _wap.instances])))
   
    def printAccessPointUsageByHour(self):
        
        print ("""

        Access Point connections by hour
        
        """)
        
        for ap in _wap.instances:
            ap.printClientCountByHour()
                  
        
    def printRangeByDay(self):    
        rangeCounts= {}
        for c in list(ClientDict.values()):
            for day, count in c.ClientRange():
                if not day in rangeCounts:
                    rangeCounts[day] = {}
                    
                if not count in rangeCounts[day]:
                    rangeCounts[day][count] = 0

                rangeCounts[day][count] += 1
                    
        
        Commentary = '''
        Range shows the mobility of clients. The table shows the number of clients
        that visited the number of APs given in the column header on the given date
        '''        
        
        columns = 20
        
        print(Commentary)        
        print("%16s" % ("Range "), end=' ')        
        for count in range(1,columns): 
            print("%5d" % (count), end=' ')
        print()    
        
        for day in sorted(rangeCounts.keys()):      
            print("%16s" % (day2String(day)), end=' ')        
            for count in range(1,columns): 
                print("%5d" % (rangeCounts[day].get(count,0)), end=' ')
              
            for count in sorted(set(rangeCounts[day]) - set(range(columns))):    
                print("(%d,%d)" % (count, rangeCounts[day][count]), end=' ')
            print()    

    def printClientWhenSeenCounts(self):
        
        seenDays = Counter([c.ClientDays() for c in list(ClientDict.values())])
        seenHours =  Counter([c.ClientHours() for c in list(ClientDict.values())])
            
        print ("\n\nCount of calendar days during which clients were observed")    
            
        j=0
        for i in sorted(seenDays):
            print("%4d: %4d,"  % (i, seenDays[i]), end=' ')
            j+=1
            if j>=10:
                j=0
                print()
                                  
        j=0        
        print ("\n\nCount of hour buckets during which clients were observed")    
        for i in sorted(seenHours):
            print("%4d: %4d,"  % (i, seenHours[i]), end=' ')
            j+=1
            if j>=10: 
                print()
                j=0
                
                
        print("\n\nHigh Usage Clients")
        j=0
        
        for c in sorted(ClientDict.values(), key=lambda c: c.ClientDays(), reverse=True):
            print("{:>18s} {:>12.2f} {:>4d} {:>6d} {:<33s} {}".format(c.mac, 
                                                                      c.clientConnectTime, 
                                                                      c.ClientDays(), 
                                                                      c.ClientHours(), 
                                                                      c.name, 
                                                                      manufacturer(c.mac)))
            j+=1
            
            if j >=200:
                break
            
                            
        print("\n\nHigh Hours Per Day")
        j=0
        for c in sorted((c for c in ClientDict.values() if c.ClientHours()/c.ClientDays() > 18), key=lambda c: c.ClientDays(), reverse=True):
            print("{:>18s} {:>6.2f}  {:>12.2f} {:>4d} {:>6d} {:<33s} {}".format(c.mac,
                                                    c.ClientHours()/c.ClientDays(), 
                                                    c.clientConnectTime, 
                                                    c.ClientDays(), 
                                                    c.ClientHours(), 
                                                    c.name,
                                                    manufacturer(c.mac)))
            j+=1
            
            if j >=200:
                break
                
        print("\n\nLow Hours Per Day")
        j=0
        for c in sorted((c for c in ClientDict.values() if c.ClientHours()/c.ClientDays() < 2), key=lambda c: c.ClientDays(), reverse=True):
            print("{:>18s} {:>6.2f}  {:>12.2f} {:>4d} {:>6d} {:<33s} {}".format(c.mac, 
                                                    c.ClientHours()/c.ClientDays(), 
                                                    c.clientConnectTime, 
                                                    c.ClientDays(), 
                                                    c.ClientHours(), 
                                                    c.name,
                                                    manufacturer(c.mac)))
            j+=1
            
            if j >=200:
                break
                
                                
        
                                    
       
    def clientConnectDurations(self):
       
        print("\n\n\nClient Connect time distribution in hour buckets, for 1000 most active clients(AP)", end=' ')
       
        durations = sorted([(int(c.ClientDuration()/3600),int(c.ClientAPDuration()/3600)) 
                            for c in list(ClientDict.values())], reverse=True)
        
        maxduration, maxmac = max([(int(c.ClientDuration()/3600),c.mac) 
                                   for c in list(ClientDict.values())])
            
        c = 0
        
        
        limit = min(len(durations),500)
        while (c < limit):  
            if (c % 10 == 0):  
                print()
            print("%4d(%4d)," % (durations[c]), end=' ')
                        
            c+=1
           
        print("\nMax Duration %d on mac %s" % (maxduration, maxmac))
            
        
        
    def clientConnectionTimes(self):
        
        print("\n\nConnected clients by day of week" )
        
        clientCounts = [[0 for x in range(24)] for y in range(7)]
        
        for c in list(ClientDict.values()):
            for day in c.days:
                w_day = datetime.utcfromtimestamp(day*24*3600).weekday()
                for hour in (set(range(day*24, (day+1)*24)) & c.hours):
                    hourInDay = hour % 24
                    clientCounts[w_day][hourInDay] += 1
                    
        print("\n%6s" % (" "), end=' ')
        for hour in range(24):
            print("%6d" % (hour), end=' ')
        print()
        
        for day in range(7):
            print("\n%3s" % (DoW[day]), end=' ')
            for hour in range(24):
                print("%6d" % (clientCounts[day][hour]), end=' ')
        
        print()
    
    def clientDailyConnectionTimes(self):
        print("\nNumber of client MACs connected to all Access Points by hour")
        
        clientHours =  dict(Counter(itertools.chain(*(c.hours for c in list(ClientDict.values())))))
        
        start = min(clientHours) 
        startDay=int(start//24)
        
        stop = max(clientHours)
        stopDay=int(stop//24)

        print("{:>18s} {}".format("Hour:", ''.join("{:>5d}".format(h) for h in range(24))))
       
        results = {}
        clients = []
        formatString = '{:<18s}' + "{:>5d}"*24
        
        for day in range(startDay,stopDay+1):
            date = day2String(day)
            clients=[clientHours.get(hour,0) for hour in range(day*24,(day+1)*24)]
            
            results[day]=dict(date=day, clients=clients)
            print(formatString.format(date, *clients))
            
        with open("clientdaily.json","w") as jsonFile:
            json.dump(results, jsonFile,indent=3, sort_keys=True)        
 
            
def ReadMerakiLogs():
    log = eventLog()

    return log
        
def __test():
    log=ReadMerakiLogs()
 
    
    print("\n\nAccess Points")
    for ap in _wap.instances:
        print("{:>4d}  {:>16d}   {:<30}".format(ap.index, ap.ap_id, ap.name))
    
    printMovementGraph()
    printClientGraph()
    printCountToClientRatioGraph()
    log.printAccessPointUsageColumns()
    log.printAccessPointUsageByHour()
    log.clientDailyConnectionTimes()
    log.clientConnectionTimes()
    log.clientConnectDurations()
    log.printClientWhenSeenCounts()
    log.printRangeByDay()
    
    for client in list(ClientDict.values()):
        client.APNetwork()

    PrintAPNClientCountDistribution()
    
    #  This is pathologically slow:  
 
    #  APNHierarchy()

    print("End")
    
                    
if __name__ == '__main__':
    __test()            

