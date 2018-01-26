
import json
import string
import operator
import time
import itertools
import os
from collections import Counter
import  re
re_digits= re.compile(r'(\d+)')

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

os.environ['TZ']='US/Eastern'
time.tzset()
print("Processing event times in the %s time zone" % (",".join(time.tzname)))

DoW =['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

def time2DayAndHour(eventTime):  
    # Returns day,hour since 1 Jam 1970
    hour = int(eventTime - time.timezone)//3600 
    day = hour//24
    return day,hour

def day2String(day):
    return evtime2String(day*24*3600)
    
def evtime2String(evtime):   
    return time.strftime("%a %b %d %Y", time.localtime(evtime))
 
with open('data/AccessPoints.json') as APfile:    
    APInformation = json.load(APfile)
    APfile.close()
    
class NetworkGraphEdge(object):
    def __init__(self,parent,child):
        self.parent=parent
        self.child=child
        self.difference=tuple(sorted(set(parent.uid) - set(child.uid)))
        # Additional edge attributes go here

APNetworkDict= {}    #  Look up by UID, a tuple of sorted AP IDs in the network

def APNetwork(APList, client):
    uid = tuple(sorted([ap.index for ap in APList]))
    if uid in APNetworkDict:
        apn = APNetworkDict[uid]
        apn.clients.append(client)
        return apn
    else:
        return _APNetwork(uid,client)

class _APNetwork(object):
    def __init__(self, uid, client): 
        self.uid=uid
        self.clients=[]    # list of client devices that have connected with this network 
        if client: self.clients.append(client)       
        self.size = len(uid)            # Number of APs in the network
        APNetworkDict[uid] = self       # Add to the dictionary
        self.children=[]            
        self.parent=None             
        self.parentEdge=None
        
    def __str__(self):
        return str(self.uid)
        
    def ancestory(self, parent):
        self.parent=parent
        self.parentEdge = NetworkGraphEdge(parent,self)
        parent.children.append((self, self.parentEdge))
      
    def APs(self): 
        # Return a list of instances of class ap that represent APs in the network
        return sorted([wap.instances[i] for i in self.uid])
    
    def totalClients(self):
        return len(self.clients) + sum(child.totalClients() for child,edge in self.children)
 
def PrintAPNClientCountDistribution():
    """
    For each apn, list by node count the number of clients that subscribe to the network
    """
    apnsBySize={}
    
    print("\nAP subnetworks are sets of APs visited by the same clients.")
    print("""
    The count of AP networks of a given size is followed by a list of [i,j] values, 
    where i is the count of subnetworks with j clients """)
           
    """
    The potential networks are combinations of k to n APs taken from the population of n APs.
    So for example, there are n potential networks of size 1, and only one potential network of size n.
    We list only networks that have at least 1 client.  
    """
    
    for apn in sorted(APNetworkDict.values(), key=lambda x: x.size):
        apnsBySize.setdefault(apn.size,[]).append(apn)
        
    for size in apnsBySize:
        clientCounts = [len(apn.clients) for apn in apnsBySize[size]]
        counts = Counter(clientCounts)
        distribution = Counter(counts)

        print(("%5d networks with %d APs and %d clients" % (len(apnsBySize[size]),size, sum(clientCounts))), end=' ')
        
        for instances,clientCount in sorted(((instance, count) for count,instance in list(distribution.items())), reverse=True):
            print(("[%d:%d]" % (instances,clientCount)), end=' ') 
        print()


def APNHierarchy():
    
    '''
    Build a graph that shows how the AP networks are related 
    Start with all APs, then diff with Networks with fewer nodes
    
    Create a graph with the universal set at the root
    
    Child nodes are created by eliminating APs from a parent node.
    The edges list the APs that were eliminated

    '''
    candidates=sorted(([(len(apn.uid), apn) for apn in list(APNetworkDict.values())]) ,reverse=True)
    rootuid = tuple(range(len(wap.instances)))
    root=_APNetwork(rootuid,None)        

    # list of all Networks in reverse size order:
    parentSet = set([root])

    for length,apn in candidates:
        # Parent network is networks with the fewest number of additional nodes and the largest number of clients
        distance, size, parent = min(((len(set(p.uid)-set(apn.uid)),len(p.clients),p) for p in parentSet 
                                      if set(apn.uid).issubset(set(p.uid))))
        apn.ancestory(parent) 
        
        parentSet.add(apn)  # Potentially a parent of a smaller network
   
   
    # Print the Network Hierarchy
    print ('\n\nNetwork Hierarchy')
    
    ListAPNHierarchy(root)
   
def ListAPNHierarchy(node, offset=0):   
    
    print(''' 
Relationships between groupings of APs based on the number of client devices that associated with them,
The hierarchy shows the relationship between these groupings or "AP Networks". 

The display is  x (y)  [z]

x is the number of APs in the grouping
y is the number of clients that connected with all nodes in this grouping 
z is the number of clients that connected to this grouping, including connections to clients that connected to
  any subset.  i.e. the number of clients that connected to any of the nodes in the grouping 
    ''')
    
    print("%s %d (%d) [%d]" % (" "*offset, len(node.uid), len(node.clients),node.totalClients()))

    for __,__,child in sorted(((-child.totalClients(), len(edge.difference), child)  for child,edge in node.children)):
        if len(child.children):
            ListAPNHierarchy(child, offset+3)
            
edgeDict = {}
clientGraph= {}       

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
    The graph shows movement of clients between APs.  Graph format it x:  [y: count, ...]
    where count is the number of movements of clients between APs x and y. 
    
    In this case the count is incremented each time the observed client changes Access Point,
    so this is indicator of traffic between locations
    ''')
    
    for n1 in sorted(clientGraph, key=lambda x: x.index):
        print("{:>2}:".format(n1.index), end=' ')
        print(("[%s]") % (", ".join([", ".join("{:>2}: {:>5d}".format(n2.index,e.totalCount()) 
                                                for n2,e in sorted(clientGraph[n1], key=lambda x : x[0].index )) ])))
 
def printClientGraph():
    print('''
    The graph shows movement of clients between APs.  Graph format it x:  [y: count, ...]
    where count is the number of clients that moved between APs x and y

    In this case the count is incremented the first time the client mac changes Access Point. 
    so this is indicator of how many unique visitors moved between locations over the full
    period of analysis.
    
    ''')
    
    for n1 in sorted(clientGraph, key=lambda x: x.index):
        print("{:>2}:".format(n1.index), end=' ')
        print(("[%s]") % (", ".join([", ".join("{:>2}: {:>5d}".format(n2.index,e.movementClients()) 
                                                for n2,e in sorted(clientGraph[n1], key=lambda x : x[0].index )) ]))) 
 
def printUniqueClientRatioGraph(): 
    for n1 in sorted(clientGraph, key=lambda x: x.index):
        print("{:>2}:".format(n1.index), end=' ')
        print(("[%s]") % (", ".join([", ".join("{:>2}: {:>5.2f}".format(n2.index,e.uniqueClientRatio()) 
                                                for n2,e in sorted(clientGraph[n1], key=lambda x : x[0].index )) ]))) 
 
           
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
        ap1.edges.append((ap2,self))
        ap2.edges.append((ap1,self))
        self.count={}
        clientGraph.setdefault(ap1,[]).append((ap2, self))
        clientGraph.setdefault(ap2,[]).append((ap1, self))
        
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
    
    def uniqueClientRatio(self, start=0, end=None):
        return self.movementCount(clients=[],start=start, end=end)/self.movementClients(start=start,end=end) 
        
        
    def totalCount(self):
        return sum([len(self.count[client]) for client in self.count])
    
class wap(object):
    instances =[]  # Keep track of instances
    
    def __init__(self,event):
        self.ap_id = event["ap_id"]

        if not str(self.ap_id) in APInformation:
            print("Key Error - AP ID %s missing from APInformation json file" % (str(self.ap_id)))
            self.name="Unknown"
            self.location= [0,0] 
        else:
            self.name = APInformation[str(self.ap_id)]['name']
            self.location = tuple(APInformation[str(self.ap_id)]['location'])
        self.clients = {}
        self.edges=[]
        # Number the APs sequentially from 0 to n-1
        # Use wap.instances to lookup the AP using its index
        
        self.index = len(wap.instances) 
        wap.instances.append(self)
        
    def __str__(self):
        return str(self.ap_id)
        
    def appendEvent(self, event, client):
        
        eventTime = float(event["time_f"])
        day,hour = time2DayAndHour(eventTime)

        self.clients.setdefault(day,set()).add(client)

    def printClientCountByDay(self):

        for day in sorted(self.clients.keys()):
            print(("%12s: %6d") % (day2String(day), len(self.clients[day])))
        
        
    def printClientCountByWeekDay(self):
        weekdayTotals = [0]*7
        
        for day in (list(self.clients)):
            dayofweek = time.localtime(day*24*3600+40000).tm_wday
            weekdayTotals[dayofweek] += len(self.clients[day])
                                            
        for wDay in range (7):
            print(("%4s: %6d") % (DoW[wDay], weekdayTotals[wDay]))
    
    def networkEdges(self, start=0, end= None):
        if end == None:
            end = time.time()
        
        return [(nap,e) for nap,e in self.edges 
                for client in e.count 
               if (min(itertools.chain(*list(e.count[client].values()))) < end 
                   and max(itertools.chain(*list(e.count[client].values()))) > start)]
       
        
            
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
            
            
        
class client(object):
    
    def __init__(self, event):
        self.mac = event["cli_mac"]
        self.name = event["cli_name"]
        #self.events = {}
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
            if (self.AnyAPExpectedAssociationTime 
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
                
            
            if duration < 24*3600:
                
                # Events in the log file are read in reverse time order.  Thus
                # the disassociation event is read before the matching association event.
                #  
            
                expectedAssTime = eventTime - duration
        
                # Two types of running total:
                # 1 Disassociations and associations with any AP
                # 2 Disassociations and associations on a per AP basis
                
                
                if not self.AnyAPExpectedAssociationTime:
                    fix=0
                    self.AnyAPExpectedAssociationTime = expectedAssTime
                    self.AllAPAssociatedDuration = duration
                else:
                    if self.AnyAPExpectedAssociationTime - eventTime > 1:
                        # Expected association didn't happen 
                        # Process the event now
                        
                        ProcessAnyAPAssociation()
                        fix=0
                        self.AnyAPExpectedAssociationTime = expectedAssTime
                        self.AllAPAssociatedDuration = duration                        
                    else:    
                        fix = self.AnyAPExpectedAssociationTime - expectedAssTime
                        if fix > 0:
                            self.AnyAPExpectedAssociationTime = expectedAssTime
                            self.AllAPAssociatedDuration += fix
                        
                if not ap in self.ExpectedAssociationTime:
                    fixap=0
                    self.ExpectedAssociationTime[ap] = expectedAssTime
                    self.associatedDuration[ap] = duration
                else:
                    if self.ExpectedAssociationTime[ap] - eventTime > 1:
                        # The expected association should have already been processed
                        # So process it now, and simply queue the new disassociation
                        ProcessAPAssociation()
                        fixap=0
                        self.ExpectedAssociationTime[ap] = expectedAssTime
                        self.associatedDuration[ap] = duration                        
                    else:
                        fixap = self.ExpectedAssociationTime[ap] - expectedAssTime
                        if fixap > 1000000:
                            print("Large Fixap value %d on mac %s at time %f" %(fixap, self.mac, eventTime))
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
        

class eventFile(object):

    def __init__(self): 
                 
        filenames=['allEvents0.json', 
                   'allEvents1.json', 
                   'allEvents2.json', 
                   'allEvents3.json', 
                   'allEvents4.json']
   
        self.ClientDict = {}
        self.APDict = {}
        
        self.eventsProcessed = 0
    
        global eventfp 
        
        eventfp = open('allEvents.csv','w')  
   
        for jsonfilename in filenames:
            filename = 'data/'+jsonfilename
            print("Getting event data from file %s " % (filename))

            with open(filename) as json_data:
                d = json.load(json_data)
                json_data.close()
 
            self.LoadEvents(d) 
            
        eventfp.close()
        
        
    
    def LoadEvents(self,d):
        for event in d:
            eventTime = event["time_f"]
            date = time.strftime("%Y-%m-%d", time.localtime(eventTime))
            
            apID = event["ap_id"]
            # Add AP to a dictionary if we haven't seen it before
            if not apID in self.APDict:
                a = self.APDict[apID] = wap(event)
            else:
                a = self.APDict[apID]

            clientMac = event["cli_mac"]
            # Add MAC address to the dictionary if we haven't seen it before
            if not clientMac in self.ClientDict:
                c = self.ClientDict[clientMac] = client(event)
            else:    
                c = self.ClientDict[clientMac]
            
            c.appendEvent(event, a)
            a.appendEvent(event, c)

            self.eventsProcessed+=1
            
            
        print(("%d events loaded" % (self.eventsProcessed)))

    def printAccessPointUsage(self):   

        print(("\n\nCount of clients connected to each of the %d Access Points" % (len(self.APDict))))
        
        for apID in sorted(self.APDict.keys()):
            a = self.APDict[apID]
            print("AP %d: (%s)" % (apID, a.name)) 
            a.printClientCountByDay()
  
        print("\n Access by Day of Week")  
  
        for apID in sorted(self.APDict.keys()):
            a = self.APDict[apID]
            print("AP %d: (%s)" % (apID, a.name)) 
            a.printClientCountByWeekDay()  
            
        
    def printRangeByDay(self):    
        rangeCounts= {}
        for c in list(self.ClientDict.values()):
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
        
        seenDays = Counter([c.ClientDays() for c in list(self.ClientDict.values())])
        seenHours =  Counter([c.ClientHours() for c in list(self.ClientDict.values())])
            
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
                                    
       
    def clientConnectDurations(self):
       
        print("\n\n\nClient Connect time distribution in hour buckets, for 1000 most active clients(AP)", end=' ')
       
        durations = sorted([(int(c.ClientDuration()/3600),int(c.ClientAPDuration()/3600)) 
                            for c in list(self.ClientDict.values())], reverse=True)
        
        maxduration, maxmac = max([(int(c.ClientDuration()/3600),c.mac) 
                                   for c in list(self.ClientDict.values())])
            
        c = 0
        
        while (c < 500):   #  Print 1000 entries. For all, change to len(durations)):
            if (c % 10 == 0):  
                print()
            print("%4d(%4d)," % (durations[c]), end=' ')
                        
            c+=1
           
        print("\nMax Duration %d on mac %s" % (maxduration, maxmac))
            
        
        
    def clientConnectionTimes(self):
        
        clientCounts = [[0 for x in range(24)] for y in range(7)]
        
        for c in list(self.ClientDict.values()):
            for day in c.days:
                w_day = time.localtime(day*24*3600).tm_wday
                for hour in (set(range(day*24, (day+1)*24)) & c.hours):
                    hourInDay = hour % 24
                    clientCounts[w_day][hourInDay] += 1
                    
        print("\n%6s" % (" "), end=' ')
        for hour in range(24):
            print("%6d" % (hour), end=' ')
        
        for day in range(7):
            print("\n%6s" % (DoW[day]), end=' ')
            for hour in range(24):
                print("%6d" % (clientCounts[day][hour]), end=' ')
        
        print()
    
    def clientDailyConnectionTimes(self):
            
        clientHours =  dict(Counter(itertools.chain(*(c.hours for c in list(self.ClientDict.values())))))
        
        start = min(clientHours) 
        start-=start%24
        
        stop = max(clientHours)
        stop-=(stop)%24

        print("\n%18s" % (" "), end=' ')
        for hour in range(24):
            print("%4d" % (hour), end=' ')

        results = {}
        clients = []
        formatString = '{:<18}' + "{:>5}"*24
        
        for hour in range(start,stop+24):
            
            clients.append(clientHours.get(hour, 0))
            
            if (hour%24) == 23:
                date=day2String(hour//24)
                results[int(hour//24)]=dict(date=date, clients=clients)
                print(formatString.format(date, *clients))
                clients=[]
        
        with open("clientdaily.json","w") as jsonFile:
            json.dump(results, jsonFile,indent=3, sort_keys=True)        
        
def __test():
    
    log = eventFile()
    
    print("\n\nAccess Points")
    for ap in wap.instances:
        print("{:>4d}  {:>16d}   {:<30}".format(ap.index, ap.ap_id, ap.name))
    
    printMovementGraph()
    printClientGraph()
    printUniqueClientRatioGraph()
    log.clientDailyConnectionTimes()
    log.clientConnectionTimes()
    log.clientConnectDurations()
    log.printClientWhenSeenCounts()
    log.printAccessPointUsage()
    log.printRangeByDay()
    
    for client in list(log.ClientDict.values()):
        client.APNetwork()

    PrintAPNClientCountDistribution()
    
    #  This is pathologically slow:  
 
    #  APNHierarchy()

    print("End")
    
                    
if __name__ == '__main__':
    __test()            

