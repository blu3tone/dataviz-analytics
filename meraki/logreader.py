
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
    pieces[1::2]=map(int,pieces[1::2])
    return pieces

def cmp_strings_with_embedded_numbers(a,b):
    ap=embedded_numbers(a)
    bp=embedded_numbers(b)
    tup=zip(ap,bp)
    for x,y in tup:
        if cmp(x,y): return cmp(x,y)
    return cmp(len(ap),len(bp))

os.environ['TZ']='US/Eastern'
time.tzset()
print "Processing event times in the %s time zone" % (",".join(time.tzname))

DoW =['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

def time2DayAndHour(eventTime):  
    # Returns day,hour since 1 Jam 1970
    hour = int(eventTime - time.timezone)//3600 
    day = hour//24
    return day,hour

def day2String(day):
    return time.strftime("%a %b %d %Y", time.localtime(day*24*3600))
 
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
apList = []  # List of ap instances indexed by ap_id
    
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
        return sorted([apList[i] for i in self.uid])
    
    def totalClients(self):
        return len(self.clients) + sum(child.totalClients() for child,edge in self.children)
 
def PrintAPNClientCountDistribution():
    """
    For each apn, list by node count the number of clients that subscribe to the network
    """
    apnsBySize={}
    
    print "\nAP subnetworks are sets of APs visited by the same clients."
    print "The count of AP networks of a given size is followed by a list of [i,j] values, where i is the count of subnetworks with j clients "
    """
    
    The potential networks are combinations of k to n APs taken from the population of n APs.
    So for example, there are n potential networks of size 1, and only one potential network of size n.
    We list only networks that have at least 1 client.  
       
    """
    
    for size, apn in sorted([(n.size, n) for n in APNetworkDict.values()]):
        apnsBySize.setdefault(size,[]).append(apn)
        
    for size in apnsBySize:
        clientCounts = [len(apn.clients) for apn in apnsBySize[size]]
        counts = Counter(clientCounts)
        distribution = Counter(counts)

        print ("%5d networks with %d APs and %d clients" % (len(apnsBySize[size]),size, sum(clientCounts))),
        
        for instances,clientCount in sorted(((instance, count) for count,instance in distribution.items()), reverse=True):
            print ("[%d:%d]" % (instances,clientCount)), 
        print




def APNHierarchy():
    
    '''
    Build a graph that shows how the AP networks are related 
    Start with all APs, then diff with Networks with fewer nodes
    
    Create a graph with the universal set at the root
    
    Child nodes are created by eliminating APs from a parent node.
    The edges list the APs that were eliminated

    '''
    candidates=sorted(([(len(apn.uid), apn) for apn in APNetworkDict.values()]) ,reverse=True)
    rootuid = tuple(range(len(apList)))
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
    
    print ''' 
Relationships between groupings of APs based on the number of client devices that associated with them,
The hierarchy shows the relationship between these groupings or "AP Networks". 

The display is  x (y)  [z]

x is the number of APs in the grouping
y is the number of clients that connected with all nodes in this grouping 
z is the number of clients that connected to this grouping, including connections to clients that connected to
  any subset.  i.e. the number of clients that connected to any of the nodes in the grouping 
    '''
    
    print "%s %d (%d) [%d]" % (" "*offset, len(node.uid), len(node.clients),node.totalClients())

    for __,__,child in sorted(((-child.totalClients(), len(edge.difference), child)  for child,edge in node.children)):
        if len(child.children):
            ListAPNHierarchy(child, offset+3)
    
class ap(object):
    def __init__(self,event):
        self.ap_id = event["ap_id"]

        if not str(self.ap_id) in APInformation:
            print "Key Error - AP ID %s missing from APInformation json file" % (str(self.ap_id))
            self.name="Unknown"
            self.location= [0,0] 
        else:
            self.name = APInformation[str(self.ap_id)]['name']
            self.location = tuple(APInformation[str(self.ap_id)]['location'])
        self.clients = {}
        
        # Number the APs sequentially from 0 to n-1
        # Use apList to lookup the AP using its index
        
        apList.append(self)
        self.index = len(apList) - 1
        
    def appendEvent(self, event, client):
        
        eventTime = float(event["time_f"])
        day,hour = time2DayAndHour(eventTime)

        self.clients.setdefault(day,set()).add(client)

    def printClientCountByDay(self):

        for day in sorted(self.clients.keys()):
            print ("%12s: %6d") % (day2String(day), len(self.clients[day]))
        
        
    def printClientCountByWeekDay(self):
        weekdayTotals = [0]*7
        
        for day in (self.clients.keys()):
            dayofweek = time.localtime(day*24*3600+40000).tm_wday
            weekdayTotals[dayofweek] += len(self.clients[day])
                                            
        for wDay in range (7):
            print ("%4s: %6d") % (DoW[wDay], weekdayTotals[wDay])
            
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
        self.events = {}
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
        
        
    def appendEvent(self, event, ap):
        
        def ProcessAnyAPAssociation():
            if (self.AnyAPExpectedAssociationTime 
                and eventTime - self.AnyAPExpectedAssociationTime) < 0.2:
                self.clientConnectTime += self.AllAPAssociatedDuration
                self.AnyAPExpectedAssociationTime=0
                self.AllAPAssociatedDuration=0
                
        def ProcessAPAssociation():
            if (ap in self.ExpectedAssociationTime 
                and eventTime - self.ExpectedAssociationTime[ap] < 0.5):
                lap = ap
                et = self.ExpectedAssociationTime.pop(lap)
                duration = self.associatedDuration.pop(lap,0)
    
                self.duration.setdefault(day,{})
                if not lap in self.duration[day]:
                    self.duration[day][lap]=duration
                else:
                    self.duration[day][lap]+=duration
    
                if not lap in self.totalConnectTime:
                    self.totalConnectTime[lap] = duration
                else: 
                    self.totalConnectTime[lap] += duration
            
        eventTime = event["time_f"]
        day, hour = time2DayAndHour(eventTime)
       
        self.events.setdefault(day,[]).append(event)
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
            
            if duration < 12*3600:
                
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
                            print "Large Fixap value %d on mac %s at time %f" %(fixap, self.mac, eventTime)
                        if fixap > 0:
                            self.ExpectedAssociationTime[ap] = expectedAssTime
                            self.associatedDuration[ap] += fixap 
                   
                starthour = hour - int(duration//3600)     # hours since jan 1 1970  
                startday = day - int(duration//3600//24)
    
                # Mark the days and hours client is connected 
                self.days.update(set(range(startday, day + 1)))
                self.hours.update(set(range(starthour, hour + 1)))                

    def ClientRange(self):
        # The count of APs connected to by this client for each calendar day
        return [(day, len(self.duration[day])) for day in self.duration.keys()]


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
                 
        filenames=['allEvents4.json', 
                   'allEvents3.json', 
                   'allEvents2.json', 
                   'allEvents1.json', 
                   'allEvents0.json']
   
        self.ClientDict = {}
        self.APDict = {}
        
        self.eventsProcessed = 0
   
        for jsonfilename in filenames:
            filename = 'data/'+jsonfilename
            print "Getting event data from file %s " % (filename)

            with open(filename) as json_data:
                d = json.load(json_data)
                json_data.close()
 
            self.LoadEvents(d) 
            
        self.clientDailyConnectionTimes()
        self.clientConnectionTimes()
        self.clientConnectDurations()
        self.printClientWhenSeenCounts()
        self.printAccessPointUsage()
        self.printRangeByDay()
        
        for client in self.ClientDict.values():
            client.APNetwork()

        PrintAPNClientCountDistribution()
        APNHierarchy()
                        
       
    def LoadEvents(self,d):
        
        with open("events.csv","w") as csvFile:
        
            for event in d:
                eventTime = event["time_f"]
                date = time.strftime("%Y-%m-%d", time.localtime(eventTime))
                
                apID = event["ap_id"]
                # Add AP to a dictionary if we haven't seen it before
                if not apID in self.APDict:
                    a = self.APDict[apID] = ap(event)
                else:
                    a = self.APDict[apID]

                clientMac = event["cli_mac"]
                # Add MAC address to the dictionary if we haven't seen it before
                if not clientMac in self.ClientDict:
                    c = self.ClientDict[clientMac] = client(event)
                else:    
                    c = self.ClientDict[clientMac]

                if (event["el_type"] == "disassociation"): 
                    etype = "D"
                    duration = float(event["details"]["duration"]) 
                else:
                    etype = "A"
                    duration = 0
                
                print >> csvFile, "%20s, %s, %16s, %s, %s, %f" % (clientMac, date, eventTime, apID, etype, duration)
                
                c.appendEvent(event, a)
                a.appendEvent(event, c)
          
                self.eventsProcessed+=1
                
            csvFile.close()
            
            
        print ("%d events loaded" % (self.eventsProcessed))

    def printAccessPointUsage(self):   

        print ("\n\nCount of clients connected to each of the %d Access Points" % (len(self.APDict)))
        
        for apID in sorted(self.APDict.keys()):
            a = self.APDict[apID]
            print "AP %d: (%s)" % (apID, a.name) 
            a.printClientCountByDay()
  
        print("\n Access by Day of Week")  
  
        for apID in sorted(self.APDict.keys()):
            a = self.APDict[apID]
            print "AP %d: (%s)" % (apID, a.name) 
            a.printClientCountByWeekDay()  
            
        
    def printRangeByDay(self):    
        rangeCounts= {}
        for c in self.ClientDict.values():
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
        
        print Commentary        
        print "%16s" % ("Range "),        
        for count in range(1,columns): 
            print "%5d" % (count),
        print    
        
        for day in sorted(rangeCounts.keys()):      
            print "%16s" % (day2String(day)),        
            for count in range(1,columns): 
                print "%5d" % (rangeCounts[day].get(count,0)),
              
            for count in sorted(set(rangeCounts[day]) - set(range(columns))):    
                print "(%d,%d)" % (count, rangeCounts[day][count]),
            print    

    def printClientWhenSeenCounts(self):
        
        seenDays = Counter([c.ClientDays() for c in self.ClientDict.values()])
        seenHours =  Counter([c.ClientHours() for c in self.ClientDict.values()])
            
        print ("\n\nCount of calendar days during which clients were observed")    
            
        j=0
        for i in sorted(seenDays):
            print "%4d: %4d,"  % (i, seenDays[i]),
            j+=1
            if j>=10:
                j=0
                print
                                  
        j=0        
        print ("\n\nCount of hour buckets during which clients were observed")    
        for i in sorted(seenHours):
            print "%4d: %4d,"  % (i, seenHours[i]),
            j+=1
            if j>=10: 
                print
                j=0
                                    
       
    def clientConnectDurations(self):
       
        print "\n\n\nClient Connect time distribution in hour buckets, for 1000 most active clients(AP)",
       
        durations = sorted([(int(c.ClientDuration()/3600),int(c.ClientAPDuration()/3600)) 
                            for c in self.ClientDict.values()], reverse=True)
        
        maxduration, maxmac = max([(int(c.ClientDuration()/3600),c.mac) 
                                   for c in self.ClientDict.values()])
        
        c = 0
        
        while (c < 1000):   #  Print 1000 entries. For all, change to len(durations)):
            if (c % 10 == 0):  
                print
            print "%4d(%4d)," % (durations[c]),
                        
            c+=1
           
        print "\nMax Duration %d on mac %s" % (maxduration, maxmac)
        
    def clientConnectionTimes(self):
        
        clientCounts = [[0 for x in range(24)] for y in range(7)]
        
        for c in self.ClientDict.values():
            for day in c.days:
                w_day = time.localtime(day*24*3600).tm_wday
                for hour in (set(range(day*24, (day+1)*24)) & c.hours):
                    hourInDay = hour % 24
                    clientCounts[w_day][hourInDay] += 1
                    
        print "\n%6s" % (" "),
        for hour in range(24):
            print "%6d" % (hour),
        
        for day in range(7):
            print "\n%6s" % (DoW[day]),
            for hour in range(24):
                print "%6d" % (clientCounts[day][hour]),
        
        print
    
    def clientDailyConnectionTimes(self):
            
        clientHours =  dict(Counter(itertools.chain(*(c.hours for c in self.ClientDict.values()))))
        
        start = min(clientHours) 
        start-=start%24
        
        stop = max(clientHours)
        stop-=(stop)%24

        print "\n%18s" % (" "),
        for hour in range(24):
            print "%4d" % (hour),
   
        
        for hour in range(start,stop+24):
            if (hour%24) == 0:
                print "\n%18s" % (day2String(hour//24)),
                
            print "%4d" % (clientHours.get(hour, 0)),
       
      
        print
            
     
def __test():
    
    log = eventFile()
    
    print "End"
    
                    
if __name__ == '__main__':
    __test()            

