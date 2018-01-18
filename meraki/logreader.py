
import json
import string
import operator
import time
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


def day2String(day):
    return time.strftime("%a %b %d %Y", time.localtime(day*24*3600))
    

with open('data/AccessPoints.json') as APfile:    
    APInformation = json.load(APfile)
    APfile.close()

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
        self.clients=[client]       # a list of clients that connect to this network 
        self.size = len(uid)        # Number of APs in the network
        APNetworkDict[uid] = self   # Add to the dictionary
          
    def APs(self): 
        # Return a list of instances of class ap that represent APs in the network
        return sorted([apList[i] for i in self.uid])

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
        hour = float(event["time_f"]) // 3600
        day = hour // 24        

        self.clients.setdefault(day,set()).add(client)

    def printClientCountByDay(self):

        for day in sorted(self.clients.keys()):
            print ("%12s: %6d") % (day2String(day), len(self.clients[day]))
        

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
        eventTime = event["time_f"]
       
        hour = int(eventTime//3600)
        day = int(hour//24)
        
        self.events.setdefault(day,[]).append(event)
        self.days.add(day)
        self.hours.add(hour)
        self.aps.add(ap)

        if (event["el_type"] == "association"): 
            
            if (self.AnyAPExpectedAssociationTime 
                and eventTime - self.AnyAPExpectedAssociationTime) < 0.2:
                self.clientConnectTime += self.AllAPAssociatedDuration
                self.AnyAPExpectedAssociationTime=0
                self.AllAPAssociatedDuration=0
            
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
                 
                # Mark the days and hours client was connected 
                self.days.update(set(range(day, day + int(duration//24//3600) + 1)))
                self.hours.update(set(range(hour, hour + int(duration//3600) + 1)))
                 
      
        elif (event["el_type"] == "disassociation"): 
            duration = float(event["details"]["duration"])  
            
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
                fix = self.AnyAPExpectedAssociationTime - expectedAssTime
                if fix > 0:
                    self.AnyAPExpectedAssociationTime = expectedAssTime
                    self.AllAPAssociatedDuration += fix
                    
            if not ap in self.ExpectedAssociationTime:
                fixap=0
                self.ExpectedAssociationTime[ap] = expectedAssTime
                self.associatedDuration[ap] = duration
            else:
               
                fixap = self.ExpectedAssociationTime[ap] - expectedAssTime
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
            
        self.clientConnectDurations()
        self.printClientWhenSeenCounts()
        self.printAccessPointUsage()
        self.printRangeByDay()
        
        for client in self.ClientDict.values():
            client.APNetwork()

        PrintAPNClientCountDistribution()
                        
       
    def LoadEvents(self,d):
        
        with open("events.csv","w") as csvFile:
        
            for event in d:
                eventTime = event["time_f"]
                day = time.strftime("%Y-%m-%d", time.localtime(eventTime))
                
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
                
                print >> csvFile, "%20s, %s, %16s, %s, %s, %f" % (clientMac, day, eventTime, apID, etype, duration)
                
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
        
        print Commentary        
        print "%16s" % ("Range "),        
        for count in range(1,11): 
            print "%5d" % (count),
        print    
        
        for day in sorted(rangeCounts.keys()):      
            print "%16s" % (day2String(day)),        
            for count in range(1,11): 
                print "%5d" % (rangeCounts[day].get(count,0)),
              
            for count in sorted(set(rangeCounts[day]) - set(range(11))):    
                print "(%d,%d)" % (count, rangeCounts[day][count]),
            print    

    def printClientWhenSeenCounts(self):
        
        seenDays = Counter([c.ClientDays() for c in self.ClientDict.values()])
        seenHours =  Counter([c.ClientHours() for c in self.ClientDict.values()])
            
        print ("\n\nCount of calendar days during which clients were observed")    
            
        
        for i in sorted(seenDays):
            print "%4d: %4d,"  % (i, seenDays[i]),
            if i%10 == 0: 
                print
                                  
        print ("\n\nCount of hours during which clients were observed")    
        for i in sorted(seenHours):
            print "%4d: %4d,"  % (i, seenHours[i]),
            if i%10 == 0: 
                print
                                    
       
    def clientConnectDurations(self):
       
        print "\nClient Connect time distribution [hours] (AP)",
       
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
        
           
     
def __test():
    
    log = eventFile()
    
    print "End"
    
                    
if __name__ == '__main__':
    __test()            

