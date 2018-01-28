from collections import Counter

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
