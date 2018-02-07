
import json
import itertools
import operator
from math import sqrt, sin, cos, pi

from streetmap import orthographicTransform

from logreader import WAPGraph, ReadMerakiLogs

#from networkVertex import Node, NodeLayer, Vertex, vertices
#from networkEdge import Link, Trail, Demand, Bypass, Edge

LayerDict = {}
NodeList = []
EdgeList = []


#Given an a ap return the equivalent Node for a given layer

nodeLayerDict = {}

class Edge(object):
    '''
    Represents a layer unique version of an _edge object
    '''
    def __init__(self, e, layer):
        self.edge=e
        self.layer=layer
        self.layerIndex=layer.index
        
        ap1,ap2= e.edgeId
        
        n1=nodeLayerDict[(ap1,layer)]
        n2=nodeLayerDict[(ap2,layer)]
        
        self.edgeId= tuple(sorted([n1, n2],key=operator.attrgetter('index')))
        self.points= (n1.index,n2.index)
                
        EdgeList.append (self)
        
    def __getattr__(self,name):
        return getattr(self.edge,name)
 

class Node(object):
    '''
    Object that stores a variant of an _wap object adding
    layer detail
    '''
   
    def __init__(self, ap, layer):
        self.layer=layer       

        # Graph is in X-Y plane, layer index is used to calculate offset of plane in z direction   
        self.layerIndex = layer.index
 
        # Node index is entry number in the VBO. Unique for all node-layer combinations
        self.index = len(NodeList)    
        self.layerOffset=0.0
        self.wap=ap
        nodeLayerDict[(ap,layer)] = self
        self.edges=[]
        NodeList.append(self)
    
        
    def __getattr__(self, name):
        return getattr(self.wap, name)

   
class Layer(object):
    '''
    A 2d Graph that overlays a map
    '''
  
    def __init__(self, name, graph, layerNumber=0):
        # edges is a graph in standard dictionary of lists of node edge tuple pairs
        
        self.name = name
        self.index=layerNumber
        LayerDict[name] = self
        self.edgeList=[]
        self.nodeList=[]
        self.loadLayerNetwork(graph)
        
    def loadLayerNetwork(self,graph):
        # Create layer unique node and edge instances
        
        
        for ap in graph:
            node = Node(ap, self)
            self.nodeList.append(node)
        
        for e in set([ed for __,ed in itertools.chain(*graph.values())]):
            edge = Edge(e, self)
            self.edgeList.append(edge)
      
            n1,n2 = edge.edgeId
            
            n1.edges.append((n2,edge))
            n2.edges.append((n1,edge))
        
       #assert max(self.NodeDict) == len(self.NodeDict)-1 ,"Node Index Error"
            
    @property
    def edges(self):
        return self.edgeList

    @property
    def edgeIndices(self):
        return [edge.points for edge in self.edgeDict.values()]

    @property 
    def nodes(self):
        return self.NodeDict.values()
    
    def MinLength(self):
        if len(self.LinkList):
            return min(
                sqrt(
                    (e.n1.location[0] -
                     e.n2.location[0]) ** 2 +
                    (
                        e.n1.location[1] -
                        e.n2.location[1]) ** 2) for e in self.edgeDict.values())
        else:
            return 0


    

class Network(object):

    def __init__(self, layers=[]):

        self.nodeList = NodeList
        self.edgeList = EdgeList

        self.layerCount=0
        self.LayerList=[]
        self.networkLayers=[]
             
        if layers==[]:
            log=ReadMerakiLogs()
            layers.append(('WiFi', WAPGraph))
                         
            WeekendGraph=log.subGraph(days=[5], hours=[3])
            layers.append(('Weekend', WeekendGraph))

            morningCommute = log.subGraph(days=[4], hours=[8])
            layers.append(('morningCommute', morningCommute))
            
        
        for layerName, graph  in layers:
            """
            layers is a dictionary where the keys are the names of the layers.
            Example layers are "WiFi", "MotionLoft", "BigBelly", "Streets", "Subways"
            The value file points to a dictionary that contains a Graph
            """
            l = Layer(layerName, graph, layerNumber=self.layerCount)
            LayerDict[layerName] = l
            self.LayerList.append(l)
            self.layerCount+=1    

        self.NormalizeCoords()
        
        self.networkLayers=[n.layerIndex for n in NodeList]
        
        self.NormalizedBoundingBox = [x/1.0 for x in self.bbox]
        
        print(self.NormalizedBoundingBox)
 
        print ("network created")
        
        
        
  
    def NormalizeCoords(self):
        
        
        minLat, minLong, maxLat, maxLong = self.bbox = self.BoundingBox()
        self.centLat, self.centLong = (minLat + maxLat) / 2.0, (minLong + maxLong) / 2.0        

        maxX,maxY = orthographicTransform(maxLat, maxLong, self.centLat, self.centLong)
        minX,minY = orthographicTransform(minLat, minLong, self.centLat, self.centLong)

        self.scale = max([maxX - minX, maxY - minY]) / 1.8        

        layerCount = len(LayerDict)
        self.layerHeights = [ 2.0*idx/layerCount - 1.0
                              for idx in range(layerCount) ]        

        self.nodeLocations = [[0,0]] * len(NodeList)
        self.vertexLocations = []
        idx=0
        for n in NodeList:
            if n.location:
                lat, lng = n.location
            else:
                print ("Node {} in layer {} has no location".format(n.ap_id, n.layer.name))
                lat,lng = minLat, minLong

            # Convert Latitude and Longitude to an orthogonal projection:
   
            x,y = orthographicTransform(lat,lng, self.centLat, self.centLong)
            
            #Scale to fit within a 1x1 cube for OpenGL 
            n.coords = (x/self.scale, y/self.scale, n.layerIndex)
            
            n.idx=idx
            self.nodeLocations[n.idx] = n.coords
            idx+=1

            self.vertexLocations.append((n.coords[0], n.coords[1], n.layerIndex))
     

    def BoundingBox(self):
        # Find a bounding box for all nodes in all layers
        nodeLocations = [n.location for n in self.nodeList if n.location != None]
        maxLat, maxLong = (max(lat for lat,__ in nodeLocations),
                      max(lng for __,lng in nodeLocations))
                      
        minLat, minLong = (min(lat for lat,__ in nodeLocations),
                      min(lng for __,lng in nodeLocations))

        return (minLat, minLong, maxLat, maxLong)

    def MinLength(self):
        return min(lyr.MinLength()
                   for lyr in self.LayersDict.values() if lyr.MinLength > 0)

    def ClientsAndServers(self, obj):
        W = P = set()

        if hasattr(obj, 'edgeId'):
            #W = set(obj.EdgeId) 
            W = set(e for ap in obj.edgeId for nap,e in ap.edges)
            
            
        elif hasattr(obj, 'edges'):
            W = set(e for nap, e in obj.edges)

        return (W, P)


 
def __test():

    log=ReadMerakiLogs()
    network = Network([('WiFi', WAPGraph)])
        
                        
if __name__ == '__main__':
    __test()            

    