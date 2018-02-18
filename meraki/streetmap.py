#Street Map

import json
from math import cos, sin, pi
import numpy as np

def orthographicTransform(lat,lng, centLat, centLong):
    #  http://mathworld.wolfram.com/OrthographicProjection.html
    x=6371*cos(lat*pi/180)*sin((lng-centLong)*pi/180)
    y=6371*(cos(centLat*pi/180)*sin(lat*pi/180)-
            sin(centLat*pi/180)*cos(lat*pi/180)*cos((lng-centLong)*pi/180))         
    return x,y
    
class Street(object):
    def __init__(self, record):
        self.id = record['properties']['LINEARID']
        self.name = record['properties']['FULLNAME']
        self.lineSegments =[]
        
        if record['geometry']['type']=='MultiLineString':
            for lineSeg in record['geometry']['coordinates']:
                self.lineSegments.append([(lng,lat) for lng,lat in lineSeg])
        elif record['geometry']['type']=='LineString':
            self.lineSegments.append([(lng,lat) for lng,lat in record['geometry']['coordinates']])
        else:
            print("Linearid {} has unprocessed record of type {}".format(self.id, record['geometery']['type']))
             
def streetBufferObj(vertices):
    '''
    Build a VBO loaded with point data. Points are vertices in a 3D graph
    and represent Access Points, Cameras and GIS points of interest. 
    
    Position is 2d. The third dimension is determined in part by the layer number. 
    Layer number is mapped into a display layer, and an associated z coord.  The Z coord 
    may also be offset by 'varyings' data, e.g. traffic or usage counts.

    The vertex data may be used to plot edges or points.  Different attributes
    support each drawing mode, as determined by the drawing program.
    
    VBO data is a 'uniform' because vertices are not expected to move.

    '''

    n = len(vertices)

    nodedata = np.zeros(n, dtype=[('a_position', np.float32, 3),
                                  ('a_layer', np.float32, 1),
                                  ('a_fg_color', np.float32, 4),
                                  ('a_bg_color', np.float32, 4),
                                  ('a_size', np.float32, 1),
                                  ('a_zAdjust', np.float32, 1),
                                  ('a_colorAdjust', np.float32, 1),
                                  ('a_linewidth', np.float32, 1),
                                  ])

    nodedata['a_position'] = np.array([(vtx[0],
                                       vtx[1],
                                       -0.045)
                                      for vtx in vertices])

    nodedata['a_layer'] = np.array([0.0 for vtx in vertices])

    nodedata['a_fg_color'] = 0.4, 0.4, 0.8, 1

    color = np.random.uniform(0.5, 1., (n, 3))
    # nodedata['a_point_bg_color'] = np.hstack((color, np.ones((n, 1))))

    
    nodedata['a_size'] = 2.0
    nodedata['a_linewidth'] = 2.0

    return nodedata

 
             
def StreetModel(centLat, centLng, scale):
    
    with open('data/nyc-streets.geojson') as f:
        streets = json.load(f)
    
    vertexLocations = []  
    edgeList = []
    vertices = []
    
    
    for streetRecord in streets['features']:
        street = Street(streetRecord)
        ## print("{:<18s} {}:".format(street.id, street.name))
        inScope=False
        for segment in street.lineSegments:
            index0=None
            #print("   [{}]".format(", ".join(("({},{})".format(lng,lat) for lng,lat in segment))))
                                             
            for lng,lat in segment:
                
                
                x, y = orthographicTransform(lat, lng, centLat, centLng)
                
                if -1.5 < x/scale < 1.5 and -1.5 < y/scale < 1.5:
                    index1=len(vertexLocations)    
                    vertexLocations.append((lng,lat))
                    vertices.append((x/scale,y/scale))        
                    
                    # Add a vertex for each point
                    # Add an edge for each pairing of points

                    if index0 is not None:
                        edgeList.append((index0,index1))
                        inScope=True
                    index0=index1
                else:
                    # Both ends must be in the box
                    index0 = None
            
            if inScope:    
                print("{:<18s} {}:".format(street.id, street.name))
            
            
    maxLat=max([lat for lng,lat in vertexLocations])
    minLat=min([lat for lng,lat in vertexLocations])
    
    maxLng=max([lng for lng,lat in vertexLocations])
    minLng=min([lng for lng,lat in vertexLocations])        

    print("Map extent is {},{} to {},{}".format(minLat,minLng,maxLat,maxLng))
    
    vbo = streetBufferObj(vertices)
    edges = np.array(edgeList, dtype=(np.uint32, 2))

    print("Done")
    return vbo,edges


if __name__ == '__main__':

    vbo,edges= StreetModel()
