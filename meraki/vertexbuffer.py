from __future__ import division
import numpy as np
from math import log



def colorSelect(wt):
    
    # Color code from Red for max to cyan for min
    return (0.5*(1+wt), (1-wt), (1-wt), 0.2 + 0.8*wt)



def vertexBufferObj(vertices,minWt=None):
    '''
    Build a VBO loaded with point data. Points are vertices in a 3D graph
    and represent Access Points, Cameras and GIS points of interest. 
    
    Position is 2d. The third dimension is determined in part by the layer number. 
    Layer number is mapped into a display layer, and an associated z coord.  The Z coord 
    may also be offset by 'varyings' data, e.g. traffic or usage counts.

    The vertex data may be used to plot edges or points.  Different attributes
    support each drawing mode, as determined by the drawing program.
    
    '''
    n = len(vertices)    
    
    nodedata = np.zeros(n, dtype=[('a_position', np.float32, 3),
                                  ('a_layer', np.float32, 1),
                                  ('a_fg_color', np.float32, 4),
                                  ('a_bg_color', np.float32, 4),
                                  ('a_size', np.float32, 1),
                                  ('a_linewidth', np.float32, 1),
                                  ])
    

    nodedata['a_position'] = np.array([vtx.coords                                         
                                       for vtx in vertices])
                                      
    nodedata['a_layer'] = np.array([vtx.layerIndex for vtx in vertices])
   
    weights = np.log(nodedata['a_position'][:,2])
    if minWt==None:  minWt = np.min(weights)
    
    nodedata['a_fg_color'] = np.array([colorSelect(1-wt/minWt) for wt in (weights)], dtype=(np.float32,4))
    
    nodedata['a_bg_color'] = (1.0, 1.0, 1.0, 1.0)

    nodedata['a_size'] = (1 - weights/minWt) * 4 + 1.0
    nodedata['a_linewidth'] = (1-weights/minWt) + 1
    
    return nodedata


def nodeLayerIndices(nodeLayers):
    nodeLayerIds = [nl.index for nl in nodeLayers]

    return np.array(nodeLayerIds, dtype=np.uint32)


def edgeIndices(edges):
    edgeEndPoints = [e.endPoints for e in edges]
    return np.array(edgeEndPoints, dtype=(np.uint32, 2))


