
import numpy as np


def vertexBufferObj(vertices):
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
                                  ('a_linewidth', np.float32, 1),
                                  ])

    nodedata['a_position'] = np.array([(vtx.coords[0],
                                       vtx.coords[1],
                                       vtx.layerOffset)
                                      for vtx in vertices])

    nodedata['a_layer'] = np.array([vtx.layerIndex for vtx in vertices])

    nodedata['a_fg_color'] = 0.4, 0.4, 0.8, 1

    color = np.random.uniform(0.5, 1., (n, 3))
    # nodedata['a_point_bg_color'] = np.hstack((color, np.ones((n, 1))))

    capacities = np.array([vtx.Capacity for vtx in vertices],
                          dtype=np.float32)
    maxC = np.max(capacities)

    nodedata['a_size'] = capacities * 8.0 / maxC + 2.0
    nodedata['a_linewidth'] = 2

    return nodedata


def nodeLayerIndices(nodeLayers):
    nodeLayerIds = [nl.index for nl in nodeLayers]

    return np.array(nodeLayerIds, dtype=np.uint32)


def edgeIndices(edges):
    edgeEndPoints = [e.endPoints for e in edges]
    return np.array(edgeEndPoints, dtype=(np.uint32, 2))



