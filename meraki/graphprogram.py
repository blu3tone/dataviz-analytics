from __future__ import division

import numpy as np
from vispy import gloo
from vispy.util.transforms import translate, rotate
from bisect import bisect_left

from picker import pickEdge, pickPoint
from streetmap import StreetModel

vert = """
#version 120

// Uniforms
// ------------------------------------
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_antialias;
uniform float u_size;
uniform float u_offset;
uniform vec4 u_color;
uniform int  u_layerMap[32];
uniform vec4 u_bg_color;


// Attributes
// ------------------------------------
attribute vec3  a_position;
attribute vec4  a_fg_color;
attribute vec4  a_bg_color;
attribute float a_linewidth;
attribute float a_size;
attribute float a_layer;
attribute float a_zAdjust;
attribute float a_colorAdjust;

// Varyings
// ------------------------------------
varying vec4 v_fg_color;
varying vec4 v_bg_color;
varying float v_size;
varying float v_linewidth;
varying float v_antialias;


float z_coord(int layerIdx, float heightOffset) {
            int lyr = u_layerMap[layerIdx];
            float z = lyr  * u_offset - 1.0 + heightOffset;
            return z;
        }

void main (void) {
    int lyridx = int(a_layer);

    v_size = a_size * u_size;
    v_linewidth = a_linewidth;
    v_antialias = u_antialias;

    v_fg_color  = a_fg_color;

    if (u_color.a >= 0.0)
         {
          v_fg_color  = u_color;
         }

    v_bg_color  = a_bg_color;

    if (u_bg_color.a >= 0.0)
         {
          v_bg_color  = u_bg_color;
         }

    float ht = z_coord(lyridx, a_position.z+ a_zAdjust);

    gl_Position = u_projection * u_view * u_model *
                        vec4(a_position.x, a_position.y, ht, 1.0);
    gl_PointSize = v_size + 2*(v_linewidth + 1.5*v_antialias);
}
"""

frag = """
#version 120

// Constants
// ------------------------------------

// Varyings
// ------------------------------------
varying vec4 v_fg_color;
varying vec4 v_bg_color;
varying float v_size;
varying float v_linewidth;
varying float v_antialias;

// Functions
// ------------------------------------
float marker(vec2 P, float size);


// Main
// ------------------------------------
void main()
{
    float size = v_size +2*(v_linewidth + 1.5*v_antialias);
    float t = v_linewidth/2.0-v_antialias;

    // The marker function needs to be linked with this shader
    float r = marker(gl_PointCoord, size);

    float d = abs(r) - t;
    if( r > (v_linewidth/2.0+v_antialias))
    {
        discard;
    }
    else if( d < 0.0 )
    {
       gl_FragColor = v_fg_color;
    }
    else
    {
        float alpha = d/v_antialias;
        alpha = exp(-alpha*alpha);
        if (r > 0)
            gl_FragColor = vec4(v_fg_color.rgb, alpha*v_fg_color.a);
        else
            gl_FragColor = mix(v_bg_color, v_fg_color, alpha);
    }
}

float marker(vec2 P, float size)
{
    float r = length((P.xy - vec2(0.5,0.5))*size);
    r -= v_size/2;
    return r;
}
"""

fs = """
#version 120

// Varyings
// ------------------------------------
varying vec4 v_fg_color;

void main(){
    gl_FragColor = v_fg_color;
}
"""

vs_plane = """
#version 120

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_offset;
uniform int    u_layerMap[32];
uniform vec4 u_lyr_fg_color[32];

attribute vec3 a_position;
attribute float a_layer;

varying vec4 v_color;

float z_coord(int layerIdx, float heightOffset) {
            int lyr = u_layerMap[layerIdx];
            float z = lyr  * u_offset - 1.0 + heightOffset;
            return z;
        }

void main() {
    int lyridx = int(a_layer);
    float ht = z_coord(lyridx, a_position.z);
    gl_Position = u_projection * u_view * u_model *
                         vec4(a_position.x, a_position.y, ht, 1);


    if (u_layerMap[lyridx] <= - 1)
         v_color = vec4 (1.,1.,1.,0.);
    else
         v_color =  u_lyr_fg_color[lyridx];
    }
"""

fs_plane = """
#version 120

varying vec4 v_color;

void main(){
    gl_FragColor = v_color;

}
"""



class GlProgram(gloo.Program):

    def __init__(self, **kwargs):
        self.parent = kwargs.pop('parent', None)
        canvas = self.canvas = kwargs.pop('canvas')
        self.heightOffset = kwargs.pop('heightOffset', 0.0)

        super(GlProgram, self).__init__(**kwargs)

        #  Register for View and Projection updates with the canvas
        canvas.registerDependent(self)

        self['u_view'] = canvas.view
        self['u_projection'] = canvas.projection

        #  A program is transformed by the chain of model matrices
        #  from parent to the root.

        self.__dict__['children'] = []

        if self.parent is not None:
            self.parent.registerChild(self)
        else:
            self.root = self

        self.model = np.eye(4, dtype=np.float32)

    def registerChild(self, child):
        self.children.append(child)
        child.root = self.root

    def modelProduct(self):
        """
        Model matrix is the product of all model transformations from here
        down to the root

        The model must be recalculated if there is a change to the model on
        any of the ancestors on the chain down to root
        """
        if self.parent is None:
            return self._model
        else:
            return self._model.dot(self.parent.modelProduct())

    @property
    def height(self):
        if self.parent is None:
            return self.canvas.height
        else:
            return self.parent.height

    def __getitem__(self, name):
        if name == 'u_model':
            return self.modelProduct()
        else:
            return super(GlProgram, self).__getitem__(name)

    def updateChildModels(self):
        self['u_model'] = self.modelProduct()
        for child in self.children:
            child.updateChildModels()

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value
        self.updateChildModels()

class geojsonProgram(GlProgram):
    
    def __init__(self, **kwargs):
        
        self.centLat = kwargs.pop('centLat')
        self.centLong = kwargs.pop('centLong')
        self.scale = kwargs.pop('scale')
                        
        self.streetVertices, self.streetEdges = StreetModel(self.centLat, 
                                                            self.centLong, 
                                                            self.scale)
        self.vbo = gloo.VertexBuffer(self.streetVertices)
        self.index = gloo.IndexBuffer(self.streetEdges)
    
        kwargs['vert'] = vert
        kwargs['frag'] = fs
    
        super(geojsonProgram,self).__init__(**kwargs)
        self.bind(self.vbo)
        
        for lyr in range(32):
            self['u_layerMap[%d]' % lyr] = lyr
    
        self['u_size'] = 1
        self['u_color'] = (0.6, 0.6, 0.6, 1.0)
        self['u_bg_color'] = (0.2, 0.2, 0.2, -0.6)
        self['u_antialias'] = True
        self['u_offset'] = -0.05

    def draw(self):
        super(geojsonProgram,self).draw('lines', self.index)

class planesProgram(GlProgram):
    
    def __init__(self, **kwargs):
        
        self.network = kwargs.pop('network')
        
        kwargs['vert'] = vs_plane
        kwargs['frag'] = fs_plane

        super(planesProgram,self).__init__(**kwargs)

        layerdata = self.loadPlanes(self.network)
        self.vbo = gloo.VertexBuffer(layerdata)
        self.bind(self.vbo)    

        color = np.random.uniform(0.5, 1, (32, 3))
        colorTable = np.hstack((color, np.ones((32, 1))))
        colorTable[0,:3] = (1,0.87,0.68)
        colorTable[:, 3] = 0.15
    
        for lyr in range(32):
            self['u_lyr_fg_color[%d]' % lyr] = colorTable[lyr]
            self['u_layerMap[%d]' % lyr] = lyr

        
    def loadPlanes(self, network):
            
        nl = network.layerCount
        layerdata = np.zeros(nl * 6, dtype=[('a_position',
                                             np.float32, 3),
                                            ('a_layer',
                                             np.float32, 1),
                                            ])
    
        x0, y0, x1, y1 = [a*1.05 for a in network.NormalizedBoundingBox]
    
        layerdata['a_position'] = np.array([(x, y, -0.002) for z in range(nl)
                                            for x, y in [[x0, y0],
                                                         [x0, y1],
                                                         [x1, y0],
                                                         [x1, y1],
                                                         [x1, y0],
                                                         [x0, y1]]],
                                           dtype=np.float32)
    
        layerdata['a_layer'] = np.array([list(range(nl))]*6, dtype=np.float32).transpose().ravel()
    
        return layerdata

    def draw(self):
        super(planesProgram,self).draw('triangles')


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
                                  ('a_zAdjust', np.float32,1),
                                  ('a_colorAdjust', np.float32,1)
                                  ])
    

    nodedata['a_position'][:,:2] = np.array([(vtx.coords[0],vtx.coords[1])                                         
                                       for vtx in vertices])
    nodedata['a_zAdjust'] = np.array([vtx.coords[2]
                                     for vtx in vertices])
    nodedata['a_layer'] = np.array([vtx.layerIndex for vtx in vertices])


    weights = np.log(nodedata['a_zAdjust'])
    if minWt==None:  minWt = np.min(weights)
    
    nodedata['a_fg_color'] = np.array([colorSelect(1-wt/minWt) for wt in (weights)], dtype=(np.float32,4))
    
    nodedata['a_bg_color'] = (1.0, 1.0, 1.0, 1.0)
    
  
    nodedata['a_size'] = (1 - weights/minWt) * 4 + 1.0
    nodedata['a_linewidth'] = (1-weights/minWt) + 1
    
    return nodedata


class graphProgram(gloo.Program):
    
    def __init__(self, **kwargs):
        ''' 
        Create vertical lines between the plane and the node
        Read the xyz location from the a_position array
        For each x,y choose the maximum z
        Create vertices at x,y,zmax and x,y,0
        Create an index table that we will use to draw lines between these points
    
        '''
        vtxs = kwargs.pop('vert',vert)
        frags = kwargs.pop('frag',fs)
        
        super(graphProgram, self).__init__(vert=vtxs, frag=frags)
        
        self.nodeData = kwargs.pop('nodeData')
  
        vbo= gloo.VertexBuffer(self.nodeData)        
        self.bind(vbo)
        
        self.canvas=kwargs.pop('canvas')
        self.canvas.registerDependent(self)
        self.parent=kwargs.pop('parent',None)
        
        if self.parent is not None:
            self.parent.registerChild(self)
        else:
            self.root = self
            
        self.__dict__['children'] = []
    
        self.model = kwargs.pop('model', np.eye(4, dtype=np.float32))
        
        self['u_projection']= self.canvas.projection
        self['u_view']= self.canvas.view
        for lyr in range(32):
            self['u_layerMap[%d]' % lyr] = lyr        

        uniforms = {'u_size':1, 
                    'u_offset':0,
                    'u_color':(0.1,0.1,0.1, 0.3),
                    'u_bg_color':(1.,1.,1., -0.3),
                    'u_antialias':True
                    }
        
        uniforms.update({k:v for k,v in kwargs.items() if k in uniforms})    
        for k,v in uniforms.items(): self[k]=v

    def _updateWorldCoordinates(self):
        """
        Locate the plot in the 3D world space for picking
        """
        xyCoords = self.nodeData['a_position'][:, :-1]
        zCoords = self.nodeData['a_zAdjust'] + self.layerOffset

        nl = len(xyCoords)

        # Create an array of Vec4s  by augmenting with a 1:
        self.worldVertexPositions = np.hstack((xyCoords.reshape(nl, 2),
                                            zCoords.reshape(nl, 1),
                                            np.ones((nl, 1))))    

    def updateAttributes(self, nodeZPositions):
        self.nodeData['a_zAdjust'] = nodeZPositions
        vbo= gloo.VertexBuffer(self.nodeData)
        self.bind(vbo)
        
    def registerChild(self, child):
        self.children.append(child)
        child.root = self.root

    def modelProduct(self):
        """
        Model matrix is the product of all model transformations from here
        down to the root

        The model must be recalculated if there is a change to the model on
        any of the ancestors on the chain down to root
        """
        if self.parent is None:
            return self._model
        else:
            return self._model.dot(self.parent.modelProduct())

    @property
    def height(self):
        if self.parent is None:
            return self.canvas.height
        else:
            return self.parent.height

    def __getitem__(self, name):
        if name == 'u_model':
            return self.modelProduct()
        else:
            return super(graphProgram,self).__getitem__(name)

    def updateChildModels(self):
        self['u_model'] = self.modelProduct()
        for child in self.children:
            child.updateChildModels()

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value
        self.updateChildModels()
        
    
class verticalsProgram(graphProgram):
    def __init__(self, **kwargs):
        ''' 
        Create vertical lines between the plane and the node
        Read the xyz location from the a_position array
        For each x,y choose the maximum z
        Create vertices at x,y,zmax and x,y,0
        Create an index table that we will use to draw lines between these points
        '''
        nodeData = kwargs.pop('nodeData')
        nl = self.apCount = nodeData.shape[0]
        
        # Double up the vertices.  The second half has z Adjust values of zero,
        # The first half has z values proportional to the attribute being graphed

        aData = np.concatenate((nodeData,nodeData),axis=0)  
        aData['a_position'][:,2]=0.
        aData['a_zAdjust'][nl:]=0
        
        kwargs['nodeData']= aData
        kwargs['frag']=fs
        
        super(verticalsProgram, self).__init__(**kwargs)
          
        #self['a_zAdjust'] = aData['a_zAdjust']
        #self['a_zAdjust'][nl:] = np.zeros(nl,dtype=np.float32)
        #self['a_colorAdjust'] = aData['a_colorAdjust']
      
        self.Index = self.IndexBuffer([(x,x+nl) for x in range(nl)])
        self.selectIndex = None
        
    def updateAttributes(self, nodeZPositions):
        self.nodeData['a_zAdjust'][:self.apCount] = nodeZPositions
        #self['a_zAdjust'][:self.apCount] = nodeZPositions
        # self['a_zAdjust'] = np.concatenate(nodeZPositions, np.zeros(self.apCount, dtype=np.float32))
        vbo= gloo.VertexBuffer(self.nodeData)
        self.bind(vbo)
  
    def IndexBuffer(self, vertices):
        edgePoints = np.array(vertices, dtype=(np.uint32, 2))
        return gloo.IndexBuffer(edgePoints)
        
    def select(self, nodeIndexArray):
        if len(nodeIndexArray):
            self.selectIndex=self.IndexBuffer([(x,x+self.apCount) for x in nodeIndexArray])
        else:
            self.selectIndex=None
        return self.selectIndex

    def unselect(self):
        self.selectIndex=None

    def draw(self):
        if self.selectIndex:
            super(verticalsProgram,self).draw('lines', self.selectIndex)
        else:
            super(verticalsProgram,self).draw('lines', self.Index)

def colorSelect(wt):
    # Color code from Red for max to cyan for min
    return (0.5*(1+wt), (1-wt), (1-wt), 0.2 + 0.8*wt)

class edgesProgram(graphProgram):
    # GL setup to draw edges in a graph structure
    
    def __init__(self, edgeList, **kwargs):
        self.edgeList = edgeList  # List of Python objects
       
        # The graph is defined by its endpoints.  The order is important 
        # Sort based on the endpoint's index attribute 
        
        self.layerOffset=-1
        
        vertices = sorted((ep for edge in edgeList for ep in edge.edgeId), key = lambda x : x.index)
        nodeData = vertexBufferObj(vertices)
        nodeData['a_position'][:,2]=0.
        
        kwargs['nodeData']= nodeData
        kwargs['vert'] = vert
        kwargs['frag'] = fs
        super(edgesProgram, self).__init__(**kwargs)

        edgeEnds = [edge.points for edge in self.edgeList]
        self.edges = np.array(edgeEnds, dtype=(np.uint32, 2))        
        self.edgesIndex = gloo.IndexBuffer(self.edges)
      
        self.highlightColor = (0.0, 1.0, 0.0, 1.0)
        self.selectedColor = (0.7, 0.375, 0.375, 1.0)
        self.edgeColor = self.edgeColorDefault = (0.9, 0.6, 0.6, -1)
      
        self.selectedEdges=[]
        self.selectedPointIndex=None
        self.selectedIndex=None
        self.highlightIndex=None

        for lyr in range(32):
            self['u_layerMap[%d]' % lyr] = lyr
        
        self['u_size'] = 1
        self['u_color'] = (0.9, 0.2, 0.2, -1.0)
        self['u_bg_color'] = (1.0, 1.0, 1.0, 1)
        #self['u_bg_color'] = (0.2, 0.2, 0.2, -0.6)
        self['u_antialias'] = 1  
        
        self._updateWorldCoordinates()
           
    def updateAttributes(self, values):
        # Values has a value to be graphed for each edge, 
        # for example the count of movement of clients between 
        # nodes
        # Update the z coordinate of the edge endpoints:
        
        for edgeIdx, edge in enumerate(self.edgeList):
            for nidx in edge.points:
                self.nodeData['a_zAdjust'][nidx]= values[edgeIdx]
        
        # Code a transition in color based on ordinate in sorted list
        weights = sorted(self.nodeData['a_zAdjust'])
        count = len(weights)
        
        self.nodeData['a_fg_color'] = np.array([colorSelect(bisect_left(weights, wt)/count) 
                                                      for wt in self.nodeData['a_zAdjust']], dtype=(np.float32,4))
        vbo= gloo.VertexBuffer(self.nodeData)
        self.bind(vbo) 
 
        self._updateWorldCoordinates()

    def closestEdge(self, x, y, tol=0.01):
        """
        Finds an edge that is closest to a ray
        from camera through x,y, the mouse pointer location 
        """

        X1, Y1 = self.canvas.pixelToNearPlaneCoords(x, y)
        #  Perspective Transformation to get a normalized screen view
        mView = self.canvas.view.dot(self.canvas.projection)
        # Now adjust the matrix so that the normalized perspective
        # model is moved to mouse location 0,0
        mView = mView.dot(translate([-X1, -Y1, 0]))

        # Use Numpy to multiply all vertex locations by the mView matrix:
        vertices = self.worldVertexPositions.dot(mView)

        def searchForClosestEdge(candidates):
            # Create an array that contains edge end point locations.
            # Return the edge that is closest to the mouse cursor,
            # and within tolerance tol

            coords = [(vertices[l.points[0]][:3],
                       vertices[l.points[1]][:3])
                      for l in candidates]
            res = pickEdge(coords, tol=tol)

            if res:
                closest, z, t = res
                return (candidates[closest], z, t)

        if self.selectedEdges:
            # Search within selected edges
            links = list(set(self.edgeList)
                         & (self.selectedEdges ))
            return searchForClosestEdge(links)
        else:
            # Or global search
            return searchForClosestEdge(self.edgeList)        
    
    def setHighlight(self, obj):
        if obj and hasattr(obj,'points'):
            highlightedPoints = np.array(obj.points, dtype=(np.uint32, 1))
            self.highlightIndex = gloo.IndexBuffer(highlightedPoints)
        else:
            self.highlightIndex=None
        
    def setSelectedItem(self, obj):
        # Index to draw edge highlight
    
        if obj and hasattr(obj, 'points'):
            #edgeList= np.array(obj.points)          # end Points of selected edge
            #pointList = list(obj.nodepoints)    # nodes
    
            selectedEdge = np.array(obj.points, dtype=(np.uint32, 1))
            self.selectedIndex = gloo.IndexBuffer(selectedEdge)
        else:
            self.seletedIndex = None
        
    def setSelectionList(self, selectedEdges):
        # Highlight the selected subset of edges
        self.selectedEdges = selectedEdges
        if selectedEdges:
            EdgeNodeList = [e.points for e in selectedEdges]
            edges = np.array(EdgeNodeList, dtype=(np.uint32, 2))            
            self.selectedIndex = gloo.IndexBuffer(edges)
        else:
            self.selectedIndex = None

        if selectedEdges:
            self.edgeColor = self.edgeColorDefault[:-1] + (0.1, )
        else:
            self.edgeColor = self.edgeColorDefault
            
        self._updateWorldCoordinates()
        
    def unselect(self):
        self.selectedIndex=None
        
    def draw(self):
        if self.selectedIndex:
            self['u_color'] = self.selectedColor
            super(edgesProgram,self).draw('lines', self.selectedIndex)
        else:
            self['u_color'] = self.edgeColor
            super(edgesProgram,self).draw('lines', self.edgesIndex)
        
        if self.highlightIndex:
            self['u_color'] = self.highlightColor
            super(edgesProgram,self).draw('lines', self.highlightIndex)

class nodesProgram(graphProgram):
    # GL setup to draw edges in a graph structure
    
    def __init__(self, nodeList, **kwargs):
        self.nodeList = nodeList  # List of Python objects
        
        # The graph is defined by its endpoints.  The order is important 
        # Sort based on the endpoint's index attribute 
        
        
        self.layerOffset=-1
        
        nodeData = vertexBufferObj(nodeList)
        nodeData['a_position'][:,2]=0.
        
        kwargs['nodeData']= nodeData
        kwargs['vert'] = vert
        kwargs['frag'] = frag        
        super(nodesProgram, self).__init__(**kwargs)

        points = [n.index for n in nodeList]
        self.points = np.array(points, dtype=(np.uint32, 1))        
        self.Index = gloo.IndexBuffer(self.points)
      
        self.pointColor = (0.3, 0.3, 0.9, 1)
        self.highlightColor = (0.0, 1.0, 0.0, 1.0)
        self.selectedColor = (0.7, 0.375, 0.375, 1.0)

        self['u_size'] = 1
        self['u_color'] = self.pointColor
        self['u_bg_color'] = (1.0, 1.0, 1.0, 1)
        self['u_antialias'] = 1

        self._updateWorldCoordinates()
        self.selectedIndex=None
        self.selectedEdges=[]
        self.highlightIndex=None
           
    def updateAttributes(self, values):
        # Values has a value to be graphed for each edge, 
        # for example the number of clients connected to an AP
        
        # Update the z coordinate of the edge endpoints:
        
        self.nodeData['a_zAdjust']= np.array(values, dtype=(np.float32,1))
        
        vbo= gloo.VertexBuffer(self.nodeData)
        self.bind(vbo) 
        self._updateWorldCoordinates()
        
    def closestNode(self, x, y, tol=0.01):
        """
        Finds an edge that is closest to a ray
        from camera through x,y, the mouse pointer location 
        """

        X1, Y1 = self.canvas.pixelToNearPlaneCoords(x, y)
        #  Perspective Transformation to get a normalized screen view
        mView = self.canvas.view.dot(self.canvas.projection)
        # Now adjust the matrix so that the normalized perspective
        # model is moved to mouse location 0,0
        mView = mView.dot(translate([-X1, -Y1, 0]))

        # Use Numpy to multiply all vertex locations by the mView matrix:
        vertices = self.worldVertexPositions.dot(mView)
        
        def searchForClosestNode(candidates):
            # Create an array that contains point locations.
            # Return the node closest to the mouse cursor,
            # and within tolerance tol
    
            coords = [(vertices[n.index][:3])
                      for n in candidates]
            res = pickPoint(coords, tol=0.1)
    
            if res:
                closest, z = res
                return (candidates[closest], z)
            else:
                return None

        if self.selectedEdges:
            # Search nodes within selection

            nodes = list(n for e in self.canvas.selectedEdges for n in e.nodes)
            return searchForClosestNode(nodes)
        else:
            # Or global search
            return searchForClosestNode(self.nodeList)

    def setHighlight(self, obj):
        if obj and hasattr(obj,'index'):
            highlightedPoints = np.array([obj.index], dtype=(np.uint32, 1))
            self.highlightIndex = gloo.IndexBuffer(highlightedPoints)
        else:
            self.highlightIndex=None
  
    def setSelectedNode(self,obj=None):  
        if obj and hasattr(obj,'index'):
            print("Node {} selected".format(obj.name))
            nodeIndexes= np.array([obj.index], dtype=(np.uint32, 1))
            self.selectedIndex = gloo.IndexBuffer(nodeIndexes)
        else:
            self.selectedIndex = None

    def setSelectedEdge(self, obj):
        # Index to draw highlight
        if obj and hasattr(obj, 'nodepoints'):
            print("Edge {} selected".format(obj.nodepoints))
            nodeIndexes = np.array(obj.nodepoints, dtype=(np.uint32, 1))
            self.selectedIndex = gloo.IndexBuffer(nodeIndexes)
        else:
            self.selectedIndex = None
        
    def setSelectionList(self, selectedEdges):
        # Show the nodes that a list of edges connect to
        self.selectedEdges = selectedEdges
        
        if selectedEdges:
            nodeIndexes = [n for e in selectedEdges for n in e.nodepoints]
            self.selectedIndex = gloo.IndexBuffer(nodeIndexes)
        else:
            self.selectedIndex = None
    
        self._updateWorldCoordinates()
        
    def unselect(self):
        self.selectedIndex=None
        
    def draw(self):
        if self.selectedIndex != []:
            self['u_color']=self.selectedColor
            super(nodesProgram,self).draw('points', self.selectedIndex)
        else:
            self['u_color']=self.pointColor
            super(nodesProgram,self).draw('points', self.Index)
            
        if self.highlightIndex:
            self['u_color']=self.highlightColor
            super(nodesProgram,self).draw('points', self.highlightIndex)
        
        