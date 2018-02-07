


import numpy as np
from numpy.linalg import inv

from math import hypot
import itertools


import wx
import time

import io
import cProfile
import pstats

from vispy import gloo, app
from vispy.gloo import set_viewport, set_state, clear
from vispy.util.transforms import frustum, translate, rotate
from vispy.util.keys import CONTROL

import textProgram as txt
from network import Network

import picker
from undo import UndoBuffer

#from networkVertex import Vertex, vertices
from vertexbuffer import vertexBufferObj, nodeLayerIndices, edgeIndices
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
uniform vec4 u_lyr_fg_color[32];
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

    v_fg_color  = u_lyr_fg_color[lyridx];

    if (u_color.a >= 0.0)
         {
          v_fg_color  = u_color;
         }

    v_bg_color  = a_bg_color;

    if (u_bg_color.a >= 0.0)
         {
          v_bg_color  = u_bg_color;
         }

    float ht = z_coord(lyridx, a_position.z);

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


#def Verticals(network):
    #res = []
    #for n in network.nodeList:

        #maxNodeLyr = max([(nl.layer.idx, nl.vtx) for nl in n.nodeLayerList])
        #minNodeLyr = min([(nl.layer.idx, nl.vtx) for nl in n.nodeLayerList])

        #if maxNodeLyr != minNodeLyr:
            #res.append((maxNodeLyr[1], minNodeLyr[1]))

    #return res


#def SelectVerticals(edges):
    #NodeEdges = {}  # Key is Node, value is NodeLayer

    #for n, nl in list(set((nl.node, nl) for e in edges for nl in e.points)):
        #NodeEdges.setdefault(n, []).append(nl)

    #res = []
    #for n in NodeEdges:
        #maxNodeLyr = max([(nl.layer.idx, nl.vtx) for nl in NodeEdges[n]])
        #minNodeLyr = min([(nl.layer.idx, nl.vtx) for nl in NodeEdges[n]])

        #if maxNodeLyr != minNodeLyr:
            #res.append((maxNodeLyr[1], minNodeLyr[1]))
    #return res


def SelectVertices(edges):
    Nodes = list(set(itertools.chain(*[e.points for e in edges])))
    pointArray = np.array(Nodes, dtype=(np.uint32, 1))
    return pointArray


def LoadPlanes(network):
    nl = network.layerCount
    layerdata = np.zeros(nl * 6, dtype=[('a_position',
                                         np.float32, 3),
                                        ('a_layer',
                                         np.float32, 1),
                                        ])

    x0, y0, x1, y1 = network.NormalizedBoundingBox

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


def LoadSelectedEdges(edgeList):

    EdgeNodeList = [e.points for e in edgeList]
    edges = np.array(EdgeNodeList, dtype=(np.uint32, 2))
    
    return edges


def baseLayerEdges(network):
    '''
    Create a ghost image of the physical network
    to provide context when viewing subset selections
    '''
    baseLayer = network.LayerList[0]
    links = baseLayer.edgeIndices
    edgeVertices = np.array(links, dtype=(np.uint32, 2))
    return edgeVertices


#def loadSelectedTrailLinks(edgeList):
    #'''
    #Returns the vertex endpoints of the links that serve any
    #trail that is in the client or server list.  Used to create
    #a ghost image to provide context.
    #'''

    #linkList = []
    #trails = [t for t in edgeList if isinstance(t, Trail)]

    #for t in trails:
        #linkList.extend(t.servers)

    #linkList = list(set(linkList))

    #edgeVertices = np.array([(l.points[0].vtx, l.points[1].vtx)
                             #for l in linkList],
                            #dtype=(np.uint32, 2))

    #return edgeVertices


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


class Canvas(app.Canvas):

    def __init__(self, **kwargs):
        # Initialize the canvas for real
        u_antialias = 1

        self.parent = kwargs['parent']

        self.model = model = kwargs.pop('model')

        app.Canvas.__init__(self, keys='interactive', **kwargs)
        self.size = 800, 778
        self.position = 50, 50

        self.zoomRatio = 1.0
        self.zoomIdx = 0
        self.savedZoomIdx = self.zoomIdx
        self.zoomLimit = 40
        self.zoomWheelRatio = 1.1
        self.zoomFactors = [self.zoomWheelRatio ** i
                            for i in range(self.zoomLimit)]

        self.ClickStart = self.DCTime = time.time()

        self.near = -1
        self.far = -6

        self.worldCenterZ = self.far + 2.0

        self.work = self.prot = None

        self.workIndex = None
        self.protIndex = None
        self.sVerticalsIndex = None
        self.highlightIndex = None
        self.selectedIndex = None
        self.selectedObject = None
        self.highlight = None
        self.highlighted = None
        self.highlightT = 0.5

        self.layerMap = list(range(32))
        self.savedLayerMap = list(self.layerMap)

        self.trailHeightOffset = 0.005
        self.highlightColor = (0.0, 1.0, 0.0, 1.0)
        self.selectedColor = (0.7, 0.375, 0.375, 1.0)
        self.baseColorDefault = (0.8, 0.8, 0.8, 1)
        self.edgeColor = self.edgeColorDefault = (0.9, 0.6, 0.6, 1)
        self.trailColor = self.trailColorDefault = (0.6, 0.6, 0.9, 0.5)
        self.pointColor = self.pointColorDefault = (0.3, 0.3, 0.8, 1)
        self.verticalColor = self.verticalColorDefault = (0.6, 0.6, 0.2, 0.4)

        self.pickEdgeCoords = None

        self.markerIndex = gloo.IndexBuffer(model.markers)
        self.edgesIndex = gloo.IndexBuffer(model.edges)
        
        self.streetVertices, self.streetEdges = StreetModel(self.model.network.centLat, 
                                                            self.model.network.centLong, 
                                                            self.model.network.scale)
        self.streetVbo = gloo.VertexBuffer(self.streetVertices)
        
        layerdata = LoadPlanes(model.network)
        self.planeVbo = gloo.VertexBuffer(layerdata)

        # The base layer is the street network
        self.baseIndex = gloo.IndexBuffer(self.streetEdges)
        
        self.view = np.eye(4, dtype=np.float32)
        self.projection = np.eye(4, dtype=np.float32)

        self.xRot = -65.0  # Pitch
        self.yRot = 0.0     # Yaw

        self.savedXRot = self.xRot
        self.savedYRot = self.yRot

        self.view = self.view.dot(rotate(self.xRot, [1, 0, 0]))

        # Where the Mouse is pointing in world and view coords
        self.wFocus = np.array([0., 0., 0., 1.], dtype=np.float32)
        self.vFocus = np.array([0., 0., self.worldCenterZ, 1.],
                               dtype=np.float32)

        self.savedWFocus = self.wFocus.copy()
        self.savedVFocus = self.vFocus.copy()

        self.spacing = 20
        self.spacingRatio = 20
        self.activeLayers = list(range(model.layerCount))

        self.dependents = []
        self.dependencies = {}

        self.updateLayerSpacing(self.spacing)

        self.undo = UndoBuffer(operation=self._updateView,
                               rState=(self.wFocus, self.vFocus,
                                       self.xRot, self.yRot, self.zoomIdx),
                               uState=(self.wFocus, self.vFocus,
                                       self.xRot, self.yRot, self.zoomIdx))

        self.view = self.view.dot(translate(self.vFocus[0:3]))

        self.registerDependency(u_view='view',
                                u_projection='projection',
                                u_offset='offset')

        self.program = GlProgram(canvas=self, vert=vert, frag=frag)
        self.program.bind(model.vbo)
        
        for lyr in range(32):
            self.program['u_lyr_fg_color[%d]' % lyr] = self.pointColorDefault
            self.program['u_layerMap[%d]' % lyr] = lyr

        self.program['u_size'] = 1
        self.program['u_color'] = self.pointColor
        self.program['u_bg_color'] = (0.2, 0.2, 0.2, -0.6)
        self.program['u_antialias'] = u_antialias
        
        
        self.program_b = GlProgram(canvas=self, vert=vert, frag=fs,
                                   parent=self.program)
        self.program_b.bind(self.streetVbo)
        for lyr in range(32):
            self.program_b['u_lyr_fg_color[%d]' % lyr] = self.baseColorDefault
            self.program_b['u_layerMap[%d]' % lyr] = lyr

        self.program_b['u_size'] = 1
        self.program_b['u_color'] = (0.6, 0.6, 0.6, 1)
        self.program_b['u_bg_color'] = (0.2, 0.2, 0.2, -0.6)
        self.program_b['u_antialias'] = u_antialias
        self.program_b['u_offset'] = -0.05
        

        self.program_e = GlProgram(canvas=self, vert=vert, frag=fs,
                                   parent=self.program)
        self.program_e.bind(model.vbo)
        
        for lyr in range(32):
            self.program_e['u_lyr_fg_color[%d]' % lyr] = self.edgeColorDefault
            self.program_e['u_layerMap[%d]' % lyr] = lyr

        self.program_e['u_size'] = 1
        self.program_e['u_color'] = (0.2, 0.2, 0.8, -0.6)
        self.program_e['u_bg_color'] = (0.2, 0.2, 0.2, -0.6)
        self.program_e['u_antialias'] = u_antialias

        # Planes (Stack of translucent Rectangles)
        self.program_p = GlProgram(canvas=self, vert=vs_plane, frag=fs_plane,
                                   parent=self.program, heightOffset=-0.001)
        self.program_p.bind(self.planeVbo)

        color = np.random.uniform(0.5, 1, (32, 3))
        colorTable = np.hstack((color, np.ones((32, 1))))
        colorTable[:, 3] = 0.15

        for lyr in range(32):
            self.program_p['u_lyr_fg_color[%d]' % lyr] = colorTable[lyr]
            self.program_p['u_layerMap[%d]' % lyr] = lyr

        # Highlight

        self.highlight_e = GlProgram(canvas=self, vert=vert, frag=fs,
                                     parent=self.program, heightOffset=0.001)
        self.highlight_e.bind(model.vbo)

        for lyr in range(32):
            self.highlight_e[
                'u_lyr_fg_color[%d]' %
                lyr] = self.edgeColorDefault
            self.highlight_e['u_layerMap[%d]' % lyr] = lyr

        self.highlight_e['u_size'] = 1.5
        self.highlight_e['u_color'] = self.highlightColor
        self.highlight_e['u_bg_color'] = (1.0, 1.0, 1.0, 1)
        self.highlight_e['u_antialias'] = u_antialias

        self.updateFrustum(zoomIdx=0, size=self.size)

        self.loadLayerLabels()
        self.loadNodeLabels(self.model.edgeList)        
        
        self.updateView()

    def registerDependent(self, dep):
        '''
        Append dep to the dependents list of GL objects that are
        viewed in this canvas so that view and projection
        changes can be updated
        '''
        self.dependents.append(dep)

    def registerDependency(self, **kwargs):
        """
        Maintains a dictionary with key (e.g. 'u_view')
        and an attribute e.g. self.view
        """
        self.dependencies.update(kwargs)

    def updateDependents(self, *args):
        '''
        These attributes hide behind access methods.
        So we have to actively update them -
        '''

        depList = list(args)

        if len(depList) == 0 and self.dependencies:
            depList = list(self.dependencies.keys())

        # for dependency in depList:
        #   for dep in self.dependents:
        #       dep[dependency] = getattr(self, self.dependencies[dependency])

        for dep in self.dependents:
            dep['u_view'] = self.view
            dep['u_projection'] = self.projection
            dep['u_offset'] = self.offset

        if ('u_layerMap' in depList):
            for lyr, val in enumerate(self.layerMap):
                if val != self.savedLayerMap[lyr]:

                    for dep in self.dependents:
                        dep['u_layerMap[%d]' % lyr] = val
                    self.savedLayerMap[lyr] = val

    def loadLayerLabels(self):
        """
        Program labels all layer planes in their bottom left hand corner
        """
        # self.layerLabels = []
        pos = self.model.NormalizedBoundingBox[:2]

        labels = [lyr.name for lyr in self.model.layerList]
        coords = [(pos[0], pos[1], i) for i in range(len(labels))]

        model = rotate(90, (1, 0, 0))

        self.layerLabels = txt.textLabels(self, labels, coords,
                                           font_size=32,  model=model,
                                           anchor_x='left', anchor_y='bottom')

    def loadNodeLabels(self, edges):
        """
        Text labels for listed edges
        """
        vertices = list(set([v for e in edges for v in e.edgeId]))

        labels = [str(node.name) for node in vertices]
        layerIndices = np.array(self.model.networkLayers)
        
        coords = [node.coords for node in vertices]
        model = rotate(90, (0, 0, 1))
        
        if not hasattr(self,'nodeLabels') or self.nodeLabels is None:

            self.nodeLabels = txt.textLabels(self, labels, coords,
                                             font_size=5, billboard=True,
                                             heightOffset=-0.005, model=model,
                                             anchor_x='right', anchor_y='center')
        else:
            self.nodeLabels.texts = labels
            self.nodeLabels.coords = coords

    def updatePickVertices(self):
        """
        Transform the vertex locations to vec4 by looking up the layer
        offset, and augmenting with a 1.0
        """
        # Create an array of Vec4s  by adding a 1 at the end
        coords = np.array(self.model.vertexLocations)
        layerIndices = np.array(self.model.networkLayers)

        nl = len(coords)

        zCoords = np.array([self.layerZCoordinates[self.layerMap[int(i)]]
                            for i in layerIndices], dtype=np.float32)

        #zOffsets = np.array(coords[:, -1], dtype=np.float32)

        self.worldVertexPositions = np.hstack((coords[:, :-1],
                                               zCoords.reshape(nl, 1),
                                               np.ones((nl, 1))))


    def on_initialize(self, event):
        set_state(clear_color='white', depth_test=True, blend=True,
                  blend_func=('src_alpha', 'one_minus_src_alpha'))

    def zoomSpaceAndPan(self, **kwargs):
        '''
        Zoom in or out adjusting the pan so that the
        focus point X1, Y1, doesn't move on screen.
        Do not save in the undo / redo buffers
        '''
        worldFocus = kwargs.pop('worldFocus')
        spacing = kwargs.pop('spacing')

        # The target must not move in  projection space
        # Snap shot the current target position in eye coords

        zoomCenter = worldFocus.dot(self.view)
        pCenter = zoomCenter.dot(self.projection)

        #  Change the z coordinate of the focus object
        #  to reflect changes in layer spacing
        #
        # Get the layer number of the focus object

        if spacing != self.spacing:

            focusLayer = min(list(range(32)),
                             key=lambda i:
                             abs(self.wFocus[2] - self.layerZCoordinates[i]))

            # Change the layer spacing
            self.updateLayerSpacing(spacing)

            # Move the z-coordinate of the focus object
            #  to match its new layer position
            self.wFocus[2] = self.layerZCoordinates[focusLayer]

            self.updateView()

        if ('zoomRatio' in kwargs) or ('zoomIdx' in kwargs):
            self.updateFrustum(**kwargs)

        if (self.zoomRatio != self.savedZoomRatio or
                self.savedSpacing != self.spacing):

            invProjection = inv(self.projection)
            newCenter = pCenter.dot(invProjection)

            # Pan the difference between newCenter and zoomCenter
            Pan = newCenter - zoomCenter
            PanX, PanY, PanZ, PanW = Pan

            self.vFocus[0] += PanX
            self.vFocus[1] += PanY
            self.vFocus[2] += PanZ

            self.updateView()

    def zoomAndPan(self, **kwargs):
        '''
        Zoom in or out adjusting the pan so that the focus
        point X1, Y1, doesn't move on screen.

        Do not save in the undo / redo buffers
        '''
        zoomCenter = kwargs.pop('zoomCenter')

        self.savedZoomRatio = self.zoomRatio
        self.savedVFocus = self.vFocus.copy()
        self.savedWFocus = self.wFocus.copy()

        pCenter = zoomCenter.dot(self.projection)

        self.updateFrustum(**kwargs)

        if self.zoomRatio != self.savedZoomRatio:

            invProjection = inv(self.projection)
            newCenter = pCenter.dot(invProjection)

            if self.zoomIdx == 0:
                # At Full View, reset the view

                PanX = -self.view[3, 0]
                PanY = -self.view[3, 1]
                PanZ = -self.view[3, 2] + self.worldCenterZ
            else:
                # Pan the difference between newCenter and zoomCenter
                Pan = newCenter - zoomCenter
                PanX, PanY, PanZ, PanW = Pan
                print("Pan ", Pan)

            self.vFocus[0] += PanX
            self.vFocus[1] += PanY
            self.vFocus[2] += PanZ

            self.updateView()

    def saveZoomSpaceAndPan(self):
        '''
        Save the old and new state in the undo
        buffer
        '''

        if (self.zoomRatio != self.savedZoomRatio or
                self.savedSpacing != self.spacing or
                not np.array_equal(self.wFocus, self.savedWFocus) or
                not np.array_equal(self.vFocus, self.savedVFocus)):

            uState = (self.savedSpacing, self.savedZoomRatio,
                      self.savedWFocus, self.savedVFocus)
            rState = (self.spacing, self.zoomRatio, self.wFocus, self.vFocus)

            self.undo.save(operation=self._SpaceAndZoom,
                           rState=rState, uState=uState)

    def _SpaceAndZoom(self, state):
        '''
        Used by Undo and Redo:  Zoom and Pan
        as specified in state 3-tuple
        (zoomRatio, wFocus, vFocus)
        '''

        kwargs = dict(list(zip(['spacing', 'zoomRatio', 'wFocus', 'vFocus'], state)))
        self.updateSpacing(**kwargs)
        self.updateFrustum(**kwargs)
        self.updateView(**kwargs)

    _SpaceAndZoom.chainable = True

    def updateSpacing(self, **kwargs):
        if ('spacing' in kwargs):
            spacing = kwargs.pop('spacing')
            self.updateLayerSpacing(spacing)

    def saveZoomAndPan(self, **kwargs):
        '''
        Save the old and new state in the undo
        buffer
        '''

        if (self.zoomRatio != self.savedZoomRatio or
                not np.array_equal(self.wFocus, self.savedWFocus) or
                not np.array_equal(self.vFocus, self.savedVFocus)):

            uState = (self.savedZoomRatio, self.savedWFocus, self.savedVFocus)
            rState = (self.zoomRatio, self.wFocus, self.vFocus)

            self.undo.save(operation=self._Zoom, rState=rState, uState=uState)

            self.savedZoomRatio = self.zoomRatio
            self.savedZoomIdx = self.zoomIdx
            self.savedWFocus = self.wFocus.copy()
            self.savedVFocus = self.vFocus.copy()

    def _Zoom(self, state):
        '''
        Used by Undo and Redo:  Zoom and Pan
        as specified in state 3-tuple
        (zoomRatio, wFocus, vFocus)
        '''

        kwargs = dict(list(zip(['zoomRatio', 'wFocus', 'vFocus'], state)))
        self.updateFrustum(**kwargs)
        self.updateView(**kwargs)

    _Zoom.chainable = True

    def updateFrustum(self, **kwargs):

        if 'zoomRatio' in kwargs:
            self.zoomRatio = kwargs.pop('zoomRatio')

            self.zoomIdx = min(list(range(self.zoomLimit)),
                               key=lambda i:
                               abs(self.zoomRatio - self.zoomFactors[i]))

        elif 'zoomIdx' in kwargs:

            zoomIdx = kwargs['zoomIdx']
            if zoomIdx < 0:
                zoomIdx = 0
            elif zoomIdx >= self.zoomLimit:
                zoomIdx = self.zoomLimit - 1

            self.zoomIdx = zoomIdx
            self.zoomRatio = self.zoomFactors[zoomIdx]
        elif 'size' in kwargs:
            self.size = kwargs['size']
        else:
            raise ValueError

        width, height = self.size
        set_viewport(0, 0, width, height)

        x = 1.3 * self.near / self.worldCenterZ

        if float(height)/ width < 1.0:
            x /= float(height) / width

        self.zoomLeft = -x / self.zoomRatio
        self.zoomRight = -self.zoomLeft
        self.zoomBottom = self.zoomLeft * height / width
        self.zoomTop = -self.zoomBottom

        self.projection = frustum(self.zoomLeft, self.zoomRight,
                                  self.zoomBottom, self.zoomTop,
                                  -self.near, -self.far)

        self.program['u_projection'] = self.projection

    def on_resize(self, event):
        self.updateFrustum(size=event.size)
        self.updateView()

    def resize(self, width, height):
        self.updateFrustum(size=(width, height))
        self.updateView()

    def updateView(self, **kwargs):
        pass

    def GlFrameImage(self):

        buffer = glReadPixels(0, 0, self.Width, self.Height, GL_RGB, GL_UNSIGNED_BYTE)

        # Use PIL to convert raw RGB buffer and flip the right way up
        image = Image.fromstring(mode="RGB", size=(self.Width, self.Height), data=buffer)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)

        return image


    def OnAnimate(self):

        for frame in range(self.movie.endFrame()):
            for attr,val in self.movie.update(frame):
                setattr(self,attr,val)

            self.updateView(Animate = True)
            self.OnDraw()

            img = self.GlFrameImage()
            filename = "/tmp/nm%04d.png" % (frame)
            img.save(filename)


    def showHighlight(self, obj):

        if hasattr(obj, 'edgeId'):
            p1, p2 = obj.edgeId
            edgeList = []
            edgeList.append(obj.points)
            pointList = obj.points
            highlightEdge = np.array(edgeList, dtype=(np.uint32, 2))
            self.highlightIndex = gloo.IndexBuffer(highlightEdge)
            self.highlightHeightOffset = 0

            highlightPoints = np.array(pointList, dtype=(np.uint32, 1))
            self.highlightPointIndex = gloo.IndexBuffer(highlightPoints)
        elif hasattr(obj, 'index'):
            edgeList = []
            highlightEdge = np.array(edgeList, dtype=(np.uint32, 2))
            self.highlightIndex = gloo.IndexBuffer(highlightEdge)
            pointList = [obj.index]
            highlightPoints = np.array(pointList, dtype=(np.uint32, 1))
            self.highlightPointIndex = gloo.IndexBuffer(highlightPoints)
            self.highlightHeightOffset = 0

    def SelectAndNotify(self, obj, work, prot):
        self.Select(obj, work, prot)
        #self.parent.NotifySelected(obj, work, prot)

    def Reset (self):
        vFocus = np.array([0., 0., self.worldCenterZ, 1.],
                          dtype=np.float32)

        self.zoomAndPan(zoomCenter=vFocus, zoomIdx=0)
        self.saveZoomAndPan()

        W = P = None
        self.Select(self.highlighted, W, P)


    def Select(self, obj, work, prot):

        self.undo.save(operation=self._Select,
                       rState=(obj, work, prot),
                       uState=(self.selectedObject, self.work, self.prot))
        self.selectedObject, self.work, self.prot = obj, work, prot
        self._Select((obj, work, prot))

    def _Select(self, state):
        obj, work, prot = state
        self._setSelectedItem(obj)
        self._SetSelectionList((work, prot))

    def _setSelectedItem(self, obj):
        self.selectedObject = obj

        self.highlightIndex = None
        self.highlightPointIndex = None

        if obj is not None:

            self.highlight = obj

            if hasattr(obj,'edgeId'):
                p1, p2 = obj.points
                edgeList = []
                edgeList.append(obj.points)
                pointList = list(obj.points)
                selectedEdge = np.array(edgeList, dtype=(np.uint32, 2))
                self.selectedIndex = gloo.IndexBuffer(selectedEdge)
                selectedPoints = np.array(pointList, dtype=(np.uint32, 1))
                self.selectedPointIndex = gloo.IndexBuffer(selectedPoints)

            elif hasattr(obj, 'index'):
                edgeList = []
                selectedEdge = np.array(edgeList, dtype=(np.uint32, 2))
                self.selectedIndex = gloo.IndexBuffer(selectedEdge)
                pointList = [obj.index]
                selectedPoints = np.array(pointList, dtype=(np.uint32, 1))
                self.selectedPointIndex = gloo.IndexBuffer(selectedPoints)
                
            if self.highlight != self.highlighted:
                print(("Selector %s" % (self.highlight.name)))

                self.showHighlight(self.highlight)
                self.highlighted = self.highlight            
        else:
            self.selectedIndex = None
            self.selectedPointIndex = None
            
    def SetSelectionList(self, work, prot):
        self.undo.save(operation=self._SetSelectionList,
                       rState=(work, prot),
                       uState=(self.work, self.prot))
        self.work, self.prot = work, prot

    def _SetSelectionList(self, state):

        # print "Change selection:  Work %s   Prot %s " % (
        #   ", ".join(obj.name for obj in work),
        #   ", ".join(obj.name for obj in prot))

        self.work, self.prot = work, prot = state

        if work:
            self.workEdges = LoadSelectedEdges(work)
            self.workIndex = gloo.IndexBuffer(self.workEdges)
        else:
            self.workIndex = self.workTrailIndex = None

        if prot:
            self.protEdges, self.protTrails = LoadSelectedEdges(prot)
            self.protIndex = gloo.IndexBuffer(self.protEdges)
            self.protTrailIndex = gloo.IndexBuffer(self.protTrails)
        else:
            self.protIndex = self.protTrailIndex = None

        if work or prot:
            #verticals = SelectVerticals(work | prot)
            #self.sVerticalsIndex = gloo.IndexBuffer(verticals)

            points = SelectVertices(work | prot)
            self.sPointsIndex = gloo.IndexBuffer(points)

            self.loadNodeLabels(work | prot)

            #if isinstance(self.selectedObject, Vertex):
                #self.loadNodeLabels([self.selectedObject])
            #elif isinstance(self.selectedObject, Edge):
                #self.loadNodeLabels(self.selectedObject.points)

            #selectedTrails = set([t for t in work | prot
            #                      if isinstance(t, Trail)])

            #if selectedTrails:
            #    trailLinks = loadSelectedTrailLinks(selectedTrails)
            #    self.trailsLinksIndex = gloo.IndexBuffer(trailLinks)
            #else:
            #    self.trailsLinksIndex = None

            # Dim whatever's not selected by setting a to 0.1
            self.edgeColor = self.edgeColorDefault[:-1] + (0.1, )
            self.pointColor = self.pointColorDefault[:-1] + (0.1, )
            self.verticalColor = self.verticalColorDefault[:-1] + (0.1,)

            #self.activeLayers = list(set(e.layer.idx for e in work | prot))
            #layerMap = dict(list(zip(self.activeLayers,
                                #list(range(len(self.activeLayers))))))

            #self.layerMap = [layerMap.get(k, -2)
                             #for k in range(self.model.layerCount)]

        else:
            self.sVerticalsIndex = None
            self.edgeColor = self.edgeColorDefault
            self.pointColor = self.pointColorDefault
            self.verticalColor = self.verticalColorDefault

            self.activeLayers = list(range(self.model.layerCount))
            self.layerMap = list(range(32))

        self.updateDependents('u_layerMap')
        self.updateLayerSpacing()
        self.updatePickVertices()

        # Change the world focus to account for the change in layer positions

        self.wFocus = self.worldFocus()
        # Removing the next line prevents the selected object from moving
        self.vFocus = self.wFocus.dot(self.view)
        self.updateView()

    def on_draw(self, event):

        # pr = cProfile.Profile()
        # pr.enable()

        clear(color=True, depth=True)
        self.program_p.draw('triangles')

        if self.protIndex or self.workIndex:

            self.program_b['u_color'] = self.baseColorDefault

            self.program_b.draw('lines', self.baseIndex)
            #self.program_e.draw('lines', self.trailsLinksIndex)

            self.program_e['u_color'] = self.edgeColorDefault

            if self.workIndex:
                self.program_e.draw('lines', self.workIndex)
                #self.program_e['u_color'] = self.trailColor
                #self.program_e.draw('lines', self.workTrailIndex)

            if self.protIndex:
                self.program_e.draw('lines', self.protIndex)
                #self.program_e['u_color'] = self.trailColor
                #self.program_e.draw('lines', self.protTrailIndex)

            #self.program_e['u_color'] = self.verticalColorDefault
            #self.program_e.draw('lines', self.sVerticalsIndex)

            self.program['u_color'] = self.pointColorDefault
            self.program.draw('points', self.sPointsIndex)

            # pr = cProfile.Profile()
            # pr.enable()

            self.nodeLabels.draw()

            # pr.disable()
            # s = StringIO.StringIO()
            # sortby = 'cumulative'
            # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            # ps.print_stats()
            # print s.getvalue()

        else:
            
            self.program_b['u_size'] = 1
            self.program_b['u_color'] = self.baseColorDefault            
            self.program_b.draw('lines', self.baseIndex)            
            
            self.program_e['u_size'] = 1
            self.program_e['u_color'] = self.edgeColor
            self.program_e.draw('lines', self.edgesIndex)

            #self.program_e['u_color'] = self.trailColor
            #self.program_e.draw('lines', self.trailsIndex)

            self.program['u_color'] = self.pointColor
            self.program.draw('points', self.markerIndex)
            # self.nodeLabels.draw()

            #self.program_e['u_color'] = self.verticalColor
            #self.program_e.draw('lines', self.vindex)

        if self.selectedIndex is not None:
            self.highlight_e['u_color'] = self.selectedColor
            self.highlight_e.draw('lines', self.selectedIndex)
            self.program['u_color'] = self.selectedColor
            self.program.draw('points', self.selectedPointIndex)

        if self.highlightIndex is not None:
            self.highlight_e['u_color'] = self.highlightColor
            self.highlight_e.draw('lines', self.highlightIndex)
            self.program['u_color'] = self.highlightColor
            self.program.draw('points', self.highlightPointIndex)

        # pr = cProfile.Profile()
        # pr.enable()

        self.layerLabels.draw()
        
 
        # pr.disable()
        # s = Strion_drawngIO.StringIO()
        # sortby = 'cumulative'
        # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        # ps.print_stats()
        # print s.getvalue()

    def updateLayerSpacing(self, spacing=None):

        if spacing is None:
            spacing = self.spacing

        self.offset = spacing * 0.1 / len(self.activeLayers)

        # Save zCoordinates of all layers for pick operation
        self.layerZCoordinates = [lyr * self.offset - 1 for lyr in range(32)]

        # self.height = np.array(self.layerZCoordinates,
        # dtype=np.float32).reshape(-1,1)

        self.updateDependents('u_offset')

        # load the updated world vertex locations into the pick search array
        self.spacing = spacing

        self.updatePickVertices()

    def on_key_press(self, event):
        """
        Undo and Redo Keys
        """
        mods = event.modifiers

        if ((event.text == 'Z') and (CONTROL in mods)):
            operation, state, ck = self.undo.undo()
            operation(state)
            print("Undo")
            print("Spacing ", self.spacing)
            print("zRatio: ", self.zoomRatio)
            print("wFocus: ", self.wFocus)
            print("vFocus: ", self.vFocus)

        elif ((event.text == 'Y') and (CONTROL in mods)):
            operation, state, ck = self.undo.redo()
            operation(state)
            print("Undo")
            print("Spacing ", self.spacing)
            print("zRatio: ", self.zoomRatio)
            print("wFocus: ", self.wFocus)
            print("vFocus: ", self.vFocus)

        else:
            print("Key Press", event.key)

    def on_key_release(self, event):
        print("Key Release", event.key)

    def on_mouse_wheel(self, event):
        mods = event.modifiers

        if CONTROL in mods:
            '''
            Adjust the spacing between all layers. Pan to keep the same focus
            '''
            self.savedSpacing = self.spacing
            self.savedZoomRatio = self.zoomRatio
            self.savedVFocus = self.vFocus.copy()
            self.savedWFocus = self.wFocus.copy()

            spacing = self.spacing - event.delta[1]
            spacing = max(0, spacing)

            self.zoomSpaceAndPan(worldFocus=self.wFocus, spacing=spacing)
            self.saveZoomSpaceAndPan()

        else:
            if event.delta[1] > 0:
                zoomIdx = self.zoomIdx + 1
            else:
                zoomIdx = self.zoomIdx - 1

            self.zoomAndPan(zoomCenter=self.vFocus, zoomIdx=zoomIdx)
            self.saveZoomAndPan()

    def worldToPixelCoords(self, wPoint):

        mView = self.view.dot(self.projection)
        x, y, z, w = wPoint.dot(mView)

        return self.nearPlaneToPixelCoords(x / w, y / w)

    def viewToPixelCoords(self, vPoint):
        x, y, z, w = vPoint.dot(self.projection)
        # Convert from homogenous Coordinates to NDC
        # and then scale to pixels

        return self.nearPlaneToPixelCoords(x / w, y / w)

    def nearPlaneToPixelCoords(self, nx, ny):
        w, h = self.size
        x = w * (nx + 1) / 2
        y = h * (1.0 - ny) / 2

        return (x, y)

    def pixelToNearPlaneCoords(self, x, y):
        """
        Convert pixel coordinates to normalized
        coordinates in the range -1 to +1 in the
        frustum near plane.
        """

        w, h = self.size

        nx = 2.0 * x / w - 1
        ny = 1 - 2.0 * y / h

        return [nx, ny]

    def pixelToAngle(self, x, y):
        """
        Convert pixel coordinates to approx rotate angle.
        """
        rx, ry = self.size
        ax = (x / rx - 0.5) * 90
        ay = ((ry - y) / ry - 0.5) * 90
        return [ax, ay]

    def updateView(self, **kwargs):
        """
        View Transformation
        ===================
        What:

        Rotate the image around model cordinate wFocus.
        Pan the image so that world coordinate wFocus is located
        at view coordinate vFocus.

        How:

        The model is scaled to fit within a -1, to +1 cube

        Pan, zoom and rotate operations should all appear
         to occur around the 3d mouse focus point
         The full history of incremental pan, and rotate around
         different conceptually can be modeled as
         three consecutive operations:

         1-5.  Translate the model so that the focus point is at 0,0,0
         6.  Yaw the model around the Z access,
              then pitch the model up around x
         7.  Translate the model to locate the model aligned with the mouse
         8.  Projection with zoom

        View control involves the following variables;
                               Model   World    View   Projection   Screen

         Mouse                            2      3                     1
                                      ---------- 4 ----------
         Pan                                     5
         xRot, yRot                              6
         Pan                                     7
         zoom                                              8

         Mouse coordinates are transformed  between normalized-screen,
         View and world.

         1.  Read mouse screen coords
         2   Search for object and determine mouse coordinates in 3D
               world space - wFocus
         3.  Mulitply world coordinates by view matrix to get mouse
               position in view coordinates  vFocus
         4.   User input may changee Rot, Pan, Zoom and / or layer
                height spacing and mapping table.
         5.  Pan identity matrix by -wFocus, to move the whole
               world model so that the focus point is 0,0,0
         6.  Yaw then pitch View matrix to updated Rot values
         7.  Pan view matrix to return the focus point to updated vFocus
         8.  If zoom changed recalc projection matrix

        """

        self.__dict__.update(kwargs)

        # Yaw and Pitch
        self.yRot = min(self.yRot, 120)
        self.yRot = max(self.yRot, -120)
        self.xRot = min(self.xRot, 120)
        self.xRot = max(self.xRot, -120)

        # Prepare by moving the world so that the focus is at 0,0,0
        self.view = translate(-self.wFocus[0:3])                    # Step5

        # Yaw then pitch                                                      #
        # Step 6
        self.view = self.view.dot(rotate(
            self.yRot, [0, 0, 1])).dot(rotate(self.xRot, [1, 0, 0]))

        # Now translate so that focus is at required location
        self.view = self.view.dot(translate(self.vFocus[0:3]))  # Step 7

        self.updateDependents()

        # print self.view
        self.update()

    def _updateView(self, state):
        self.updateView(dict(list(zip(['wFocus', 'vFocus',
                                  'xRot', 'yRot', 'zoomIdx'],
                                 state))))

    def on_mouse_move(self, event):
        """Pan the view based on the change in mouse position."""
        if event.is_dragging and event.buttons[0] == 1:
            x0, y0 = event.last_event.pos[0], event.last_event.pos[1]
            x1, y1 = event.pos[0], event.pos[1]
            X0, Y0 = self.pixelToNearPlaneCoords(float(x0), float(y0))
            X1, Y1 = self.pixelToNearPlaneCoords(float(x1), float(y1))

            start = self.vFocus.dot(self.projection)
            invProjection = inv(self.projection)
            start = start.dot(translate((X1 - X0, Y1 - Y0, 0)))
            end = start.dot(invProjection)

            self.vFocus[0] = end[0]
            self.vFocus[1] = end[1]
            self.updateView()

        # Rotate and tilt the model
        elif event.is_dragging and event.buttons[0] == 0:
            self.view = np.eye(4, dtype=np.float32)
            x0, y0 = event.last_event.pos[0], event.last_event.pos[1]
            x1, y1 = event.pos[0], event.pos[1]
            XA0, YA0 = self.pixelToAngle(float(x0), float(y0))
            XA1, YA1 = self.pixelToAngle(float(x1), float(y1))

            self.xRot -= YA1 - YA0
            self.yRot += XA1 - XA0

            self.updateView()

        elif event.is_dragging and event.buttons[0] == 2:  # Zoom and space
            x0, y0 = event.last_event.pos[0], event.last_event.pos[1]
            x1, y1 = event.pos[0], event.pos[1]
            X0, Y0 = self.pixelToNearPlaneCoords(float(x0), float(y0))
            X1, Y1 = self.pixelToNearPlaneCoords(float(x1), float(y1))

            spacing = self.spacing + (Y1 - Y0 + X0 - X1) * self.spacingRatio
            spacing = max(0, spacing)

            zoomRatio = (self.zoomRatio *
                         self.zoomWheelRatio ** (self.zoomLimit * (X1 - X0)))

            zoomRatio = max(1.0, zoomRatio)
            zoomRatio = min(
                zoomRatio, self.zoomWheelRatio**(self.zoomLimit - 1))

            self.zoomSpaceAndPan(spacing=spacing,
                                 worldFocus=self.wZoomCenter,
                                 zoomRatio=zoomRatio)

        else:

            # Find objects under the mouse
            # Find edges and vertices that are close to a ray from the
            # camera (0,0,0) through the mouse position on the front
            # face of the frustum

            x, y = event.pos

            # Pick within +- p pixels.  Translating p to near plane coords
            p = 5
            tolerance = p * 2 * self.zoomRight / self.size[0]

            res = self.closestEdge(x, y, tol=tolerance)

            if res:
                edge, Z, t = res
                p1, p2 = edge.edgeId

                self.highlightT = t

                if t < 0.15:
                    self.highlight = p1
                elif t > 0.85:
                    self.highlight = p2
                else:
                    self.highlight = edge

                # Get the mouse hit coordinate in world coordinate space,
                # by linear interpolation between the end points of the edge

                self.wFocus = self.worldFocus(edge, t)

                X, Y = self.pixelToNearPlaneCoords(x, y)

                x1, y1 = self.worldToPixelCoords(self.wFocus)

                # and map the coordinate to view space (rotated and panned)
                self.vFocus = self.wFocus.dot(self.view)

                if self.highlight != self.highlighted:
                    print(("Mouseover %s %d %f" % (self.highlight.name, self.highlight.layerIndex, t)))

                    self.showHighlight(self.highlight)
                    self.highlighted = self.highlight

                    self.update()

    def worldFocus(self, obj=None, t=0.5):
        """
        Return the position in model space of on edge
        t is the fractional distance between the end-points
        """
        if obj is None:
            obj = self.highlight
            t = self.highlightT

        if obj is None:
            pos = ([0,0,0,1])
            
        elif hasattr(obj,'index' ):
            pos = self.worldVertexPositions[obj.index]

        elif hasattr(obj,'points'):
            p1, p2 = obj.points

            loc1 = self.worldVertexPositions[p1]
            loc2 = self.worldVertexPositions[p2]

            # Linear interpolation -
            pos = loc1 + (loc2 - loc1) * t
        else:
            pos = ([0,0,0,1])

        return pos

    def closestEdge(self, x, y, tol=0.01):
        """
        Finds an edge that is closest to a ray
        from camera through the mouse pointer
        """

        X1, Y1 = self.pixelToNearPlaneCoords(x, y)
        #  Perspective Transformation to get a normalized screen view
        mView = self.view.dot(self.projection)
        # Now adjust the matrix so that the normalized perspective
        # model is moved to mouse location 0,0
        mView = mView.dot(translate([-X1, -Y1, 0]))

        # Use Numpy to multiply all vertex locations by the mView matrix:
        vertices = self.worldVertexPositions.dot(mView)

        def searchForClosest(candidates):
            # Create an array that contains edge end point locations.
            # Return the edge that is closest to the mouse cursor,
            # and within tolerance tol

            coords = [(vertices[l.points[0]][:3],
                       vertices[l.points[1]][:3])
                      for l in candidates]
            res = picker.pickEdge(coords, tol=tol)

            if res:
                closest, z, t = res
                return (candidates[closest], z, t)

        if self.work or self.prot:
            links = list(set(self.model.edgeList)
                         & (self.work | self.prot))
            return searchForClosest(links)

        else:
            return searchForClosest(self.model.edgeList)

    def on_mouse_press(self, event):
        if event.button == 0:
            now = time.time()
            # print "Press Time %f %f" % ( now , now-self.ClickStart)

            if now - self.ClickStart < 0.6:
                self.DCTime = now

            self.ClickStart = now

        elif event.button == 2:

            print("Mouse press:")
            print("Spacing ", self.spacing)
            print("zRatio: ", self.zoomRatio)
            print("wFocus: ", self.wFocus)
            print("vFocus: ", self.vFocus)

            self.savedSpacing = self.spacing
            self.savedZoomRatio = self.zoomRatio
            self.savedVFocus = self.vFocus.copy()
            self.savedWFocus = self.wFocus.copy()
            self.wZoomCenter = self.wFocus

    def on_mouse_release(self, event):

        if event.button == 0:
            now = self.clickStop = time.time()
            # print "Release Time %f %f" % ( now , now-self.ClickStart)
            x, y = event.pos
            x1, y1 = self.worldToPixelCoords(self.wFocus)
            d = hypot(x - x1, y - y1)

            if self.highlighted:
                if d < 20:
                    if ((now - self.ClickStart) < 0.25):
                        # Mouse Left-Click
                        W=P=[]
                        W, P = self.model.network.ClientsAndServers(self.highlighted)
                        # print "Selected %s: %s  " % (self.highlighted.name ,
                        #          ", ".join(e.name for e in W | P) )
                        self.SelectAndNotify(self.highlighted, W, P)

                    #if ((now - self.DCTime) < 0.35):
                        ## Double Click handler might open a popup menu
                        ## related to the highlighted item
                        #self.popupMenu(self.highlighted)

                elif ((now - self.ClickStart) < 0.25):
                    # click with distance between
                    # mouse and selected item more than 20 pixels
                    W = P = None
                    self.SelectAndNotify(self.highlighted, W, P)

                else:
                    # End of left mouse drag operation
                    # Save Rotation in the undo buffer
                    if ((self.xRot != self.savedXRot)
                       or (self.yRot != self.savedYRot)):
                        uState = (self.savedXRot, self.savedYRot,
                                  self.savedWFocus, self.savedVFocus)
                        rState = (self.xRot, self.yRot, self.wFocus,
                                  self.vFocus)
                        self.undo.save(operation=self._Rotate,
                                       uState=uState, rState=rState)
                        self.savedXRot = self.xRot
                        self.savedYRot = self.yRot
                        self.savedWFocus = self.wFocus.copy()
                        self.savedVFocus = self.vFocus.copy()

        elif event.button == 1:    # Middle Button - Pan
            if not np.array_equal(self.vFocus, self.savedVFocus):

                self.undo.save(operation=self._Pan,
                               rState=(self.wFocus, self.vFocus),
                               uState=(self.savedWFocus, self.savedVFocus))

                self.savedVFocus = self.vFocus.copy()
                self.savedWFocus = self.wFocus.copy()

        elif event.button == 2:
            print("Mouse Release:")
            print("Spacing ", self.spacing, self.savedSpacing)
            print("zRatio: ", self.zoomRatio, self.savedZoomRatio)
            print("wFocus: ", self.wFocus, self.savedWFocus)
            print("vFocus: ", self.vFocus, self.savedVFocus)

            self.saveZoomSpaceAndPan()

    def _Rotate(self, state):
        self.xRot, self.yRot, self.wFocus, self.vFocus = state
        self.updateView()

    _Rotate.chainable = True

    def _Pan(self, state):
        self.wFocus, self.vFocus = state
        self.updateView()

    _Pan.chainable = True

    def popupMenu(self, item):
        #  "Call to popup menu goes here"
        print("Selected %s: %s  " % (
            self.highlighted.name,
            ", ".join(e.name for e in self.work | self.prot)))


class NetworkModel3D(object):
    """
    Class that contains model accessors for graphical rendering of
    a multilayer network.
    """

    def __init__(self, **kwargs):

        network = self.network = kwargs.pop('network')
        vertices = kwargs.pop('vertices')
        self.viewLayers = kwargs.pop('viewLayers', None)

        data = vertexBufferObj(vertices)
        self.vbo = gloo.VertexBuffer(data)

        self.markers = np.array(nodeLayerIndices(network.nodeList),
                           dtype=np.uint32)
        self.edgeList=network.edgeList
        
        edgeEnds = [edge.points for edge in network.edgeList]
        self.edges = np.array(edgeEnds, dtype=(np.uint32, 2))
      
        # self.edgeArray = network.edgeArray

        if self.viewLayers is None:
            self.layerCount = network.layerCount
            self.layerList = network.LayerList
        else:
            self.layerList = list(set(network.LayerList)&set(self.viewLayers))
            self.layerCount = len (self.layerList)

        self.vertexLocations = network.vertexLocations
        self.networkLayers = network.networkLayers

        self.NormalizedBoundingBox = network.NormalizedBoundingBox

class Panel3D(wx.Panel):

    def __init__(self, parent, net, viewLayers=None):

        self.net = net
        self.parent = parent
        wx.Panel.__init__(self, parent, -1)

        self.frameSize = parent.Size

        self.model = NetworkModel3D(network=net, viewLayers=viewLayers, vertices=net.nodeList)

        self.canvas = Canvas(app="wx", parent=self, model=self.model, position=(0,0))

        self.Bind(wx.EVT_SIZE, self.OnSize)

        self.canvas.native.Show()

    def OnSize(self, event):

        w, h = event.GetSize()
        self.canvas.resize(w, h)

    def Select(self, obj, work, prot):
        self.canvas.Select(obj, work, prot)

    def NotifySelected(self, obj, work, prot):
        print("Notify Selected Objects ...{}".format(obj))
        ##self.parent.Set2DSelectionList(work, prot)
        ##self.parent.context.SetItems([obj])

    def setAnimator(self, keyFrames=None):
        self.canvas.setAnimator()

    def OnAnimate(self):
        self.canvas.OnAnimate()

    def StartAnimation(self, event):
        self.canvas.StartAnimation(event)

    def AddKeyFrame(self, event):
        self.canvas.AddKeyFrame(event)

    def CloseAnimation(self, event):
        self.canvas.CloseAnimation()

    def Reset(self):
        self.canvas.Reset()


class TestFrame(wx.Frame):

    def __init__(self):
        wx.Frame.__init__(self, None, -1, "WX Test",
                          wx.DefaultPosition, size=(800, 800))

        MenuBar = wx.MenuBar()
        file_menu = wx.Menu()
        file_menu.Append(wx.ID_EXIT, "&Quit")
        self.Bind(wx.EVT_MENU, self.on_quit, id=wx.ID_EXIT)
        MenuBar.Append(file_menu, "&File")
        self.SetMenuBar(MenuBar)
        net = Network()

        self.panel = Panel3D(self, net)

    def on_quit(self, event):
        self.Close(True)

if __name__ == '__main__':
    myapp = wx.App(0)

    frame = TestFrame()

    frame.Show(True)
    myapp.MainLoop()


# if __name__ == '__main__':

    # gloo.gl.use_gl('desktop debug')
    # net = Network(filename= 'DT4.js')
    # c = Canvas(title="Graph", network=net)

    # c.show()
    # app.run()
