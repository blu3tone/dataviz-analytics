

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
from vispy.gloo import set_viewport, set_state, clear, read_pixels
from vispy.util.transforms import frustum, translate, rotate
from vispy.util.keys import CONTROL

from PIL import Image

import textProgram as txt
from network import Network

from undo import UndoBuffer

from graphprogram import verticalsProgram, edgesProgram, nodesProgram, GlProgram, geojsonProgram, planesProgram

from logreader import edgeMovementsSequence, WapAssociationCountSequence, evtime2String





class Canvas(app.Canvas):

    def __init__(self, **kwargs):
        # Initialize the canvas for real
        u_antialias = 1
        self.hourAnimator=0
        self.parent = kwargs['parent']

        self.model = model = kwargs.pop('model')
        self.network = model.network
        
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
        self.layerMap = list(range(32))
        self.savedLayerMap = list(self.layerMap)

        self.baseColorDefault = (0.8, 0.8, 0.8, 1)
        self.layerList=self.model.network.LayerList

        self.pitch = self.view = np.eye(4, dtype=np.float32)
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

        self.undo = UndoBuffer(operation=self._updateView,
                               rState=(self.wFocus, self.vFocus,
                                       self.xRot, self.yRot, self.zoomIdx),
                               uState=(self.wFocus, self.vFocus,
                                       self.xRot, self.yRot, self.zoomIdx))

        self.view = self.view.dot(translate(self.vFocus[0:3]))

        self.registerDependency(u_view='view',
                                u_pitch='pitch',
                                u_projection='projection',
                                u_offset='offset')

        self.markers = nodesProgram(self.network.nodeList, canvas=self)
        self.basemap = geojsonProgram(canvas=self, 
                                      centLat=self.network.centLat, 
                                      centLong=self.network.centLong, 
                                      scale=self.network.scale)
        self.edges = edgesProgram(self.network.edgeList, canvas=self, parent=self.markers)
        self.planes = planesProgram(canvas=self, parent=self.markers, heightOffset=-0.001,
                                    network=self.network)
     
        self.updateFrustum(zoomIdx=0, size=self.size)

        self.updateLayerSpacing(self.spacing)
        self.loadLayerLabels()

        self.loadNodeLabels(self.network.nodeList) 
       
        self.verticals = verticalsProgram(canvas=self, nodeData=self.markers.nodeData, 
                                          parent=self.markers)

        self.selectedEdges=[]
        self.selectedObject=None
        self.highlight=self.highlighted=None
        
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
            if 'u_pitch' in dep._user_variables:
                dep['u_pitch'] = self.pitch

        if ('u_layerMap' in depList):
            for lyr, val in enumerate(self.layerMap):
                if val != self.savedLayerMap[lyr]:

                    for dep in self.dependents:
                        dep['u_layerMap[%d]' % lyr] = val
                    self.savedLayerMap[lyr] = val
            
                    
        for dep in self.dependents:
            if '_updateWorldCoordinates' in dep.__dict__:
                dep._updateWorldCoordinates()

    def loadLayerLabels(self):
        """
        Program labels all layer planes in their bottom left hand corner
        """
        # self.layerLabels = []
        pos = self.model.NormalizedBoundingBox[:2]

        labels = [lyr.name for lyr in self.model.layerList]
        coords = [(pos[0], pos[1], 0) for i in range(len(labels))]
        layers = [lyr.index for lyr in self.model.layerList]

        model = rotate(90, (1, 0, 0))

        self.layerLabels = txt.textLabels(self, labels, coords, layers,
                                           font_size=32,   billboard="sphere",
                                           anchor_x='center', anchor_y='bottom')

    def loadNodeLabels(self, nodes, force=False):
        """
        Text labels for listed nodes
        """
        
        labels = [str(node.name) for node in nodes]
        coords = [(node.coords[0], node.coords[1], 0) for node in nodes]
        layers = [node.layerIndex for node in nodes]
        
        if force or not hasattr(self,'nodeLabels') or self.nodeLabels is None:
            model = rotate(90, (0, 0, 1)).dot(rotate(90,(1,0,0)))
            self.nodeLabels = txt.textLabels(self, labels, coords, layers,
                                             font_size=8, billboard="cylinder",
                                             heightOffset=-0.005, model=model,
                                             anchor_x='right', anchor_y='center')
        else:
            self.nodeLabels.texts = labels
            self.nodeLabels.coords = coords
            self.nodeLabels.layers = layers
            
    def updateNodeLabelZCoordinates(self, zLocs):
        pass
        #self.nodeLabels.zCoords= zLocs

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

        self.markers['u_projection'] = self.projection

    def on_resize(self, event):
        self.updateFrustum(size=event.size)
        self.updateView()

    def resize(self, width, height):
        self.updateFrustum(size=(width, height))
        self.updateView()

    def updateView(self, **kwargs):
        pass

    def GlFrameImage(self):
        buffer = read_pixels()
        image = Image.fromarray(buffer, mode="RGBA")
        return image

    def clientCountAnimation(self):
        self.apAssociationCounts, self.apFrameStart, self.apFrameStep = WapAssociationCountSequence(self.network.nodeList, hours=range(24), 
                                                                                            days=range(7), period='days')
        
        self.edgeMovementCounts, self.frameStart, self.frameStep = edgeMovementsSequence(self.network.edgeList, hours=range(24), 
                                    days=range(7), period='days')
        
        self.maxFrame=len(self.edgeMovementCounts[:,0]) - 1
        self.frame=0
        self._showFrame(self.frame)
            
    def nextFrame(self):
        self.frame += 1
        if self.frame > self.maxFrame: self.frame = self.maxFrame
        self._showFrame(self.frame)      
        
    def prevFrame(self):
        self.frame -= 1
        if self.frame < 0: self.frame = 0
        self._showFrame(self.frame)              
        
    def _showFrame(self,frame):
        
        # Plots represent data in the z dimension over a gis map.
        # The plot uses 3d Equivents of bar charts, line charts and point cloud data.
        
        # An example, the number of wifi uses can be plotted using a bar or stacked bar. 
        # where the components of the stack may for divide the user cound based on 
        # some classification like duration.
        
        # A line chart may be used to indicate traffic between wireless access points
        # The color or width of the line encodes the number of users that move between the locations
        
        # A point cloud can indicate other location based data - for example Taxi / Uber drop off location
        # vs trip distance
        
        # This version of code animates over time.  The frames might model data for a sequence of days.
        
        self.markers.setHighlight(None)
        self.edges.setHighlight(None)
        
        self.frameTime = self.frameStart + frame*self.frameStep
    
        frameTimeString = evtime2String(self.frameTime)
        self.layerLabels.texts=[frameTimeString]
    
        print ("Frame {} {}".format(frame, frameTimeString))


        self.edges.updateAttributes(self.edgeMovementCounts[frame])
        self.markers.updateAttributes(self.apAssociationCounts[frame])
       
        self.verticals.updateAttributes(self.markers.nodeData['a_zAdjust'])
        self.loadNodeLabels(self.markers.nodeList)
        
        self.on_draw()
        self.swap_buffers()  
    
        img = self.GlFrameImage()
        
        img.load()  
        background = Image.new('RGB', img.size, color=(255,255,255))
        background.paste(img, mask=img.split()[3])  # 3 is the alpha channel        
        
        filename = "/tmp/dg%04d.png" % (frame)
        background.save(filename)   
        
    def OnAnimate(self):

        for frame in range(self.movie.endFrame()):
            for attr,val in self.movie.update(frame):
                setattr(self,attr,val)

            self.updateView(Animate = True)
            self.on_draw()

            img = self.GlFrameImage()
            filename = "/tmp/nm%04d.png" % (frame)
            img.save(filename)


    def showHighlight(self, obj):

        if hasattr(obj, 'edgeId'):
            self.edges.setHighlight(obj)
           
        elif hasattr(obj, 'index'):
            self.markers.setHighlight(obj)
            
    def SelectAndNotify(self, obj, selectedEdges):
        self.Select(obj, selectedEdges)
        
        # Let the higher order app know so that it can update other frames
        # self.parent.NotifySelected(obj, selectedEdges)

    def Reset (self):
        vFocus = np.array([0., 0., self.worldCenterZ, 1.],
                          dtype=np.float32)

        self.zoomAndPan(zoomCenter=vFocus, zoomIdx=0)
        self.saveZoomAndPan()

        selectedEdges = None
        self.Select(self.highlighted, selectedEdges)

    def Select(self, obj, selectedEdges):

        self.undo.save(operation=self._Select,
                       rState=(obj, selectedEdges),
                       uState=(self.selectedObject, self.selectedEdges))
        self.selectedObject, self.selectedEdges = obj, selectedEdges
        self._Select(obj, selectedEdges)

    def _Select(self, obj, selectedEdges):
        self._setSelectedItem(obj)
        self._SetSelectionList(selectedEdges)

    def _setSelectedItem(self, obj):
        self.selectedObject = obj
        if obj is not None:
            self.highlight = obj

            if hasattr(obj,'edgeId'):
                self.edges.setSelectedItem(obj)
                self.markers.setSelectedEdge(obj)   
                
            elif hasattr(obj, 'index'):
                self.markers.setSelectedNode(obj)
                self.edges.unselect()
        else:
            self.edges.unselect()
            self.markers.unselect()
            self.verticals.unselect()
  
    def selectVertices(self,selectedEdges):
        
        points = list(set([n.index for edge in selectedEdges for n in edge.nodes]))
        
        return np.array(points,dtype=np.uint32)

    def _SetSelectionList(self, selectedEdges):
        self.selectedEdges = selectedEdges 
        
        self.edges.setSelectionList(selectedEdges)
        self.markers.setSelectionList(selectedEdges)

        if selectedEdges:
            # To be refactored...
            points = self.selectVertices(selectedEdges )
            self.sPointsIndex = gloo.IndexBuffer(points)
            self.selectedVerticalsIndex = self.verticals.select(points)

            nodes = sorted((set(n for edge in selectedEdges for n in edge.nodes)), key=lambda x: x.index)
            self.loadNodeLabels(nodes, force=True)

        self.updateDependents('u_layerMap')
        self.updateLayerSpacing()

        # Change the world focus to account for the change in layer positions

        self.wFocus = self.worldFocus()
        # Removing the next line prevents the selected object from moving
        self.vFocus = self.wFocus.dot(self.view)
        self.updateView()

    def on_draw(self, event=None):
        # pr = cProfile.Profile()
        # pr.enable()

        clear(color=True, depth=True)

        self.nodeLabels.draw()
        self.basemap.draw()
        self.planes.draw()
        
        self.verticals.draw()
        self.edges.draw()
        self.markers.draw()
        
        self.layerLabels.draw()


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

        elif (event.text == 'G'):
            print("Client Count Animation")
            self.clientCountAnimation()

        elif (event.text == 'N'):
            self.nextFrame()
            
        elif (event.text == 'P'):
            self.prevFrame()        

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
        self.pitch  = self.view = translate(-self.wFocus[0:3])                    # Step5

        # Yaw then pitch                                                      #
        # Step 6
        
        self.view = self.view.dot(rotate(
            self.yRot, [0, 0, 1])).dot(rotate(self.xRot, [1, 0, 0]))


        # For cylindrical Billboards adjust the pitch of the label to 
        # match the camera, but maintain fixed roll and yaw.   The 
        # effect is counter intuitive - by tracking pitch, the label appears 
        # to maintain a constant pitch wrt the rest of the model, like for example 
        # a lighthouse standing vertical on a plane.  By not adjusting yaw we get 
        # cylindrical effect - in the yaw dimension the label always faces the camera:
        
        self.pitch = self.pitch.dot(rotate(self.xRot, [1, 0, 0]))      
        
        # Now translate so that focus is at required location
        self.view = self.view.dot(translate(self.vFocus[0:3]))  # Step 7
        self.pitch = self.pitch.dot(translate(self.vFocus[0:3]))

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
  
            res = self.markers.closestNode(x, y, tol=tolerance)
            if res:
                node, Z = res
                t=0
                self.markers.setHighlight(node)
                self.edges.setHighlight(None)
                self.highlight=node
                self.wFocus=self.worldFocus(node)
            else:
                res = self.edges.closestEdge(x, y, tol=tolerance)
                if res:
                    edge, Z, t = res
                    p1, p2 = edge.edgeId
    
                    self.highlightT = t
                    
                    
                    if 0.15 < t < 0.85:
                        self.highlight=edge
                        self.markers.setHighlight(None)
                        self.edges.setHighlight(edge)
                        
                        # Get the mouse hit coordinate in world coordinate space,
                        # by linear interpolation between the end points of the edge
        
                    self.wFocus = self.worldFocus(edge, t)
                else:
                    
                    return

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
            pos =  np.array([0., 0., 0., 1.], dtype=np.float32)
            
        elif hasattr(obj,'index' ):
            pos = self.markers.worldVertexPositions[obj.index]

        elif hasattr(obj,'points'):
            p1, p2 = obj.points

            loc1 = self.edges.worldVertexPositions[p1]
            loc2 = self.edges.worldVertexPositions[p2]

            # Linear interpolation -
            pos = loc1 + (loc2 - loc1) * t
        else:
            pos =  np.array([0., 0., 0., 1.], dtype=np.float32)

        return pos


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
                        selectedEdgeList = self.model.network.ClientsAndServers(self.highlighted)
                        # print "Selected %s: %s  " % (self.highlighted.name ,
                        #          ", ".join(e.name for e in selectedEdges | P) )
                        self.SelectAndNotify(self.highlighted, selectedEdgeList)

                    #if ((now - self.DCTime) < 0.35):
                        ## Double Click handler might open a popup menu
                        ## related to the highlighted item
                        #self.popupMenu(self.highlighted)

                elif ((now - self.ClickStart) < 0.25):
                    # click with distance between
                    # mouse and selected item more than 20 pixels
                    selectedEdgeList = None
                    self.SelectAndNotify(None, selectedEdgeList)

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
            ", ".join(e.name for e in self.selectedEdges )))


class NetworkModel3D(object):
    """
    Class that contains model accessors for graphical rendering of
    a multilayer network.
    """

    def __init__(self, **kwargs):

        network = self.network = kwargs.pop('network')
        vertices = kwargs.pop('vertices')
        edgeVertices = kwargs.pop('edgeVertices')
        self.viewLayers = kwargs.pop('viewLayers', None)
        #self.edgeList=network.edgeList
        #self.nodeList=network.nodeList
 

        if self.viewLayers is None:
            self.layerCount = network.layerCount
            self.layerList = network.LayerList
        else:
            self.layerList = list(set(network.LayerList)&set(self.viewLayers))
            self.layerCount = len (self.layerList)

        self.networkLayers = network.networkLayers

        self.NormalizedBoundingBox = network.NormalizedBoundingBox

class Panel3D(wx.Panel):

    def __init__(self, parent, net, viewLayers=None):

        self.net = net
        self.parent = parent
        wx.Panel.__init__(self, parent, -1)

        self.frameSize = parent.Size

        self.model = NetworkModel3D(network=net, 
                                    viewLayers=viewLayers, 
                                    vertices=net.nodeList,
                                    edgeVertices=net.endPointList)

        self.canvas = Canvas(app="wx", parent=self, model=self.model, position=(0,0))

        self.Bind(wx.EVT_SIZE, self.OnSize)
        
        self.canvas.native.Show()

    def OnSize(self, event):

        w, h = event.GetSize()
        self.canvas.resize(w, h)

    def Select(self, obj, selectedEdges):
        self.canvas.Select(obj, selectedEdges)

    def NotifySelected(self, obj, selectedEdges):
        print("Notify Selected Objects ...{}".format(obj))
        ##self.parent.Set2DSelectionList(selectedEdges)
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


