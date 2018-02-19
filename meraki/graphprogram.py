import numpy as np
from vispy import gloo


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





class graphProgram(gloo.Program):
    
    def __init__(self, **kwargs):
        ''' 
        Create vertical lines between the plane and the node
        Read the xyz location from the a_position array
        For each x,y choose the maximum z
        Create vertices at x,y,zmax and x,y,0
        Create an index table that we will use to draw lines between these points
    
        '''
        super(graphProgram, self).__init__(vert=vert, frag=fs)
        
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

    def updateAttributes(self, nodeZPositions):
        
        self.nodeData['a_zAdjust'][:self.apCount] = nodeZPositions
        #self['a_zAdjust'][:self.apCount] = nodeZPositions
        # self['a_zAdjust'] = np.concatenate(nodeZPositions, np.zeros(self.apCount, dtype=np.float32))
        vbo= gloo.VertexBuffer(self.nodeData)
        self.bind(vbo)
        

        
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
            return super(verticalsProgram,self).__getitem__(name)

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
        
        
        #self.aData = np.zeros(2*nl, dtype=[('a_position', np.float32, 3),
                                     #('a_layer', np.float32, 1),
                                     #('a_fg_color', np.float32, 4),
                                     #('a_bg_color', np.float32, 4),
                                     #('a_size', np.float32, 1),
                                     #('a_linewidth', np.float32, 1),
                                     #('a_zAdjust', np.float32, 1),
                                     #('a_colorAdjust', np.float32, 1)
                                     #])
        

        #Double up node data
        # Double up the vertices.  The second half has z Adjust values of zero,
        # The first half has z values proportional to the attribute being graphed

        aData = np.concatenate((nodeData,nodeData),axis=0)  
        aData['a_position'][:,2]=0.
        aData['a_zAdjust'][nl:]=0
        
        kwargs['nodeData']= aData
        
        super(verticalsProgram, self).__init__(**kwargs)
          
        #self['a_zAdjust'] = aData['a_zAdjust']
        #self['a_zAdjust'][nl:] = np.zeros(nl,dtype=np.float32)
        #self['a_colorAdjust'] = aData['a_colorAdjust']
      
        self.Index = self.IndexBuffer([(x,x+nl) for x in range(nl)])
        self.selectIndex = None
      
      
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
            super(graphProgram,self).draw('lines', self.selectIndex)
        else:
            super(graphProgram,self).draw('lines', self.Index)


