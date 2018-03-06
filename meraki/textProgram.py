# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2014, Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------

##############################################################################
# Load font into texture

from __future__ import division

import numpy as np
from copy import deepcopy
from os import path as op
import sys

from sdf import SDFRenderer
from vispy.gloo import (TextureAtlas, set_state, IndexBuffer, VertexBuffer,
                     set_viewport)
from vispy.gloo import gl
from vispy.gloo.wrappers import _check_valid
from vispy.ext.six import string_types
from vispy.util.fonts import _load_glyph

from vispy.gloo import Program
#from ..shaders import ModularProgram

from vispy.util.transforms import scale

from vispy.color import Color
#from ..visual import Visual
from vispy.io import _data_dir

try:
    import StringIO
except:
    from io import StringIO
import cProfile
import pstats

class TextureFont(object):
    """Gather a set of glyphs relative to a given font name and size

    Parameters
    ----------
    font : dict
        Dict with entries "face", "size", "bold", "italic".
    renderer : instance of SDFRenderer
        SDF renderer to use.
    """
    def __init__(self, font, renderer):
        self._atlas = TextureAtlas()
        self._atlas.wrapping = 'clamp_to_edge'
        self._kernel = np.load(op.join(_data_dir, 'spatial-filters.npy'))
        self._renderer = renderer
        self._font = deepcopy(font)
        self._font['size'] = 256  # use high resolution point size for SDF
        self._lowres_size = 64  # end at this point size for storage
        assert (self._font['size'] % self._lowres_size) == 0
        # spread/border at the high-res for SDF calculation; must be chosen
        # relative to fragment_insert.glsl multiplication factor to ensure we
        # get to zero at the edges of characters
        self._spread = 32
        assert self._spread % self.ratio == 0
        self._glyphs = {}

    @property
    def ratio(self):
        """Ratio of the initial high-res to final stored low-res glyph"""
        return self._font['size'] // self._lowres_size

    @property
    def slop(self):
        """Extra space along each glyph edge due to SDF borders"""
        return self._spread // self.ratio

    def __getitem__(self, char):
        if not (isinstance(char, string_types) and len(char) == 1):
            raise TypeError('index must be a 1-character string')
        if char not in self._glyphs:
            self._load_char(char)
        return self._glyphs[char]

    def _load_char(self, char):
        """Build and store a glyph corresponding to an individual character

        Parameters:
        -----------
        char : str
            A single character to be represented.
        """
        assert isinstance(char, string_types) and len(char) == 1
        assert char not in self._glyphs
        # load new glyph data from font
        _load_glyph(self._font, char, self._glyphs)
        # put new glyph into the texture
        glyph = self._glyphs[char]
        bitmap = glyph['bitmap']

        # convert to padded array
        data = np.zeros((bitmap.shape[0] + 2*self._spread,
                         bitmap.shape[1] + 2*self._spread), np.uint8)
        data[self._spread:-self._spread, self._spread:-self._spread] = bitmap

        # Store, while scaling down to proper size
        height = data.shape[0] // self.ratio
        width = data.shape[1] // self.ratio
        region = self._atlas.get_free_region(width + 2, height + 2)
        if region is None:
            raise RuntimeError('Cannot store glyph')
        x, y, w, h = region
        x, y, w, h = x + 1, y + 1, w - 2, h - 2

        self._renderer.render_to_texture(data, self._atlas, (x, y), (w, h))
        u0 = x / float(self._atlas.shape[1])
        v0 = y / float(self._atlas.shape[0])
        u1 = (x+w) / float(self._atlas.shape[1])
        v1 = (y+h) / float(self._atlas.shape[0])
        texcoords = (u0, v0, u1, v1)
        glyph.update(dict(size=(w, h), texcoords=texcoords))


class FontManager(object):
    """Helper to create TextureFont instances and reuse them when possible"""
    # todo: store a font-manager on each context,
    # or let TextureFont use a TextureAtlas for each context
    def __init__(self):
        self._fonts = {}
        self._renderer = SDFRenderer()

    def get_font(self, face, bold=False, italic=False):
        """Get a font described by face and size"""
        key = '%s-%s-%s' % (face, bold, italic)
        if key not in self._fonts:
            font = dict(face=face, bold=bold, italic=italic)
            self._fonts[key] = TextureFont(font, self._renderer)
        return self._fonts[key]

##############################################################################


def text_to_vbo(texts, coords, layers, font, anchor_x, anchor_y, lowres_size):
    """Convert text characters to VBO"""
    text_vtype = np.dtype([('a_position', 'f4', 3),
                           ('a_layer', 'f4', 1),
                           ('a_anchor', 'f4', 3),
                           ('a_texcoord', 'f4', 3)])

    chars = sum(len(text) for text in texts)
    vertices = np.zeros(chars * 4, dtype=text_vtype)
    prev = None

    ratio, slop = 1. / font.ratio, font.slop

    # Need to make sure we have a unicode string here (Py2.7 mis-interprets
    # characters like "•" otherwise)

    orig_viewport = gl.glGetParameter(gl.GL_VIEWPORT)

    if sys.version[0] == '2' :
        for ii, text in enumerate(texts):
            if isinstance(text, str):
                texts[ii] = text.decode('utf-8')

    for char in 'hy':
        glyph = font[char]
        y0 = glyph['offset'][1] * ratio + slop
        y1 = y0 - glyph['size'][1]
        hyAscender = y0 - slop
        hyDescender = y1 + slop
        hyHeight = glyph['size'][1] - 2*slop

    wi = 0                                      # Word Index
    for idx,text in enumerate(texts):
        x_off = -slop
        width = height = 0
        ascender = hyAscender
        descender = hyDescender

        anchor = coords[idx]
        layer = layers[idx]

        for ii, char in enumerate(text):
            glyph = font[char]
            kerning = glyph['kerning'].get(prev, 0.) * ratio
            x0 = x_off + glyph['offset'][0] * ratio + kerning
            y0 = glyph['offset'][1] * ratio + slop
            x1 = x0 + glyph['size'][0]
            y1 = y0 - glyph['size'][1]
            u0, v0, u1, v1 = glyph['texcoords']
            position = [[x0, y0, 0], [x0, y1, 0], [x1, y1, 0], [x1, y0, 0]]
            texcoords = [[u0, v0, 0], [u0, v1, 0], [u1, v1, 0], [u1, v0, 0]]
            vi = wi + ii * 4
            vertices['a_position'][vi:vi+4] = position
            vertices['a_layer'][vi:vi+4] = layer
            vertices['a_anchor'][vi:vi+4] = anchor
            vertices['a_texcoord'][vi:vi+4] = texcoords
            x_move = glyph['advance'] * ratio + kerning
            x_off += x_move
            ascender = max(ascender, y0 - slop)
            descender = min(descender, y1 + slop)
            width += x_move
            height = max(height, glyph['size'][1] - 2*slop)
            prev = char

        # Normalize vertical alignment

        width -= glyph['advance'] * ratio - (glyph['size'][0] - 2*slop)
        dx = dy = 0
        if anchor_y == 'top':
            dy = -ascender
        elif anchor_y in ('center', 'middle'):
            dy = -(height / 2 + descender)
        elif anchor_y == 'bottom':
            dy = -descender

        if anchor_x == 'right':
            dx = -width
        elif anchor_x == 'center':
            dx = -width / 2.

        vertices['a_position'] [wi:vi+4] += (dx, dy,0)
        wi = vi+4


    # Restore original viewport  because font[char] changed it
    set_viewport(*orig_viewport)

    vertices['a_position']  /= (lowres_size,lowres_size, 1)

    return VertexBuffer(vertices)


class textLabels(Program):
    """Program/ shaders that display 3d text

    Parameters
    ----------
    text : str
        Text to display.
    color : instance of Color
        Color to use.
    bold : bool
        Bold face.
    italic : bool
        Italic face.
    face : str
        Font face to use.
    font_size : float
        Point size to use.
    a_position : tuple
        Position (x, y) of the text.
    anchor_x : str
        Horizontal text anchor.
    anchor_y : str
        Vertical text anchor.
    """

    VERTEX_SHADER = """
        uniform mat4 u_model;
        uniform mat4 u_view;
        uniform mat4 u_pitch;
        uniform mat4 u_projection;
        uniform int  u_billboard;
        uniform vec4 u_color;

        uniform float u_offset;
        uniform float u_heightOffset;
        uniform int   u_layerMap[32];


        attribute vec3 a_position;
        attribute vec3 a_anchor;
        attribute float a_layer;
        attribute vec3 a_texcoord;

        varying vec2 v_texcoord;
        varying vec4 v_color;

        float z_coord(float pos_z) {
            int z_idx = int(a_layer);
            int lyr = u_layerMap[z_idx];

            float z = lyr*u_offset - 1.0 + u_heightOffset + pos_z;
            return z;
        }

        void main(void) {
            float height = z_coord(a_anchor.z);

            if (u_billboard == 2) {
                 vec4 pos = (u_pitch * u_model * vec4(a_position.xy, 0, 0))
                             + (u_view * vec4(a_anchor.xy, height, 1));
                 gl_Position = u_projection * pos;
                 } 
            else if (u_billboard == 1) {
                 vec4 pos = (u_model * vec4(a_position.xy, 0, 0))
                             + (u_view * vec4(a_anchor.xy, height, 1));
                 gl_Position = u_projection * pos;
                 }
            else {
                 vec4 pos = u_view * ((u_model * vec4(a_position.xy, 0, 0))
                                      + vec4(a_anchor.xy, height, 1));
                 gl_Position = u_projection * pos;
                 }

            v_texcoord = a_texcoord.xy;
            int idx = int(a_layer);

            if (u_layerMap[idx]  <= -1)
                 v_color = vec4(0,0,0,0);
            else
                 v_color = u_color;

        }
        """

    FRAGMENT_SHADER = """
        // Adapted from vispy
        const float M_SQRT1_2 = 0.707106781186547524400844362104849039;
        const float kernel_bias  = -0.234377;
        const float kernel_scale = 1.241974;

        uniform sampler2D u_font_atlas;
        uniform vec2 u_font_atlas_shape;
        uniform float u_npix;
        uniform sampler2D u_kernel;

        varying vec4 v_color;
        varying vec2 v_texcoord;
        const float center = 0.5;

        // CatRom interpolation code
        vec4 filter1D_radius2(sampler2D kernel, float index, float x,
                              vec4 c0, vec4 c1, vec4 c2, vec4 c3) {
            float w, w_sum = 0.0;
            vec4 r = vec4(0.0,0.0,0.0,0.0);
            w = texture2D(kernel, vec2(0.500000+(x/2.0),index) ).r;
            w = w*kernel_scale + kernel_bias;
            r += c0 * w;
            w = texture2D(kernel, vec2(0.500000-(x/2.0),index) ).r;
            w = w*kernel_scale + kernel_bias;
            r += c2 * w;
            w = texture2D(kernel, vec2(0.000000+(x/2.0),index) ).r;
            w = w*kernel_scale + kernel_bias;
            r += c1 * w;
            w = texture2D(kernel, vec2(1.000000-(x/2.0),index) ).r;
            w = w*kernel_scale + kernel_bias;
            r += c3 * w;
            return r;
        }

        vec4 filter2D_radius2(sampler2D texture, sampler2D kernel, float index,
                              vec2 uv, vec2 pixel) {
            vec2 texel = uv/pixel - vec2(0.0,0.0) ;
            vec2 f = fract(texel);
            texel = (texel-fract(texel)+vec2(0.001,0.001))*pixel;
            vec4 t0 = filter1D_radius2(kernel, index, f.x,
                texture2D( texture, texel + vec2(-1,-1)*pixel),
                texture2D( texture, texel + vec2(0,-1)*pixel),
                texture2D( texture, texel + vec2(1,-1)*pixel),
                texture2D( texture, texel + vec2(2,-1)*pixel));
            vec4 t1 = filter1D_radius2(kernel, index, f.x,
                texture2D( texture, texel + vec2(-1,0)*pixel),
                texture2D( texture, texel + vec2(0,0)*pixel),
                texture2D( texture, texel + vec2(1,0)*pixel),
                texture2D( texture, texel + vec2(2,0)*pixel));
            vec4 t2 = filter1D_radius2(kernel, index, f.x,
                texture2D( texture, texel + vec2(-1,1)*pixel),
                texture2D( texture, texel + vec2(0,1)*pixel),
                texture2D( texture, texel + vec2(1,1)*pixel),
                texture2D( texture, texel + vec2(2,1)*pixel));
            vec4 t3 = filter1D_radius2(kernel, index, f.x,
                texture2D( texture, texel + vec2(-1,2)*pixel),
                texture2D( texture, texel + vec2(0,2)*pixel),
                texture2D( texture, texel + vec2(1,2)*pixel),
                texture2D( texture, texel + vec2(2,2)*pixel));
            return filter1D_radius2(kernel, index, f.y, t0, t1, t2, t3);
        }

        vec4 CatRom(sampler2D texture, vec2 shape, vec2 uv) {
            return filter2D_radius2(texture, u_kernel, 0.468750,
                                    uv, 1.0/shape);
        }

        float contour(in float d, in float w)
        {
            return smoothstep(center - w, center + w, d);
        }

        float sample(sampler2D texture, vec2 uv, float w)
        {
            return contour(texture2D(texture, uv).r, w);
        }

        void main(void) {
            vec4 color = v_color;
            vec2 uv = v_texcoord.xy;
            vec4 rgb;

            // Use interpolation at high font sizes
            if(u_npix >= 50.0)
                rgb = CatRom(u_font_atlas, u_font_atlas_shape, uv);
            else
                rgb = texture2D(u_font_atlas, uv);
            float distance = rgb.r;

            // GLSL's fwidth = abs(dFdx(uv)) + abs(dFdy(uv))
            float width = 0.5 * fwidth(distance);  // sharpens a bit

            // Regular SDF
            float alpha = contour(distance, width);

            if (u_npix < 30.) {
                // Supersample, 4 extra points
                // Half of 1/sqrt2; you can play with this
                float dscale = 0.5 * M_SQRT1_2;
                vec2 duv = dscale * (dFdx(v_texcoord) + dFdy(v_texcoord));
                vec4 box = vec4(v_texcoord-duv, v_texcoord+duv);
                float asum = sample(u_font_atlas, box.xy, width)
                           + sample(u_font_atlas, box.zw, width)
                           + sample(u_font_atlas, box.xw, width)
                           + sample(u_font_atlas, box.zy, width);
                // weighted average, with 4 extra points having 0.5 weight
                // each, so 1 + 0.5*4 = 3 is the divisor
                alpha = (alpha + 0.5 * asum) / 3.0;
            }

            gl_FragColor = vec4(color.rgb, color.a * alpha);
        }
        """

    def __init__(self, canvas, texts, coords, layers, color='black', bold=False,
                 italic=False, face='OpenSans', font_size=12,
                 anchor_x='center', anchor_y='center',
                 font_manager=None, parent = None, **kwargs):


        # Check input
        valid_keys = ('top', 'center', 'middle', 'baseline', 'bottom')
        _check_valid('anchor_y', anchor_y, valid_keys)
        valid_keys = ('left', 'center', 'right')
        _check_valid('anchor_x', anchor_x, valid_keys)
        # Init font handling stuff
        # _font_manager is a temporary solution to use global mananger
        self._font_manager = font_manager or FontManager()
        self._font = self._font_manager.get_font(face, bold, italic)

        self.heightOffset = kwargs.pop('heightOffset', 0.0)

        super(textLabels, self).__init__(vert = self.VERTEX_SHADER,
                                       frag = self.FRAGMENT_SHADER)
        self.canvas = canvas
        self.parent = parent
        self._vertices = None
        self._anchors = (anchor_x, anchor_y)
        # Init text properties
        self.color = color
        self.texts = texts
        self.coords = coords
        self.layers = layers
        self.font_size = font_size
        self.textScale = font_size / 720

        #  Register for View and Projection updates with the canvas
        canvas.registerDependent(self)

        self['u_view'] = canvas.view
        self['u_pitch'] = canvas.pitch
        self['u_projection']  = canvas.projection

        for lyr in range(32):
            self['u_layerMap[%d]' % lyr] = lyr

        n_pix = (self._font_size / 720.)
        self['u_npix'] = n_pix
        self['u_kernel'] = self._font._kernel
        self['u_color'] = self._color.rgba
        self['u_font_atlas'] = self._font._atlas
        self['u_font_atlas_shape'] = self._font._atlas.shape[:2]
        self['u_offset'] = self.offset
        self['u_heightOffset']  = self.heightOffset
        self.billboard = kwargs.pop('billboard', None)

        set_state(blend=True, depth_test=False,
                  blend_func=('src_alpha', 'one_minus_src_alpha'))

        #  A program is transformed by the chain of model matrices
        #  from self through parent to the root.

        if self.parent is not None:
            self.parent.registerChild(self)
        else:
            self.root = self

        model = kwargs.pop('model', np.eye(4, dtype=np.float32))

        self.model = scale((self.textScale,
                            self.textScale,
                            self.textScale)).dot(model)

    @property
    def texts(self):
        """The text string"""
        return self._texts

    @texts.setter
    def texts(self, txts):
        for text in txts:
            assert isinstance(text, string_types)
        self._texts = txts
        self._vertices = None

    @property
    def coords(self):
        return self._coords

    @coords.setter
    def coords(self, cds):
        for coord in cds:
            assert len(coord) == 3
        self._coords = cds
        self._vertices = None

    @property
    def zCoords(self):
        return list(np.array(self._coords[:,2]))

    @zCoords.setter
    def zCoords(self, zLoc):
        assert len(zLoc) == len(self._coords)
        
        for i,v in enumerate(zLoc):
            self._coords[i][2]=v
        
        self._vertices = None
    
    @property
    def layers(self):
        return self._coords
    
    @layers.setter
    def layers(self,lyrs):
        for lyr in lyrs:
            assert lyr >=0 and lyr<8
        self._layers = lyrs
        self._vertices = None

    @property
    def font_size(self):
        """ The font size (in points) of the text
        """
        return self._font_size

    @font_size.setter
    def font_size(self, size):
        self._font_size = max(0.0, float(size))

    @property
    def color(self):
        """ The color of the text
        """
        return self._color

    @color.setter
    def color(self, color):
        self._color = Color(color)

    @property
    def billboard(self):
        return self._billboard

    @billboard.setter
    def billboard(self, mode):
        if mode=='cylinder':
            self._billboard = 2
        elif mode == 'sphere':
            self._billboard = 1
        else:
            self._billboard = 0

        self['u_billboard'] = self._billboard

    @property
    def offset(self):
        """
        Height derived from layer
        """
        return self.canvas.offset


    def modelProduct(self):
        """
        Model matrix is the product of all model transformations from here
        down to the root.
        The model must be recalculated if there is a change to the model on
        any of the ancestors on the chain down to root
        """
        if self.parent is None:
            return self._model
        else:
            return self._model.dot(self.parent.modelProduct())

    def __getitem__(self, name):
        if name == 'u_model':
            return self.modelProduct()
        else:
            return super(Program, self).__getitem__(name)

    def updateChildModels(self):
        self['u_model'] = self.modelProduct()


    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value
        self.updateChildModels()


    def draw(self):
        # attributes / uniforms are not available until program is built
        if len(self.texts) == 0:
            return

        #pr = cProfile.Profile()
        #pr.enable()

        if self._vertices is None:
            # we delay creating vertices because it requires a context,
            # which may or may not exist when the object is initialized
            self.canvas.context.flush_commands()  # flush GLIR commands
            self._vertices = text_to_vbo(self._texts, self._coords, self._layers, self._font,
                                          self._anchors[0], self._anchors[1],
                                          self._font._lowres_size)

            charCount = sum(len(text) for text in self.texts)

            start = 0
            stop = 4*charCount
            idx = (np.array([0, 1, 2, 0, 2, 3], np.uint32) +
                   np.arange(start, stop, 4,
                             dtype=np.uint32)[:, np.newaxis])

            self.ibo = IndexBuffer(idx.ravel())

            self.bind(self._vertices)
            set_state(blend=True, depth_test=False,
                      blend_func=('src_alpha', 'one_minus_src_alpha'))

        super(textLabels,self).draw('triangles', self.ibo)

        #pr.disable()
        #s = StringIO.StringIO()
        #sortby = 'cumulative'
        #ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        #ps.print_stats()
        #print s.getvalue()
