
from numpy import cross, array
from numpy.linalg import norm, det
from math import hypot


def pickDistanceToEdge(edge):
    '''
    Given a line segment a0, a1 translated so that the mouse coordinates are
    (0,0, Zn) Return

    - The shortest distance from line segment to a ray from the camera
    through the mouse position (translated to (0.,0., Znear)

    - The Z coord of the pick point,

    - A value t with value between 0.0 and 1.0 that defines the closest point
    on the line-segment. A value of 0.0 corresponds with point a0, 1.0 with
    a1. See:

        http://stackoverflow.com/questions/2824478/
                shortest-distance-between-two-line-segments
    '''
    a0, a1 = edge

    A = a1 - a0
    _A = A / norm(A)

    # b0 is 0,0,0  the camera position
    # b1 is 0,0,-1  the coordinates of the mouse (m) after translation
    _B = m = array((0, 0, -1))

    xp = cross(_A, _B)
    denom = norm(xp)**2

    t = (m - a0)
    det0 = det([t, _B, xp])
    det1 = det([t, _A, xp])

    t0 = det0/denom
    t1 = det1/denom

    pA = a0 + (_A * t0)
    pB = m + (_B * t1)

    # Clamp results to line segment
    if t0 < 0:
        pA = a0
    elif t0 > norm(A):
        pA = a1

    d = norm(pA-pB)

    return d, pB[2], t0/norm(A)


def excludeFarAwayEdges(edges, tol=0.01):
    """
    Return a list of indices of edges that
    pass close to the origin.

    Excludes obviously bad candidates. A few False positives are okay,
    because final selection is deferred to a more complex and costly
    algorithm...
    """
    res = []

    for idx, edge in enumerate(edges):
        x0, y0, z0 = edge[0]
        x1, y1, z1 = edge[1]

        if (min(x0, x1) < tol and max(x0, x1) > -tol
                and min(y0, y1) < tol and max(y0, y1) > -tol):
            res.append(idx)
    return res




def pickEdge(edges, tol=0.01):
    candidates = excludeFarAwayEdges(edges, tol)

    pickList = [(pickDistanceToEdge(edges[idx]) + (idx,))
                for idx in candidates]

    if pickList:
        d, z, t,  closest = min(pickList, key=lambda x: x[0])
        # print ("Pick depth: %f [%d candidates]  idx: %d distance: %f"
        #        %  (z, len(candidates), closest, d))
        return (closest, z, t)

    return None


def pickPoint(points, tol=0.01):
    '''
    Given a point translated so that the mouse coordinates are
    (0,0, Zn) Return:

    - The shortest distance from the point to a ray from the camera
    through the mouse position (translated to (0.,0., Znear)

    - The Z coord of the pick point,

    '''

    d, z, idx = min((hypot(p[0], p[1]), p[2], idx) for idx, p in enumerate(points))   

    #print ("Pickpoint d ={} z={} idx={}".format(d,z,idx))
    
    if d<tol:
        return idx, z      
    else:
        return None
    