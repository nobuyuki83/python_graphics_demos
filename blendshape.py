import numpy
import cvxopt
from cvxopt import matrix


def direct_manipulation(shape2pos, markers):
    B = []
    num_shape = shape2pos.shape[0]
    for vtx_marker in markers.keys():
        for idx_shape in range(num_shape):
            pos0 = shape2pos[idx_shape].reshape(-1, 3)[vtx_marker].copy()
            pos0 = markers[vtx_marker][0].dot(numpy.append(pos0, 1.0))[0:2]
            B.append(pos0)
    B = numpy.vstack(B).transpose().reshape(-1, num_shape)
    T = []
    for vtx_marker in markers.keys():
        T.append(markers[vtx_marker][1])
    T = numpy.array(T).transpose().flatten()
    print(T.shape, B.shape)
    P = B.transpose().dot(B) + numpy.eye(num_shape) * 0.001
    q = -B.transpose().dot(T)
    A = numpy.ones((1, num_shape)).astype(numpy.double)
    b = numpy.array([1.]).reshape(1, 1)
    G = numpy.vstack([numpy.eye(num_shape), -numpy.eye(num_shape)]).astype(numpy.double)
    h = numpy.vstack([numpy.ones((num_shape, 1)), numpy.zeros((num_shape, 1))]).astype(numpy.double)
    sol = cvxopt.solvers.qp(P=matrix(P), q=matrix(q),
                            A=matrix(A), b=matrix(b),
                            G=matrix(G), h=matrix(h))
    return numpy.array(sol['x'], dtype=numpy.float32)
