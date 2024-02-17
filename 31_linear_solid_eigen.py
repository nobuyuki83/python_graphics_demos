import numpy
import scipy
import moderngl
from PyQt5 import QtWidgets
from util_moderngl_qt import DrawerMesh, QGLWidgetViewer3
from del_msh import TriMesh, PolyLoop

if __name__ == "__main__":
    vtxi2xyi = numpy.array([
        [0, 0],
        [1, 0],
        [1, 0.6],
        [0.6, 0.6],
        [0.6, 1.0],
        [0, 1]], dtype=numpy.float32)
    ##
    tri2vtx, vtx2xy = PolyLoop.tesselation2d(
        vtxi2xyi, resolution_edge=0.04, resolution_face=0.04)
    print("# vtx: ", vtx2xy.shape[0])
    row2idx, idx2col = TriMesh.vtx2vtx(tri2vtx, vtx2xy.shape[0], True)
    idx2val = numpy.zeros((idx2col.shape[0],2,2), dtype=numpy.float32)
    vtx2area = TriMesh.vtx2area(tri2vtx, vtx2xy)
    mmat = scipy.sparse.dia_matrix(
        (vtx2area.repeat(2).reshape(1, -1), numpy.array([0])),
        shape=(vtx2xy.shape[0]*2, vtx2xy.shape[0]*2))
    from del_fem.del_fem import merge_linear_solid_to_csr_for_meshtri2
    merge_linear_solid_to_csr_for_meshtri2(tri2vtx, vtx2xy, row2idx, idx2col, idx2val)
    smat = scipy.sparse.bsr_matrix((idx2val, idx2col, row2idx))
    assert scipy.sparse.linalg.norm(smat - smat.transpose()) < 1.0e-5
    assert scipy.linalg.norm(smat * numpy.ones((vtx2xy.shape[0]*2))) < 1.0e-4
    eig = scipy.sparse.linalg.eigsh(smat, M=mmat, sigma=0.0)
    i_eig = 4
    evec0 = eig[1].transpose()[i_eig].copy()
    eval0 = eig[0][i_eig]
    print(f"eigenvalue: {eval0}")
    evec0 /= scipy.linalg.norm(evec0)
    assert abs(eval0*(mmat * evec0).dot(evec0)-(smat * evec0).dot(evec0)) < 1.0e-4
    vtx2val = evec0.reshape(-1, 2) * 10
    edge2vtx = TriMesh.edge2vtx(tri2vtx, vtx2xy.shape[0])
    drawer_ini = DrawerMesh.Drawer(
        vtx2xyz=vtx2xy,
        list_elem2vtx=[
            DrawerMesh.ElementInfo(index=edge2vtx, color=(0, 0, 0.5), mode=moderngl.LINES),
            # DrawerMesh.ElementInfo(index=tri2vtx, color=(1, 1, 1), mode=moderngl.TRIANGLES)
        ]
    )
    drawer_def = DrawerMesh.Drawer(
        vtx2xyz=vtx2xy + vtx2val*0.1,
        list_elem2vtx=[
            DrawerMesh.ElementInfo(index=edge2vtx, color=(0, 0, 0), mode=moderngl.LINES),
            DrawerMesh.ElementInfo(index=tri2vtx, color=(1, 1, 1), mode=moderngl.TRIANGLES)
        ]
    )

    with QtWidgets.QApplication([]) as app:
        win = QGLWidgetViewer3.QtGLWidget_Viewer3(
            [drawer_ini, drawer_def])
        win.show()
        app.exec()
