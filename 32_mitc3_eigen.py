import numpy
import scipy
import moderngl
from PyQt5 import QtWidgets
from util_moderngl_qt import DrawerMesh, QGLWidgetViewer3
from del_msh import TriMesh, PolyLoop
from del_fem import Mitc3

if __name__ == "__main__":
    vtxi2xyi = numpy.array([
        [0, 0],
        [1, 0],
        [1, 0.1],
        [0, 0.1]], dtype=numpy.float32)
    ##
    tri2vtx, vtx2xy = PolyLoop.tesselation2d(
        vtxi2xyi, resolution_edge=0.03, resolution_face=0.03)
    print("# vtx: ", vtx2xy.shape[0])
    thickness = 0.01
    smat = Mitc3.stiffness_matrix_from_uniform_mesh(thickness, 1.0, 1.0, tri2vtx, vtx2xy)
    mmat = Mitc3.mass_matrix_from_uniform_mesh(thickness, 1.0, tri2vtx, vtx2xy)
    assert scipy.sparse.linalg.norm(smat - smat.transpose()) < 1.0e-5
    # assert scipy.linalg.norm(smat * numpy.ones((vtx2xy.shape[0]*3))) < 1.0e-4
    eig = scipy.sparse.linalg.eigsh(smat, M=mmat, sigma=0.0, k=8)
    print(eig[0])
    i_eig = 3
    evec0 = eig[1].transpose()[i_eig].copy()
    eval0 = eig[0][i_eig]
    print(f"eigenvalue: {eval0}")
    evec0 /= scipy.linalg.norm(evec0)
    assert abs(eval0*(mmat * evec0).dot(evec0)-(smat * evec0).dot(evec0)) < 1.0e-4
    evec0 = evec0.reshape(-1, 3)
    vtx2xyz = numpy.hstack([vtx2xy, evec0[:, 0].copy().reshape(-1, 1)*2.0])
    print(vtx2xyz.shape)
    edge2vtx = TriMesh.edge2vtx(tri2vtx, vtx2xy.shape[0])
    drawer_ini = DrawerMesh.Drawer(
        vtx2xyz=vtx2xy,
        list_elem2vtx=[
            DrawerMesh.ElementInfo(index=edge2vtx, color=(0, 0, 0.5), mode=moderngl.LINES),
            # DrawerMesh.ElementInfo(index=tri2vtx, color=(1, 1, 1), mode=moderngl.TRIANGLES)
        ]
    )
    drawer_def = DrawerMesh.Drawer(
        vtx2xyz=vtx2xyz,
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