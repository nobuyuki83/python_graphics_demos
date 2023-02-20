from pathlib import Path

import cvxopt
import del_msh
import del_srch
import moderngl
import numpy
from PyQt5 import QtWidgets, QtCore
from cvxopt import matrix
from pyrr import Matrix44
from util_moderngl_qt.drawer_meshpos import DrawerMesPos, ElementInfo
from util_moderngl_qt.drawer_transform import DrawerTransform
from util_moderngl_qt.qtglwidget_viewer3 import QtGLWidget_Viewer3


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, paths):
        vtx2xyz, _, elem2idx, idx2vtx_xyz, _ = del_msh.load_wavefront_obj(paths[0])
        idx2vtx_xyz = idx2vtx_xyz.astype(numpy.uint64)
        E = del_msh.edges_of_polygon_mesh(elem2idx, idx2vtx_xyz, vtx2xyz.shape[0])
        self.tri_vtx = del_msh.triangles_from_polygon_mesh(elem2idx, idx2vtx_xyz)
        print(self.tri_vtx.shape, elem2idx.shape)
        self.drawer_mesh = DrawerMesPos(
            V=vtx2xyz.astype(numpy.float32),
            element=[
                ElementInfo(index=E.astype(numpy.uint32), color=(0, 0, 0), mode=moderngl.LINES),
                ElementInfo(index=self.tri_vtx.astype(numpy.uint32), color=(1, 1, 1), mode=moderngl.TRIANGLES)]
        )

        F,V = del_msh.sphere_meshtri3(1., 32, 32)
        self.drawer_sphere = DrawerMesPos(V, element=[
            ElementInfo(index=F.astype(numpy.uint32), color=(1., 0., 0.), mode=moderngl.TRIANGLES)])
        self.drawer_sphere = DrawerTransform(self.drawer_sphere)
        self.drawer_sphere.is_visible = False

        self.shape_pos = numpy.array([vtx2xyz.flatten().copy()], dtype=numpy.float32)
        self.weights = numpy.array([1.], dtype=numpy.float32)
        for path in paths[1:]:
            self.add_shape(path)

        self.vtx_idx = -1

        QtWidgets.QMainWindow.__init__(self)
        self.resize(640, 480)
        self.setWindowTitle('Mesh Viewer')
        self.glwidget = QtGLWidget_Viewer3([self.drawer_mesh, self.drawer_sphere])
        self.glwidget.mousePressCallBack.append(self.mouse_press_callback)
        self.glwidget.mouseMoveCallBack.append(self.mouse_move_callback)
        self.setCentralWidget(self.glwidget)

    def add_shape(self, path1):
        vtx2xyz1, _, elem2idx, idx2vtx_xyz, _ = del_msh.load_wavefront_obj(path1)
        vtx2xyz1 = vtx2xyz1.flatten().astype(numpy.float32)
        self.shape_pos = numpy.vstack([self.shape_pos, vtx2xyz1])
        self.weights = numpy.append(self.weights, 0).astype(numpy.float32)

    def mouse_press_callback(self, event):
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
            return
        if event.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier:
            return
        vtx_xyz = self.weights.transpose().dot(self.shape_pos).reshape(-1, 3)
        src, dir = self.glwidget.nav.picking_ray()
        pos, tri_index = del_srch.first_intersection_ray_meshtri3(
            numpy.array(src.xyz).astype(numpy.float32),
            numpy.array(dir.xyz).astype(numpy.float32),
            vtx_xyz.astype(numpy.float32), self.tri_vtx)
        self.drawer_sphere.is_visible = False
        self.vtx_idx = -1
        if tri_index != -1:
            print("hit", tri_index)
            self.vtx_idx = self.tri_vtx[tri_index][0]
            assert self.vtx_idx < self.tri_vtx.shape[0]
            pos0 = vtx_xyz[self.vtx_idx].copy()
            self.drawer_sphere.is_visible = True
            rad = self.glwidget.nav.view_height / self.glwidget.nav.scale * 0.03
            self.drawer_sphere.transform = Matrix44.from_translation(pos0) * Matrix44.from_scale((rad, rad, rad))
        self.glwidget.updateGL()

    def mouse_move_callback(self, event):
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
            return
        if event.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier:
            return
        if self.vtx_idx == -1:
            return
        mvp = self.glwidget.nav.projection_matrix() * self.glwidget.nav.modelview_matrix()
        mvp = numpy.array(mvp).transpose()
        trg = (self.glwidget.nav.cursor_x, self.glwidget.nav.cursor_y)
        B = []
        nshape = self.shape_pos.shape[0]
        for ishape in range(nshape):
            pos0 = self.shape_pos[ishape].reshape(-1, 3)[self.vtx_idx].copy()
            pos0 = mvp.dot(numpy.append(pos0, 1.0))[0:2]
            B.append(pos0)
        B = numpy.vstack(B).transpose()
        P = B.transpose().dot(B) + numpy.eye(nshape) * 0.001
        q = -B.transpose().dot(trg)
        A = numpy.ones((1, nshape)).astype(numpy.double)
        b = numpy.array([1.]).reshape(1, 1)
        G = numpy.vstack([numpy.eye(nshape), -numpy.eye(nshape)]).astype(numpy.double)
        h = numpy.vstack([numpy.ones((nshape, 1)), numpy.zeros((nshape, 1))]).astype(numpy.double)
        sol = cvxopt.solvers.qp(P=matrix(P), q=matrix(q),
                                A=matrix(A), b=matrix(b),
                                G=matrix(G), h=matrix(h))
        self.weights = numpy.array(sol['x'], dtype=numpy.float32)
        vtx_xyz = self.weights.transpose().dot(self.shape_pos).reshape(-1, 3).copy()
        self.drawer_mesh.update_position(vtx_xyz)
        pos0 = vtx_xyz[self.vtx_idx].copy()
        self.drawer_sphere.is_visible = True
        # self.drawer_sphere.transform = Matrix44.from_translation(pos0) * Matrix44.from_scale((0.03, 0.03, 0.03))
        rad = self.glwidget.nav.view_height / self.glwidget.nav.scale * 0.03
        self.drawer_sphere.transform = Matrix44.from_translation(pos0) * Matrix44.from_scale((rad, rad, rad))
        self.glwidget.updateGL()


if __name__ == "__main__":
    path_dir = Path('.') / 'asset'
    paths = [str(path_dir / 'suzanne0.obj'),
             str(path_dir / 'suzanne1.obj'),
             str(path_dir / 'suzanne2.obj')]

    with QtWidgets.QApplication([]) as app:
        win = MainWindow(paths)
        win.show()
        app.exec()
