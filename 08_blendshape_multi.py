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
from util_moderngl_qt.drawer_transform_multi import DrawerTransformMulti
from util_moderngl_qt.qtglwidget_viewer3 import QtGLWidget_Viewer3

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, paths):
        vtx2xyz, _, elem2idx, idx2vtx_xyz, _ = del_msh.load_wavefront_obj(paths[0])
        idx2vtx_xyz = idx2vtx_xyz.astype(numpy.uint64)
        edge2vtx = del_msh.edges_of_polygon_mesh(elem2idx, idx2vtx_xyz, vtx2xyz.shape[0])
        self.tri2vtx = del_msh.triangles_from_polygon_mesh(elem2idx, idx2vtx_xyz)
        print(self.tri2vtx.shape, elem2idx.shape)
        self.drawer_mesh = DrawerMesPos(
            V=vtx2xyz.astype(numpy.float32),
            element=[
                ElementInfo(index=edge2vtx.astype(numpy.uint32), color=(0, 0, 0), mode=moderngl.LINES),
                ElementInfo(index=self.tri2vtx.astype(numpy.uint32), color=(1, 1, 1), mode=moderngl.TRIANGLES)]
        )

        # sphere
        F, V = del_msh.sphere_meshtri3(1., 32, 32)
        self.drawer_sphere = DrawerMesPos(V, element=[
            ElementInfo(index=F.astype(numpy.uint32), color=(1., 0., 0.), mode=moderngl.TRIANGLES)])
        self.drawer_sphere = DrawerTransformMulti(self.drawer_sphere)
        self.drawer_sphere.is_visible = False

        # add shapes
        self.shape2pos = numpy.array([vtx2xyz.flatten().copy()], dtype=numpy.float32)
        self.weights = numpy.array([1.], dtype=numpy.float32)
        for path in paths[1:]:
            self.add_shape(path)

        self.vtx_pick = -1
        self.markers = {}

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
        self.shape2pos = numpy.vstack([self.shape2pos, vtx2xyz1])
        self.weights = numpy.append(self.weights, 0).astype(numpy.float32)

    def mouse_press_callback(self, event):
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
            return
        if event.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier:
            return
        vtx2xyz = self.weights.transpose().dot(self.shape2pos).reshape(-1, 3)  # current shape
        src, dir = self.glwidget.nav.picking_ray()
        pos, tri_index = del_srch.first_intersection_ray_meshtri3(
            numpy.array(src.xyz).astype(numpy.float32),
            numpy.array(dir.xyz).astype(numpy.float32),
            vtx2xyz.astype(numpy.float32), self.tri2vtx)
        self.drawer_sphere.is_visible = False
        self.vtx_pick = -1
        if tri_index == -1:
            return
        print("hit", tri_index)
        self.vtx_pick = self.tri2vtx[tri_index][0]
        assert self.vtx_pick < self.tri2vtx.shape[0]
        mvp = self.glwidget.nav.projection_matrix() * self.glwidget.nav.modelview_matrix()
        mvp = numpy.array(mvp).transpose()
        trg = (self.glwidget.nav.cursor_x, self.glwidget.nav.cursor_y)
        self.markers[self.vtx_pick] = [mvp, trg]
        rad = self.glwidget.nav.view_height / self.glwidget.nav.scale * 0.03
        self.drawer_sphere.list_transform = []
        for key_markers in self.markers.keys():
            scale = Matrix44.from_scale((rad, rad, rad))
            translate = Matrix44.from_translation(vtx2xyz[key_markers].copy())
            self.drawer_sphere.list_transform.append(translate * scale)
        print(self.markers)
        self.glwidget.updateGL()

    def mouse_move_callback(self, event):
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
            return
        if event.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier:
            return
        if self.vtx_pick == -1:
            return
        assert self.vtx_pick in self.markers
        mvp = self.glwidget.nav.projection_matrix() * self.glwidget.nav.modelview_matrix()
        mvp = numpy.array(mvp).transpose()
        trg = (self.glwidget.nav.cursor_x, self.glwidget.nav.cursor_y)
        self.markers[self.vtx_pick] = [mvp, trg]
        B = []
        num_shape = self.shape2pos.shape[0]
        for vtx_marker in self.markers.keys():
            for idx_shape in range(num_shape):
                pos0 = self.shape2pos[idx_shape].reshape(-1, 3)[vtx_marker].copy()
                pos0 = self.markers[vtx_marker][0].dot(numpy.append(pos0, 1.0))[0:2]
                B.append(pos0)
        B = numpy.vstack(B).transpose().reshape(-1, num_shape)
        T = []
        for vtx_marker in self.markers.keys():
            T.append(self.markers[vtx_marker][1])
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
        self.weights = numpy.array(sol['x'], dtype=numpy.float32)
        vtx2xyz = self.weights.transpose().dot(self.shape2pos).reshape(-1, 3).copy()
        self.drawer_mesh.update_position(vtx2xyz)
        rad = self.glwidget.nav.view_height / self.glwidget.nav.scale * 0.03
        self.drawer_sphere.list_transform = []
        for key_markers in self.markers.keys():
            scale = Matrix44.from_scale((rad, rad, rad))
            translate = Matrix44.from_translation(vtx2xyz[key_markers].copy())
            self.drawer_sphere.list_transform.append(translate * scale)
        self.glwidget.updateGL()


if __name__ == "__main__":
    path_dir = Path('.') / 'asset'
    paths = [str(path_dir / 'suzanne0.obj'),
             str(path_dir / 'suzanne1.obj'),
             str(path_dir / 'suzanne2.obj'),
             str(path_dir / 'suzanne3.obj'),
             str(path_dir / 'suzanne4.obj')]

    with QtWidgets.QApplication([]) as app:
        win = MainWindow(paths)
        win.show()
        app.exec()
