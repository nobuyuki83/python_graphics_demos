from pathlib import Path

import blendshape
import del_msh
import del_srch
import moderngl
import numpy
from PyQt5 import QtWidgets, QtCore
from pyrr import Matrix44
from util_moderngl_qt.drawer_meshpos import DrawerMesPos, ElementInfo
from util_moderngl_qt.drawer_transform import DrawerTransform
from util_moderngl_qt.qtglwidget_viewer3 import QtGLWidget_Viewer3


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, paths_obj):
        vtx2xyz, _, elem2idx, idx2vtx_xyz, _ = del_msh.load_wavefront_obj(paths_obj[0])
        idx2vtx_xyz = idx2vtx_xyz.astype(numpy.uint64)
        edge2vtx = del_msh.edges_of_polygon_mesh(elem2idx, idx2vtx_xyz, vtx2xyz.shape[0])
        self.tri2vtx = del_msh.triangles_from_polygon_mesh(elem2idx, idx2vtx_xyz)
        print(self.tri2vtx.shape, elem2idx.shape)
        self.drawer_mesh = DrawerMesPos(
            vtx2xyz=vtx2xyz,
            list_elem2vtx=[
                ElementInfo(index=edge2vtx, color=(0, 0, 0), mode=moderngl.LINES),
                ElementInfo(index=self.tri2vtx, color=(1, 1, 1), mode=moderngl.TRIANGLES)]
        )

        sphere_tri2vtx, sphere_vtx2xyz = del_msh.sphere_meshtri3(1., 32, 32)
        self.drawer_sphere = DrawerMesPos(vtx2xyz=sphere_vtx2xyz, list_elem2vtx=[
            ElementInfo(index=sphere_tri2vtx, color=(1., 0., 0.), mode=moderngl.TRIANGLES)])
        self.drawer_sphere = DrawerTransform(self.drawer_sphere)
        self.drawer_sphere.is_visible = False

        self.shape_pos = numpy.array([vtx2xyz.flatten().copy()], dtype=numpy.float32)
        self.weights = numpy.array([1.], dtype=numpy.float32)
        for path in paths_obj[1:]:
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
        src, direction = self.glwidget.nav.picking_ray()
        self.vtx_idx = del_srch.pick_vertex_meshtri3(
            numpy.array(src.xyz).astype(numpy.float32),
            numpy.array(direction.xyz).astype(numpy.float32),
            vtx_xyz.astype(numpy.float32), self.tri2vtx)
        print(self.vtx_idx)
        self.drawer_sphere.is_visible = False
        if self.vtx_idx != -1:
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
        self.weights = blendshape.direct_manipulation(self.shape_pos, {self.vtx_idx: [mvp,trg]})
        vtx_xyz = self.weights.transpose().dot(self.shape_pos).reshape(-1, 3).copy()
        self.drawer_mesh.update_position(vtx_xyz)
        pos0 = vtx_xyz[self.vtx_idx].copy()
        self.drawer_sphere.is_visible = True
        # self.drawer_sphere.transform = Matrix44.from_translation(pos0) * Matrix44.from_scale((0.03, 0.03, 0.03))
        rad = self.glwidget.nav.view_height / self.glwidget.nav.scale * 0.03
        self.drawer_sphere.transform = Matrix44.from_translation(pos0) * Matrix44.from_scale((rad, rad, rad))
        self.glwidget.updateGL()


def main():
    path_dir = Path('.') / 'asset'
    paths = [str(path_dir / 'suzanne0.obj'),
             str(path_dir / 'suzanne1.obj'),
             str(path_dir / 'suzanne2.obj')]

    with QtWidgets.QApplication([]) as app:
        win = MainWindow(paths)
        win.show()
        app.exec()


if __name__ == "__main__":
    main()
