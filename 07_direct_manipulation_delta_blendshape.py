from pathlib import Path
import del_msh
import del_srch
import moderngl
import numpy
import blendshape_delta
from PyQt5 import QtWidgets, QtCore
from pyrr import Matrix44
from util_moderngl_qt import DrawerMesh, QGLWidgetViewer3
from util_moderngl_qt.drawer_transform_multi import DrawerTransformMulti


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, paths):
        vtx2xyz, _, _, \
            elem2idx, idx2vtx_xyz, _, _, \
            _, _, _, _, _ = del_msh.load_wavefront_obj(paths[0])
        idx2vtx_xyz = idx2vtx_xyz.astype(numpy.uint64)
        edge2vtx = del_msh.edges_of_polygon_mesh(elem2idx, idx2vtx_xyz, vtx2xyz.shape[0])
        self.tri2vtx = del_msh.triangles_from_polygon_mesh(elem2idx, idx2vtx_xyz)
        self.drawer_mesh = DrawerMesh.Drawer(
            vtx2xyz=vtx2xyz,
            list_elem2vtx=[
                DrawerMesh.ElementInfo(index=edge2vtx, color=(0, 0, 0), mode=moderngl.LINES),
                DrawerMesh.ElementInfo(index=self.tri2vtx, color=(1, 1, 1), mode=moderngl.TRIANGLES)]
        )

        # sphere
        self.rad_sphere = 0.05
        sphere_tri2vtx, sphere_vtx2xyz = del_msh.sphere_meshtri3(1., 32, 32)
        self.drawer_sphere = DrawerMesh.Drawer(sphere_vtx2xyz, list_elem2vtx=[
            DrawerMesh.ElementInfo(index=sphere_tri2vtx, color=(1., 0., 0.), mode=moderngl.TRIANGLES)])
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
        self.glwidget = QGLWidgetViewer3.QtGLWidget_Viewer3(
            [self.drawer_mesh, self.drawer_sphere])
        self.glwidget.nav.view_height = 1.
        self.glwidget.mousePressCallBack.append(self.mouse_press_callback)
        self.glwidget.mouseMoveCallBack.append(self.mouse_move_callback)
        self.glwidget.mouseDoubleClickCallBack.append(self.mouse_doubleclick_callback)
        self.setCentralWidget(self.glwidget)

    def add_shape(self, path1):
        vtx2xyz1, _, _, \
            elem2idx, idx2vtx_xyz, _, _, \
            _, _, _, _, _ = del_msh.load_wavefront_obj(path1)
        vtx2xyz1 = vtx2xyz1.flatten().astype(numpy.float32)
        self.shape2pos = numpy.vstack([self.shape2pos, vtx2xyz1])
        self.weights = numpy.append(self.weights, 0).astype(numpy.float32)

    def mouse_press_callback(self, event):
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
            return
        if event.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier:
            return
        vtx2xyz = self.weights.transpose().dot(self.shape2pos).reshape(-1, 3)  # current shape
        src, direction = self.glwidget.nav.picking_ray()
        self.vtx_pick = del_srch.pick_vertex_meshtri3(
            numpy.array(src.xyz).astype(numpy.float32),
            numpy.array(direction.xyz).astype(numpy.float32),
            vtx2xyz.astype(numpy.float32), self.tri2vtx)
        if self.vtx_pick == -1:
            return
        assert self.vtx_pick < self.tri2vtx.shape[0]
        mvp = self.glwidget.nav.projection_matrix() * self.glwidget.nav.modelview_matrix()
        mvp = numpy.array(mvp).transpose()
        trg = (self.glwidget.nav.cursor_x, self.glwidget.nav.cursor_y)
        self.markers[self.vtx_pick] = [mvp, trg]
        self.drawer_sphere.list_transform = []
        for key_markers in self.markers.keys():
            scale = Matrix44.from_scale((self.rad_sphere, self.rad_sphere, self.rad_sphere))
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
        dweights = blendshape_delta.direct_manipulation(self.shape2pos, self.markers)
        print(dweights)
        self.weights = numpy.append(1 - numpy.sum(dweights), dweights).astype(numpy.float32)
        vtx2xyz = self.weights.transpose().dot(self.shape2pos).reshape(-1, 3).copy()
        self.drawer_mesh.update_position(vtx2xyz)
        self.drawer_sphere.list_transform = []
        for key_markers in self.markers.keys():
            scale = Matrix44.from_scale((self.rad_sphere, self.rad_sphere, self.rad_sphere))
            translate = Matrix44.from_translation(vtx2xyz[key_markers].copy())
            self.drawer_sphere.list_transform.append(translate * scale)
        self.glwidget.updateGL()

    def mouse_doubleclick_callback(self, event):
        vtx2xyz = self.weights.transpose().dot(self.shape2pos).reshape(-1, 3)  # current shape
        src, direction = self.glwidget.nav.picking_ray()
        vtx_pick = del_srch.pick_vertex_meshtri3(
            numpy.array(src.xyz).astype(numpy.float32),
            numpy.array(direction.xyz).astype(numpy.float32),
            vtx2xyz.astype(numpy.float32), self.tri2vtx)
        for vtx in self.markers.copy():
            len = numpy.linalg.norm(vtx2xyz[vtx_pick] - vtx2xyz[vtx])
            print(len, self.rad_sphere, vtx, vtx_pick)
            if len < self.rad_sphere:
                self.markers.pop(vtx)
        self.vtx_pick = -1
        dweights = blendshape_delta.direct_manipulation(self.shape2pos, self.markers)
        self.weights = numpy.append(1 - numpy.sum(dweights), dweights).astype(numpy.float32)
        vtx2xyz = self.weights.transpose().dot(self.shape2pos).reshape(-1, 3).copy()
        self.drawer_mesh.update_position(vtx2xyz)
        rad = self.glwidget.nav.view_height / self.glwidget.nav.scale * 0.03
        self.drawer_sphere.list_transform = []
        for key_markers in self.markers.keys():
            scale = Matrix44.from_scale((rad, rad, rad))
            translate = Matrix44.from_translation(vtx2xyz[key_markers].copy())
            self.drawer_sphere.list_transform.append(translate * scale)
        self.glwidget.updateGL()


def main():
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


if __name__ == "__main__":
    main()
