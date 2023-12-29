from pathlib import Path
import moderngl
import numpy
from PyQt5 import QtWidgets, QtCore
from pyrr import Matrix44
from util_moderngl_qt import DrawerMesh, QGLWidgetViewer3, DrawerSpheres
# from util_moderngl_qt.drawer_transform_multi import DrawerTransformMulti
from del_msh import WavefrontObj, PolygonMesh, TriMesh, BlendShape


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, paths):
        obj = WavefrontObj.load(paths[0])
        edge2vtx = PolygonMesh.edges(obj.elem2idx, obj.idx2vtxxyz, obj.vtxxyz2xyz.shape[0])
        self.tri2vtx = PolygonMesh.triangles(obj.elem2idx, obj.idx2vtxxyz)
        print(self.tri2vtx.shape, obj.elem2idx.shape)
        self.drawer_mesh = DrawerMesh.Drawer(
            vtx2xyz=obj.vtxxyz2xyz,
            list_elem2vtx=[
                DrawerMesh.ElementInfo(index=edge2vtx, color=(0, 0, 0), mode=moderngl.LINES),
                DrawerMesh.ElementInfo(index=self.tri2vtx, color=(1, 1, 1), mode=moderngl.TRIANGLES)]
        )

        # sphere
        self.drawer_sphere = DrawerSpheres.Drawer()

        # add shapes
        self.shape2pos = numpy.array([obj.vtxxyz2xyz.flatten().copy()], dtype=numpy.float32)
        self.weights = numpy.array([1.], dtype=numpy.float32)
        for path in paths[1:]:
            self.add_shape(path)

        self.vtx_pick = -1
        self.markers = {}

        QtWidgets.QMainWindow.__init__(self)
        self.resize(640, 480)
        self.setWindowTitle('Mesh Viewer')
        self.glwidget = QGLWidgetViewer3.QtGLWidget_Viewer3([self.drawer_mesh, self.drawer_sphere])
        self.glwidget.mousePressCallBack.append(self.mouse_press_callback)
        self.glwidget.mouseMoveCallBack.append(self.mouse_move_callback)
        self.glwidget.mouseDoubleClickCallBack.append(self.mouse_doubleclick_callback)
        self.setCentralWidget(self.glwidget)

    def add_shape(self, path1):
        obj = WavefrontObj.load(path1)
        vtx2xyz1 = obj.vtxxyz2xyz.flatten().astype(numpy.float32)
        self.shape2pos = numpy.vstack([self.shape2pos, vtx2xyz1])
        self.weights = numpy.append(self.weights, 0).astype(numpy.float32)

    def mouse_press_callback(self, event):
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
            return
        if event.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier:
            return
        vtx2xyz = self.weights.transpose().dot(self.shape2pos).reshape(-1, 3)  # current shape
        src, direction = self.glwidget.nav.picking_ray()
        self.vtx_pick = TriMesh.pick_vertex(
            self.tri2vtx, vtx2xyz.astype(numpy.float32),
            numpy.array(src.xyz).astype(numpy.float32),
            numpy.array(direction.xyz).astype(numpy.float32))
        print(self.vtx_pick)
        if self.vtx_pick == -1:
            return
        assert self.vtx_pick < self.tri2vtx.shape[0]
        mvp = self.glwidget.nav.projection_matrix() * self.glwidget.nav.modelview_matrix()
        mvp = numpy.array(mvp).transpose()
        trg = (self.glwidget.nav.cursor_x, self.glwidget.nav.cursor_y)
        self.markers[self.vtx_pick] = [mvp, trg]
        rad = self.glwidget.nav.view_height / self.glwidget.nav.scale * 0.03
        self.drawer_sphere.list_sphere = []
        for key_markers in self.markers.keys():
            self.drawer_sphere.list_sphere.append(
                DrawerSpheres.SphereInfo(rad=0.03, pos=vtx2xyz[key_markers], color=(1., 0., 0)))
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
        self.weights = BlendShape.direct_manipulation_absolute(self.shape2pos, self.markers)
        vtx2xyz = self.weights.transpose().dot(self.shape2pos).reshape(-1, 3).copy()
        self.drawer_mesh.update_position(vtx2xyz)
        rad = self.glwidget.nav.view_height / self.glwidget.nav.scale * 0.03
        self.drawer_sphere.list_sphere = []
        for key_markers in self.markers.keys():
            self.drawer_sphere.list_sphere.append(
                DrawerSpheres.SphereInfo(rad=0.03, pos=vtx2xyz[key_markers], color=(1., 0., 0)))
        self.glwidget.updateGL()

    def mouse_doubleclick_callback(self, event):
        vtx2xyz = self.weights.transpose().dot(self.shape2pos).reshape(-1, 3)  # current shape
        src, direction = self.glwidget.nav.picking_ray()
        vtx_pick = TriMesh.pick_vertex(
            numpy.array(src.xyz).astype(numpy.float32),
            numpy.array(direction.xyz).astype(numpy.float32),
            vtx2xyz.astype(numpy.float32), self.tri2vtx)
        if vtx_pick not in self.markers:
            return
        self.markers.pop(vtx_pick)
        self.vtx_pick = -1
        self.weights = BlendShape.direct_manipulation_absolute(self.shape2pos, self.markers)
        vtx2xyz = self.weights.transpose().dot(self.shape2pos).reshape(-1, 3).copy()
        self.drawer_mesh.update_position(vtx2xyz)
        rad = self.glwidget.nav.view_height / self.glwidget.nav.scale * 0.03
        self.drawer_sphere.list_sphere = []
        for key_markers in self.markers.keys():
            self.drawer_sphere.list_sphere.append(
                DrawerSpheres.SphereInfo(rad=0.03, pos=vtx2xyz[key_markers], color=(1., 0., 0)))
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
