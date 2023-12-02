import pathlib
import moderngl
from PyQt5 import QtWidgets, QtCore, QtGui
import numpy

from util_moderngl_qt import DrawerMesh, QGLWidgetViewer3, DrawerMeshUnindex
from del_msh import TriMesh


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        path_file = pathlib.Path('.') / 'asset' / 'bunny_1k.obj'
        self.tri2vtx, self.vtx2xyz = TriMesh.load_wavefront_obj(str(path_file), is_centerize=True, normalized_size=1.8)

        edge2vtx = TriMesh.edge2vtx(self.tri2vtx, self.vtx2xyz.shape[0])
        drawer_edge = DrawerMesh.Drawer(
            vtx2xyz=self.vtx2xyz,
            list_elem2vtx=[
                DrawerMesh.ElementInfo(index=edge2vtx, color=(0, 0, 0), mode=moderngl.LINES)]
        )

        tri2vtx2xyz = TriMesh.unindexing(self.tri2vtx, self.vtx2xyz)
        drawer_face = DrawerMeshUnindex.Drawer(
            elem2node2xyz=tri2vtx2xyz.astype(numpy.float32),
        )

        self.cur_dist = -1
        self.tri2dist = numpy.zeros(self.tri2vtx.shape[0], dtype=numpy.uint64)
        self.tri2flag = numpy.zeros(self.tri2vtx.shape[0], dtype=numpy.int32)
        self.tri2tri = TriMesh.tri2tri(self.tri2vtx, self.vtx2xyz.shape[0])

        super().__init__()
        self.resize(640, 480)
        self.setWindowTitle('Mesh Viewer')
        self.glwidget = QGLWidgetViewer3.QtGLWidget_Viewer3(
            [drawer_face, drawer_edge])
        self.glwidget.mousePressCallBack.append(self.mouse_press_callback)
        self.glwidget.mouseMoveCallBack.append(self.mouse_move_callback)
        self.glwidget.mouseReleaseCallBack.append(self.mouse_release_callback)
        self.setCentralWidget(self.glwidget)

    def mouse_press_callback(self, event):
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
            return
        if event.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier:
            return
        src, direction = self.glwidget.nav.picking_ray()
        pos, tri_index = TriMesh.first_intersection_ray(
            numpy.array(src.xyz).astype(numpy.float32), numpy.array(direction.xyz).astype(numpy.float32),
            self.vtx2xyz, self.tri2vtx)
        if tri_index < 0:
            return
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            self.tri2flag[tri_index] = 0
        else:
            self.tri2flag[tri_index] = 1
        self.cur_dist = -1
        self.update_visualization(True)
        self.tri2dist = TriMesh.tri2distance(tri_index, self.tri2tri)

    def mouse_move_callback(self, event: QtGui.QMouseEvent) -> None:
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
            return
        if event.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier:
            return
        src, direction = self.glwidget.nav.picking_ray()
        pos, tri_index = TriMesh.first_intersection_ray(
            numpy.array(src.xyz).astype(numpy.float32), numpy.array(direction.xyz).astype(numpy.float32),
            self.vtx2xyz, self.tri2vtx)
        if tri_index == -1:
            return
        self.cur_dist = self.tri2dist[tri_index]
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            self.update_visualization(True)
        else:
            self.update_visualization(False)

    def mouse_release_callback(self, event: QtGui.QMouseEvent) -> None:
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            self.tri2flag[self.tri2dist <= self.cur_dist] = 0
        else:
            self.tri2flag[self.tri2dist <= self.cur_dist] = 1
        self.cur_dist = -1
        self.update_visualization(True)

    def update_visualization(self, is_unselect: bool):
        num_tri = self.tri2vtx.shape[0]
        tri2node2rgb = numpy.zeros([num_tri, 3, 3], dtype=numpy.float32)
        tri2node2rgb[self.tri2flag == 0, :, 0] = 1.
        tri2node2rgb[self.tri2flag == 0, :, 1] = 1.
        tri2node2rgb[self.tri2flag == 0, :, 2] = 1.
        tri2node2rgb[self.tri2flag == 1, :, 0] = 1
        tri2node2rgb[self.tri2flag == 1, :, 1] = 0
        tri2node2rgb[self.tri2flag == 1, :, 2] = 0
        if not is_unselect:
            tri2node2rgb[self.tri2dist <= self.cur_dist, :, 0] = 1.0
            tri2node2rgb[self.tri2dist <= self.cur_dist, :, 1] = 0.5
            tri2node2rgb[self.tri2dist <= self.cur_dist, :, 2] = 0.5
        else:
            tri2node2rgb[self.tri2dist <= self.cur_dist, :, 0] = 1.0
            tri2node2rgb[self.tri2dist <= self.cur_dist, :, 1] = 0.7
            tri2node2rgb[self.tri2dist <= self.cur_dist, :, 2] = 1.0
        self.glwidget.list_drawer[0].update_color(tri2node2rgb)
        self.glwidget.update()


def main():
    with QtWidgets.QApplication([]) as app:
        win = MainWindow()
        win.show()
        app.exec()


if __name__ == "__main__":
    main()
