import pathlib
import moderngl
from PyQt5 import QtWidgets, QtCore, QtGui
import numpy

from util_moderngl_qt.drawer_meshpos import DrawerMesPos, ElementInfo
from util_moderngl_qt.drawer_meshunindex import DrawerMeshUnindex
import util_moderngl_qt.qtglwidget_viewer3
import del_msh
import del_srch


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        path_file = pathlib.Path('.') / 'asset' / 'bunny_1k.obj'
        self.tri2vtx, vtx2xyz = del_msh.load_wavefront_obj_as_triangle_mesh(str(path_file))
        self.vtx2xyz = del_msh.centerize_scale_3d_points(vtx2xyz)

        edge2vtx = del_msh.edges_of_uniform_mesh(self.tri2vtx, self.vtx2xyz.shape[0])
        drawer_edge = DrawerMesPos(
            vtx2xyz=self.vtx2xyz.astype(numpy.float32),
            list_elem2vtx=[
                ElementInfo(index=edge2vtx, color=(0, 0, 0), mode=moderngl.LINES)]
        )

        tri2vtx2xyz = del_msh.unidex_vertex_attribute_for_triangle_mesh(self.tri2vtx, self.vtx2xyz)
        drawer_face = DrawerMeshUnindex(
            tri2node2xyz=tri2vtx2xyz.astype(numpy.float32),
        )

        self.cur_dist = -1
        self.tri2dist = numpy.zeros(self.tri2vtx.shape[0], dtype=numpy.uint64)
        self.tri2flag = numpy.zeros(self.tri2vtx.shape[0], dtype=numpy.int32)
        self.elsuel = del_msh.elsuel_uniform_mesh_polygon(self.tri2vtx, self.vtx2xyz.shape[0])

        super().__init__()
        self.resize(640, 480)
        self.setWindowTitle('Mesh Viewer')
        self.glwidget = util_moderngl_qt.qtglwidget_viewer3.QtGLWidget_Viewer3(
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
        pos, tri_index = del_srch.first_intersection_ray_meshtri3(
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
        self.tri2dist = del_msh.topological_distance_on_uniform_mesh(tri_index, self.elsuel)

    def mouse_move_callback(self, event: QtGui.QMouseEvent) -> None:
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
            return
        if event.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier:
            return
        src, direction = self.glwidget.nav.picking_ray()
        pos, tri_index = del_srch.first_intersection_ray_meshtri3(
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
