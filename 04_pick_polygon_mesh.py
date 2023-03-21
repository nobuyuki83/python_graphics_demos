from pathlib import Path

import del_msh
import del_srch
import moderngl
import numpy
import pyrr
import util_moderngl_qt.qtglwidget_viewer3
from PyQt5 import QtWidgets, QtCore
from util_moderngl_qt.drawer_meshpos import DrawerMesPos, ElementInfo
from util_moderngl_qt.drawer_transform import DrawerTransform


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):

        newpath = Path('.') / 'asset' / 'HorseSwap.obj'
        vtx2xyz, vtx2uv, elem2idx, idx2vtx_xyz, idx2vtx_uv = del_msh.load_wavefront_obj(str(newpath))
        self.vtx2xyz = del_msh.centerize_scale_3d_points(vtx2xyz)
        idx2vtx_xyz = idx2vtx_xyz.astype(numpy.uint64)

        edge2vtx = del_msh.edges_of_polygon_mesh(elem2idx, idx2vtx_xyz, self.vtx2xyz.shape[0])
        self.tri2vtx = del_msh.triangles_from_polygon_mesh(elem2idx, idx2vtx_xyz)

        drawer_triquadmesh3 = DrawerMesPos(
            vtx2xyz=self.vtx2xyz.astype(numpy.float32),
            list_elem2vtx=[
                ElementInfo(index=edge2vtx, color=(0, 0, 0), mode=moderngl.LINES),
                ElementInfo(index=self.tri2vtx, color=(1, 1, 1), mode=moderngl.TRIANGLES)]
        )

        sphere_tri2vtx, shere_vtx2xyz = del_msh.sphere_meshtri3(1., 32, 32)
        self.drawer_sphere = DrawerMesPos(vtx2xyz=shere_vtx2xyz, list_elem2vtx=[
            ElementInfo(index=sphere_tri2vtx, color=(1., 0., 0.), mode=moderngl.TRIANGLES)])
        self.drawer_sphere = DrawerTransform(self.drawer_sphere)
        self.drawer_sphere.transform = pyrr.Matrix44.from_scale((0.05, 0.05, 0.05))

        QtWidgets.QMainWindow.__init__(self)
        self.resize(640, 480)
        self.setWindowTitle('Mesh Viewer')
        self.glwidget = util_moderngl_qt.qtglwidget_viewer3.QtGLWidget_Viewer3(
            [drawer_triquadmesh3, self.drawer_sphere])
        self.glwidget.mousePressCallBack.append(self.mouse_press_callback)
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
        self.drawer_sphere.is_visible = False
        if tri_index != -1:
            self.drawer_sphere.is_visible = True
            self.drawer_sphere.transform = pyrr.Matrix44.from_translation(pos) * pyrr.Matrix44.from_scale(
                (0.03, 0.03, 0.03))
        self.glwidget.updateGL()


if __name__ == "__main__":
    with QtWidgets.QApplication([]) as app:
        win = MainWindow()
        win.show()
        app.exec()
