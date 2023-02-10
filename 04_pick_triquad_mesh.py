import moderngl
import pyrr
from PyQt5 import QtWidgets, QtCore
import numpy

from pathlib import Path

from util_moderngl_qt.drawer_meshpos import DrawerMesPos, ElementInfo
from util_moderngl_qt.drawer_transform import DrawerTransformer
import util_moderngl_qt.qtglwidget_viewer3
import del_msh, del_srch


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):

        newpath = Path('.') / 'asset' / 'HorseSwap.obj'
        vtx2xyz, elem2idx, idx2vtx_xyz = del_msh.load_wavefront_obj(str(newpath))
        self.vtx_xyz = del_msh.centerize_scale_3d_points(vtx2xyz)
        idx2vtx_xyz = idx2vtx_xyz.astype(numpy.uint64)

        edge2vtx = del_msh.edges_of_triquad_mesh(elem2idx, idx2vtx_xyz, self.vtx_xyz.shape[0])
        self.tri_vtx = del_msh.triangles_from_triquad_mesh(elem2idx, idx2vtx_xyz)

        drawer_triquadmesh3 = DrawerMesPos(
            V=self.vtx_xyz.astype(numpy.float32),
            element=[
                ElementInfo(index=edge2vtx.astype(numpy.uint32), color=(0, 0, 0), mode=moderngl.LINES),
                ElementInfo(index=self.tri_vtx.astype(numpy.uint32), color=(1, 1, 1), mode=moderngl.TRIANGLES)]
        )

        shere_vtx2xyz, sphere_tri2vtx = del_msh.sphere_meshtri3(1., 32, 32)
        self.drawer_sphere = DrawerMesPos(shere_vtx2xyz, element=[
            ElementInfo(index=sphere_tri2vtx.astype(numpy.uint32), color=(1.,0.,0.), mode=moderngl.TRIANGLES)])
        self.drawer_sphere = DrawerTransformer(self.drawer_sphere)
        self.drawer_sphere.transform = pyrr.Matrix44.from_scale((0.05,0.05,0.05))

        QtWidgets.QMainWindow.__init__(self)
        self.resize(640, 480)
        self.setWindowTitle('Mesh Viewer')
        self.glwidget = util_moderngl_qt.qtglwidget_viewer3.QtGLWidget_Viewer3([drawer_triquadmesh3, self.drawer_sphere])
        self.glwidget.mousePressCallBack.append(self.mouse_press_callback)
        self.setCentralWidget(self.glwidget)

    def mouse_press_callback(self, event):
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
            return
        if event.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier:
            return
        src, dir = self.glwidget.nav.picking_ray()
        pos, tri_index = del_srch.first_intersection_ray_meshtri3(
            numpy.array(src.xyz).astype(numpy.float32), numpy.array(dir.xyz).astype(numpy.float32),
            self.vtx_xyz, self.tri_vtx)
        self.drawer_sphere.is_visible = False
        if tri_index != -1:
            self.drawer_sphere.is_visible = True
            self.drawer_sphere.transform = pyrr.Matrix44.from_translation(pos) * pyrr.Matrix44.from_scale((0.03, 0.03, 0.03))
        self.glwidget.updateGL()




if __name__ == "__main__":

    with QtWidgets.QApplication([]) as app:
        win = MainWindow()
        win.show()
        app.exec()




