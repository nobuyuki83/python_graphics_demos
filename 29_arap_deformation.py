import math
#
import numpy
import moderngl
import pyrr
from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer
#
from util_moderngl_qt import DrawerMesh, QGLWidgetViewer3
from del_msh import TriMesh
import del_ls

class DialogForParameters(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.btn_spoke = QtWidgets.QRadioButton('Spoke', self)
        self.btn_spokerim = QtWidgets.QRadioButton('Spoke && Rim', self)
        rb_layout = QtWidgets.QHBoxLayout()
        rb_layout.addWidget(self.btn_spoke)
        rb_layout.addWidget(self.btn_spokerim)
        self.btn_spoke.setChecked(True)

        from PyQt5.QtCore import Qt
        self.label_iter = QtWidgets.QLabel('iteration: 2', self)
        self.slider = QtWidgets.QSlider(orientation=Qt.Orientation.Horizontal, parent=self)
        self.slider.setTickInterval(1)
        self.slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksAbove)
        self.slider.setMaximum(5)
        self.slider.setMinimum(1)
        self.slider.setValue(2)
        self.slider.valueChanged.connect(self.slider_value_change)
        sl_layout = QtWidgets.QHBoxLayout()
        sl_layout.addWidget(self.slider)
        sl_layout.addWidget(self.label_iter)

        v_layout = QtWidgets.QVBoxLayout()
        v_layout.addLayout(rb_layout)
        v_layout.addLayout(sl_layout)

        self.setLayout(v_layout)
        #
        self.setGeometry(300, 300, 250, 50)
        self.setWindowTitle('Parameter Input Dialog')
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.show()

    def slider_value_change(self):
        val = self.slider.value()
        self.label_iter.setText(f"iteration: {val}")


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        self.tri2vtx, self.vtx2xyz_ini = TriMesh.capsule(
            radius=0.2, height=1.6,
            ndiv_height=18, ndiv_theta=18, ndiv_longtitude=4)
        self.vtx2rot = numpy.zeros((self.vtx2xyz_ini.shape[0], 3, 3))
        self.vtxs0 = []
        self.vtxs1 = []
        for iv in range(self.vtx2xyz_ini.shape[0]):
            y = self.vtx2xyz_ini[iv, 1]
            if y < -0.65:
                self.vtxs0.append(iv)
            if y > +0.65:
                self.vtxs1.append(iv)
        self.vtx2xyz_def = self.vtx2xyz_ini.copy()
        #
        vtx2idx, idx2vtx = TriMesh.vtx2vtx(self.tri2vtx, self.vtx2xyz_ini.shape[0])
        # print(vtx2idx, idx2vtx)
        self.sparse = del_ls.SparseSquareMatrix(vtx2idx, idx2vtx)
        self.sparse.set_zero()
        from del_fem.del_fem import merge_hessian_mesh_laplacian_on_trimesh3
        merge_hessian_mesh_laplacian_on_trimesh3(
            self.tri2vtx, self.vtx2xyz_ini,
            self.sparse.row2idx, self.sparse.idx2col,
            self.sparse.row2val, self.sparse.idx2val)
        self.r_vec = numpy.zeros_like(self.vtx2xyz_ini)
        self.penalty = 1.0e+2
        self.sparse.row2val[self.vtxs0] += self.penalty
        self.sparse.row2val[self.vtxs1] += self.penalty
        #
        edge2vtx = TriMesh.edge2vtx(self.tri2vtx, self.vtx2xyz_ini.shape[0])
        self.drawer_edge = DrawerMesh.Drawer(
            vtx2xyz=self.vtx2xyz_def,
            list_elem2vtx=[
                DrawerMesh.ElementInfo(index=edge2vtx, color=(0, 0, 0), mode=moderngl.LINES),
                DrawerMesh.ElementInfo(index=self.tri2vtx, color=(1, 1, 1), mode=moderngl.TRIANGLES)
            ]
        )

        self.dialog = DialogForParameters()

        super().__init__()
        self.resize(640, 480)
        self.setWindowTitle('Mesh Viewer')
        self.glwidget = QGLWidgetViewer3.QtGLWidget_Viewer3(
            [self.drawer_edge])
        self.setCentralWidget(self.glwidget)
        #
        self.timer = QTimer()
        self.timer.setInterval(30)
        self.timer.timeout.connect(self.step_time)
        self.timer.start()
        self.iframe = 0
        self.dialog.show()

    def step_time(self):
        iframe = self.iframe
        t0 = pyrr.Matrix44.from_translation((0., -0.6, 0.))
        ry = pyrr.Matrix44.from_y_rotation(0.5 * math.sin(0.05 * iframe))
        rz = pyrr.Matrix44.from_z_rotation(0.8 * math.sin(0.03 * iframe))
        t1 = pyrr.Matrix44.from_translation((0.4 * math.sin(0.03 * iframe), +0.4 + 0.2 * math.cos(0.03 * iframe), 0.))
        # t0 = pyrr.Matrix44.from_translation((0., -0.8, 0.))
        # ry = pyrr.Matrix44.from_y_rotation(0.5 * math.sin(0.05 * iframe))
        # rz = pyrr.Matrix44.from_z_rotation(0.3 * math.sin(0.07 * iframe))
        # t1 = pyrr.Matrix44.from_translation((0.4 * math.sin(0.03 * iframe), +0.6 + 0.3 * math.cos(0.05 * iframe), 0.))
        a = t1 * rz * ry * t0
        a = a.transpose()
        self.vtx2xyz_def[self.vtxs0] = self.vtx2xyz_ini[self.vtxs0]
        # affine transformation
        self.vtx2xyz_def[self.vtxs1] = a.matrix33.dot(self.vtx2xyz_ini[self.vtxs1].transpose()).transpose()
        self.vtx2xyz_def[self.vtxs1] += numpy.array([a.m14, a.m24, a.m34])
        for iter in range(0, self.dialog.slider.value()):
            if self.dialog.btn_spokerim.isChecked():
                from del_fem.del_fem import optimal_rotations_arap_spoke_rim_trimesh3
                optimal_rotations_arap_spoke_rim_trimesh3(
                    self.tri2vtx,
                    self.vtx2xyz_ini, self.vtx2xyz_def,
                    self.vtx2rot)
                #
                from del_fem.del_fem import residual_arap_spoke_rim_trimesh3
                residual_arap_spoke_rim_trimesh3(
                    self.tri2vtx,
                    self.vtx2xyz_ini, self.vtx2xyz_def,
                    self.vtx2rot,
                    self.r_vec)
            else:
                from del_fem.del_fem import optimal_rotations_arap_spoke
                optimal_rotations_arap_spoke(
                    self.vtx2xyz_ini, self.vtx2xyz_def,
                    self.sparse.row2idx, self.sparse.idx2col, self.sparse.idx2val,
                    self.vtx2rot)
                #
                from del_fem.del_fem import residual_arap_spoke
                residual_arap_spoke(
                    self.vtx2xyz_ini, self.vtx2xyz_def,
                    self.sparse.row2idx, self.sparse.idx2col, self.sparse.idx2val,
                    self.vtx2rot,
                    self.r_vec)
            norm_r0 = numpy.linalg.norm(self.r_vec)
            x_vec, conv_hist = self.sparse.solve_cg(self.r_vec, conv_ratio_tol=1.0e-5)
            # print("cg_iterations =", len(conv_hist), conv_hist[len(conv_hist) - 1], norm_r0)
            self.vtx2xyz_def += x_vec
        #
        self.drawer_edge.update_position(self.vtx2xyz_def)
        self.iframe += 1
        self.glwidget.update()


def main():
    with QtWidgets.QApplication([]) as app:
        win = MainWindow()
        win.show()
        app.exec()


if __name__ == "__main__":
    main()
