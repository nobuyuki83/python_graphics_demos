import moderngl
import pyrr
from PyQt5 import QtOpenGL, QtWidgets, QtCore
import numpy

from util_moderngl_qt import DrawerMesh


class MyQtGLWidget(QtOpenGL.QGLWidget):

    def __init__(self, drawer, parent=None):
        self.parent = parent
        fmt = QtOpenGL.QGLFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QtOpenGL.QGLFormat.CoreProfile)
        fmt.setSampleBuffers(True)
        super(MyQtGLWidget, self).__init__(fmt, None)
        #
        self.resize(640, 480)
        self.setWindowTitle('Mesh Viewer')
        self.drawer = drawer
        self.ctx = None

    def initializeGL(self):
        self.ctx = moderngl.create_context()
        self.drawer.init_gl(self.ctx)

    def paintGL(self):
        self.ctx.clear(1.0, 0.8, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        mvp = pyrr.Matrix44.identity()
        self.drawer.paint_gl(mvp)

    def resizeGL(self, width, height):
        width = max(2, width)
        height = max(2, height)
        self.ctx.viewport = (0, 0, width, height)


if __name__ == '__main__':
    V = numpy.array([
        [-0.5, -0.5, 0],
        [+0.5, -0.5, 0],
        [+0, +0.5, 0]], dtype=numpy.float32)
    F = numpy.array([
        [0, 1, 2]], dtype=numpy.uint32)
    E = numpy.array([
        [0, 1],
        [1, 2],
        [2, 0]], dtype=numpy.uint32)

    with QtWidgets.QApplication([]) as app:
        drawer = DrawerMesh.Drawer(
            vtx2xyz=V,
            list_elem2vtx=[
                DrawerMesh.ElementInfo(index=F, color=(1, 0, 0), mode=moderngl.TRIANGLES),
                DrawerMesh.ElementInfo(index=E, color=(0, 0, 0), mode=moderngl.LINES)]
        )
        win = MyQtGLWidget(drawer)
        win.show()
        app.exec()
