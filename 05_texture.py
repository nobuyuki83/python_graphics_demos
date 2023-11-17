import moderngl
import pyrr
from PyQt5 import QtOpenGL, QtWidgets
import numpy
from PIL import Image
from util_moderngl_qt import DrawerMeshTexture


class MyQtGLWidget(QtOpenGL.QGLWidget):

    def __init__(self, drawer, img: numpy.ndarray, parent=None):
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
        self.img = img

    def initializeGL(self):
        self.ctx = moderngl.create_context()
        self.drawer.init_gl(self.ctx)
        img2 = numpy.flip(self.img, axis=0)  # flip upside down
        self.texture = self.ctx.texture((img.shape[0], img.shape[1]), img.shape[2], img2.tobytes())
        del self.img

    def paintGL(self):
        self.ctx.clear(1.0, 0.8, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        mvp = pyrr.Matrix44.identity()
        self.texture.use(location=0)
        self.drawer.paint_gl_texture(mvp, texture_location=0)

    def resizeGL(self, width, height):
        width = max(2, width)
        height = max(2, height)
        self.ctx.viewport = (0, 0, width, height)


if __name__ == '__main__':
    vtx2xyz = numpy.array([
        [-0.5, -0.5, 0],
        [+0.5, -0.5, 0],
        [+0.5, +0.5, 0],
        [-0.5, +0.5, 0]], dtype=numpy.float32)
    tri2vtx = numpy.array([
        [0, 1, 2],
        [0, 2, 3]], dtype=numpy.uint32)
    vtx2uv = numpy.array([
        [0., 0.],
        [1., 0.],
        [1., 1.],
        [0., 1.]], dtype=numpy.float32)
    edge2vtx = numpy.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [0, 2]], dtype=numpy.uint32)

    img = Image.open("asset/tesla.png")
    img = numpy.asarray(img)

    with QtWidgets.QApplication([]) as app:
        drawer = DrawerMeshTexture.Drawer(
            list_elem2vtx=[
                DrawerMeshTexture.ElementInfo(index=tri2vtx, color=None, mode=moderngl.TRIANGLES),
                DrawerMeshTexture.ElementInfo(index=edge2vtx, color=(0, 0, 0), mode=moderngl.LINES)],
            vtx2xyz=vtx2xyz,
            vtx2uv=vtx2uv
        )
        win = MyQtGLWidget(drawer, img)
        win.show()
        app.exec()
