from pathlib import Path
import moderngl
import numpy
from PIL import Image
from PyQt5 import QtWidgets

from del_msh import WavefrontObj
from util_moderngl_qt import DrawerMeshTexture, QGLWidgetViewer3Texture

if __name__ == '__main__':
    path_obj = Path('.') / 'asset' / 'Babi' / 'Babi.obj'
    obj = WavefrontObj.load(str(path_obj), is_centerize=True, normalized_size=1.8)
    tri2uni, uni2xyz, uni2uv, _, _ = obj.triangle_mesh_with_uv()

    img = Image.open("asset/Babi/Tex.png")
    img = numpy.asarray(img)

    with QtWidgets.QApplication([]) as app:
        drawer = DrawerMeshTexture.Drawer(
            list_elem2vtx=[
                DrawerMeshTexture.ElementInfo(index=tri2uni, color=None, mode=moderngl.TRIANGLES)],
            vtx2xyz=uni2xyz,
            vtx2uv=uni2uv
        )
        win = QGLWidgetViewer3Texture.QtGLWidget_Viewer3_Texture([drawer], img)
        win.show()
        app.exec()
