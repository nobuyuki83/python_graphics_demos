from pathlib import Path
import moderngl
import numpy
from PIL import Image
from PyQt5 import QtWidgets
import del_msh
from del_msh import WavefrontObj
from util_moderngl_qt import DrawerMeshTexture, QGLWidgetViewer3Texture

if __name__ == '__main__':
    path_obj = Path('.') / 'asset' / 'Babi' / 'Babi.obj'
    obj = WavefrontObj.load(str(path_obj))
    obj.vtxxyz2xyz = del_msh.centerize_scale_points(obj.vtxxyz2xyz, scale=1.8)

    tri2vtxxyz = del_msh.triangles_from_polygon_mesh(obj.elem2idx, obj.idx2vtxxyz)
    tri2vtxuv = del_msh.triangles_from_polygon_mesh(obj.elem2idx, obj.idx2vtxuv)

    tri2uni, uni2vtxxyz, uni2vtxuv = del_msh.unify_two_indices_of_triangle_mesh(tri2vtxxyz, tri2vtxuv)

    uni2xyz = numpy.ndarray((uni2vtxxyz.shape[0], 3), obj.vtxxyz2xyz.dtype)
    uni2xyz[:, :] = obj.vtxxyz2xyz[uni2vtxxyz[:], :]

    uni2uv = numpy.ndarray((uni2vtxuv.shape[0], 2), obj.vtxuv2uv.dtype)
    uni2uv[:, :] = obj.vtxuv2uv[uni2vtxuv[:], :]

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
