from pathlib import Path
import moderngl
import numpy
from PIL import Image
from PyQt5 import QtWidgets
import del_msh
from util_moderngl_qt.drawer_meshpostex import DrawerMesPosTex, ElementInfo
from util_moderngl_qt.qtglwidget_viewer3_texture import QtGLWidget_Viewer3_Texture

if __name__ == '__main__':
    path_obj = Path('.') / 'asset' / 'Babi' / 'Babi.obj'
    vtx2xyz, vtx2uv, elem2idx, idx2vtx_xyz, idx2vtx_uv = del_msh.load_wavefront_obj(str(path_obj))
    vtx2xyz[:, 0] -= (vtx2xyz[:, 0].max() + vtx2xyz[:, 0].min()) * 0.5
    vtx2xyz[:, 1] -= (vtx2xyz[:, 1].max() + vtx2xyz[:, 1].min()) * 0.5
    vtx2xyz[:, 2] -= (vtx2xyz[:, 2].max() + vtx2xyz[:, 2].min()) * 0.5
    idx2vtx_xyz = idx2vtx_xyz.astype(numpy.uint64)

    tri2vtx_xyz = del_msh.triangles_from_polygon_mesh(elem2idx, idx2vtx_xyz)
    tri2vtx_uv = del_msh.triangles_from_polygon_mesh(elem2idx, idx2vtx_uv)

    tri2uni, uni2xyz, uni2uv = del_msh.unify_triangle_indices_of_xyz_and_uv(tri2vtx_xyz, vtx2xyz, tri2vtx_uv, vtx2uv)

    img = Image.open("asset/Babi/Tex.png")
    img = numpy.asarray(img)

    with QtWidgets.QApplication([]) as app:
        drawer = DrawerMesPosTex(
            list_elem2vtx=[
                ElementInfo(index=tri2uni, color=None, mode=moderngl.TRIANGLES)],
            vtx2xyz=uni2xyz,
            vtx2uv=uni2uv
        )
        win = QtGLWidget_Viewer3_Texture(drawer, img)
        win.show()
        app.exec()
