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
    vtx2xyz, vtx2uv, vtx2nrm, \
        elem2idx, idx2vtx_xyz, idx2vtx_uv, idx2vtx_nrm, \
        elem2group, group2name,\
        elem2mtl, mtl2name, mtl_file_name =  del_msh.load_wavefront_obj(str(path_obj))
    vtx2xyz[:, 0] -= (vtx2xyz[:, 0].max() + vtx2xyz[:, 0].min()) * 0.5
    vtx2xyz[:, 1] -= (vtx2xyz[:, 1].max() + vtx2xyz[:, 1].min()) * 0.5
    vtx2xyz[:, 2] -= (vtx2xyz[:, 2].max() + vtx2xyz[:, 2].min()) * 0.5
    idx2vtx_xyz = idx2vtx_xyz.astype(numpy.uint64)

    tri2vtx_xyz = del_msh.triangles_from_polygon_mesh(elem2idx, idx2vtx_xyz)
    tri2vtx_uv = del_msh.triangles_from_polygon_mesh(elem2idx, idx2vtx_uv)

    tri2uni, uni2vtxxyz, uni2vtxuv = del_msh.unify_two_indices_of_triangle_mesh(tri2vtx_xyz, tri2vtx_uv)

    uni2xyz = numpy.ndarray((uni2vtxxyz.shape[0],3),vtx2xyz.dtype)
    uni2xyz[:,:] = vtx2xyz[uni2vtxxyz[:],:]

    uni2uv = numpy.ndarray((uni2vtxuv.shape[0],2),vtx2uv.dtype)
    uni2uv[:,:] = vtx2uv[uni2vtxuv[:],:]

    img = Image.open("asset/Babi/Tex.png")
    img = numpy.asarray(img)

    with QtWidgets.QApplication([]) as app:
        drawer = DrawerMesPosTex(
            list_elem2vtx=[
                ElementInfo(index=tri2uni, color=None, mode=moderngl.TRIANGLES)],
            vtx2xyz=uni2xyz,
            vtx2uv=uni2uv
        )
        win = QtGLWidget_Viewer3_Texture([drawer], img)
        win.show()
        app.exec()
