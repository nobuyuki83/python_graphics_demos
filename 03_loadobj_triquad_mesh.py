import moderngl
from PyQt5 import QtWidgets
import numpy

from pathlib import Path

from util_moderngl_qt.drawer_meshpos import DrawerMesPos, ElementInfo
import util_moderngl_qt.qtglwidget_viewer3
import del_msh

if __name__ == "__main__":

    newpath = Path('.') / 'asset' / 'HorseSwap.obj'
    # newpath = Path('.') / 'asset' / 'Babi' / 'Babi.obj'
    vtx2xyz, vtx2uv, elem2idx, idx2vtx_xyz, idx2vtx_uv = del_msh.load_wavefront_obj(str(newpath))
    vtx2xyz[:, 0] -= (vtx2xyz[:, 0].max() + vtx2xyz[:, 0].min()) * 0.5
    vtx2xyz[:, 1] -= (vtx2xyz[:, 1].max() + vtx2xyz[:, 1].min()) * 0.5
    vtx2xyz[:, 2] -= (vtx2xyz[:, 2].max() + vtx2xyz[:, 2].min()) * 0.5
    idx2vtx_xyz = idx2vtx_xyz.astype(numpy.uint64)

    edge2vtx = del_msh.edges_of_polygon_mesh(elem2idx, idx2vtx_xyz, vtx2xyz.shape[0])
    tri2vtx = del_msh.triangles_from_polygon_mesh(elem2idx, idx2vtx_xyz)

    with QtWidgets.QApplication([]) as app:
        drawer = DrawerMesPos(
            V=vtx2xyz.astype(numpy.float32),
            element=[
                ElementInfo(index=edge2vtx.astype(numpy.uint32), color=(0, 0, 0), mode=moderngl.LINES),
                ElementInfo(index=tri2vtx.astype(numpy.uint32), color=(1, 1, 1), mode=moderngl.TRIANGLES)]
        )
        win = util_moderngl_qt.qtglwidget_viewer3.QtGLWidget_Viewer3([drawer])
        win.show()
        app.exec()




