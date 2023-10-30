import moderngl
from PyQt5 import QtWidgets
import numpy

from pathlib import Path

from util_moderngl_qt.drawer_mesh import DrawerMesh, ElementInfo
import util_moderngl_qt.qtglwidget_viewer3

import del_msh
from del_msh import WavefrontObj

if __name__ == "__main__":
    newpath = Path('.') / 'asset' / 'HorseSwap.obj'
    # newpath = Path('.') / 'asset' / 'Babi' / 'Babi.obj'

    obj = WavefrontObj.load(str(newpath))
    del_msh.centerize_scale_points(obj.vtxxyz2xyz, scale=1.8)
    edge2vtx = del_msh.edges_of_polygon_mesh(obj.elem2idx, obj.idx2vtxxyz, obj.vtxxyz2xyz.shape[0])
    tri2vtx = del_msh.triangles_from_polygon_mesh(obj.elem2idx, obj.idx2vtxxyz)

    with QtWidgets.QApplication([]) as app:
        drawer = DrawerMesh(
            vtx2xyz=obj.vtxxyz2xyz,
            list_elem2vtx=[
                ElementInfo(index=edge2vtx, color=(0, 0, 0), mode=moderngl.LINES),
                ElementInfo(index=tri2vtx, color=(1, 1, 1), mode=moderngl.TRIANGLES)]
        )
        win = util_moderngl_qt.qtglwidget_viewer3.QtGLWidget_Viewer3([drawer])
        win.show()
        app.exec()




