import typing
from pathlib import Path
import moderngl
from PyQt5 import QtWidgets
from util_moderngl_qt import DrawerMesh, QGLWidgetViewer3
from del_msh import WavefrontObj

if __name__ == "__main__":
    # newpath = Path('.') / 'asset' / 'HorseSwap.obj'
    newpath = Path('.') / 'asset' / 'Babi' / 'Babi.obj'

    obj = WavefrontObj.load(str(newpath), is_centerize=True, normalized_size=1.8)
    edge2vtx = obj.edge2vtxxyz()
    tri2vtx = obj.tri2vtxxyz()

    with QtWidgets.QApplication([]) as app:
        drawer = DrawerMesh.Drawer(
            vtx2xyz=obj.vtxxyz2xyz,
            list_elem2vtx=[
                DrawerMesh.ElementInfo(index=edge2vtx, color=(0, 0, 0), mode=moderngl.LINES),
                DrawerMesh.ElementInfo(index=tri2vtx, color=(1, 1, 1), mode=moderngl.TRIANGLES)]
        )
        win = QGLWidgetViewer3.QtGLWidget_Viewer3([drawer])
        win.show()
        app.exec()




