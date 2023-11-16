import moderngl
from PyQt5 import QtWidgets
from pathlib import Path
from util_moderngl_qt import DrawerMesh, QGLWidgetViewer3
from del_msh import WavefrontObj, PolygonMesh

if __name__ == "__main__":

    path_dir = Path('.') / 'asset' / 'shapenet_car4' / 'models'
    obj = WavefrontObj.load(str(path_dir / 'model_normalized.obj'), is_centerize=True, normalized_size=1.8)

    dict_mtl = WavefrontObj.read_material(str(path_dir / obj.mtl_file_name))
    dict_mtl['_default'] = {'Kd': [0., 0., 0.]}

    with QtWidgets.QApplication([]) as app:
        list_elem2vtx = []
        list_elem2vtx.append(DrawerMesh.ElementInfo(index=obj.edges(), color=(0., 0., 0.), mode=moderngl.LINES))
        for mtl_idx, mtl_name in enumerate(obj.mtl2name):
            pelem2pidx, pidx2vtxxyz = obj.extract_polygon_mesh_of_material(mtl_idx)
            if len(pelem2pidx) == 1:
                continue
            ptri2vtx = PolygonMesh.triangles(pelem2pidx, pidx2vtxxyz)
            color = dict_mtl[mtl_name]['Kd']
            list_elem2vtx.append(DrawerMesh.ElementInfo(index=ptri2vtx, color=color, mode=moderngl.TRIANGLES))

        drawer = DrawerMesh.Drawer(
            vtx2xyz=obj.vtxxyz2xyz,
            list_elem2vtx=list_elem2vtx
        )
        win = QGLWidgetViewer3.QtGLWidget_Viewer3([drawer])
        win.show()
        app.exec()
