from pathlib import Path
import moderngl
from PyQt5 import QtWidgets
from util_moderngl_qt import DrawerMeshNormal, QGLWidgetViewer3
from del_msh import WavefrontObj, PolygonMesh

if __name__ == "__main__":

    path_dir = Path('.') / 'asset' / 'shapenet_car1' / 'models'
    obj = WavefrontObj.load(str(path_dir / 'model_normalized.obj'), is_centerize=True, normalized_size=1.8)

    idx2uni, uni2xyz, uni2nrm, _, _ = obj.polygon_mesh_with_normal()

    dict_mtl = WavefrontObj.read_material(str(path_dir / obj.mtl_file_name))
    dict_mtl['_default'] = {'Kd': [0., 0., 0.]}

    with QtWidgets.QApplication([]) as app:
        list_elem2vtx = []
        # edge2vtx = PolygonMesh.edges(obj.elem2idx, idx2uni, uni2xyz.shape[0])
        # list_elem2vtx.append(DrawerMeshNormal.ElementInfo(index=edge2vtx, color=(0., 0., 0.), mode=moderngl.LINES))
        for mtl_idx, mtl_name in enumerate(obj.mtl2name):
            pelem2pidx, pidx2uni = PolygonMesh.extract(obj.elem2idx, idx2uni, obj.elem2mtl[:] == mtl_idx)
            if len(pelem2pidx) == 1:
                continue
            ptri2vtx = PolygonMesh.triangles(pelem2pidx, pidx2uni)
            color = dict_mtl[mtl_name]['Kd']
            list_elem2vtx.append(DrawerMeshNormal.ElementInfo(index=ptri2vtx, color=color, mode=moderngl.TRIANGLES))
        drawer = DrawerMeshNormal.Drawer(
            vtx2xyz=uni2xyz,
            vtx2nrm=uni2nrm,
            list_elem2vtx=list_elem2vtx
        )
        win = QGLWidgetViewer3.QtGLWidget_Viewer3([drawer])
        win.show()
        app.exec()
