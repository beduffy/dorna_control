# <!-- https://stackoverflow.com/questions/62596854/aligning-a-point-cloud-with-the-floor-plane-using-open3d -->

import open3d as o3d

aruco_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                                        size=0.5, origin=[0.0, 0.0, 0.0])

mesh_box = o3d.geometry.TriangleMesh.create_box(width=10.0, height=10.0, depth=0.1)

list_of_geometry_elements = [aruco_coordinate_frame, mesh_box]
o3d.visualization.draw_geometries(list_of_geometry_elements)