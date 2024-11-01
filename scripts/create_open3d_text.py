import open3d as o3d
import numpy as np


def create_text_visualization():
    # Create text geometry
    text = o3d.geometry.Text3D("Hello Open3D", size=1.0)
    text.paint_uniform_color([1, 0, 0])  # Red text

    # Create coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

    # Create a sphere to show origin
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    sphere.paint_uniform_color([0, 1, 0])  # Green sphere

    # Position text above origin
    text.translate([0, 0, 1])

    # Visualize geometries
    o3d.visualization.draw_geometries([text, coord_frame, sphere])

def create_text_visualization2():
    # Create coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

    # Create a sphere to show origin
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    sphere.paint_uniform_color([0, 1, 0])  # Green sphere

    # Use standard visualization instead of GUI/WebRTC
    vis = o3d.visualization.Visualizer()
    vis.create_window("3D Text Demo", width=1024, height=768)
    
    # Add geometries
    vis.add_geometry(coord_frame)
    vis.add_geometry(sphere)
    
    # Add text using standard text geometry
    text = o3d.geometry.Text3D("Hello Open3D", size=1.0)
    text.paint_uniform_color([1, 0, 0])  # Red text
    text.translate([0, 0, 1])
    vis.add_geometry(text)
    
    # Set camera view
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    
    # Run visualization
    vis.run()
    vis.destroy_window()

def create_text_so():
    import open3d as o3d
    from open3d.t.geometry import TriangleMesh

    hello_open3d_mesh: TriangleMesh = o3d.t.geometry.TriangleMesh.create_text("Hello Open3D", depth=0.1).to_legacy()
    hello_open3d_mesh.paint_uniform_color((0.4, 0.1, 0.9))


    # Scale down since default mesh is quite big
    # Location
    location = (1, 3, 6)  # The mesh is not centered at origin, so there is already an offset.
    # I am adding another location shift as an example.
    hello_open3d_mesh.transform([[0.1, 0, 0, location[0]], [0, 0.1, 0, location[1]], [0, 0, 0.1, location[2]],
                                [0, 0, 0, 1]])
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])

    o3d.visualization.draw_geometries([origin, hello_open3d_mesh], mesh_show_back_face=True)


import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

def make_point_cloud(npts, center, radius):
    pts = np.random.uniform(-radius, radius, size=[npts, 3]) + center
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    colors = np.random.uniform(0.0, 1.0, size=[npts, 3])
    cloud.colors = o3d.utility.Vector3dVector(colors)

def main():
    app = gui.Application.instance
    app.initialize()

    points = make_point_cloud(100, (0, 0, 0), 1.0)

    w = app.create_window("Open3D - 3D Text", 1024, 768)
    widget3d = gui.SceneWidget()
    widget3d.scene = rendering.Open3DScene(w.renderer)
    mat = rendering.Material()
    mat.shader = "defaultUnlit"
    mat.point_size = 5 * w.scaling
    widget3d.scene.add_geometry("Points", points, mat)
    for idx in range(0, len(points.points)):
        widget3d.add_3d_label(points.points[idx], "{}".format(idx))
    bbox = widget3d.scene.bounding_box
    widget3d.setup_camera(60.0, bbox, bbox.get_center())
    w.add_child(widget3d)

    app.run()

# if __name__ == "__main__:
#     main()

if __name__ == "__main__":
    # create_text_visualization()
    # create_text_visualization2()
    # create_text_so()
    # https://github.com/isl-org/Open3D/issues/3894
    main()