import time

import open3d as o3d
from open3d.t.geometry import TriangleMesh
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

def set_camera_view(vis):
    ctr = vis.get_view_control()
    
    # Set camera position/view
    ctr.set_front([0, 0, -1])  # Camera direction
    ctr.set_lookat([0, 0, 0])  # Point camera looks at
    ctr.set_up([0, 1, 0])      # Camera up direction
    ctr.set_zoom(0.7)          # Zoom level

    # Other useful methods:
    # ctr.rotate(x, y)         # Rotate camera
    # ctr.translate(x, y)      # Pan camera
    # ctr.set_constant_z_near(z_near)  # Set near clipping plane
    # ctr.set_constant_z_far(z_far)  


def visualize_with_custom_camera():
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Add geometries
    pcd = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis.add_geometry(pcd)
    
    # Set initial view
    ctr = vis.get_view_control()
    ctr.set_front([1, 0, -1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, 1, 0])
    ctr.set_zoom(0.8)
    
    # Optional: Set render options
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.5, 0.5, 0.5])  # Gray background
    opt.point_size = 2.0
    
    vis.run()
    vis.destroy_window()





def visualize_with_custom_camera2():
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Add geometries
    pcd = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis.add_geometry(pcd)
    
    # Set initial view
    ctr = vis.get_view_control()
    ctr.set_front([1, 0, -1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, 1, 0])
    ctr.set_zoom(0.8)
    
    # Optional: Set render options
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.5, 0.5, 0.5])
    opt.point_size = 2.0
    
    # Animation loop
    while True:
        # Check if window is closed
        if not vis.poll_events():
            break
            
        # Example: Rotate camera slowly around Y axis
        current_time = time.time()
        angle = current_time % (2 * np.pi)  # Full rotation
        
        # Update camera position
        ctr.set_front([np.sin(angle), 0, np.cos(angle)])
        
        # Update visualization
        vis.update_renderer()
        
        # Optional: Add small delay
        time.sleep(0.01)
    
    vis.destroy_window()


def visualize_with_custom_camera3():
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Add geometries
    pcd = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis.add_geometry(pcd)
    
    # Set initial view
    ctr = vis.get_view_control()
    initial_cam_pos = [1, 0, -1]
    ctr.set_front(initial_cam_pos)
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, 1, 0])
    ctr.set_zoom(0.8)
    
    # Camera control variables
    cam_pos = initial_cam_pos
    rotation_speed = 0.02
    is_rotating = False
    
    def key_callback(vis, action, mods):
        nonlocal is_rotating
        if action == ord('R'):  # Press 'R' to toggle rotation
            is_rotating = not is_rotating
            return True
        return False
    
    def animation_callback(vis):
        nonlocal cam_pos
        if is_rotating:
            # Rotate camera around Y axis
            angle = rotation_speed
            x, y, z = cam_pos
            cam_pos = [
                x * np.cos(angle) + z * np.sin(angle),
                y,
                -x * np.sin(angle) + z * np.cos(angle)
            ]
            ctr.set_front(cam_pos)
        return False
    
    # Register callbacks
    vis.register_key_callback(ord('R'), key_callback)
    vis.register_animation_callback(animation_callback)
    
    # Run visualization
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    # create_text_visualization()
    # create_text_visualization2()
    # create_text_so()

    # https://github.com/isl-org/Open3D/issues/3894
    # main()

    # visualize_with_custom_camera()
    visualize_with_custom_camera2()
