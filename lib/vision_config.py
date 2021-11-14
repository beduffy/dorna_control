import os

try:
    import open3d as o3d
    open3d_import_success = True
except Exception as e:
    print(e)
    print('Tried to import open3d but not installed')
    open3d_import_success = False


# globals
# with open(os.path.abspath(os.path.join(__file__, '..', '..', 'resources', 'coco_labels_91.txt')), 'r') as f:
#     class_names = [line.strip() for line in f.readlines()]
#     class_name_to_id = {n: idx for idx, n in enumerate(class_names)}

# intrinsics
# from better calibration RMS 0.18 and mean error:  0.02500
width, height, fx, fy, ppx, ppy = (640.0, 480.0, 612.14801862, 611.29345062, 340.03640321,
                                   230.06928807)
if open3d_import_success:
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            int(width), int(height), fx, fy, ppx, ppy)
# todo make sure this is used everywhere
