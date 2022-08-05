from wild_visual_navigation import WVN_ROOT_DIR
import os
from os.path import join
import torch
from torchvision import transforms as T
from kornia.geometry.camera.pinhole import PinholeCamera
from kornia.geometry.linalg import transform_points
from kornia.utils.draw import draw_convex_polygon
from scipy.spatial import ConvexHull

from liegroups.torch import SE3, SO3


class ImageProjector:
    def __init__(self, K: torch.tensor, h: int, w: int, new_h: int = 448, new_w: int = None):
        """Initializes the projector using the pinhole model, without distortion

        Args:
            K (torch.Tensor, dtype=torch.float32, shape=(B, 4, 4)): Camera matrices
            pose_camera_in_world (torch.Tensor, dtype=torch.float32, shape=(B, 4, 4)): Extrinsics SE(3) matrix
            h (torch.Tensor, dtype=torch.int64): Image height
            w (torch.Tensor, dtype=torch.int64): Image width
            new_h (int): New height size
            new_w (int): New width size

        Returns:
            None
        """

        # TODO: Add shape checks

        # Get device for later
        device = K.device

        # Initialize pinhole model (no extrinsics)
        E = torch.eye(4).expand(K.shape).to(device)

        # Store original parameters
        self.K = K
        self.height = h
        self.width = w

        # Compute scale
        sy = new_h / h
        sx = (new_w / w) if (new_w is not None) else sy

        # Compute scaled parameters
        sh = new_h
        sw = new_w if new_w is not None else sh

        # Prepare image cropper
        if new_w is None or new_w == new_h:
            self.image_crop = T.Compose([T.Resize(new_h, T.InterpolationMode.NEAREST), T.CenterCrop(new_h)])
        else:
            self.image_crop = T.Resize([new_h, new_w], T.InterpolationMode.NEAREST)

        # Adjust camera matrix
        # Fill values
        sK = K.clone()
        if new_w is None or new_w == new_h:
            sK[-1, 0, 0] = K[-1, 1, 1] * sy
            sK[-1, 0, 2] = K[-1, 1, 2] * sy
            sK[-1, 1, 1] = K[-1, 1, 1] * sy
            sK[-1, 1, 2] = K[-1, 1, 2] * sy
        else:
            sK[-1, 0, 0] = K[-1, 0, 0] * sx
            sK[-1, 0, 2] = K[-1, 0, 2] * sx
            sK[-1, 1, 1] = K[-1, 1, 1] * sy
            sK[-1, 1, 2] = K[-1, 1, 2] * sy

        # Initialize camera with scaled parameters
        sh = torch.IntTensor([sh]).to(device)
        sw = torch.IntTensor([sw]).to(device)
        self.camera = PinholeCamera(sK, E, sh, sw)

    def change_device(self, device):
        """Changes the device of all the class members

        Args:
            device (str): new device
        """
        self.K = self.K.to(device)
        self.camera = PinholeCamera(
            self.camera.intrinsics.to(device),
            self.camera.extrinsics.to(device),
            self.camera.height.to(device),
            self.camera.width.to(device),
        )

    def check_validity(self, points_3d: torch.tensor, points_2d: torch.tensor) -> torch.tensor:
        """Check that the points are valid after projecting them on the image

        Args:
            points_3d: (torch.Tensor, dtype=torch.float32, shape=(B, N, 3)): B batches of N points in camera frame
            points_2d: (torch.Tensor, dtype=torch.float32, shape=(B, N, 2)): B batches of N points on the image

        Returns:
            valid_points: (torch.Tensor, dtype=torch.bool, shape=(B, N, 1)): B batches of N bools
        """

        # Check cheirality (if points are behind the camera, i.e, negative z)
        valid_z = points_3d[..., 2] >= 0
        # # Check if projection is within image range
        valid_xmin = points_2d[..., 0] >= 0
        valid_xmax = points_2d[..., 0] <= self.camera.width
        valid_ymin = points_2d[..., 1] >= 0
        valid_ymax = points_2d[..., 1] <= self.camera.height

        # Return validity
        return valid_z & valid_xmax & valid_xmin & valid_ymax & valid_ymin

    def project(self, pose_camera_in_world: torch.tensor, points_W: torch.tensor):
        """Applies the pinhole projection model to a batch of points

        Args:
            points: (torch.Tensor, dtype=torch.float32, shape=(B, N, 3)): B batches of N input points in world frame

        Returns:
            projected_points: (torch.Tensor, dtype=torch.float32, shape=(B, N, 2)): B batches of N output points on image space
        """

        # Adjust input points depending on the extrinsics
        T_CW = pose_camera_in_world.inverse()
        # Convert from fixed to camera frame
        points_C = transform_points(T_CW, points_W)

        # Project points to image
        projected_points = self.camera.project(points_C)

        # Validity check (if points are out of the field of view)
        valid_points = self.check_validity(points_C, projected_points)

        # # Clamp values
        # N = self.camera.width.shape[0]
        # zeros = self.camera.width * 0
        # projected_points[..., 0] = torch.clamp(projected_points[..., 0], min=zeros, max=self.camera.width-1)
        # projected_points[..., 1] = torch.clamp(projected_points[..., 1], min=zeros, max=self.camera.height-1)
        # projected_points = projected_points.unique(dim=2)

        # Return projected points and validity
        return projected_points, valid_points

    def project_and_render(
        self, pose_camera_in_world: torch.tensor, points: torch.tensor, colors: torch.tensor, image: torch.tensor = None
    ):
        """Projects the points and returns an image with the projection

        Args:
            points: (torch.Tensor, dtype=torch.float32, shape=(B, N, 3)): B batches, of N input points in 3D space
            colors: (torch.Tensor, rtype=torch.float32, shape=(B, 3))

        Returns:
            out_img (torch.tensor, dtype=torch.int64): Image with projected points
        """

        B = self.camera.batch_size
        C = 3  # RGB channel output
        H = self.camera.height.item()
        W = self.camera.width.item()

        # Create output mask
        masks = torch.zeros((B, C, H, W), dtype=torch.float32)
        image_overlay = image

        # Project points
        projected_points, valid_points = self.project(pose_camera_in_world, points)
        projected_points = projected_points[valid_points].reshape(B, -1, 2)
        np_projected_points = projected_points.squeeze(0).cpu().numpy()

        # Get convex hull
        if valid_points.sum() > 3:
            hull = ConvexHull(np_projected_points, qhull_options="QJ")

            # Get subset of points that are part of the convex hull
            indices = torch.LongTensor(hull.vertices)
            projected_hull = projected_points[..., indices, :].to(torch.int32)

            # Fill the mask
            masks = draw_convex_polygon(masks, projected_hull.to(masks.device), colors.to(masks.device))

            # Draw on image (if applies)
            if image is not None:
                if len(image.shape) != 4:
                    image = image[None]
                image_overlay = draw_convex_polygon(image, projected_hull.to(image.device), colors.to(image.device))

        # Return torch masks
        masks[masks == 0.0] = torch.nan
        return masks, image_overlay, projected_points, valid_points

    def resize_image(self, image: torch.tensor):
        return self.image_crop(image)


def run_image_projector():
    """Projects 3D points to example images and returns an image with the projection"""

    from wild_visual_navigation.utils import get_img_from_fig
    from wild_visual_navigation.utils import make_plane
    from PIL import Image
    import matplotlib.pyplot as plt
    import torch
    import torchvision.transforms as transforms
    from kornia.utils import tensor_to_image
    from stego.src import remove_axes

    to_tensor = transforms.ToTensor()

    # Create test directory
    os.makedirs(join(WVN_ROOT_DIR, "results", "test_image_projector"), exist_ok=True)

    # Prepare single pinhole model
    # Camera is created 1.5m backward, and 1m upwards, 0deg towards the origin
    # Intrinsics
    K = torch.FloatTensor([[720, 0, 720, 0], [0, 720, 540, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    K = K.unsqueeze(0)
    # Extrisics
    rho = torch.FloatTensor([-1.5, 0, 1])  # Translation vector (x, y, z)
    phi = torch.FloatTensor([-2 * torch.pi / 4, 0.0, -torch.pi / 2])  # roll-pitch-yaw
    R_WC = SO3.from_rpy(phi)  # Rotation matrix from roll-pitch-yaw
    pose_camera_in_world = SE3(R_WC, rho).as_matrix()  # Pose matrix of camera in world frame
    pose_camera_in_world = pose_camera_in_world.unsqueeze(0)
    # Image size
    H = 1080
    W = 1440

    # Create projector
    im = ImageProjector(K, H, W)

    # Load image
    pil_img = Image.open(join(WVN_ROOT_DIR, "assets/images/forest_clean.png"))

    # Convert to torch
    k_img = to_tensor(pil_img)
    k_img = k_img.unsqueeze(0)
    k_img = im.resize_image(k_img)

    # Create 3D points around origin
    X = make_plane(x=0.8, y=0.5, pose=torch.eye(4)).unsqueeze(0)
    colors = torch.tensor([0, 1, 0]).expand(1, 3)

    # Project points to image
    k_mask, k_img_overlay, k_points, k_valid = im.project_and_render(pose_camera_in_world, X, colors, k_img)
    
    # Plot points independently
    k_points_overlay = k_img.clone()
    for p in k_points[0]:
        print(p)
        idx = torch.round(p).to(torch.int32)
        print(idx, k_points_overlay.shape)
        for y in range(-3, 3, 1):
            for x in range(-3, 3, 1):
                try:
                    k_points_overlay[0][:, idx[1].item() + y, idx[0].item() + x] = torch.tensor([0, 255, 0])
                except Exception as e:
                    continue

    # Plot result as in colab
    fig, ax = plt.subplots(1, 4, figsize=(5 * 4, 5))

    ax[0].imshow(tensor_to_image(k_img))
    ax[0].set_title("Image")
    ax[1].imshow(tensor_to_image(k_mask))
    ax[1].set_title("Labels")
    ax[2].imshow(tensor_to_image(k_img_overlay))
    ax[2].set_title("Overlay")
    ax[3].imshow(tensor_to_image(k_points_overlay))
    ax[3].set_title("Overlay - dots")
    remove_axes(ax)
    plt.tight_layout()

    # Store results to test directory
    img = get_img_from_fig(fig)
    img.save(join(WVN_ROOT_DIR, "results", "test_image_projector", "forest_clean_image_projector.png"))


if __name__ == "__main__":
    run_image_projector()

    # K = torch.tensor([[[347.5481,   0.0000, 342.4544,   0.0000],
    #      [  0.0000, 347.4347, 271.3680,   0.0000],
    #      [  0.0000,   0.0000,   1.0000,   0.0000],
    #      [  0.0000,   0.0000,   0.0000,   1.0000]]], device='cuda:0')
    
    # H = 540
    # W = 720
    # image_projector = ImageProjector(K, H, W, new_h=448, new_w=448)

    # fp1 = torch.tensor([[[-14.4048,   7.3462,  -0.5022],
    #      [-14.4065,   7.3774,  -0.4979],
    #      [-14.4081,   7.4086,  -0.4935],
    #      [-14.4098,   7.4399,  -0.4892],
    #      [-14.4115,   7.4711,  -0.4848],
    #      [-14.4132,   7.5023,  -0.4805],
    #      [-14.4149,   7.5336,  -0.4761],
    #      [-14.4166,   7.5648,  -0.4718],
    #      [-14.4183,   7.5960,  -0.4674],
    #      [-14.4200,   7.6273,  -0.4631],
    #      [-14.4216,   7.6585,  -0.4587],
    #      [-14.4233,   7.6897,  -0.4544],
    #      [-14.4250,   7.7210,  -0.4500],
    #      [-14.4267,   7.7522,  -0.4457],
    #      [-14.4284,   7.7834,  -0.4413],
    #      [-14.4301,   7.8147,  -0.4370],
    #      [-14.4318,   7.8459,  -0.4326],
    #      [-14.4334,   7.8771,  -0.4283],
    #      [-14.4351,   7.9084,  -0.4239],
    #      [-14.4368,   7.9396,  -0.4196],
    #      [-14.4368,   7.9396,  -0.4196],
    #      [-14.4351,   7.9084,  -0.4239],
    #      [-14.4334,   7.8771,  -0.4283],
    #      [-14.4318,   7.8459,  -0.4326],
    #      [-14.4301,   7.8147,  -0.4370],
    #      [-14.4284,   7.7834,  -0.4413],
    #      [-14.4267,   7.7522,  -0.4457],
    #      [-14.4250,   7.7210,  -0.4500],
    #      [-14.4233,   7.6897,  -0.4544],
    #      [-14.4216,   7.6585,  -0.4587],
    #      [-14.4200,   7.6273,  -0.4631],
    #      [-14.4183,   7.5960,  -0.4674],
    #      [-14.4166,   7.5648,  -0.4718],
    #      [-14.4149,   7.5336,  -0.4761],
    #      [-14.4132,   7.5023,  -0.4805],
    #      [-14.4115,   7.4711,  -0.4848],
    #      [-14.4098,   7.4399,  -0.4892],
    #      [-14.4081,   7.4086,  -0.4935],
    #      [-14.4065,   7.3774,  -0.4979],
    #      [-14.4048,   7.3462,  -0.5022],
    #      [-14.4048,   7.3462,  -0.5022],
    #      [-14.4065,   7.3774,  -0.4979],
    #      [-14.4081,   7.4086,  -0.4935],
    #      [-14.4098,   7.4399,  -0.4892],
    #      [-14.4115,   7.4711,  -0.4848],
    #      [-14.4132,   7.5023,  -0.4805],
    #      [-14.4149,   7.5336,  -0.4761],
    #      [-14.4166,   7.5648,  -0.4718],
    #      [-14.4183,   7.5960,  -0.4674],
    #      [-14.4200,   7.6273,  -0.4631],
    #      [-14.4216,   7.6585,  -0.4587],
    #      [-14.4233,   7.6897,  -0.4544],
    #      [-14.4250,   7.7210,  -0.4500],
    #      [-14.4267,   7.7522,  -0.4457],
    #      [-14.4284,   7.7834,  -0.4413],
    #      [-14.4301,   7.8147,  -0.4370],
    #      [-14.4318,   7.8459,  -0.4326],
    #      [-14.4334,   7.8771,  -0.4283],
    #      [-14.4351,   7.9084,  -0.4239],
    #      [-14.4368,   7.9396,  -0.4196],
    #      [-14.4368,   7.9396,  -0.4196],
    #      [-14.4351,   7.9084,  -0.4239],
    #      [-14.4334,   7.8771,  -0.4283],
    #      [-14.4318,   7.8459,  -0.4326],
    #      [-14.4301,   7.8147,  -0.4370],
    #      [-14.4284,   7.7834,  -0.4413],
    #      [-14.4267,   7.7522,  -0.4457],
    #      [-14.4250,   7.7210,  -0.4500],
    #      [-14.4233,   7.6897,  -0.4544],
    #      [-14.4216,   7.6585,  -0.4587],
    #      [-14.4200,   7.6273,  -0.4631],
    #      [-14.4183,   7.5960,  -0.4674],
    #      [-14.4166,   7.5648,  -0.4718],
    #      [-14.4149,   7.5336,  -0.4761],
    #      [-14.4132,   7.5023,  -0.4805],
    #      [-14.4115,   7.4711,  -0.4848],
    #      [-14.4098,   7.4399,  -0.4892],
    #      [-14.4081,   7.4086,  -0.4935],
    #      [-14.4065,   7.3774,  -0.4979],
    #      [-14.4048,   7.3462,  -0.5022]]], device='cuda:0')
    
    # fp2 = torch.tensor([[[-14.9040,   7.3189,  -0.4999],
    #      [-14.9054,   7.3436,  -0.4965],
    #      [-14.9067,   7.3683,  -0.4930],
    #      [-14.9080,   7.3931,  -0.4896],
    #      [-14.9094,   7.4178,  -0.4861],
    #      [-14.9107,   7.4425,  -0.4827],
    #      [-14.9120,   7.4672,  -0.4792],
    #      [-14.9134,   7.4920,  -0.4758],
    #      [-14.9147,   7.5167,  -0.4724],
    #      [-14.9160,   7.5414,  -0.4689],
    #      [-14.9174,   7.5661,  -0.4655],
    #      [-14.9187,   7.5909,  -0.4620],
    #      [-14.9200,   7.6156,  -0.4586],
    #      [-14.9214,   7.6403,  -0.4551],
    #      [-14.9227,   7.6651,  -0.4517],
    #      [-14.9240,   7.6898,  -0.4482],
    #      [-14.9254,   7.7145,  -0.4448],
    #      [-14.9267,   7.7392,  -0.4414],
    #      [-14.9281,   7.7640,  -0.4379],
    #      [-14.9294,   7.7887,  -0.4345],
    #      [-14.9307,   7.8134,  -0.4310],
    #      [-14.9321,   7.8381,  -0.4276],
    #      [-14.9334,   7.8629,  -0.4241],
    #      [-14.9347,   7.8876,  -0.4207],
    #      [-14.9361,   7.9123,  -0.4172],
    #      [-14.8945,   7.9146,  -0.4174],
    #      [-14.8624,   7.3212,  -0.5001],
    #      [-14.8208,   7.3234,  -0.5003],
    #      [-14.8529,   7.9169,  -0.4176],
    #      [-14.7792,   7.3257,  -0.5005],
    #      [-14.8113,   7.9191,  -0.4178],
    #      [-14.7376,   7.3280,  -0.5007],
    #      [-14.7696,   7.9214,  -0.4180],
    #      [-14.6960,   7.3303,  -0.5009],
    #      [-14.7280,   7.9237,  -0.4182],
    #      [-14.6544,   7.3325,  -0.5011],
    #      [-14.6864,   7.9259,  -0.4184],
    #      [-14.6128,   7.3348,  -0.5013],
    #      [-14.6448,   7.9282,  -0.4186],
    #      [-14.6032,   7.9305,  -0.4188],
    #      [-14.5712,   7.3371,  -0.5015],
    #      [-14.5296,   7.3394,  -0.5017],
    #      [-14.5616,   7.9328,  -0.4190],
    #      [-14.4880,   7.3416,  -0.5019],
    #      [-14.5200,   7.9350,  -0.4192],
    #      [-14.4464,   7.3439,  -0.5021],
    #      [-14.4784,   7.9373,  -0.4194],
    #      [-14.4048,   7.3462,  -0.5022],
    #      [-14.4368,   7.9396,  -0.4196],
    #      [-14.3632,   7.3484,  -0.5024],
    #      [-14.3952,   7.9419,  -0.4198],
    #      [-14.3216,   7.3507,  -0.5026],
    #      [-14.3536,   7.9441,  -0.4200],
    #      [-14.2800,   7.3530,  -0.5028],
    #      [-14.3120,   7.9464,  -0.4202],
    #      [-14.2704,   7.9487,  -0.4204],
    #      [-14.2384,   7.3553,  -0.5030],
    #      [-14.1968,   7.3575,  -0.5032],
    #      [-14.2288,   7.9510,  -0.4205],
    #      [-14.1551,   7.3598,  -0.5034],
    #      [-14.1872,   7.9532,  -0.4207],
    #      [-14.1135,   7.3621,  -0.5036],
    #      [-14.1456,   7.9555,  -0.4209],
    #      [-14.0719,   7.3644,  -0.5038],
    #      [-14.1040,   7.9578,  -0.4211],
    #      [-14.0303,   7.3666,  -0.5040],
    #      [-14.0624,   7.9601,  -0.4213],
    #      [-13.9887,   7.3689,  -0.5042],
    #      [-14.0208,   7.9623,  -0.4215],
    #      [-13.9792,   7.9646,  -0.4217],
    #      [-13.9471,   7.3712,  -0.5044],
    #      [-13.9055,   7.3735,  -0.5046],
    #      [-13.9069,   7.3982,  -0.5011],
    #      [-13.9082,   7.4229,  -0.4977],
    #      [-13.9095,   7.4476,  -0.4942],
    #      [-13.9109,   7.4724,  -0.4908],
    #      [-13.9122,   7.4971,  -0.4874],
    #      [-13.9135,   7.5218,  -0.4839],
    #      [-13.9149,   7.5465,  -0.4805],
    #      [-13.9162,   7.5713,  -0.4770],
    #      [-13.9175,   7.5960,  -0.4736],
    #      [-13.9189,   7.6207,  -0.4701],
    #      [-13.9202,   7.6454,  -0.4667],
    #      [-13.9215,   7.6702,  -0.4632],
    #      [-13.9229,   7.6949,  -0.4598],
    #      [-13.9242,   7.7196,  -0.4564],
    #      [-13.9255,   7.7443,  -0.4529],
    #      [-13.9269,   7.7691,  -0.4495],
    #      [-13.9282,   7.7938,  -0.4460],
    #      [-13.9296,   7.8185,  -0.4426],
    #      [-13.9309,   7.8432,  -0.4391],
    #      [-13.9322,   7.8680,  -0.4357],
    #      [-13.9336,   7.8927,  -0.4322],
    #      [-13.9349,   7.9174,  -0.4288],
    #      [-13.9362,   7.9421,  -0.4254],
    #      [-13.9376,   7.9669,  -0.4219]]], device='cuda:0')
    
    # pose_cam_in_world = torch.tensor([[[ 0.7753,  0.0568, -0.6291, -2.4353],
    #      [ 0.6303, -0.0054,  0.7763,  1.9086],
    #      [ 0.0407, -0.9984, -0.0400,  7.5109],
    #      [ 0.0000,  0.0000,  0.0000,  1.0000]]], device='cuda:0')
    
    # from liegroups.torch import SE3, SO3
    # rho = torch.FloatTensor([-14, 7.5, 0.5])  # Translation vector (x, y, z)
    # phi = torch.FloatTensor([0, torch.pi, 0])  # roll-pitch-yaw
    # R_WC = SO3.from_rpy(phi)  # Rotation matrix from roll-pitch-yaw
    # pose_cam_in_world = SE3(R_WC, rho).as_matrix()[None].to('cuda:0')  # Pose matrix of camera in world frame
    
    # color = torch.tensor([0.9870, 0.9870, 0.9870], device='cuda:0')

    # m1, im1, p1, v1 = image_projector.project_and_render(pose_cam_in_world, fp1, color)
    # m2, im2, p2, v2 = image_projector.project_and_render(pose_cam_in_world, fp2, color)

    # pix1 = torch.nan_to_num(m1) * 0
    # pix2 = torch.nan_to_num(m2) * 0

    # for p in p1[0]:
    #     idx = torch.round(p).to(torch.int32)
    #     pix1[0][:, idx[1].item(), idx[0].item()] = torch.tensor([255, 255,255]).to('cuda:0')
    
    # for p in p2[0]:
    #     idx = torch.round(p).to(torch.int32)
    #     pix2[0][:, idx[1].item(), idx[0].item()] = torch.tensor([255, 255,255]).to('cuda:0')

    # # Plot result as in colab
    # import matplotlib.pyplot as plt
    # from kornia.utils import tensor_to_image
    # from stego.src import remove_axes
    # from wild_visual_navigation.utils import get_img_from_fig

    # fig, ax = plt.subplots(1, 4, figsize=(5 * 4, 5))

    # ax[0].imshow(tensor_to_image(m1))
    # ax[0].set_title("Mask1 - slice")
    # ax[1].imshow(tensor_to_image(m2))
    # ax[1].set_title("Mask2 - footprint")
    # ax[2].imshow(tensor_to_image(pix1))
    # ax[2].set_title("Mask1 - slice - pixels")
    # ax[3].imshow(tensor_to_image(pix2))
    # ax[3].set_title("Mask2 - footprint - pixels")
    # remove_axes(ax)
    # plt.tight_layout()

    # # Store results to test directory
    # img = get_img_from_fig(fig)
    # img.save(join(WVN_ROOT_DIR, "results", "test_image_projector", "test_masks.png"))