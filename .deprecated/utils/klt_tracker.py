import torch
import kornia


def interp2_torch_batch(v, xq, yq):
    BS = xq.shape[0]
    NR = xq.shape[1]
    h = v.shape[0]
    w = v.shape[1]
    xq = xq.flatten()
    yq = yq.flatten()

    x_floor = torch.floor(xq).type(torch.long)
    y_floor = torch.floor(yq).type(torch.long)
    x_ceil = torch.ceil(xq).type(torch.long)
    y_ceil = torch.ceil(yq).type(torch.long)

    x_floor = x_floor.clip(0, w - 1)
    y_floor = y_floor.clip(0, h - 1)
    x_ceil = x_ceil.clip(0, w - 1)
    y_ceil = y_ceil.clip(0, h - 1)

    v1 = v[y_floor, x_floor]
    v2 = v[y_floor, x_ceil]
    v3 = v[y_ceil, x_floor]
    v4 = v[y_ceil, x_ceil]

    lh = yq - y_floor
    lw = xq - x_floor
    hh = 1 - lh
    hw = 1 - lw

    w1 = hh * hw
    w2 = hh * lw
    w3 = lh * hw
    w4 = lh * lw

    interp_val = v1 * w1 + w2 * v2 + w3 * v3 + w4 * v4
    return interp_val.reshape(BS, NR)


import cv2 as cv
import numpy as np


class KLTTrackerOpenCV(torch.nn.Module):
    def __init__(self, device="cpu", window_size=25, levels=15) -> None:
        super().__init__()
        self.lk_params = dict(
            winSize=(7, 7),
            maxLevel=2,
            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
        )

    def forward(
        self,
        t_startXs: torch.Tensor,
        t_startYs: torch.Tensor,
        img_prev: torch.Tensor,
        img_next: torch.Tensor,
    ) -> torch.Tensor:
        """_summary_

        Args:
            t_startXs (torch.Tensor, type torch.float32, N): _description_
            t_startYs (torch.Tensor, type torch.float32, N): _description_
            img_prev (torch.Tensor, type torch.float32, shape(C,H,W)): _description_
            img_next (torch.Tensor, type torch.float32, shape(C,H,W)): _description_

        Returns:
            torch.Tensor: _description_
        """
        pre = np.uint8((img_prev * 255).detach().cpu().numpy())
        cur = np.uint8((img_next * 255).detach().cpu().numpy())

        p0 = torch.stack([t_startXs, t_startYs], axis=1)[:, None, :].cpu().numpy()
        p1, st, err = cv.calcOpticalFlowPyrLK(pre, cur, p0, None, **self.lk_params)
        p1 = torch.from_numpy(p1).to(img_prev.device)

        return p1[:, 0, 0], p1[:, 0, 1]


class KLTTracker(torch.nn.Module):
    def __init__(self, device="cpu", window_size=25, levels=15) -> None:
        super().__init__()
        self.device = device
        self.window_size = torch.tensor(window_size, device=device)
        self.levels = levels
        self.t_mesh_x, self.t_mesh_y = torch.meshgrid(
            torch.arange(self.window_size, device=self.device),
            torch.arange(self.window_size, device=self.device),
        )

    def forward(
        self,
        t_startXs: torch.Tensor,
        t_startYs: torch.Tensor,
        img_prev: torch.Tensor,
        img_next: torch.Tensor,
    ) -> torch.Tensor:
        """_summary_

        Args:
            t_startXs (torch.Tensor, type torch.float32, N): _description_
            t_startYs (torch.Tensor, type torch.float32, N): _description_
            img_prev (torch.Tensor, type torch.float32, shape(C,H,W)): _description_
            img_next (torch.Tensor, type torch.float32, shape(C,H,W)): _description_

        Returns:
            torch.Tensor: _description_
        """
        t_img_prev_gray = kornia.color.rgb_to_grayscale(img_prev)
        t_img_next_gray = kornia.color.rgb_to_grayscale(img_next)

        gb = kornia.filters.GaussianBlur2d((5, 5), (0.2, 0.2))
        t_I = gb(t_img_prev_gray[None])
        t_Iy, t_Ix = torch.gradient(t_I[0, 0] * 255)  # Same just 255 smaller

        t_startXs_flat = t_startXs.flatten()
        t_startYs_flat = t_startYs.flatten()

        t_newXs = torch.full(t_startXs_flat.shape, -1, dtype=torch.float32, device=self.device)
        t_newYs = torch.full(t_startYs_flat.shape, -1, dtype=torch.float32, device=self.device)

        t_newXs, t_newYs = self.estimateFeatureTranslationBatch(
            t_startXs_flat,
            t_startYs_flat,
            t_Ix,
            t_Iy,
            (t_img_prev_gray * 255)[0],
            (t_img_next_gray * 255)[0],
        )

        t_newXs = torch.reshape(t_newXs, t_startXs.shape)
        t_newYs = torch.reshape(t_newYs, t_startYs.shape)

        return t_newXs, t_newYs

    def estimateFeatureTranslationBatch(self, t_startX, t_startY, t_Ix, t_Iy, t_img1_gray, t_img2_gray):
        BS = t_startX.shape[0]
        t_X = t_startX
        t_Y = t_startY
        t_mesh_x, t_mesh_y = self.t_mesh_x.clone(), self.t_mesh_y.clone()

        t_mesh_x_flat_fix = t_mesh_x.flatten()[None].repeat(BS, 1) + t_X[:, None] - torch.floor(self.window_size / 2)
        t_mesh_y_flat_fix = t_mesh_y.flatten()[None].repeat(BS, 1) + t_Y[:, None] - torch.floor(self.window_size / 2)

        t_coor_fix = torch.stack((t_mesh_x_flat_fix, t_mesh_y_flat_fix), dim=1)

        t_I1_value = interp2_torch_batch(t_img1_gray, t_coor_fix[:, 0, :], t_coor_fix[:, 1, :])
        t_Ix_value = interp2_torch_batch(t_Ix, t_coor_fix[:, 0, :], t_coor_fix[:, 1, :])
        t_Iy_value = interp2_torch_batch(t_Iy, t_coor_fix[:, 0, :], t_coor_fix[:, 1, :])

        t_I = torch.stack([t_Ix_value, t_Iy_value], dim=1)
        t_A = torch.bmm(t_I, t_I.transpose(1, 2))

        for _ in range(self.levels):
            t_mesh_x_flat = (
                t_mesh_x.clone().flatten()[None].repeat(BS, 1) + t_X[:, None] - torch.floor(self.window_size / 2)
            )
            t_mesh_y_flat = (
                t_mesh_y.clone().flatten()[None].repeat(BS, 1) + t_Y[:, None] - torch.floor(self.window_size / 2)
            )

            t_coor = torch.stack((t_mesh_x_flat, t_mesh_y_flat), dim=1)
            t_I2_value = interp2_torch_batch(
                t_img2_gray.clone(),
                t_coor[:, 0, :].contiguous(),
                t_coor[:, 1, :].contiguous(),
            ).contiguous()

            t_Ip = (t_I2_value - t_I1_value)[:, :, None]
            t_b = -torch.bmm(t_I, t_Ip)
            t_solution = torch.bmm(torch.linalg.inv(t_A), t_b)
            t_X += t_solution[0, 0]
            t_Y += t_solution[1, 0]

        return t_X, t_Y


if __name__ == "__main__":
    # TODO write a test for it
    pre = torch.load("/home/jonfrey/git/wild_visual_navigation/previous.pt")
    cur = torch.load("/home/jonfrey/git/wild_visual_navigation/current.pt")

    pre_c = np.uint8(pre.cpu().permute(1, 2, 0).numpy() * 255)
    cur_c = np.uint8(cur.cpu().permute(1, 2, 0).numpy() * 255)

    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    pre = cv.cvtColor(pre_c, cv.COLOR_BGR2GRAY)[:, :, None]
    p0 = cv.goodFeaturesToTrack(pre, mask=None, **feature_params)

    t_img_prev = kornia.image_to_tensor(pre_c) / 255
    t_img_next = kornia.image_to_tensor(cur_c) / 255

    klt = KLTTracker(device="cuda")
    gn = torch.from_numpy(p0).to(device="cuda")
    res = klt(gn[:, 0, 0], gn[:, 0, 1], t_img_prev.to("cuda"), t_img_next.to("cuda"))
