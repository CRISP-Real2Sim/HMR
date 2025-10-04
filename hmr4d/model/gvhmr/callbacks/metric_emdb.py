import os
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from hmr4d.configs import MainStore, builds

from hmr4d.utils.comm.gather import all_gather
from hmr4d.utils.pylogger import Log

from hmr4d.utils.eval.eval_utils import (
    compute_camcoord_metrics,
    compute_global_metrics,
    compute_camcoord_perjoint_metrics,
    rearrange_by_mask,
    as_np_array,
)
from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from hmr4d.utils.smplx_utils import make_smplx
from einops import einsum, rearrange

from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines
from hmr4d.utils.vis.renderer import Renderer, get_global_cameras_static
from hmr4d.utils.geo.hmr_cam import estimate_focal_length
from hmr4d.utils.video_io_utils import read_video_np, save_video
import imageio
from tqdm import tqdm
import numpy as np
import cv2


class MetricMocap(pl.Callback):
    def __init__(
        self,
        emdb_split=1,
        emdb2_crop_json_path="/data3/zihanwa3/_Robotics/__results/emdb/emdb_crop.json",
    ):
        """
        Args:
            emdb_split: 1 -> evaluate in-cam; 2 -> evaluate global
            emdb2_crop_json_path: JSON with {seq_name: [start, end]} for EMDB-2
        """
        super().__init__()
        self.emdb_split = emdb_split

        # vid->result
        if emdb_split == 1:
            self.target_dataset_id = "EMDB_1"
            self.metric_aggregator = {
                "pa_mpjpe": {},
                "mpjpe": {},
                "pve": {},
                "accel": {},
            }
        elif emdb_split == 2:
            self.target_dataset_id = "EMDB_2"
            self.metric_aggregator = {
                "wa2_mpjpe": {},
                "waa_mpjpe": {},
                "rte": {},
                "jitter": {},
                "fs": {},
            }
        else:
            raise ValueError(f"Unknown emdb_split: {emdb_split}")

        # SMPL / SMPL-X
        self.smplx = make_smplx("supermotion")
        self.smpl_model = {
            "male": make_smplx("smpl", gender="male"),
            "female": make_smplx("smpl", gender="female"),
        }

        self.J_regressor = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt")
        self.smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt")
        self.faces_smpl = self.smpl_model["male"].faces
        self.faces_smplx = self.smplx.faces

        # EMDB-2: load crop config (sequence -> [start, end], inclusive)
        self.emdb2_crop_cfg = {}
        if self.emdb_split == 2:

            with open(emdb2_crop_json_path, "r") as f:
                self.emdb2_crop_cfg = json.load(f)
            Log.info(f"[MetricMocap] Loaded EMDB-2 crop JSON with {len(self.emdb2_crop_cfg)} entries: {emdb2_crop_json_path}")

        # Batch hooks are shared across splits
        self.on_test_batch_end = self.on_validation_batch_end = self.on_predict_batch_end
        self.on_test_epoch_end = self.on_validation_epoch_end = self.on_predict_epoch_end

    @staticmethod
    def _canon_seq_name(raw_name):
        """
        Try to canonicalize sequence names so they match JSON keys like '09_outdoor_walk'.
        """
        if raw_name is None:
            return None
        s = str(raw_name)
        # Drop any directory prefix
        s = Path(s).name
        # Drop extension if any
        s = Path(s).stem
        return s

    def _apply_emdb2_cropping(self, batch, mask):
        """
        For EMDB-2, use the JSON to select sequences and frame ranges.
        Returns:
            (bool keep_sequence, torch.BoolTensor new_mask)
        If keep_sequence is False, we should skip evaluating this batch.
        """
        # Identify sequence name robustly
        vid = batch["meta"][0]["vid"]
        print(vid, 'vide')
        name_candidates = [
            batch.get("meta_render", [{}])[0].get("name", None),
            batch["meta"][0].get("name", None) if isinstance(batch["meta"][0], dict) else None,
            vid,
        ]
        seq_keys = [self._canon_seq_name(x) for x in name_candidates if x is not None]
        seq_key = None
        for k in seq_keys:
            if k[3:] in self.emdb2_crop_cfg:
                
                seq_key = k[3:]
                break

        # If a crop config exists: only evaluate sequences listed in the JSON
        if len(self.emdb2_crop_cfg) > 0:
            if seq_key is None:
                # Not listed -> skip evaluation for this batch
                if vid is not None:
                    Log.info(f"[MetricMocap] Skip (EMDB-2): sequence not in crop JSON. vid/name candidates={seq_keys}")
                return False, mask

            # Build new mask within [start, end]
            start, end = self.emdb2_crop_cfg[seq_key]
            L = mask.shape[0]
            start = max(0, min(int(start), L - 1))
            end = max(start, min(int(end), L - 1))  # inclusive
            print(seq_key, start, end)
            idx = torch.arange(L, device=mask.device)
            range_mask = (idx >= start) & (idx <= end)
            new_mask = range_mask & mask.bool()
            # Edge case: if nothing left after cropping, skip
            if new_mask.sum().item() == 0:
                Log.info(f"[MetricMocap] Skip (EMDB-2): empty mask after cropping [{start}, {end}] for seq={seq_key}")
                return False, new_mask
            return True, new_mask

        # No crop cfg -> keep all as-is
        return True, mask

    # ================== Batch-based Computation  ================== #
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """The behaviour is the same for val/test/predict"""
        assert batch["B"] == 1
        dataset_id = batch["meta"][0]["dataset_id"]
        if dataset_id != self.target_dataset_id:
            return

        # Move to cuda if not
        self.smplx = self.smplx.cuda()
        for g in ["male", "female"]:
            self.smpl_model[g] = self.smpl_model[g].cuda()
        self.J_regressor = self.J_regressor.cuda()
        self.smplx2smpl = self.smplx2smpl.cuda()

        vid = batch["meta"][0]["vid"]
        seq_length = batch["length"][0].item()
        gender = batch["gender"][0]
        T_w2c = batch["T_w2c"][0]
        mask = batch["mask"][0].bool()

        # ---------- EMDB-2 cropping logic ----------
        if self.emdb_split == 2:
            keep, mask = self._apply_emdb2_cropping(batch, mask)
            if not keep:
                return
        # ------------------------------------------

        # Groundtruth (world, cam)
        target_w_params = {k: v[0] for k, v in batch["smpl_params"].items()}
        target_w_output = self.smpl_model[gender](**target_w_params)
        target_w_verts = target_w_output.vertices
        target_w_j3d = torch.matmul(self.J_regressor, target_w_verts)
        target_c_verts = apply_T_on_points(target_w_verts, T_w2c)
        target_c_j3d = apply_T_on_points(target_w_j3d, T_w2c)

        # + Prediction -> Metric
        if self.target_dataset_id == "EMDB_1":  # in camera metrics
            pred_smpl_params_incam = outputs["pred_smpl_params_incam"]
            smpl_out = self.smplx(**pred_smpl_params_incam)
            pred_c_verts = torch.stack([torch.matmul(self.smplx2smpl, v_) for v_ in smpl_out.vertices])
            pred_c_j3d = einsum(self.J_regressor, pred_c_verts, "j v, l v i -> l j i")
            del smpl_out  # Prevent OOM

            batch_eval = {
                "pred_j3d": pred_c_j3d,
                "target_j3d": target_c_j3d,
                "pred_verts": pred_c_verts,
                "target_verts": target_c_verts,
            }
            camcoord_metrics = compute_camcoord_metrics(batch_eval, mask=mask)
            for k in camcoord_metrics:
                self.metric_aggregator[k][vid] = as_np_array(camcoord_metrics[k])

        elif self.target_dataset_id == "EMDB_2":  # global metrics
            pred_smpl_params_global = outputs["pred_smpl_params_global"]
            smpl_out = self.smplx(**pred_smpl_params_global)
            pred_ay_verts = torch.stack([torch.matmul(self.smplx2smpl, v_) for v_ in smpl_out.vertices])
            pred_ay_j3d = einsum(self.J_regressor, pred_ay_verts, "j v, l v i -> l j i")
            del smpl_out  # Prevent OOM

            batch_eval = {
                "pred_j3d_glob": pred_ay_j3d,
                "target_j3d_glob": target_w_j3d,
                "pred_verts_glob": pred_ay_verts,
                "target_verts_glob": target_w_verts,
            }
            global_metrics = compute_global_metrics(batch_eval, mask=mask)
            for k in global_metrics:
                self.metric_aggregator[k][vid] = as_np_array(global_metrics[k])

        # ----- (optional debug/vis blocks kept as-is, guarded by False) -----
        if False:  # wis3d debug
            wis3d = make_wis3d(name="debug-emdb-incam")
            pred_cr_j3d = pred_c_j3d - pred_c_j3d[:, [0]]
            target_cr_j3d = target_c_j3d - target_c_j3d[:, [0]]
            add_motion_as_lines(pred_cr_j3d, wis3d, name="pred_cr_j3d", const_color="blue")
            add_motion_as_lines(target_cr_j3d, wis3d, name="target_cr_j3d", const_color="green")

        if False:  # Dump wis3d
            vid = batch["meta"][0]["vid"]
            split = batch["meta_render"][0]["split"]
            wis3d = make_wis3d(name=f"dump_emdb{split}-{vid}")
            R_cam_type = batch["meta_render"][0]["R_cam_type"]

            pred_cr_j3d = pred_c_j3d - pred_c_j3d[:, [0]]
            target_cr_j3d = target_c_j3d - target_c_j3d[:, [0]]
            add_motion_as_lines(pred_cr_j3d, wis3d, name="pred_cr_j3d", const_color="blue")
            add_motion_as_lines(target_cr_j3d, wis3d, name="target_cr_j3d", const_color="green")
            add_motion_as_lines(pred_ay_j3d, wis3d, name=f"pred_ay_j3d@{R_cam_type}")

        if False:  # Render incam
            vname = batch["meta_render"][0]["name"]
            video_path = batch["meta_render"][0]["video_path"]
            width, height = batch["meta_render"][0]["width_height"]
            K = batch["meta_render"][0]["K"]
            faces = self.faces_smpl
            split = batch["meta_render"][0]["split"]

            out_fn = f"outputs/dump_render_emdb{split}/{vname}.mp4"
            Path(out_fn).parent.mkdir(exist_ok=True, parents=True)

            renderer = Renderer(width, height, device="cuda", faces=faces, K=K)
            resize_factor = 0.25
            images = read_video_np(video_path, scale=resize_factor)
            frame_id = batch["meta_render"][0]["frame_id"]
            bbx_xys_render = batch["meta_render"][0]["bbx_xys"]
            metric_vis = rearrange_by_mask(torch.from_numpy(self.metric_aggregator["mpjpe"][vid]), mask)

            verts_incam = pred_c_verts
            output_images = []
            for i in tqdm(range(len(images)), desc=f"Rendering {vname}"):
                img = renderer.render_mesh(verts_incam[i].cuda(), images[i], [0.8, 0.8, 0.8])
                bbx_xys_ = bbx_xys_render[i].cpu().numpy()
                lu_point = (bbx_xys_[:2] - bbx_xys_[2:] / 2).astype(int)
                rd_point = (bbx_xys_[:2] + bbx_xys_[2:] / 2).astype(int)
                img = cv2.rectangle(img, lu_point, rd_point, (255, 178, 102), 2)

                if metric_vis[i] > 0:
                    text = f"pred mpjpe: {metric_vis[i]:.1f}"
                    text_color = (244, 10, 20) if metric_vis[i] > 80 else (0, 205, 0)
                    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, text_color, 2)

                output_images.append(img)
            save_video(output_images, out_fn, quality=5)

        if False:  # Visualize incam + global results
            def move_to_start_point_face_z(verts):
                "XZ to origin, Start from the ground, Face-Z"
                verts = verts.clone()
                xz_mean = verts[0].mean(0)[[0, 2]]
                y_min = verts[0, :, [1]].min()
                offset = torch.tensor([[[xz_mean[0], y_min, xz_mean[1]]]]).to(verts)
                verts = verts - offset

                T_ay2ayfz = compute_T_ayfz2ay(einsum(self.J_regressor, verts[[0]], "j v, l v i -> l j i"), inverse=True)
                verts = apply_T_on_points(verts, T_ay2ayfz)
                return verts

            verts_incam = pred_c_verts.clone()
            verts_glob = move_to_start_point_face_z(pred_ay_verts)
            global_R, global_T, global_lights = get_global_cameras_static(verts_glob.cpu())

            vname = batch["meta_render"][0]["name"]
            width, height = batch["meta_render"][0]["width_height"]
            K = batch["meta_render"][0]["K"]
            faces = self.faces_smpl
            out_fn = f"outputs/dump_render_global/{vname}.mp4"
            Path(out_fn).parent.mkdir(exist_ok=True, parents=True)
            writer = imageio.get_writer(out_fn, fps=30, mode="I", format="FFMPEG", macro_block_size=1)

            renderer_incam = Renderer(width, height, device="cuda", faces=faces, K=K)
            renderer_glob = Renderer(width, height, estimate_focal_length(width, height), device="cuda", faces=faces)

            video_path = batch["meta_render"][0]["video_path"]
            frame_id = batch["meta_render"][0]["frame_id"].cpu().numpy()
            images = read_video_np(video_path, frame_id=frame_id)

            cx, cz = (verts_glob.mean(1).max(0)[0] + verts_glob.mean(1).min(0)[0])[[0, 2]] / 2.0
            scale = (verts_glob.mean(1).max(0)[0] - verts_glob.mean(1).min(0)[0])[[0, 2]].max() * 1.5
            renderer_glob.set_ground(scale, cx.item(), cz.item())
            color = torch.ones(3).float().cuda() * 0.8

            for i in tqdm(range(seq_length), desc=f"Rendering {vname}"):
                img_overlay_pred = renderer_incam.render_mesh(verts_incam[i].cuda(), images[i], [0.8, 0.8, 0.8])
                if batch["meta_render"][0].get("bbx_xys", None) is not None:
                    bbx_xys = batch["meta_render"][0]["bbx_xys"][i].cpu().numpy()
                    lu_point = (bbx_xys[:2] - bbx_xys[2:] / 2).astype(int)
                    rd_point = (bbx_xys[:2] + bbx_xys[2:] / 2).astype(int)
                    img_overlay_pred = cv2.rectangle(img_overlay_pred, lu_point, rd_point, (255, 178, 102), 2)
                pred_mpjpe_ = self.metric_aggregator["mpjpe"][vid][i]
                text = f"pred mpjpe: {pred_mpjpe_:.1f}"
                cv2.putText(img_overlay_pred, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 100, 200), 2)

                cameras = renderer_glob.create_camera(global_R[i], global_T[i])
                img_glob = renderer_glob.render_with_ground(verts_glob[[i]], color[None], cameras, global_lights)
                img = np.concatenate([img_overlay_pred, img_glob], axis=1)
                writer.append_data(img)
            writer.close()

    # ================== Epoch Summary  ================== #
    def on_predict_epoch_end(self, trainer, pl_module):
        """Without logger"""
        local_rank, world_size = trainer.local_rank, trainer.world_size
        if "mpjpe" in self.metric_aggregator:
            monitor_metric = "mpjpe"
        else:
            monitor_metric = list(self.metric_aggregator.keys())[0]

        metric_keys = list(self.metric_aggregator.keys())
        with torch.inference_mode(False):
            metric_aggregator_gathered = all_gather(self.metric_aggregator)
        for metric_key in metric_keys:
            for d in metric_aggregator_gathered:
                self.metric_aggregator[metric_key].update(d[metric_key])

        total = len(self.metric_aggregator[monitor_metric])
        Log.info(f"{total} sequences evaluated in {self.__class__.__name__}")
        if total == 0:
            return

        mm_per_seq = {k: v.mean() for k, v in self.metric_aggregator[monitor_metric].items()}
        if len(mm_per_seq) > 0:
            sorted_mm_per_seq = sorted(mm_per_seq.items(), key=lambda x: x[1], reverse=True)
            n_worst = 5 if trainer.state.stage == "validate" else len(sorted_mm_per_seq)
            if local_rank == 0:
                Log.info(
                    f"monitored metric {monitor_metric} per sequence\n"
                    + "\n".join([f"{m:5.1f} : {s}" for s, m in sorted_mm_per_seq[:n_worst]])
                    + "\n------"
                )

        metrics_avg = {k: np.concatenate(list(v.values())).mean() for k, v in self.metric_aggregator.items()}
        if local_rank == 0:
            Log.info(
                f"[Metrics] {self.target_dataset_id}:\n"
                + "\n".join(f"{k}: {v:.1f}" for k, v in metrics_avg.items())
                + "\n------"
            )

        if pl_module.logger is not None:
            cur_epoch = pl_module.current_epoch
            for k, v in metrics_avg.items():
                pl_module.logger.log_metrics({f"val_metric_{self.target_dataset_id}/{k}": v}, step=cur_epoch)

        for k in self.metric_aggregator:
            self.metric_aggregator[k] = {}


# Register (unchanged)
emdb1_node = builds(MetricMocap, emdb_split=1)
emdb2_node = builds(MetricMocap, emdb_split=2)  # 默认使用上面的 JSON 路径
MainStore.store(name="metric_emdb1", node=emdb1_node, group="callbacks", package="callbacks.metric_emdb1")
MainStore.store(name="metric_emdb2", node=emdb2_node, group="callbacks", package="callbacks.metric_emdb2")
