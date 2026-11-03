import os
import torch
import numpy as np
import nibabel as nib
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import transforms
import dataset
from models.EKCNet import EKCNet


MODEL_PATH   = ""
DATA_ROOT    = ""
CSV_PATH     = ""
FOLD         = 0
SAVE_DIR     = "./ped_results"
GPU_ID       = 0
BASE_CHANNEL = 16

ADAPTER_REDUCTIONS = (4, 4, 4, 4, 4)

# 模态置零（0=flair, 1=t1, 2=t1ce, 3=t2）
ZERO_MODALITIES = [1]


class BottleneckAdapter(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.down = nn.Sequential(
            nn.Conv3d(channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm3d(mid),
            nn.GELU(),
        )
        self.up = nn.Conv3d(mid, channels, kernel_size=1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(self.down(x))


class CrossLayerAdapter(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4, init_scale=0.0):
        super().__init__()
        mid = max(out_channels // reduction, 4)
        self.down = nn.Sequential(
            nn.Conv3d(in_channels, mid, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(mid),
            nn.GELU(),
        )
        self.up = nn.Conv3d(mid, out_channels, kernel_size=1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.gamma = nn.Parameter(torch.full((out_channels,), init_scale))

    def forward(self, x):
        delta = self.up(self.down(x))
        return delta * self.gamma.view(1, -1, 1, 1, 1)


class EKCNetWithSelectiveAdapters(nn.Module):
    def __init__(self, base_model, baseChannel=16,
                 reductions=(2, 2, 2, 2, 2),
                 freeze_encoder='encoder2'):
        super().__init__()
        self.base = base_model
        self.freeze_encoder = freeze_encoder
        C = baseChannel

        def _make_adapter_list():
            return nn.ModuleList([
                BottleneckAdapter(C * 1, reductions[0]),
                CrossLayerAdapter(C * 1,  C * 2,  reductions[1]),
                CrossLayerAdapter(C * 2,  C * 4,  reductions[2]),
                CrossLayerAdapter(C * 4,  C * 8,  reductions[3]),
                CrossLayerAdapter(C * 8,  C * 16, reductions[4]),
            ])

        self.adapter1 = _make_adapter_list() if freeze_encoder in ('encoder1', 'both') else None
        self.adapter2 = _make_adapter_list() if freeze_encoder in ('encoder2', 'both') else None

    @staticmethod
    def _build_rest_adapters(adapter_list):
        if adapter_list is None:
            return None
        return [None, adapter_list[1], adapter_list[2], adapter_list[3], adapter_list[4]]

    def forward(self, x):
        base = self.base
        flair, t1, t1ce, t2 = x[:, :1], x[:, 1:2], x[:, 2:3], x[:, 3:]
        x_in1 = torch.cat((t1, t1ce), dim=1)
        x_in2 = torch.cat((flair, t2), dim=1)

        x1_1 = base.encoder1.forward_first_layer(x_in1)
        x1_2 = base.encoder2.forward_first_layer(x_in2)
        inj_1to2 = base.injection_1to2(x1_1)
        inj_2to1 = base.injection_2to1(x1_2)

        if self.adapter1 is not None:
            x1_1 = self.adapter1[0](x1_1)
        if self.adapter2 is not None:
            x1_2 = self.adapter2[0](x1_2)

        h_out_1 = base.encoder1.forward_rest(
            x1_1, first_injection=inj_2to1,
            adapters=self._build_rest_adapters(self.adapter1))
        h_out_2 = base.encoder2.forward_rest(
            x1_2, first_injection=inj_1to2,
            adapters=self._build_rest_adapters(self.adapter2))

        x1 = torch.cat((h_out_1[0], h_out_2[0]), dim=1)
        x2 = torch.cat((h_out_1[1], h_out_2[1]), dim=1)
        x3 = torch.cat((h_out_1[2], h_out_2[2]), dim=1)
        x4 = torch.cat((h_out_1[3], h_out_2[3]), dim=1)
        x5 = base.bottleneck(h_out_1[4], h_out_2[4])

        d5 = base.Up5(x5); d5 = torch.cat((x4, d5), dim=1); d5 = base.Up_conv5(d5)
        d4 = base.Up4(d5); d4 = torch.cat((x3, d4), dim=1); d4 = base.Up_conv4(d4)
        d3 = base.Up3(d4); d3 = torch.cat((x2, d3), dim=1); d3 = base.Up_conv3(d3)
        d2 = base.Up2(d3); d2 = torch.cat((x1, d2), dim=1); d2 = base.Up_conv2(d2)
        return base.Conv_1x1(d2)


# ==================== 模型加载（自动识别纯 EKCNet / wrapper） ====================
def load_inference_model(path, base_channel=16, reductions=(4, 4, 4, 4, 4), device='cpu'):
    sd = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(sd, dict):
        sd = sd.state_dict()

    is_wrapper = any(k.startswith('base.') for k in sd.keys())

    if is_wrapper:
        has_a1 = any(k.startswith('adapter1.') for k in sd.keys())
        has_a2 = any(k.startswith('adapter2.') for k in sd.keys())
        if has_a1 and has_a2:   freeze = 'both'
        elif has_a1:            freeze = 'encoder1'
        elif has_a2:            freeze = 'encoder2'
        else:                   freeze = 'none'

        base = EKCNet(inChannel=2, outChannel=4, baseChannel=base_channel)
        model = EKCNetWithSelectiveAdapters(
            base, baseChannel=base_channel,
            reductions=reductions, freeze_encoder=freeze)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[wrapper] freeze_encoder = {freeze}, reductions = {reductions}")
        if missing:
            print(f"  WARN: {len(missing)} missing keys (前 5: {missing[:5]})")
        if unexpected:
            print(f"  WARN: {len(unexpected)} unexpected keys (前 5: {unexpected[:5]})")
        return model, 'wrapper'
    else:
        base = EKCNet(inChannel=2, outChannel=4, baseChannel=base_channel)
        base.load_state_dict(sd)
        print("[base] Loaded pure EKCNet model.")
        return base, 'base'


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    model, kind = load_inference_model(
        MODEL_PATH,
        base_channel=BASE_CHANNEL,
        reductions=ADAPTER_REDUCTIONS,
        device=device)
    model.to(device)
    model.eval()
    print(f"Model type: {kind}, loaded from {MODEL_PATH}")


    frame = pd.read_csv(CSV_PATH)

    val_ids = list(frame["ID"][frame["Fold_{}".format(FOLD)] == 2])
    print(f"Number of validation samples: {len(val_ids)}")

    val_transforms = transforms.Compose([
        transforms.NormalizeIntensity(),
        transforms.RandomCrop(margin=(0, 0, 0),
                              target_size=(128, 128, 128),
                              original_size=(155, 240, 240)),
        transforms.ToTensor()
    ])
    val_set = dataset.MedDataset(DATA_ROOT, val_ids,
                                 transforms=val_transforms, mode="val")

    # ---------- 推理 ----------
    with torch.no_grad():
        for idx in tqdm(range(len(val_set)), desc='Inference'):
            sample = val_set[idx]
            input_tensor = sample['input'].unsqueeze(0).to(device)  # (1, 4, 128, 128, 128)

            # ★ 关键：模态置零与训练一致
            if ZERO_MODALITIES:
                input_tensor[:, ZERO_MODALITIES, ...] = 0.0

            output = model(input_tensor)                            # (1, 4, 128, 128, 128)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

            patient_id = val_ids[idx]
            save_path = os.path.join(SAVE_DIR, f'{patient_id}_seg.nii.gz')


            affine = np.eye(4)
            nifti_img = nib.Nifti1Image(pred.astype(np.uint8), affine)
            nib.save(nifti_img, save_path)

    print(f"Inference completed. Predictions saved to {SAVE_DIR}")


if __name__ == '__main__':
    main()