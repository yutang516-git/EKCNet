
import os
import models.EKCNet as net

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import losses, transforms, dataset, metrics, utils
from torch.nn import init


def param_network(model):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print("The number of parameters: {}".format(num_params))


def init_weights(net, init_type='xavier_uniform_', gain=1.0):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier_normal_':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'xavier_uniform_':
                init.xavier_uniform_(m.weight.data, gain=gain)
            elif init_type == 'kaiming_normal_':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'kaiming_uniform_':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1 or classname.find('GroupNorm') != -1:
            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


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
        delta = self.up(self.down(x))                     # [B, C_out, D/2, H/2, W/2]
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
                # idx=0：作用于本层输出（有空间信息）
                BottleneckAdapter(C * 1, reductions[0]),
                # idx=1~4：跨层，输入通道是上一层通道，输出是当前层通道
                CrossLayerAdapter(C * 1,  C * 2,  reductions[1]),
                CrossLayerAdapter(C * 2,  C * 4,  reductions[2]),
                CrossLayerAdapter(C * 4,  C * 8,  reductions[3]),
                CrossLayerAdapter(C * 8,  C * 16, reductions[4]),
            ])

        if freeze_encoder in ('encoder1', 'both'):
            self.adapter1 = _make_adapter_list()
        else:
            self.adapter1 = None

        if freeze_encoder in ('encoder2', 'both'):
            self.adapter2 = _make_adapter_list()
        else:
            self.adapter2 = None

    @staticmethod
    def _build_rest_adapters(adapter_list):

        if adapter_list is None:
            return None
        return [None, adapter_list[1], adapter_list[2],
                adapter_list[3], adapter_list[4]]

    def forward(self, x):
        base = self.base
        flair, t1, t1ce, t2 = x[:, :1], x[:, 1:2], x[:, 2:3], x[:, 3:]
        x_in1 = torch.cat((t1, t1ce), dim=1)
        x_in2 = torch.cat((flair, t2), dim=1)

        # 阶段 1：两个 encoder 各跑第一层 ----
        x1_1 = base.encoder1.forward_first_layer(x_in1)
        x1_2 = base.encoder2.forward_first_layer(x_in2)

        #  阶段 2：双向浅层注入 ----
        inj_1to2 = base.injection_1to2(x1_1)
        inj_2to1 = base.injection_2to1(x1_2)

        # 阶段 3：idx=0 的 adapter（作用于 x1 本身） ----
        if self.adapter1 is not None:
            x1_1 = self.adapter1[0](x1_1)
        if self.adapter2 is not None:
            x1_2 = self.adapter2[0](x1_2)

        #  阶段 4：idx=1~4 的跨层 adapter 交给 forward_rest ----
        adapters1_for_rest = self._build_rest_adapters(self.adapter1)
        adapters2_for_rest = self._build_rest_adapters(self.adapter2)

        h_out_1 = base.encoder1.forward_rest(
            x1_1, first_injection=inj_2to1, adapters=adapters1_for_rest)
        h_out_2 = base.encoder2.forward_rest(
            x1_2, first_injection=inj_1to2, adapters=adapters2_for_rest)

        # 解码器
        x1 = torch.cat((h_out_1[0], h_out_2[0]), dim=1)
        x2 = torch.cat((h_out_1[1], h_out_2[1]), dim=1)
        x3 = torch.cat((h_out_1[2], h_out_2[2]), dim=1)
        x4 = torch.cat((h_out_1[3], h_out_2[3]), dim=1)
        x5 = base.bottleneck(h_out_1[4], h_out_2[4])

        d5 = base.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = base.Up_conv5(d5)

        d4 = base.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = base.Up_conv4(d4)

        d3 = base.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = base.Up_conv3(d3)

        d2 = base.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = base.Up_conv2(d2)

        out = base.Conv_1x1(d2)
        return out


if __name__ == "__main__":
    date = (datetime.now()).strftime("%Y-%m-%d")
    frame = pd.read_csv("2021dataset_125_126.csv")

    fold = 0
    print("Training only fold: ", fold)

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lrate', type=float, default=0.0001)
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--val_batch_size', type=int, default=2)
    parser.add_argument('--is_thop', type=bool, default=True)
    parser.add_argument('--path_image', type=str,
                        default="/mnt/data8T/dschettler8845/brats-2021-task1/versions/1/BraTS2021_Training_Data")
    parser.add_argument('--pretrained', type=str,
                        default="./quan_2_TC_0.8946437642380998_ED_0.8318433842143497_ET_0.8162086235510336_WT_0.9094836148055824.pt")
    parser.add_argument('--modelname', type=str, default="EKCNet")
    parser.add_argument('--dataname', type=str, default="2021")
    parser.add_argument('--adapter_reductions', type=str, default='4,4,4,4,4',
                                                 help='五层 (128³,64³,32³,16³,8³) 各自的 reduction，逗号分隔')

    parser.add_argument('--adapter_init_scale', type=float, default=0.1,
                        help='跨层 adapter 的 gamma 初始值')
    # 选择冻结哪个 encoder
    parser.add_argument('--freeze_encoder', type=str, default='encoder1',
                        choices=['encoder1', 'encoder2', 'both', 'none'],
                        help='冻结哪个 encoder 并在其上插 adapter')
    args = parser.parse_args()


    reductions = tuple(int(x) for x in args.adapter_reductions.split(','))
    assert len(reductions) == 5, "adapter_reductions 必须提供 5 个值"
    print(f"Adapter reductions (idx 0~4): {reductions}")
    print(f"Freeze target: {args.freeze_encoder}")

    np.random.seed(100)
    torch.manual_seed(100)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")



    base_model = net.EKCNet(inChannel=2, outChannel=4, baseChannel=16)
    model = EKCNetWithSelectiveAdapters(
        base_model, baseChannel=16,
        reductions=reductions,
        freeze_encoder=args.freeze_encoder,
    )

    if args.adapter_init_scale != 0.0:
        for name, m in model.named_modules():
            if isinstance(m, CrossLayerAdapter):
                nn.init.constant_(m.gamma, args.adapter_init_scale)


    train_transforms = transforms.Compose([
        transforms.NormalizeIntensity(),
        transforms.RandomCrop(margin=(16, 16, 16), target_size=(128, 128, 128),
                              original_size=(155, 240, 240)),
        transforms.Mirroring(p=0.5),
        transforms.ToTensor()])
    val_transforms = transforms.Compose([
        transforms.NormalizeIntensity(),
        transforms.RandomCrop(margin=(0, 0, 0), target_size=(128, 128, 128),
                              original_size=(155, 240, 240)),
        transforms.ToTensor()])

    listTrainPatients = list(frame["ID"][frame["Fold_{}".format(fold)] == 1])
    listValPatients = list(frame["ID"][frame["Fold_{}".format(fold)] == 0])
    train_set = dataset.MedDataset(args.path_image, listTrainPatients,
                                   transforms=train_transforms, mode="train")
    val_set = dataset.MedDataset(args.path_image, listValPatients,
                                 transforms=val_transforms, mode="val")

    train_loader = DataLoader(train_set, batch_size=args.train_batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    dataloaders = {'train': train_loader, 'val': val_loader}

    model.to(device)


    if args.pretrained is not None:
        print("Loading pretrained model from:", args.pretrained)
        pretrain = torch.load(args.pretrained, map_location=device, weights_only=False)
        pretrained_dict = pretrain if isinstance(pretrain, dict) else pretrain.state_dict()

        base_state = model.base.state_dict()
        loaded = {k: v for k, v in pretrained_dict.items() if k in base_state}
        base_state.update(loaded)
        model.base.load_state_dict(base_state)
        print(f"Loaded {len(loaded)} / {len(base_state)} keys into base model.")
    else:
        print("Training from Scratch")
        model.apply(init_weights)

    # ==================== 根据 freeze_encoder 冻结对应 encoder ====================
    for name, param in model.named_parameters():
        if args.freeze_encoder in ('encoder1', 'both') and name.startswith('base.encoder1.'):
            param.requires_grad = False
        if args.freeze_encoder in ('encoder2', 'both') and name.startswith('base.encoder2.'):
            param.requires_grad = False


    if args.freeze_encoder in ('encoder1', 'both'):
        model.base.encoder1.eval()
        for module in model.base.encoder1.modules():
            if isinstance(module, nn.BatchNorm3d):
                module.track_running_stats = True
    if args.freeze_encoder in ('encoder2', 'both'):
        model.base.encoder2.eval()
        for module in model.base.encoder2.modules():
            if isinstance(module, nn.BatchNorm3d):
                module.track_running_stats = True

    print(f"Freeze config applied: {args.freeze_encoder}")


    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")


    for enc_name in ['adapter1', 'adapter2']:
        adapter_list = getattr(model, enc_name, None)
        if adapter_list is not None:
            for i, ad in enumerate(adapter_list):
                n = sum(p.numel() for p in ad.parameters())
                print(f"  {enc_name}[{i}] params: {n}")
        else:
            print(f"  {enc_name}: None (not applied)")


    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lrate, betas=(0.9, 0.99)
    )

    ce_weights = torch.tensor([1.0, 2.0, 2.0, 3.0], dtype=torch.float32).to(device)
    criterion = losses.CE_GeneralizedSoftDiceLoss(alpha=0.3, ce_weights=ce_weights)
    dice_metric = metrics.DiceMetrics()
    best_mean_dice = 0.0
    min_loss = np.inf
    filename = "model.pt"
    log_data = []

    for epoch in range(args.epochs):
        loss_train, loss_val = 0, 0
        dice_train, dice_val = [0] * 4, [0] * 4

        model.train()

        if args.freeze_encoder in ('encoder1', 'both'):
            model.base.encoder1.eval()
        if args.freeze_encoder in ('encoder2', 'both'):
            model.base.encoder2.eval()

        for sample in tqdm(dataloaders['train']):
            input, target = sample["input"].to(device), sample["target"].type(torch.LongTensor).to(device)
            # 置零 flair(0),t1ce（ 2）, t1(1), t2(3)
            input[:, [1,3], ...] = 0.0

            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            dice = dice_metric(output, target)
            loss_train += loss.item()
            for i in range(4):
                dice_train[i] += dice[i].item()

        loss_train /= len(dataloaders['train'])
        for i in range(4):
            dice_train[i] /= len(dataloaders['train'])

        model.eval()
        with torch.no_grad():
            for sample in tqdm(dataloaders['val']):
                input, target = sample["input"].to(device), sample["target"].type(torch.LongTensor).to(device)
                # 置零 flair(0),t1ce（ 2）, t1(1), t2(3)
                input[:, [1,3], ...] = 0.0

                output = model(input)
                loss = criterion(output, target)
                dice = dice_metric(output, target)
                loss_val += loss.item()
                for i in range(4):
                    dice_val[i] += dice[i].item()

        loss_val /= len(dataloaders['val'])
        for i in range(4):
            dice_val[i] /= len(dataloaders['val'])


        mean_dice = float(np.mean(dice_val))
        if mean_dice > best_mean_dice:
            best_mean_dice = mean_dice
            filename = (f"./2021data/flair+t1ce_{args.train_batch_size}"
                        f"_TC_{dice_val[0]:.4f}_ED_{dice_val[1]:.4f}"
                        f"_ET_{dice_val[2]:.4f}_WT_{dice_val[3]:.4f}.pt")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            torch.save(model.state_dict(), filename)
            print(f"New best mean Dice = {mean_dice:.4f}, saved to {filename}")

        print(f"Epoch: {epoch} | Loss_train: {loss_train:.4f} | "
              f"Dice_TC: {dice_train[0]:.4f} | Dice_ED: {dice_train[1]:.4f} | "
              f"Dice_ET: {dice_train[2]:.4f} | Dice_WT: {dice_train[3]:.4f}")
        print(f"Epoch: {epoch} | Loss_val:   {loss_val:.4f} | "
              f"Dice_TC: {dice_val[0]:.4f} | Dice_ED: {dice_val[1]:.4f} | "
              f"Dice_ET: {dice_val[2]:.4f} | Dice_WT: {dice_val[3]:.4f}")

        log_data.append([epoch, loss_train,
                         dice_train[0], dice_train[1], dice_train[2], dice_train[3],
                         loss_val,
                         dice_val[0], dice_val[1], dice_val[2], dice_val[3]])



    print("Training completed for fold {}".format(fold))