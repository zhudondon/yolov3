# Ultralytics YOLOv3 🚀, AGPL-3.0 license
"""
Train a YOLOv3 model on a custom dataset. Models and datasets download automatically from the latest YOLOv3 release.

Usage - Single-GPU training:
    $ python train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (recommended)
    $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights yolov5s.pt --img 640 --device 0,1,2,3

Models:     https://github.com/ultralytics/yolov5/tree/master/models
Datasets:   https://github.com/ultralytics/yolov5/tree/master/data
Tutorial:   https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
"""

import argparse
import math
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
try:
    import comet_ml  # must be imported before torch (if installed)
except ImportError:
    comet_ml = None

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm



FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv3 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val as validate  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download, is_url
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    check_amp,
    check_dataset,
    check_file,
    check_git_info,
    check_git_status,
    check_img_size,
    check_requirements,
    check_suffix,
    check_yaml,
    colorstr,
    get_latest_run,
    increment_path,
    init_seeds,
    intersect_dicts,
    labels_to_class_weights,
    labels_to_image_weights,
    methods,
    one_cycle,
    print_args,
    print_mutation,
    strip_optimizer,
    yaml_save,
)
from utils.loggers import Loggers
from utils.loggers.comet.comet_utils import check_comet_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve
from utils.torch_utils import (
    EarlyStopping,
    ModelEMA,
    de_parallel,
    select_device,
    smart_DDP,
    smart_optimizer,
    smart_resume,
    torch_distributed_zero_first,
)


LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
GIT_INFO = check_git_info()

# 训练本身
def train(hyp, opt, device, callbacks):  # hyp is path/to/hyp.yaml or hyp dictionary
    """
    Train a YOLOv3 model on a custom dataset and manage the training process.

    Args:
        hyp (str | dict): Path to hyperparameters yaml file or hyperparameters dictionary.
        opt (argparse.Namespace): Parsed command line arguments containing training options.
        device (torch.device): Device to load and train the model on.
        callbacks (Callbacks): Callbacks to handle various stages of the training lifecycle.

    Returns:
        None

    Usage - Single-GPU training:
        $ python train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (recommended)
        $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch

    Usage - Multi-GPU DDP training:
        $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights
            yolov5s.pt --img 640 --device 0,1,2,3

    Models: https://github.com/ultralytics/yolov5/tree/master/models
    Datasets: https://github.com/ultralytics/yolov5/tree/master/data
    Tutorial: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data

    Examples:
        ```python
        from ultralytics import train
        import argparse
        import torch
        from utils.callbacks import Callbacks

        # Example usage
        args = argparse.Namespace(
            data='coco128.yaml',
            weights='yolov5s.pt',
            cfg='yolov5s.yaml',
            img_size=640,
            epochs=50,
            batch_size=16,
            device='0'
        )

        device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
        callbacks = Callbacks()

        train(hyp='hyp.scratch.yaml', opt=args, device=device, callbacks=callbacks)
        ```
    """
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = (
        Path(opt.save_dir),
        opt.epochs,
        opt.batch_size,
        opt.weights,
        opt.single_cls,
        opt.evolve,
        opt.data,
        opt.cfg,
        opt.resume,
        opt.noval,
        opt.nosave,
        opt.workers,
        opt.freeze,
    )
    # 回调，执行
    callbacks.run("on_pretrain_routine_start")

    # Directories 权重目录 最好一次，和最后一次
    w = save_dir / "weights"  # weights dir
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / "last.pt", w / "best.pt"

    # Hyperparameters 超参数
    if isinstance(hyp, str):
        with open(hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr("hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in hyp.items()))
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints

    # Save run settings 保存超参数
    if not evolve:
        yaml_save(save_dir / "hyp.yaml", hyp)
        yaml_save(save_dir / "opt.yaml", vars(opt))

    # Loggers 日志
    data_dict = None
    if RANK in {-1, 0}:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance

        # Register actions 注册动作
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

        # Process custom dataset artifact link
        data_dict = loggers.remote_dataset
        if resume:  # If resuming runs from remote artifact 从远程恢复
            weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

    # Config 配置
    plots = not evolve and not opt.noplots  # create plots 创建图形显示
    # cpu 还是 cuda
    cuda = device.type != "cpu"
    # 初始化随机种子
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)  # check if None
    # 数据集  训练数据集和验证数据集的路径
    train_path, val_path = data_dict["train"], data_dict["val"]
    # 分类数量
    nc = 1 if single_cls else int(data_dict["nc"])  # number of classes
    # 分类名称
    names = {0: "item"} if single_cls and len(data_dict["names"]) != 1 else data_dict["names"]  # class names 分类名称
    # 是否是coco数据集
    is_coco = isinstance(val_path, str) and val_path.endswith("coco/val2017.txt")  # COCO dataset coco数据集

    # Model 检测权重文件
    check_suffix(weights, ".pt")  # check weights
    # 预训练权重
    pretrained = weights.endswith(".pt")
    if pretrained:
        # torch_distributed_zero_first 通过实现分布式训练中的进程同步，‌确保了数据加载和处理的同步性，‌提高了训练的稳定性和可靠性
        # 尝试下载权重
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)  # download if not found locally
        # 加载权重到cpu，避免cuda内存泄漏
        ckpt = torch.load(weights, map_location="cpu")  # load checkpoint to CPU to avoid CUDA memory leak
        # 初始化模型 cfg还是配置文件 3频道 分类数量 先验框 ,使用设备训练
        model = Model(cfg or ckpt["model"].yaml, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # create
        # 排除项
        exclude = ["anchor"] if (cfg or hyp.get("anchors")) and not resume else []  # exclude keys
        # 预加载的权重,记录状态
        csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
        # 交叉项,取交集,排除一些不需要的配置
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        # 模型加载 交集的配置项(预训练的配置是别人的,把自己个性化的覆盖上去)
        model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(f"Transferred {len(csd)}/{len(model.state_dict())} items from {weights}")  # report
    else:
        # 非预训练的,从头开始
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # create
    amp = check_amp(model)  # check AMP

    # Freeze 冻结一些不需要训练的参数,这里是从网络x层数的开始,把这个 grad 定为 true 训练,false 不训练
    freeze = [f"model.{x}." for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        # 这里可以顺便注册一些 事件,一般用于写日志和绘图,观察这个 训练效果,可以传入一些 函数
        # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
        if any(x in k for x in freeze):
            LOGGER.info(f"freezing {k}")
            v.requires_grad = False

    # Image size 图片的大小,检测图片大小
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # Batch size 批次大小,评估最好的批次大小
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(model, imgsz, amp)
        loggers.on_params_update({"batch_size": batch_size})

    # Optimizer 优化器,超参数的 权重衰减 优化,传入模型,动量,学习率,权重衰减 得到 优化器
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp["weight_decay"] *= batch_size * accumulate / nbs  # scale weight_decay
    optimizer = smart_optimizer(model, opt.optimizer, hyp["lr0"], hyp["momentum"], hyp["weight_decay"])

    # Scheduler 调度器,得到一个频率,可能是余弦方式,可能是线性的,学习率 超参数lrf学习
    if opt.cos_lr:
        lf = one_cycle(1, hyp["lrf"], epochs)  # cosine 1->hyp['lrf']
    else:

        def lf(x):
            """Linear learning rate scheduler function with decay calculated by epoch proportion."""
            return (1 - x / epochs) * (1.0 - hyp["lrf"]) + hyp["lrf"]  # linear

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA 指数平均数指标(Exponential Moving Average，EXPMA或EMA)
    # EMA（‌Exponential Moving Average）‌的主要作用包括趋势识别、‌避免数据二次采样带来的信息损失风险、‌提高评估指标的准确性和稳定性
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # Resume 恢复训练
    best_fitness, start_epoch = 0.0, 0
    if pretrained:
        if resume:
            # 如果是恢复预训练
            best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)
        del ckpt, csd

    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning(
            "WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n"
            "See Multi-GPU Tutorial at https://docs.ultralytics.com/yolov5/tutorials/multi_gpu_training to get started."
        )
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm 同步批量正则化
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info("Using SyncBatchNorm()")

    # Trainloader 数据加载
    train_loader, dataset = create_dataloader(
        train_path,
        imgsz,
        batch_size // WORLD_SIZE,
        gs,
        single_cls,
        hyp=hyp,
        augment=True,
        cache=None if opt.cache == "val" else opt.cache,
        rect=opt.rect,
        rank=LOCAL_RANK,
        workers=workers,
        image_weights=opt.image_weights,
        quad=opt.quad,
        prefix=colorstr("train: "),
        shuffle=True,
        seed=opt.seed,
    )
    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())  # max label class 最大标签类
    assert mlc < nc, f"Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}"

    # Process 0
    if RANK in {-1, 0}:
        val_loader = create_dataloader(
            val_path,
            imgsz,
            batch_size // WORLD_SIZE * 2,
            gs,
            single_cls,
            hyp=hyp,
            cache=None if noval else opt.cache,
            rect=True,
            rank=-1,
            workers=workers * 2,
            pad=0.5,
            prefix=colorstr("val: "),
        )[0]

        if not resume:
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp["anchor_t"], imgsz=imgsz)  # run AutoAnchor
            model.half().float()  # pre-reduce anchor precision

        callbacks.run("on_pretrain_routine_end", labels, names)

    # DDP mode 分布式模式
    if cuda and RANK != -1:
        model = smart_DDP(model)

    # Model attributes 模型属性
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps) 层数
    hyp["box"] *= 3 / nl  # scale to layers
    hyp["cls"] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp["obj"] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp["label_smoothing"] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    # 连接类别权重
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training 开始训练
    t0 = time.time()
    nb = len(train_loader)  # number of batches
    nw = max(round(hyp["warmup_epochs"] * nb), 100)  # {rations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # 提前停止的条件
    stopper, stop = EarlyStopping(patience=opt.patience), False
    # 初始化损失函数
    compute_loss = ComputeLoss(model)  # init loss class
    callbacks.run("on_train_start")
    LOGGER.info(
        f'Image sizes {imgsz} train, {imgsz} val\n'
        f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
        f"Logging results to {colorstr('bold', save_dir)}\n"
        f'Starting training for {epochs} epochs...'
    )
    # 遍历代数
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        callbacks.run("on_train_epoch_start")
        model.train()

        # Update image weights (optional, single-GPU only)
        # 分类权重,图片权重,随机选择数据集,根据图片权重,随机获取数据集
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        # 平均误差
        mloss = torch.zeros(3, device=device)  # mean losses
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(("\n" + "%11s" * 7) % ("Epoch", "GPU_mem", "box_loss", "obj_loss", "cls_loss", "Instances", "Size"))
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  # progress bar 进度条
        # 优化器,梯度归零
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            callbacks.run("on_train_batch_start")
            ni = i + nb * epoch  # number integrated batches (since train start)
            # 图片 255 归一化到0-1
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0


            # Warmup,
            # 小于 这个代数,就进行预热,初始化一些超参数
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(ni, xi, [hyp["warmup_bias_lr"] if j == 0 else 0.0, x["initial_lr"] * lf(epoch)])
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [hyp["warmup_momentum"], hyp["momentum"]])

            # Multi-scale 缩放,不是强行裁剪,优化一个比例
            if opt.multi_scale:
                sz = random.randrange(int(imgsz * 0.5), int(imgsz * 1.5) + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)

            # Forward 前向传播
            with torch.cuda.amp.autocast(amp):
                pred = model(imgs)  # forward 进行前向传播,得到预测值,target是真实值
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.0

            # Backward 方向传播,调整参数权重
            scaler.scale(loss).backward()

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            # 优化器,调整学习率
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)  # unscale gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                # 调整完后,梯度清零
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Log
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
                pbar.set_description(
                    ("%11s" * 2 + "%11.4g" * 5)
                    % (f"{epoch}/{epochs - 1}", mem, *mloss, targets.shape[0], imgs.shape[-1])
                )
                callbacks.run("on_train_batch_end", model, ni, imgs, targets, paths, list(mloss))
                if callbacks.stop_training:
                    return
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler 调度器,
        lr = [x["lr"] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        if RANK in {-1, 0}:
            # mAP
            callbacks.run("on_train_epoch_end", epoch=epoch)
            ema.update_attr(model, include=["yaml", "nc", "hyp", "names", "stride", "class_weights"])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:  # Calculate mAP
                results, maps, _ = validate.run(
                    data_dict,
                    batch_size=batch_size // WORLD_SIZE * 2,
                    imgsz=imgsz,
                    half=amp,
                    model=ema.ema,
                    single_cls=single_cls,
                    dataloader=val_loader,
                    save_dir=save_dir,
                    plots=False,
                    callbacks=callbacks,
                    compute_loss=compute_loss,
                )

            # Update best mAP 召回率这些
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            stop = stopper(epoch=epoch, fitness=fi)  # early stop check
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            callbacks.run("on_fit_epoch_end", log_vals, epoch, best_fitness, fi)

            # Save model 保存模型
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {
                    "epoch": epoch,
                    "best_fitness": best_fitness,
                    "model": deepcopy(de_parallel(model)).half(),
                    "ema": deepcopy(ema.ema).half(),
                    "updates": ema.updates,
                    "optimizer": optimizer.state_dict(),
                    "opt": vars(opt),
                    "git": GIT_INFO,  # {remote, branch, commit} if a git repo
                    "date": datetime.now().isoformat(),
                }

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f"epoch{epoch}.pt")
                del ckpt
                callbacks.run("on_model_save", last, epoch, final_epoch, best_fitness, fi)

        # EarlyStopping 提前停止,学习效果停滞不前
        if RANK != -1:  # if DDP training
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break  # must break all DDP ranks

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in {-1, 0}:
        LOGGER.info(f"\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.")
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    LOGGER.info(f"\nValidating {f}...")
                    results, _, _ = validate.run(
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device).half(),
                        iou_thres=0.65 if is_coco else 0.60,  # best pycocotools at iou 0.65
                        single_cls=single_cls,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=plots,
                        callbacks=callbacks,
                        compute_loss=compute_loss,
                    )  # val best model with plots
                    if is_coco:
                        callbacks.run("on_fit_epoch_end", list(mloss) + list(results) + lr, epoch, best_fitness, fi)

        callbacks.run("on_train_end", last, best, epoch, results)

    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    """
    Parse command line arguments for configuring the training of a YOLO model.

    Args:
        known (bool): Flag to parse known arguments only, defaults to False.

    Returns:
        (argparse.Namespace): Parsed command line arguments.

    Examples:
        ```python
        options = parse_opt()
        print(options.weights)
        ```

    Notes:
        * The default weights path is 'yolov3-tiny.pt'.
        * Set `known` to True for parsing only the known arguments, useful for partial arguments.

    References:
        * Models: https://github.com/ultralytics/yolov5/tree/master/models
        * Datasets: https://github.com/ultralytics/yolov5/tree/master/data
        * Training Tutorial: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
    """
    parser = argparse.ArgumentParser()
    # 初始化权重
    parser.add_argument("--weights", type=str, default=ROOT / "yolov3-tiny.pt", help="initial weights path")
    # 配置文件 配置了网络的层级
    parser.add_argument("--cfg", type=str, default="", help="model.yaml path")
    # 数据集 例如coco数据集
   # parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument("--data", type=str, default=ROOT / "data/voc.yaml")
    """ 超参数 """
    parser.add_argument("--hyp", type=str, default=ROOT / "data/hyps/hyp.scratch-low.yaml", help="hyperparameters path")
    # 总代数
    parser.add_argument("--epochs", type=int, default=100, help="total training epochs")
    # 批次大小每个gpu的训练批次大小
    parser.add_argument("--batch-size", type=int, default=16, help="total batch size for all GPUs, -1 for autobatch")
    # 图片大小 默认640
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="train, val image size (pixels)")
    # 矩形
    parser.add_argument("--rect", action="store_true", help="rectangular training")
    # 重新开始 恢复训练
    parser.add_argument("--resume", nargs="?", const=True, default=True, help="resume most recent training")
    # 存储规则，最后校验点才报错
    parser.add_argument("--nosave", action="store_true", help="only save final checkpoint")
    # 最后一代 才校验
    parser.add_argument("--noval", action="store_true", help="only validate final epoch")
    # 没有自动先验框
    parser.add_argument("--noautoanchor", action="store_true", help="disable AutoAnchor")
    # 主要作用是防止在程序运行过程中生成不必要的图形界面文件
    parser.add_argument("--noplots", action="store_true", help="save no plot files")
    # 调整 超参数 多少代之后进行
    parser.add_argument("--evolve", type=int, nargs="?", const=300, help="evolve hyperparameters for x generations")
    # Google Cloud Storage（‌GCS）‌
    parser.add_argument("--bucket", type=str, default="", help="gsutil bucket")
    # 图片缓存
    parser.add_argument("--cache", type=str, nargs="?", const="ram", help="image --cache ram/disk")
    # 图片权重
    parser.add_argument("--image-weights", action="store_true", help="use weighted image selection for training")
    # 设备
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    # 多个缩放比例
    parser.add_argument("--multi-scale", action="store_true", help="vary img-size +/- 50%%")
    # 当做一个类别
    parser.add_argument("--single-cls", action="store_true", help="train multi-class data as single-class")
    # 优化器 随机梯度下降，adam
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW"], default="SGD", help="optimizer")
    # 使用同步 批量正则化，只支持在分布式环境
    parser.add_argument("--sync-bn", action="store_true", help="use SyncBatchNorm, only available in DDP mode")
    # 多线程数据加载器 分布式环境 这里多线程要少一点，不然容易崩 DLL load failed while importing _cdflib: 页面文件太小，无法完成操作
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    # 项目根目录
    parser.add_argument("--project", default=ROOT / "runs/train", help="save to project/name")
    # 项目名称
    parser.add_argument("--name", default="exp", help="save to project/name")
    # 不创建
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    # 是否 用大规模数据工具
    parser.add_argument("--quad", action="store_true", help="quad")
    # 余弦 学习率调度器
    parser.add_argument("--cos-lr", action="store_true", help="cosine LR scheduler")
    # 使标签更为平滑
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing epsilon")
    # 如果没有训练效果，多久才停止
    parser.add_argument("--patience", type=int, default=100, help="EarlyStopping patience (epochs without improvement)")
    # 冻结 层数，自己训练自己的参数
    parser.add_argument("--freeze", nargs="+", type=int, default=[0], help="Freeze layers: backbone=10, first3=0 1 2")
    # 多少个代进行保存
    parser.add_argument("--save-period", type=int, default=-1, help="Save checkpoint every x epochs (disabled if < 1)")
    # 全局训练种子 使用的随机过程（‌如数据洗牌、‌随机裁剪等）‌具有一致的结果，‌从而使得每次训练都是确定性的。‌这对于研究和开发来说是非常有用的，‌
    parser.add_argument("--seed", type=int, default=0, help="Global training seed")
    #
    parser.add_argument("--local_rank", type=int, default=-1, help="Automatic DDP Multi-GPU argument, do not modify")

    # Logger arguments
    parser.add_argument("--entity", default=None, help="Entity")
    # 上传数据集
    parser.add_argument("--upload_dataset", nargs="?", const=True, default=False, help='Upload data, "val" option')
    # 定时记录 框框
    parser.add_argument("--bbox_interval", type=int, default=-1, help="Set bounding-box image logging interval")
    parser.add_argument("--artifact_alias", type=str, default="latest", help="Version of dataset artifact to use")

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt, callbacks=Callbacks()):
    """
    Main training/evolution script handling model checks, DDP setup, training, and hyperparameter evolution.

    Args:
        opt (argparse.Namespace): Parsed command-line options.
        callbacks (Callbacks, optional): Callback object for handling training events. Defaults to Callbacks().

    Returns:
        None

    Raises:
        AssertionError: If certain constraints are violated (e.g., when specific options are incompatible with DDP training).

    Notes:
       - For a tutorial on using Multi-GPU with DDP: https://docs.ultralytics.com/yolov5/tutorials/multi_gpu_training

    Example:
        Single-GPU training:
        ```python
        $ python train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (recommended)
        $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch
        ```

        Multi-GPU DDP training:
        ```python
        $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml \
        --weights yolov5s.pt --img 640 --device 0,1,2,3
        ```

        Models: https://github.com/ultralytics/yolov5/tree/master/models
        Datasets: https://github.com/ultralytics/yolov5/tree/master/data
        Tutorial: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
    """
    if RANK in {-1, 0}:
        print_args(vars(opt))
        check_git_status()
        check_requirements(ROOT / "requirements.txt")

    """ 恢复训练 指定一个point 或者最近的 
        opt.resume 是一个文件路径 
    """
    # Resume (from specified or most recent last.pt)
    if opt.resume and not check_comet_resume(opt) and not opt.evolve:
        # 从文件读取，否则 自动获取恢复点 get_latest_run 搜索文件.pt 获取时间最晚的
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        # 训练配置文件
        opt_yaml = last.parent.parent / "opt.yaml"  # train options yaml
        opt_data = opt.data  # original dataset
        if opt_yaml.is_file():
            # 如果有配置信息的 配置文件，就从这里加载
            with open(opt_yaml, errors="ignore") as f:
                d = yaml.safe_load(f)
        else:
            # 加载 最新训练的环境配置
            d = torch.load(last, map_location="cpu")["opt"]
        opt = argparse.Namespace(**d)  # replace
        opt.cfg, opt.weights, opt.resume = "", str(last), True  # reinstate 使恢复
        if is_url(opt_data):
            opt.data = check_file(opt_data)  # avoid HUB resume auth timeout 远程数据
    else:
        # 新创建一个训练项目 ，从传入配置yaml 初始化，返回真正的数据结构，例如权重，项目信息
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = (
            check_file(opt.data),
            check_yaml(opt.cfg),
            check_yaml(opt.hyp),
            str(opt.weights),
            str(opt.project),
        )  # checks
        assert len(opt.cfg) or len(opt.weights), "either --cfg or --weights must be specified" # 必传
        if opt.evolve:
            # 如果是要更新 超参数
            if opt.project == str(ROOT / "runs/train"):  # if default project name, rename to runs/evolve
                opt.project = str(ROOT / "runs/evolve")
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        if opt.name == "cfg":
            opt.name = Path(opt.cfg).stem  # use model.yaml as name
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP mode 分布式训练
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        msg = "is not compatible with YOLOv3 Multi-GPU DDP training"
        assert not opt.image_weights, f"--image-weights {msg}"
        assert not opt.evolve, f"--evolve {msg}"
        assert opt.batch_size != -1, f"AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size"
        assert opt.batch_size % WORLD_SIZE == 0, f"--batch-size {opt.batch_size} must be multiple of WORLD_SIZE"
        assert torch.cuda.device_count() > LOCAL_RANK, "insufficient CUDA devices for DDP command"
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device("cuda", LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # Train 如果不训练超参数，直接开始训练
    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks)

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        """ 超参数 
            e5就是1*（10的5次方）即100000
        """
        meta = {
            "lr0": (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
            "lrf": (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
            "momentum": (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1 动量 减少抖动
            "weight_decay": (1, 0.0, 0.001),  # optimizer weight decay 权重衰退 减少权重 对学习的影响,达到学习抽象泛化 能识别更多的
            "warmup_epochs": (1, 0.0, 5.0),  # warmup epochs (fractions ok)
            "warmup_momentum": (1, 0.0, 0.95),  # warmup initial momentum 初始化动量
            "warmup_bias_lr": (1, 0.0, 0.2),  # warmup initial bias lr 初始化 偏执和学习率
            "box": (1, 0.02, 0.2),  # box loss gain 识别框的损失得分
            "cls": (1, 0.2, 4.0),  # cls loss gain  分类损失（‌Classification Loss）‌得分
            "cls_pw": (1, 0.5, 2.0),  # cls BCELoss positive_weight 分类损失交叉熵 积极权重  二元交叉熵(Binary CrossEntropy)
            "obj": (1, 0.2, 4.0),  # obj loss gain (scale with pixels) 目标损失得分
            "obj_pw": (1, 0.5, 2.0),  # obj BCELoss positive_weight 目标交叉熵 积极权重
            "iou_t": (0, 0.1, 0.7),  # IoU training threshold iou训练阈值
            "anchor_t": (1, 2.0, 8.0),  # anchor-multiple threshold 先验框多阈值
            "anchors": (2, 2.0, 10.0),  # anchors per output grid (0 to ignore) 先验框 输出方格
            "fl_gamma": (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5) Focal Loss是一种损失函数，‌旨在解决分类问题中类别不平衡和模型训练过程中的样本难易程度问题。‌
            "hsv_h": (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            "hsv_s": (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            "hsv_v": (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
            "degrees": (1, 0.0, 45.0),  # image rotation (+/- deg)
            "translate": (1, 0.0, 0.9),  # image translation (+/- fraction)
            "scale": (1, 0.0, 0.9),  # image scale (+/- gain)
            "shear": (1, 0.0, 10.0),  # image shear (+/- deg)
            "perspective": (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
            "flipud": (1, 0.0, 1.0),  # image flip up-down (probability)
            "fliplr": (0, 0.0, 1.0),  # image flip left-right (probability)
            "mosaic": (1, 0.0, 1.0),  # image mixup (probability)
            "mixup": (1, 0.0, 1.0),  # image mixup (probability)
            "copy_paste": (1, 0.0, 1.0),
        }  # segment copy-paste (probability)
        """ 类似 try catch 忽略错误 """
        with open(opt.hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if "anchors" not in hyp:  # anchors commented in hyp.yaml
                hyp["anchors"] = 3
        if opt.noautoanchor:
            del hyp["anchors"], meta["anchors"]
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        evolve_yaml, evolve_csv = save_dir / "hyp_evolve.yaml", save_dir / "evolve.csv"
        # google bucket 不需要
        if opt.bucket:
            # download evolve.csv if exists
            subprocess.run(
                [
                    "gsutil",
                    "cp",
                    f"gs://{opt.bucket}/evolve.csv",
                    str(evolve_csv),
                ]
            )

        # 遍历 超参数 自动化地调整这些超参数，‌以优化模型的训练效果
        for _ in range(opt.evolve):  # generations to evolve
            if evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate 选取最好的超参数
                # Select parent(s)
                parent = "single"  # parent selection method: 'single' or 'weighted'
                # 加载参数文件
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=",", skiprows=1)
                # 先前结果
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations 最优
                w = fitness(x) - fitness(x).min() + 1e-6  # weights (sum > 0) 计算模型权重最优参数
                if parent == "single" or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == "weighted":
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([meta[k][0] for k in hyp.keys()])  # gains 0-1 得分
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation 训练调整
            """ 关键代码 """
            results = train(hyp.copy(), opt, device, callbacks)
            # 回调
            callbacks = Callbacks()
            # Write mutation results 写变化结果
            keys = (
                "metrics/precision",
                "metrics/recall",
                "metrics/mAP_0.5",
                "metrics/mAP_0.5:0.95",
                "val/box_loss",
                "val/obj_loss",
                "val/cls_loss",
            )
            # 打印调整 变化
            print_mutation(keys, results, hyp.copy(), save_dir, opt.bucket)

        # Plot results 画图
        plot_evolve(evolve_csv)
        LOGGER.info(
            f'Hyperparameter evolution finished {opt.evolve} generations\n'
            f"Results saved to {colorstr('bold', save_dir)}\n"
            f'Usage example: $ python train.py --hyp {evolve_yaml}'
        )


def run(**kwargs):
    """
    Run the training process for a YOLOv3 model with the specified configurations.

    Args:
        data (str): Path to the dataset YAML file.
        weights (str): Path to the pre-trained weights file or '' to train from scratch.
        cfg (str): Path to the model configuration file.
        hyp (str): Path to the hyperparameters YAML file.
        epochs (int): Total number of training epochs.
        batch_size (int): Total batch size across all GPUs.
        imgsz (int): Image size for training and validation (in pixels).
        rect (bool): Use rectangular training for better aspect ratio preservation.
        resume (bool | str): Resume most recent training if True, or resume training from a specific checkpoint if a string.
        nosave (bool): Only save the final checkpoint and not the intermediate ones.
        noval (bool): Only validate model performance in the final epoch.
        noautoanchor (bool): Disable automatic anchor generation.
        noplots (bool): Do not save any plots.
        evolve (int): Number of generations for hyperparameters evolution.
        bucket (str): Google Cloud Storage bucket name for saving run artifacts.
        cache (str | None): Cache images for faster training ('ram' or 'disk').
        image_weights (bool): Use weighted image selection for training.
        device (str): Device to use for training, e.g., '0' for first GPU or 'cpu' for CPU.
        multi_scale (bool): Use multi-scale training.
        single_cls (bool): Train a multi-class dataset as a single-class.
        optimizer (str): Optimizer to use ('SGD', 'Adam', or 'AdamW').
        sync_bn (bool): Use synchronized batch normalization (only in DDP mode).
        workers (int): Maximum number of dataloader workers (per rank in DDP mode).
        project (str): Location of the output directory.
        name (str): Unique name for the run.
        exist_ok (bool): Allow existing output directory.
        quad (bool): Use quad dataloader.
        cos_lr (bool): Use cosine learning rate scheduler.
        label_smoothing (float): Label smoothing epsilon.
        patience (int): EarlyStopping patience (epochs without improvement).
        freeze (list[int]): List of layers to freeze, e.g., [0] to freeze only the first layer.
        save_period (int): Save checkpoint every 'save_period' epochs (disabled if less than 1).
        seed (int): Global training seed for reproducibility.
        local_rank (int): For automatic DDP Multi-GPU argument parsing, do not modify.

    Returns:
        None

    Example:
        ```python
        from ultralytics import run
        run(data='coco128.yaml', weights='yolov5m.pt', imgsz=320, epochs=100, batch_size=16)
        ```

    Notes:
        - Ensure the dataset YAML file and initial weights are accessible.
        - Refer to the [Ultralytics YOLOv5 repository](https://github.com/ultralytics/yolov5) for model and data configurations.
        - Use the [Training Tutorial](https://docs.ultralytics.com/yolov5/tutorials/train_custom_data) for custom dataset training.
    """
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt

if __name__ == "__main__":
    """
    讲命令行参数 解析成 普通参数，传入main方法
    一般来说，main方法 可以通过命令行启动，传入参数
    """
    opt = parse_opt()
    main(opt)
