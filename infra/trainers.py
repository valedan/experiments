from infra.configs import TrainerParams
from infra.logger import DataLogger
import torch as t
from torch import nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

PROFILE = False


def create_profiler():
    return t.profiler.profile(
        activities=[
            t.profiler.ProfilerActivity.CPU,
            t.profiler.ProfilerActivity.CUDA,
        ],
        schedule=t.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=t.profiler.tensorboard_trace_handler("./log/pytorch_profiler"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )


def train_classifier(
    logger: DataLogger,
    model: nn.Module,
    params: TrainerParams,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device="cpu",
):
    assert train_loader and val_loader
    scaler = GradScaler()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    if params.optimizer == "sgd":
        assert params.learning_rate is not None
        optimizer = t.optim.SGD(model.parameters(), params.learning_rate, fused=True)
    elif params.optimizer == "adam":
        optimizer = t.optim.Adam(model.parameters(), fused=True)

    if PROFILE:
        prof = create_profiler()
        prof.start()

    step = 0
    for epoch in range(1, params.epochs + 1):
        for train_images, labels in train_loader:
            with t.autocast(device_type=str(device), dtype=t.float16):
                # TODO: try out non_blocking=true here
                train_images = train_images.to(device)
                labels = labels.to(device)
                # TODO: normalize images so that pixels are [0, 1] not [0, 255] and see what the impact is
                optimizer.zero_grad()
                logits = model(train_images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            # TODO: look into using clip_grad_norm here to deal with exploding gradients
            scaler.step(optimizer)
            scaler.update()
            logger.log(logits, labels, loss, step, epoch, "train")
            if PROFILE:
                prof.step()

            if step % params.val_log_interval == 0:
                # TODO: put model in eval mode here - model.eval()
                with t.autocast(device_type=str(device), dtype=t.float16):
                    with t.no_grad():
                        for val_images, labels in val_loader:
                            val_images = val_images.to(device)
                            labels = labels.to(device)
                            logits = model(val_images)
                            val_loss = criterion(logits, labels)
                            logger.log(logits, labels, val_loss, step, epoch, "val")

            step += 1
        logger.flush()

    if PROFILE:
        prof.stop()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
