import torch
import torch.nn as nn


def create_lr_scheduler(optimizer, num_step: int, epochs: int,warmup=True, warmup_epochs=1, warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):

        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中，学习率因子（learning rate factor）： warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后， warmup过程中，学习率因子（learning rate factor）：1 -> 0
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


class ConfusionMatrix:
    def __init__(self, num_classes=21):
        self.num_classes = num_classes
        self.mat = None

    def update(self, target, predict):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=target.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + predict[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def reset(self):
        if self.mat is not None:
            self.mat.zero_()

    def reduce_from_all_processes(self):

        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)

def evaluate(model, dataloader):
    model.eval()
    conf_mat = ConfusionMatrix()
    with torch.no_grad():
        for image, target in dataloader:
            image, target = image.to("cuda"), target.to("cuda")
            predict = model(image)
            predict = predict['out']
            conf_mat.update(target.flatten(), predict.argmax(1).flatten())
        conf_mat.reduce_from_all_processes()
    return conf_mat

def criterion(predict, target):
    loss = {}
    for name, ret in predict.items():
        loss[name] = nn.functional.cross_entropy(ret, target, ignore_index=255)
    if len(loss) == 1:
        return loss['out']
    return loss['out'] + 0.5 * loss['aux']


def train_one_epoch(model, optimizer, dataloader, lr_scheduler, scaler=None):
    model.train()
    for image, target in dataloader:
        image, target = image.to("cuda"), target.to("cuda")
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            predict = model(image)
            loss = criterion(predict, target)
        ##### 更新模型参数########
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        lr_scheduler.step()
    return loss, optimizer.param_groups[0]["lr"]



