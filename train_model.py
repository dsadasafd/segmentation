from FCN_model import fcn_resnet50_model
from dataset import VOCSegmentation
from torch.utils.data import DataLoader
from my_utils import create_lr_scheduler, criterion, evaluate, train_one_epoch
import argparse, os, torch, time, datetime, shutil




##### 程序运行参数设置 ##################################################
def creat_argparse():
    ap = argparse.ArgumentParser(description="pytorch fcn training")
    ap.add_argument("--data_path", required=True, help="path of VOC2012")
    args = vars(ap.parse_args())
    return args
#####################################################################

def main(args):
    # 初始化文件参数 #
    root_path = args['data_path'] # 原文件夹目录
    JPEGImages = os.path.join(root_path, 'JPEGImages') # 未分类图文件
    SegmenttationClass = os.path.join(root_path, 'SegmentationClass') # 标签，已分类图文件
    train_txt = os.path.join(root_path, 'train.txt')
    val_txt = os.path.join(root_path, 'val.txt')
    weight_path = "E:\\python_project\\FCN_Resnet50\\fcn_resnet50_coco-1167a1af.pth"

    assert os.path.exists(JPEGImages), 'JPEGImages not exists'
    assert os.path.exists(SegmenttationClass), 'SegmenttationClass not exists'
    assert os.path.exists(train_txt), 'train_txt not exists'
    assert os.path.exists(val_txt), 'val_txt not exists'

    # Dataset创建  #
    train_dataset = VOCSegmentation(JPEGImages, SegmenttationClass, train_txt, train_val="train")
    val_dataset = VOCSegmentation(JPEGImages, SegmenttationClass, val_txt, train_val='val')

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, pin_memory=True, collate_fn=train_dataset.collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=2, pin_memory=True, collate_fn=val_dataset.collate_fn)

    #创建模型, 创建优化器， 创建损失函数#
    weight_dict = torch.load(weight_path, map_location='cpu')
    for k in list(weight_dict.keys()):
        if 'classifer.4' in k:
            del weight_dict[k]
    model = fcn_resnet50_model(aux=True, num_classes=21)
    missing_keys, unexpected_keys = model.load_state_dict(weight_dict, strict=False)
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print('missing_keys:', missing_keys)
        print('unexpected_keys:', unexpected_keys)
    
    model.to("cuda")
    amp = False

    params_to_optimize = [{'params': [p for p in model.backbone.parameters() if p.requires_grad]},
                          {'params': [p for p in model.classifier.parameters() if p.requires_grad]},
                          {'params': [p for p in model.aux_classifier.parameters() if p.requires_grad], 'lr': 0.001}]

    optimizer = torch.optim.SGD(params_to_optimize, lr=0.0001, momentum=0.9, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler() if amp else None
    # 学习率更新策略：每个step更新一次 (不是每个epoch 更新一次)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_dataloader), 30, warmup=True)

    if os.path.exists('./results'):
        shutil.rmtree('./results')
    os.mkdir('./results')
    results_file = "results/result_{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    #####################从这里开始正式是train######################################################
    start_time = time.time()
    for epoch in range(30):
        mean_loss, lr = train_one_epoch(model, optimizer, train_dataloader, lr_scheduler, scaler)
        #conf_mat = evaluate(model, val_dataloader)
        #val_info = str(conf_mat)
        print(mean_loss)

        #with open(results_file, 'a') as r:
            #train_info = f"[epoch: {epoch}]\n" + f"train_loss: {mean_loss:.4f}\n" + f"lr: {lr:.6f}\n"
            #r.write(train_info + val_info + "\n\n")

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}

        if amp:
            save_file["scaler"] = scaler.state_dict()
        torch.save(save_file, "save_weights/model.pth")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


##############################调试**********************************
if __name__ == "__main__":
    args = creat_argparse()

    if os.path.exists("./save_weights"):
        shutil.rmtree('./save_weights')

    os.mkdir("./save_weights")

    main(args)


