import os, time, torch
import numpy as np
from torchvision import transforms
from PIL import Image
from FCN_model import fcn_resnet50_model
import matplotlib.pyplot as plt

def main():
    aux = False
    classes = 20

    ### check file######################
    weight_path = 'E:\\python_project\\FCN_Resnet50\\save_weights\\model.pth'
    test_path = 'E:\\python_project\\FCN_Resnet50\\11-20112G61125.jpg'
    assert os.path.exists(weight_path), f"weight {weight_path} is not found"
    assert os.path.exists(test_path), f"test image {test_path} is not found"
    ####################################
    # 删除model中的aux一切权重和层
    model = fcn_resnet50_model(aux=aux, num_classes=classes+1)
    weights_dict = torch.load(weight_path, map_location='cpu')['model']
    for key in list(weights_dict.keys()):
        if 'aux' in key:
            del weights_dict[key]
    ###加载权重给model#######################
    model.load_state_dict(weights_dict)
    model.to('cuda')
    img_trans_method = transforms.Compose([transforms.Resize(520),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    orig_img = Image.open(test_path)
    image = img_trans_method(orig_img)
    image = torch.unsqueeze(image, dim=0)

    model.eval()
    with torch.no_grad():
        imageh, imagew = image.shape[-2:]
        init_img = torch.zeros((1,3,imageh,imagew), device='cuda')
        model(init_img) #### 预先跑一次推理过程
        ### 分类我们的图片
        start_time = time.time()
        output = model(image.to('cuda'))
        end_time = time.time()
        print(f"inference time: {end_time-start_time}")

        ###########预测结果

        prediction = output['out'].argmax(1).squeeze(0)
        out_img = prediction.to('cpu').numpy().astype(np.uint8)


    plt.subplot(121)
    plt.imshow(np.array(orig_img))
    plt.subplot(122)
    plt.imshow(out_img)
    plt.show()


if __name__ == "__main__":
    main()



