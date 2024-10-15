import onnxruntime
import torch
from torchvision import transforms
from torchvision.transforms.functional import *
from torch.autograd import Variable
# from skimage.metrics import structural_similarity as ssim  

""" *********************function********************** """
# resize
def pad_image(image, target_size=[64,128]):
    iw, ih = image.size  # 原始图像的尺寸
    w, h = target_size  # 目标图像的尺寸
    scale = min(w / iw, h / ih)  # 转换的最小比例

    # 保证长或宽，至少一个符合目标图像的尺寸
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)  # 缩小图像
    # image.show()
    new_image = Image.new('RGB', target_size, (128, 128, 128))  # 生成灰色图像
    # // 为整数除法，计算图像的位置
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))  # 将图像填充为中间图像，两侧为灰色的样式
    # new_image.show()

    return new_image

# load img
def load_img(path_img):
    from PIL import Image
    image = Image.open(path_img)
    image=pad_image(image)
    transform = transforms.Compose([transforms.ToTensor(),])
    tensor_image = transform(image).unsqueeze(0)
    return tensor_image

# transfer tensor to nparray
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# extract feature
def extract_feature(img,session):
    feature_img=np.zeros((512,))
    for i in range(2):
        # horizontal flip image
        if(i==1):
            img = np.flip(img,axis=-1)
        inputs = {'input': img}
        f=session.run(None,inputs)[-1].squeeze(0)
        feature_img+=f
        return feature_img
if __name__ == '__main__':
    # load model
    file_path=r"net_last.onnx"
    session = onnxruntime.InferenceSession(file_path)

    """ *********************workplace********************** """
    img_path='try1.PNG'
    img=load_img(img_path)
    img=to_numpy(img)
    feature_img=extract_feature(img,session)
