import torch
import torch.nn
import onnx
model_name=""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
 # 路径改成训练输出模型的位置
model = torch.load(f'./model/{model_name}/net_last.pth', map_location=device)
model.eval()
 
input_names = ['input']
output_names = ['output']
 
batch_size=1
input_shape=(3, 128, 64)
x = torch.randn(batch_size, *input_shape, device=device)   # 生成张量
 # 路径改为转换onnx模型的位置
torch.onnx.export(model, x, f'./model/{model_name}/net_last.onnx', input_names=input_names, output_names=output_names, verbose='True')# export_params=True,do_constant_folding=True,