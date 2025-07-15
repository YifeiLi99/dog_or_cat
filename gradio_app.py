import gradio as gr
import torch
from inference.inference import predict_image, EfficientNetBinaryClassifier
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EfficientNetBinaryClassifier(pretrained=False)
model.load_state_dict(torch.load('./weights/efficientnet_cat_dog03.pth', map_location=device))
model.to(device)

# transform和训练时保持一致
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def gr_inference(image):
    cls, prob = predict_image(model, image, device)
    label = 'Dog 🐶' if cls == 1 else 'Cat 🐱'
    return f'{label} - 置信度 {prob:.2%}'

gr.Interface(fn=gr_inference,
             inputs=gr.Image(type='pil'),
             outputs=gr.Text(label='推理结果'),
             title='猫狗分类 Demo',
             description='拖拽或上传图片，返回猫或狗的分类结果').launch()
