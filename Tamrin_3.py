# فایل باید با Anacoda باز شود
import os
from PIL import Image
import torch
from torchvision import transforms, models
import re

# مسیر پوشه تصاویر
image_folder = "lion"

# مسیر فایل کلاس‌های ImageNet
class_labels_file = "imagenet_class_labels.txt"

# تابع برای بارگذاری کلاس‌ها از فایل
def load_classes(filepath):
    with open(filepath) as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

# بارگذاری کلاس‌ها
classes = load_classes(class_labels_file)

# تعریف تغییرات اعمال شده روی تصویر برای پیش‌بینی
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# بارگذاری مدل از پیش آموزش‌داده شده
model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)

# تابع پیش‌بینی
def predict(img_path):
    img = Image.open(img_path)
    img_tensor = transform(img)
    batch = img_tensor.unsqueeze(0)
    model.eval()
    output = model(batch)
    predicted_label = torch.argmax(output, dim=1).item()
    return predicted_label

# بررسی تصاویر و محاسبه دقت
correct = 0
total = 0

for filename in os.listdir(image_folder):
    img_path = os.path.join(image_folder, filename)
    predicted_label = predict(img_path)
    true_label = classes[predicted_label]

    if re.search("lion", true_label, re.IGNORECASE):
        correct += 1

    total += 1

accuracy = (correct / total) * 100
print(f"accuracy : {accuracy:.2f}%")
