import torch
import torch.nn as nn
import torchvision.transforms as transforms
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)
        self.encoder5 = conv_block(512, 1024)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = conv_block(128, 64)
        
        self.conv_last = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        enc5 = self.encoder5(self.pool(enc4))
        
        dec4 = self.upconv4(enc5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return self.conv_last(dec1)


def load_model(path, device):
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def rectify_paper(original_image_np, mask_np):
    
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise ValueError("No contours found in mask image")

    
    contour = max(contours, key=cv2.contourArea)

    
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) != 4:
        raise ValueError("The detected contour does not have 4 corners")

    
    points = approx.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]

   
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    
    warped = cv2.warpPerspective(original_image_np, M, (maxWidth, maxHeight))

    return warped



def blur_and_threshold(gray):

    gray = cv2.GaussianBlur(gray, (1, 1), 0)  

 
    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 15)  # Adjust block size and constant for less harsh thresholding

 
    threshold = cv2.fastNlMeansDenoising(threshold, None, 10, 15, 31)  
    
    return threshold


def flat_field_correction(image):
    
    flat_field = cv2.GaussianBlur(image, (10, 10), 0)  
    
    
    mean_flat_field = np.mean(flat_field)
    
    
    corrected_image = (image.astype(np.float32) * mean_flat_field) / flat_field.astype(np.float32)
    
    
    corrected_image = cv2.normalize(corrected_image, None, 0, 255, cv2.NORM_MINMAX)
    corrected_image = corrected_image.astype(np.uint8)
    
    return corrected_image




def open_file():
    try:
        file_path = filedialog.askopenfilename()
        if file_path:
            
            original_image = Image.open(file_path).convert("L")
            original_image_np = np.array(original_image)
            original_height, original_width = original_image_np.shape
            
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((256, 256)),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            
            img_tensor = transform(original_image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                mask = model(img_tensor).squeeze().cpu().numpy()
            
            
            mask = (mask - mask.min()) / (mask.max() - mask.min())
            mask = (mask > 0.5).astype(np.uint8)  
            
            
            mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
            
            
            original_image_rgb = Image.open(file_path).convert("RGB")
            original_image_np_rgb = np.array(original_image_rgb)
            mask_np = np.array(mask)
            
            
            rectified_image_np = rectify_paper(original_image_np_rgb, mask_np)
            
          
            rectified_image_gray = cv2.cvtColor(rectified_image_np, cv2.COLOR_RGB2GRAY)
            
           
            rectified_image_np = blur_and_threshold(rectified_image_gray)
            
           
            rectified_image_np = cv2.cvtColor(rectified_image_np, cv2.COLOR_GRAY2RGB)
            
            rectified_image = Image.fromarray(rectified_image_np.astype(np.uint8))
            
        
            rectified_image.show()
            
       
            original_image_resized = original_image_rgb.resize((256, 256))
            original_image_tk = ImageTk.PhotoImage(original_image_resized)
            
            original_label.config(image=original_image_tk)
            original_label.image = original_image_tk
    except Exception as e:
        print(f"An error occurred: {e}")


root = tk.Tk()
root.title("Image Segmentation with UNet")

open_button = tk.Button(root, text="Open Image", command=open_file)
open_button.pack()

original_label = tk.Label(root)
original_label.pack()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model('best_model.pth', device)

root.mainloop()
