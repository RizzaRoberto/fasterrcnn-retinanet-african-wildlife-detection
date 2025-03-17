import tkinter as tk
from tkinter import filedialog
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from tkinter import messagebox
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math


class ObjectDetectionApp:

    class_mapping = {
        1: "Buffalo",
        2: "Elephant",
        3: "Rhino",
        4: "Zebra"
    }

    def __init__(self, window):
        self.window = window
        self.window.title("African Animals Detection")

        self.window.resizable(False, False)  

        self.model = None
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])
        self.models = {
            "faster_rcnn": "path/fasterRCNN_checkpoint__5.pth",
            "retinanet": "path/retinaNet_checkpoint__10.pth"
        }
        self.image_dir = None
        self.image_list = []

        self.create_widgets()

    def create_widgets(self):
        button_frame = tk.Frame(self.window)
        button_frame.pack(pady=10)

        self.load_faster_rcnn_btn = tk.Button(button_frame, text="Load FasterR-CNN", command=lambda: self.load_model("faster_rcnn"), height=1, width=16)
        self.load_faster_rcnn_btn.pack(side=tk.LEFT, padx=5)

        self.load_retinanet_btn = tk.Button(button_frame, text="Load RetinaNet", command=lambda: self.load_model("retinanet"), height=1, width=16)
        self.load_retinanet_btn.pack(side=tk.LEFT, padx=5)

        self.model_status_label = tk.Label(self.window, text="No model loaded", font=("Helvetica", 10))
        self.model_status_label.pack(pady=5)

        self.canvas = tk.Canvas(self.window, width=800, height=600)
        self.canvas.pack()

        self.predict_btn = tk.Button(self.window, text="Predict", command=self.predict, width=12, height=2)
        self.predict_btn.pack()

        self.select_image_btn = tk.Button(self.window, text="Select Image or Folder", command=self.select_image_or_folder, height=2, width=20)
        self.select_image_btn.pack()

        self.exit_btn = tk.Button(self.window, text="Esci", command=self.exit_app, width=12, height=2)
        self.exit_btn.pack(pady=5)

        

    def load_model(self, model_name):
        self.model_name = model_name
        if model_name == "faster_rcnn":
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
            num_classes = 5
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        elif model_name == "retinanet":
            self.model = torchvision.models.detection.retinanet_resnet50_fpn_v2(pretrained=True)
            num_classes = 5
            in_features = self.model.head.classification_head.conv[0][0].in_channels
            num_anchors = self.model.head.classification_head.num_anchors
            self.model.head.classification_head.num_classes = num_classes
            out_channels = self.model.head.classification_head.conv[0][0].out_channels
            cls_logits = torch.nn.Conv2d(out_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
            torch.nn.init.normal_(cls_logits.weight, std=0.01)
            torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))
            self.model.head.classification_head.cls_logits = cls_logits

        checkpoint = torch.load(self.models[model_name], map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        self.model_status_label.config(text=f"Loaded model: {model_name}")

    def select_image_or_folder(self):
        choice = messagebox.askquestion("Select Image or Folder", "Do you want to select a single image file?", icon='question')
        if choice == 'yes':
            file_path = filedialog.askopenfilename(title="Select Image",
                                                   filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
            if file_path:
                self.image_list = [file_path]
                self.image_dir = os.path.dirname(file_path)
        else:
            folder_path = filedialog.askdirectory(title="Select Folder")
            if folder_path:
                self.image_dir = folder_path
                self.image_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    def get_class_name(self, class_id):
        return self.class_mapping.get(class_id, "Unknown")

    def predict(self):
        if self.model is None:
            messagebox.showwarning("Choose model", "Please load a model before predicting.")
            return

        if not self.image_list:
            messagebox.showwarning("Choose image", "Please select an image or folder before predicting.")
            return
        self.select_image_btn.pack_forget()
        image_path = random.choice(self.image_list)
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transforms(image).unsqueeze(0)

        with torch.no_grad():
            predictions = self.model([image_tensor[0]])[0]

        boxes = predictions['boxes'].cpu().detach().numpy()
        labels = predictions['labels'].cpu().detach().numpy()
        scores = predictions['scores'].cpu().detach().numpy()

        threshold = 0.5
        idx = np.where(scores > threshold)[0]
        boxes = boxes[idx]
        labels = labels[idx]
        scores = scores[idx]

        self.display_image_with_boxes(image, boxes, labels, scores)

    def display_image_with_boxes(self, image, boxes, labels, scores):
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        ax = plt.gca()
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            rect = plt.Rectangle((x1, y1), w, h, fill=False, color="red")
            ax.add_patch(rect)
            ax.text(x1, y1, f"{self.get_class_name(label)}: {score:.2f}", fontsize=15, color="white",
                    bbox=dict(facecolor="red", alpha=0.5))
        plt.axis('off')

        fig = plt.gcf()
        fig.tight_layout()
        self.canvas.figure = fig

        canvas = FigureCanvasTkAgg(fig, master=self.window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas.delete("all")
        canvas.get_tk_widget().place(x=0, y=0, width=800, height=600)

    def exit_app(self):
        self.window.quit()
        self.window.destroy()


if __name__ == "__main__":
    window = tk.Tk()
    app = ObjectDetectionApp(window)
    window.mainloop()
