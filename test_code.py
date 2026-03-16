import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageTk
import os
import tkinter as tk
from tkinter import filedialog, messagebox

# --- CẤU HÌNH HỆ THỐNG ---
MODEL_PATH = 'dog_cat_model.pth'
IMAGE_DIR = 'images'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lấy danh sách class từ thư mục images
if os.path.exists(IMAGE_DIR):
    classes = sorted(list(set(["_".join(f.split("_")[:-1])
                               for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')])))
else:
    classes = []

# --- LOAD MODEL ---
def load_trained_model():
    model_ft = models.resnet50()
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(classes))
    if os.path.exists(MODEL_PATH):
        model_ft.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model_ft.to(DEVICE)
        model_ft.eval()
        return model_ft
    return None

# Khởi tạo biến model toàn cục
global_model = load_trained_model()

# --- TIỀN XỬ LÝ ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- GIAO DIỆN TKINTER ---
class DogCatGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Nhận Diện Chó Mèo - Ha_cho_meo")
        self.root.geometry("500x700")
        self.root.configure(bg="#f5f5f5")

        # Gán model vào class để tránh lỗi AttributeError
        self.model = global_model

        # Giao diện
        self.label_title = tk.Label(root, text="Phân loại giống Chó & Mèo", font=("Arial", 18, "bold"), bg="#f5f5f5")
        self.label_title.pack(pady=20)

        self.canvas = tk.Canvas(root, width=300, height=300, bg="white", highlightthickness=1)
        self.canvas.pack(pady=10)

        self.btn_browse = tk.Button(root, text="Chọn ảnh để nhận diện", command=self.upload_image,
                                    font=("Arial", 12, "bold"), bg="#2196F3", fg="white", padx=20, pady=10)
        self.btn_browse.pack(pady=20)

        self.label_result = tk.Label(root, text="Kết quả: Chưa có ảnh", font=("Arial", 15, "bold"), fg="#333", bg="#f5f5f5")
        self.label_result.pack(pady=10)

        self.label_conf = tk.Label(root, text="Độ tin cậy: 0%", font=("Arial", 12), fg="#666", bg="#f5f5f5")
        self.label_conf.pack(pady=5)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return

        # Hiển thị ảnh lên giao diện
        img = Image.open(file_path)
        img.thumbnail((300, 300))
        self.img_tk = ImageTk.PhotoImage(img)
        self.canvas.create_image(150, 150, image=self.img_tk)

        # Kiểm tra model trước khi dự đoán
        if self.model:
            self.predict(file_path)
        else:
            messagebox.showerror("Lỗi", "Không tìm thấy file 'dog_cat_model.pth'!\nHãy chạy file huấn luyện trước.")

    def predict(self, path):
        # Tiền xử lý ảnh
        img = Image.open(path).convert("RGB")
        img_t = transform(img).unsqueeze(0).to(DEVICE)

        # Dự đoán
        with torch.no_grad():
            output = self.model(img_t)
            prob = torch.nn.functional.softmax(output[0], dim=0)
            conf, pred = torch.max(prob, 0)

        # --- BỘ LỌC VẬT THỂ LẠ ---
        # Ngưỡng tin cậy (Threshold) đang đặt là 0.6 (tức 60%)
        # Bạn có thể tăng lên 0.7 hoặc 0.8 nếu muốn AI "khắt khe" hơn
        if conf.item() < 0.60:
            self.label_result.config(text="Kết quả: Không rõ Chó hay Mèo!", fg="red")
            self.label_conf.config(text=f"Độ tin cậy quá thấp: {conf.item() * 100:.2f}%")
            return

        # Lấy tên giống từ danh sách classes
        raw_name = classes[pred.item()]

        # Logic phân loại Chó/Mèo: Viết hoa = Mèo, viết thường = Chó
        if raw_name[0].isupper():
            species_type = "Mèo"
        else:
            species_type = "Chó"

        # Định dạng lại tên loài (Ví dụ: russian_blue -> Russian Blue)
        breed_name = raw_name.replace("_", " ").title()

        # Cập nhật kết quả lên màn hình
        self.label_result.config(text=f"Kết quả: {species_type}: {breed_name}", fg="#2E7D32")
        self.label_conf.config(text=f"Độ tin cậy: {conf.item() * 100:.2f}%")

if __name__ == "__main__":
    root = tk.Tk()
    app = DogCatGUI(root)
    root.mainloop()