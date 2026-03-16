import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights  # Thêm dòng này cho chuẩn mới
import os
from PIL import Image


# 1. Định nghĩa Dataset tùy chỉnh
class OxfordPetDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        # Lấy danh sách file ảnh, bỏ qua các file không phải .jpg
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        # Trích xuất tên giống từ tên file (ví dụ: 'Abyssinian_1.jpg' -> 'Abyssinian')
        self.labels = ["_".join(f.split("_")[:-1]) for f in self.img_names]
        self.classes = sorted(list(set(self.labels)))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.class_to_idx[self.labels[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label


# 2. Cấu hình tiền xử lý ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Load dữ liệu
# Đảm bảo bạn có thư mục 'images' nằm cùng cấp với file app.py này
dataset = OxfordPetDataset(img_dir='images', transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 4. Khởi tạo mô hình ResNet50 với Weights mới nhất
# Cách viết này sẽ không bị Warning như lúc nãy bạn gặp
weights = ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)

# Thay đổi lớp cuối (Fully Connected) để phân loại 37 giống loài
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(dataset.classes))

# 5. Thiết lập thiết bị và bộ tối ưu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. Vòng lặp huấn luyện
print(f"Bắt đầu huấn luyện với {len(dataset.classes)} giống loài trên {device}...")
model.train()
for epoch in range(5):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if (i + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}, Batch {i + 1}: Loss = {loss.item():.4f}")

    print(f"--- Kết thúc Epoch {epoch + 1} - Average Loss: {running_loss / len(train_loader):.4f} ---")

# 7. QUAN TRỌNG NHẤT: LƯU MÔ HÌNH
print("\nĐang lưu mô hình...")
torch.save(model.state_dict(), 'dog_cat_model.pth')
print("Đã lưu file 'dog_cat_model.pth' thành công! Giờ bạn có thể tắt máy đi ngủ.")