import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision import models
from base import BaseModel

# ==============================================================================
# 1. CÁC MODULE ĐƠN LẺ VÀ HÀM HỖ TRỢ
# ==============================================================================

class FaceEmotionCNN(nn.Module):
    """
    Mạng CNN để nhận dạng cảm xúc khuôn mặt.
    CẬP NHẬT: Kiến trúc nn.Sequential được sắp xếp lại để khớp 100%
    với thứ tự các lớp trong code gốc (Conv->Dropout->BN->Pool->ReLU).
    """
    def __init__(self, num_classes=7):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1: Conv -> BN -> Pool -> ReLU
            nn.Conv2d(1, 8, kernel_size=3), nn.BatchNorm2d(8), nn.MaxPool2d(2, 1), nn.ReLU(inplace=True),
            # Block 2: Conv -> Dropout -> BN -> Pool -> ReLU
            nn.Conv2d(8, 16, kernel_size=3), nn.Dropout(0.3), nn.BatchNorm2d(16), nn.MaxPool2d(2, 1), nn.ReLU(inplace=True),
            # Block 3: Conv -> BN -> Pool -> ReLU
            nn.Conv2d(16, 32, kernel_size=3), nn.BatchNorm2d(32), nn.MaxPool2d(2, 1), nn.ReLU(inplace=True),
            # Block 4: Conv -> Dropout -> BN -> Pool -> ReLU
            nn.Conv2d(32, 64, kernel_size=3), nn.Dropout(0.3), nn.BatchNorm2d(64), nn.MaxPool2d(2, 1), nn.ReLU(inplace=True),
            # Block 5: Conv -> BN -> Pool -> ReLU
            nn.Conv2d(64, 128, kernel_size=3), nn.BatchNorm2d(128), nn.MaxPool2d(2, 2), nn.ReLU(inplace=True),
            # Block 6: Conv -> Dropout -> BN -> Pool -> ReLU
            nn.Conv2d(128, 256, kernel_size=3), nn.Dropout(0.3), nn.BatchNorm2d(256), nn.MaxPool2d(2, 2), nn.ReLU(inplace=True),
            # Block 7: Conv -> Dropout -> BN -> Pool -> ReLU
            nn.Conv2d(256, 256, kernel_size=3), nn.Dropout(0.3), nn.BatchNorm2d(256), nn.MaxPool2d(2, 2), nn.ReLU(inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Block FC 1: Linear -> Dropout -> ReLU
            nn.Linear(1024, 512), nn.Dropout(0.3), nn.ReLU(inplace=True),
            # Block FC 2: Linear -> Dropout -> ReLU
            nn.Linear(512, 256), nn.Dropout(0.3), nn.ReLU(inplace=True),
            # Block FC 3 (lớp cuối)
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Thay thế hàm cũ bằng hàm này trong file model/model.py

def create_swin_backbone():
    """
    Tạo một backbone Swin-T và loại bỏ lớp classifier cuối cùng.
    Swin-T đã tự thực hiện pooling và flatten bên trong.
    """
    model = models.swin_t(weights='DEFAULT')
    
    # Chỉ cần bỏ đi lớp head cuối cùng là đủ.
    # Đầu ra của backbone này đã là một vector đặc trưng 1D (shape [batch, 768]).
    backbone = nn.Sequential(*(list(model.children())[:-1]))
    
    return backbone
# ======================
# Create Resnet backbone
# ======================

# def create_resnet_backbone():


# ==============================================================================
# 2. KIẾN TRÚC KẾT HỢP
# ==============================================================================

class FeatureExtractors(nn.Module):
    """
    Module này chứa 3 bộ trích xuất đặc trưng cho Face, Body, và Context.
    CẬP NHẬT: Tự động chuyển đổi key khi tải model face.
    """
    def __init__(self, face_model_path, num_classes=7):
        super().__init__()
        
        # --- Face Extractor ---
        full_face_model = FaceEmotionCNN(num_classes=num_classes)
        
        # --- LOGIC CHUYỂN ĐỔI KEY "NÓNG" ---
        old_state_dict = torch.load(face_model_path, map_location=torch.device('cpu'))
        new_state_dict = OrderedDict()
        
        key_map = {
            'cnn1': 'features.0',  'cnn1_bn': 'features.1',
            'cnn2': 'features.4',  'cnn2_bn': 'features.6',
            'cnn3': 'features.9',  'cnn3_bn': 'features.10',
            'cnn4': 'features.13', 'cnn4_bn': 'features.15',
            'cnn5': 'features.18', 'cnn5_bn': 'features.19',
            'cnn6': 'features.22', 'cnn6_bn': 'features.24',
            'cnn7': 'features.27', 'cnn7_bn': 'features.29',
            'fc1': 'classifier.1', 'fc2': 'classifier.4', 'fc3': 'classifier.7',
        }
        
        for old_key, value in old_state_dict.items():
            parts = old_key.split('.')
            layer_name = parts[0]
            if layer_name in key_map:
                param_type = '.'.join(parts[1:])
                new_layer_name = key_map[layer_name]
                new_key = f"{new_layer_name}.{param_type}"
                new_state_dict[new_key] = value
        
        full_face_model.load_state_dict(new_state_dict)
        print(f"Đã tải và chuyển đổi thành công trọng số cho Face Model từ: {face_model_path}")
        # --- KẾT THÚC LOGIC CHUYỂN ĐỔI ---
        
        self.face_features = full_face_model.features
        self.face_classifier_head = full_face_model.classifier[:-1]

        # --- Body and Context Extractors ---
        self.body_extractor = create_swin_backbone()
        self.context_extractor = create_swin_backbone()

    def forward(self, face_img, body_img, context_img):
        face_feat = self.face_features(face_img) # [B, 256, 2, 2]
        body_feat = self.body_extractor(body_img) # [B, 768]
        context_feat = self.context_extractor(context_img) # [B, 768]
        
        face_final_feat = self.face_classifier_head(face_feat) # Convert [256, 2, 2] -> [256]
        
        return face_final_feat, body_feat, context_feat

class FusionNetwork(nn.Module):
    def __init__(self, use_face=True, use_body=True, use_context=True, num_classes=7):
        super().__init__()
        self.use_face, self.use_body, self.use_context = use_face, use_body, use_context

        face_dim, body_dim, context_dim = 256, 768, 768
        self.body_proj = nn.Linear(body_dim, 256)
        self.context_proj = nn.Linear(context_dim, 256)
        self.face_bn = nn.BatchNorm1d(256)
        self.body_bn = nn.BatchNorm1d(256)
        self.context_bn = nn.BatchNorm1d(256)
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(256, num_classes)
        )

    def forward(self, face_feat, body_feat, context_feat):


        body_proj, context_proj = self.body_proj(body_feat), self.context_proj(context_feat)
        face_bn, body_bn, context_bn = self.face_bn(face_feat), self.body_bn(body_proj), self.context_bn(context_proj)
        combined_features = torch.cat([face_bn, body_bn, context_bn], dim=1)
        return self.classifier(combined_features)

class CAERSNet(BaseModel):
    # (Giữ nguyên, không cần thay đổi)
    def __init__(self, face_model_path, num_classes=7, use_face=True, use_body=True, use_context=True):
        super().__init__()
        self.backbone = FeatureExtractors(face_model_path, num_classes)
        self.fusion_net = FusionNetwork(use_face, use_body, use_context, num_classes)

    def forward(self, face_img, body_img, context_img):
        face_feat, body_feat, context_feat = self.backbone(face_img, body_img, context_img)
        return self.fusion_net(face_feat, body_feat, context_feat)