#!/bin/bash

echo "--- BẮT ĐẦU HUẤN LUYỆN ---"

# Chạy train.py và lưu kết quả (đường dẫn model) vào một biến
# Thay config.json bằng đường dẫn file config của bạn
CONFIG_FILE="/kaggle/working/CAER-S/CAER/configs/config.json"
BEST_MODEL_PATH=$(python train.py --config $CONFIG_FILE)

# Kiểm tra xem có nhận được đường dẫn không
if [ -z "$BEST_MODEL_PATH" ]; then
    echo "Lỗi: Không nhận được đường dẫn model từ train.py. Dừng lại."
    exit 1
fi

echo "--- HUẤN LUYỆN HOÀN TẤT ---"
echo "Model tốt nhất được lưu tại: $BEST_MODEL_PATH"
echo ""
echo "--- BẮT ĐẦU KIỂM THỬ ---"

# Tự động chạy test.py với đường dẫn vừa nhận được
python test.py --config $CONFIG_FILE --resume "$BEST_MODEL_PATH"

echo "--- KIỂM THỬ HOÀN TẤT ---"