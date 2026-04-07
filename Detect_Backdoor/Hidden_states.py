import torch
import numpy as np

# 1. Đọc dữ liệu từ file .pt
print("Đang đọc file hidden_states.pt...")
data = torch.load('/home/dung/Downloads/project1/Detect_Backdoor/hidden_states_clean.pt', weights_only=False)

# 2. Rút ma trận ở layer cuối cùng và nhãn
X = data['hidden_states']['layer_-1'].numpy()
y = data['labels'].numpy()

print(f"Kích thước ban đầu của X (layer_-1): {X.shape}")
print(f"Kích thước của y (nhãn): {y.shape}")

# 3. Lưu thành định dạng Numpy để bạn tự do thao tác
np.save('X_clean_raw.npy', X)
np.save('y_clean_raw.npy', y)

print("\nĐã lưu thành công X_raw.npy và y_raw.npy!")
print("Bây giờ bạn có thể dùng np.load() để load data và tự do phát triển thuật toán QSVM của mình.")