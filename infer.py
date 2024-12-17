import torch
import argparse
import torch.nn as nn
# Define the model architecture
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        # Lớp đầu vào và lớp ẩn thứ nhất
        self.fc1 = nn.Linear(16, 128)  # Dense(128)
        self.dropout1 = nn.Dropout(0.3)  # Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(128)  # Batch Normalization

        # Lớp ẩn thứ hai
        self.fc2 = nn.Linear(128, 64)  # Dense(64)
        self.dropout2 = nn.Dropout(0.3)  # Dropout(0.3)
        self.bn2 = nn.BatchNorm1d(64)  # Batch Normalization

        # Lớp ẩn thứ ba
        self.fc3 = nn.Linear(64, 32)  # Dense(32)
        self.bn3 = nn.BatchNorm1d(32)  # Batch Normalization

        # Lớp đầu ra (linear)
        self.output = nn.Linear(32, 1)  # Dense(1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Forward pass qua các lớp
        x = self.relu(self.bn1(self.fc1(x)))  # Lớp ẩn thứ nhất với ReLU và Batch Norm
        x = self.dropout1(x)  # Dropout
        x = self.relu(self.bn2(self.fc2(x)))  # Lớp ẩn thứ hai với ReLU và Batch Norm
        x = self.dropout2(x)  # Dropout
        x = self.relu(self.bn3(self.fc3(x)))  # Lớp ẩn thứ ba với ReLU và Batch Norm
        x = self.output(x)  # Lớp đầu ra (linear, không activation)
        return x

def main():
    parser = argparse.ArgumentParser(description='Infer with a trained model.')
    parser.add_argument('--path', type=str, required=True, help='Path to the model checkpoint (best_model.pth)')
    args = parser.parse_args()

    # Load model và thông số chuẩn hóa
    checkpoint = torch.load(args.path)
    model = RegressionModel()  # Thay MyModel bằng kiến trúc model bạn sử dụng
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    x_train_min = checkpoint['x_train_min']  # Lấy giá trị x_train_min
    x_train_max = checkpoint['x_train_max']  # Lấy giá trị x_train_max

    # Nhập giá trị từ người dùng
    def get_user_input(feature_name):
        return float(input(f"Xin mời nhập giá trị {feature_name}: "))

    # Nhập từng giá trị
    t2mdew = get_user_input("t2mdew")
    t2m = get_user_input("t2m")
    ps = get_user_input("ps")
    tqv = get_user_input("tqv")
    tql = get_user_input("tql")
    h1000 = get_user_input("h1000")
    disph = get_user_input("disph")
    frcan = get_user_input("frcan")
    hlml = get_user_input("hlml")
    rhoa = get_user_input("rhoa")
    cig = get_user_input("cig")
    ws = get_user_input("ws")
    cldcr = get_user_input("cldcr")
    v_2m = get_user_input("v_2m")
    v_50m = get_user_input("v_50m")
    v_850 = get_user_input("v_850")

    # Tạo tensor từ giá trị người dùng
    x_user_tensor = torch.tensor([t2mdew, t2m, ps, tqv, tql, h1000, disph, frcan, hlml, rhoa, cig, ws, cldcr, v_2m, v_50m, v_850], dtype=torch.float32)

    # Normalize dữ liệu người dùng
    x_user_normalized = (x_user_tensor - x_train_min.squeeze()) / (x_train_max.squeeze() - x_train_min.squeeze())

    # Tạo dự đoán
    with torch.no_grad():
        prediction = model(x_user_normalized.unsqueeze(0))  # Thêm batch dimension

    # In kết quả dự đoán
    print("Dự đoán từ model pm2.5:", prediction.item())

if __name__ == '__main__':
    main()
