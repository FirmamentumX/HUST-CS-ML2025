
from utils import draw

def extract_model_type(path):
    return path.split('/')[1]

# 主程序
if __name__ == '__main__':
    log_file = '../CNN/logs/fashion_mnist_Icnn_2025-05-25T20-24-29.353102.txt' 
    # log_file = '../RNN/logs/fashion_mnist_Iwrn_40-4-0.5_2025-05-29T23-37-24.595442.txt'
    model_type = extract_model_type(log_file)
    draw(log_file, model_type)