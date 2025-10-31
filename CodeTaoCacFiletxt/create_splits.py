import os
import random
from collections import defaultdict

# ================================================================= #
# =========================== CẤU HÌNH ============================ #
# ================================================================= #

# 1. Đường dẫn đến thư mục gốc chứa các thư mục 'train' và 'test'
#    Ví dụ: 'C:/Dataset/Mubaohiem'
ROOT_DATA_FOLDER = 'C:/animal'

# 2. Đường dẫn đến thư mục bạn muốn lưu các file .txt
#    Ví dụ: 'D:/du_an_swat/data/Mubaohiem/'
OUTPUT_FOLDER = 'D:/thuyloiuniversity/Mon hoc tren lop/BigData/SWAT/data/animal'

# --- Cấu hình nâng cao (thường không cần đổi) ---
VAL_RATIO = 0.1  # 10% của tập train gốc sẽ được dùng làm validation
FEWSHOT_SIZES = [4, 8, 16] # Các kích thước few-shot bạn cần
SEEDS = [1, 2, 3] # Các seed bạn cần

# ================================================================= #
# ========================= PHẦN XỬ LÝ ============================ #
# ================================================================= #

def process_folder(folder_path, class_to_id):
    """Đọc một thư mục (train hoặc test) và trả về list các (đường dẫn, id_lớp)."""
    data_list = []
    for class_name in class_to_id.keys():
        class_path = os.path.join(folder_path, class_name)
        if not os.path.isdir(class_path):
            continue
            
        image_files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        
        for img in image_files:
            full_path = os.path.join(class_path, img)
            data_list.append((full_path, class_to_id[class_name]))
            
    return data_list

def write_to_file(filepath, data_list):
    """Ghi danh sách dữ liệu ra file txt."""
    with open(filepath, 'w') as f:
        for path, label in data_list:
            # Chuẩn hóa đường dẫn để tương thích
            path = path.replace('\\', '/')
            f.write(f"{path} {label} 1\n")
    print(f"✅ Đã ghi {len(data_list)} dòng vào file '{os.path.basename(filepath)}'")

def create_all_splits():
    """Hàm chính để tạo tất cả các file split."""
    print("Bắt đầu quá trình tạo file split...\n")
    
    # --- Bước 1: Khám phá dữ liệu ---
    train_folder = os.path.join(ROOT_DATA_FOLDER, 'train')
    test_folder = os.path.join(ROOT_DATA_FOLDER, 'test')

    if not os.path.isdir(train_folder) or not os.path.isdir(test_folder):
        print(f"Lỗi: Không tìm thấy thư mục 'train' hoặc 'test' trong '{ROOT_DATA_FOLDER}'")
        return

    # Tạo thư mục output nếu chưa có
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Tự động tìm tên các lớp từ thư mục train
    class_names = [d for d in os.listdir(train_folder) if os.path.isdir(os.path.join(train_folder, d))]
    if not class_names:
        print(f"Lỗi: Không tìm thấy thư mục lớp nào trong '{train_folder}'.")
        return
        
    class_to_id = {name: i for i, name in enumerate(class_names)}
    print(f"Tìm thấy {len(class_names)} lớp: {class_to_id}\n")

    # --- Bước 2: Xử lý và ghi file Test ---
    print("--- Xử lý tập TEST ---")
    test_data = process_folder(test_folder, class_to_id)
    random.shuffle(test_data)
    write_to_file(os.path.join(OUTPUT_FOLDER, 'test.txt'), test_data)
    print("-" * 20)

    # --- Bước 3: Xử lý tập Train gốc để tạo ra Train/Val và Fewshot ---
    original_train_data = process_folder(train_folder, class_to_id)

    # --- Bước 3a: Tạo file Train và Val ---
    print("\n--- Xử lý tập TRAIN và VALIDATION ---")
    random.seed(42) # Dùng một seed cố định để việc chia train/val luôn giống nhau
    random.shuffle(original_train_data)
    
    split_index = int(len(original_train_data) * (1 - VAL_RATIO))
    final_train_data = original_train_data[:split_index]
    val_data = original_train_data[split_index:]
    
    write_to_file(os.path.join(OUTPUT_FOLDER, 'train.txt'), final_train_data)
    write_to_file(os.path.join(OUTPUT_FOLDER, 'val.txt'), val_data)
    print("-" * 20)

    # --- Bước 3b: Tạo các file Fewshot ---
    print("\n--- Xử lý các tập FEWSHOT ---")
    
    # Gom các ảnh theo từng lớp để dễ rút mẫu
    data_by_class = defaultdict(list)
    for path, label in original_train_data:
        data_by_class[label].append((path, label))

    for seed in SEEDS:
        print(f"Đang tạo fewshot cho SEED = {seed}...")
        for size in FEWSHOT_SIZES:
            # Đặt seed để đảm bảo kết quả lặp lại cho mỗi seed
            random.seed(seed)
            
            fewshot_data = []
            for class_id, images in data_by_class.items():
                # Xáo trộn danh sách ảnh của lớp hiện tại
                random.shuffle(images)
                # Lấy `size` mẫu từ mỗi lớp
                fewshot_data.extend(images[:size])
            
            # Xáo trộn tập fewshot cuối cùng
            random.shuffle(fewshot_data)
            
            filename = f"fewshot{size}_seed{seed}.txt"
            write_to_file(os.path.join(OUTPUT_FOLDER, filename), fewshot_data)
    print("-" * 20)
    
    print("\n🎉 Hoàn tất! Tất cả các file đã được tạo thành công.")


if __name__ == '__main__':
    create_all_splits()