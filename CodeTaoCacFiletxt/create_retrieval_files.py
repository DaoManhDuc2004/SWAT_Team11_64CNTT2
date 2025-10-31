import os
import random

# ================================================================= #
# =========================== CẤU HÌNH ============================ #
# ================================================================= #

# 1. Đường dẫn đến thư mục gốc chứa ảnh đã truy xuất (retrieved)
#    Script sẽ tự động tìm các thư mục con bên trong (Mega_Blueberry, Mega_Peach)
ROOT_RETRIEVED_FOLDER = 'C:/animal/retrieved'

# 2. Đường dẫn đến thư mục bạn muốn lưu các file .txt
OUTPUT_FOLDER = 'D:/thuyloiuniversity/Mon hoc tren lop/BigData/SWAT/data/animal'

# 3. Giới hạn số lượng ảnh lấy từ mỗi lớp
MAX_IMAGES_PER_CLASS = 500

# ================================================================= #
# ========================= PHẦN XỬ LÝ ============================ #
# ================================================================= #

def create_retrieval_files():
    """
    Hàm chính để tìm ảnh, giới hạn số lượng và ghi ra các file.
    """
    print("Bắt đầu quá trình tạo file retrieval...\n")
    
    if not os.path.isdir(ROOT_RETRIEVED_FOLDER):
        print(f"❌ Lỗi: Thư mục ảnh '{ROOT_RETRIEVED_FOLDER}' không tồn tại.")
        return

    # Tạo thư mục output nếu chưa có
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Tự động tìm tên các lớp từ các thư mục con
    try:
        class_names = [d for d in os.listdir(ROOT_RETRIEVED_FOLDER) if os.path.isdir(os.path.join(ROOT_RETRIEVED_FOLDER, d))]
        if not class_names:
            print(f"❌ Lỗi: Không tìm thấy thư mục lớp nào trong '{ROOT_RETRIEVED_FOLDER}'.")
            return
    except FileNotFoundError:
        print(f"❌ Lỗi: Không thể truy cập '{ROOT_RETRIEVED_FOLDER}'. Kiểm tra lại đường dẫn.")
        return
        
    class_to_id = {name: i for i, name in enumerate(class_names)}
    print(f"🔍 Tìm thấy {len(class_names)} lớp: {class_to_id}\n")

    # Danh sách để chứa tất cả dữ liệu
    all_retrieved_data = []

    # Duyệt qua từng lớp
    for class_name, class_id in class_to_id.items():
        class_path = os.path.join(ROOT_RETRIEVED_FOLDER, class_name)
        
        # Lấy tất cả các file ảnh trong thư mục lớp
        image_files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        
        # Xáo trộn và giới hạn số lượng ảnh
        random.shuffle(image_files)
        selected_files = image_files[:MAX_IMAGES_PER_CLASS]
        
        print(f"Lớp '{class_name}': {len(image_files)} ảnh -> Lấy {len(selected_files)} ảnh.")
        
        # Thêm vào danh sách tổng với đường dẫn tương đối
        for img in selected_files:
            # Tạo đường dẫn tương đối, ví dụ: "Mega_Blueberry/image1.jpg"
            relative_path = os.path.join(class_name, img).replace('\\', '/')
            all_retrieved_data.append((relative_path, class_id))

    # Xáo trộn toàn bộ dữ liệu lần cuối
    random.shuffle(all_retrieved_data)

    # Hàm trợ giúp để ghi file
    def write_to_file(filepath, data_list):
        with open(filepath, 'w') as f:
            for path, label in data_list:
                f.write(f"{path} {label} 0\n")
        print(f"✅ Đã ghi {len(data_list)} dòng vào file '{os.path.basename(filepath)}'")

    # Ghi ra các file .txt
    output_file_1 = os.path.join(OUTPUT_FOLDER, 'T2T500.txt')
    output_file_2 = os.path.join(OUTPUT_FOLDER, 'T2T500+T2I0.25.txt')
    
    write_to_file(output_file_1, all_retrieved_data)
    write_to_file(output_file_2, all_retrieved_data)
    
    print("\n🎉 Hoàn tất! Hai file retrieval đã được tạo thành công.")

if __name__ == '__main__':
    create_retrieval_files()