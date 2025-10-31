import torch
import os
import argparse
import numpy as np
import logging
import sys
from multiprocessing import Pool, set_start_method
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import traceback
# Import các hàm cần thiết từ code của bạn
from utils.models import set_model, MyLinear
from testing import load_model
from utils.datasets.dataset_utils import load_dataset, NUM_CLASSES_DICT
from utils.prompt import set_prompt
# (Không cần import parse_args ở đây nữa, vì main.py sẽ truyền args vào)

# --- Logger cho từng tiến trình con ---
def setup_process_logger(process_id):
    logger = logging.getLogger(f"Mapper-{process_id}")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(f'[Mapper {process_id}] %(message)s'))
    if not logger.hasHandlers():
        logger.addHandler(handler)
    return logger

# --- Dataset đơn giản để đọc ảnh từ danh sách ---
class SimpleImageDataset(Dataset):
    def __init__(self, file_list, preprocess):
        self.paths = [item[0] for item in file_list]
        self.labels = [item[1] for item in file_list]
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image_path = self.paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.preprocess(image)
        except Exception as e:
            # Ghi lại lỗi nhưng không crash
            print(f"[PID {os.getpid()}] Lỗi khi tải ảnh {image_path}: {e}")
            return None # Sẽ được lọc ra sau

        return image_tensor, label, image_path

# Hàm lọc ra các item bị lỗi (None)
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None, None, None # Trả về None nếu cả batch lỗi
    return torch.utils.data.dataloader.default_collate(batch)

# --- GIAI ĐOẠN MAP ---
def map_task(process_id, file_chunk, model_path, cli_args, temp_dir):
    """
    Đây là hàm MAPPER. Chạy trên một tiến trình CPU riêng.
    """
    logger = setup_process_logger(process_id)
    logger.info(f"Bắt đầu, xử lý {len(file_chunk)} ảnh.")

    # ----- BỌC TOÀN BỘ LOGIC XỬ LÝ VÀO TRY...EXCEPT -----
    try:
        # 1. Thiết lập thiết bị (Device)
        device = "cpu"
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            device = f'cuda:{process_id % num_gpus}'
            logger.info(f"Sẽ dùng GPU: {device} (trên tổng số {num_gpus} GPUs)")
        else:
            logger.info("Sẽ dùng CPU.")

        # 2. Tải model config
        # Tạm thời gán device vào cli_args để set_model sử dụng
        cli_args.device = device
        model, preprocess, tokenizer = set_model(cli_args, logger)

        # Tải classifier head
        num_classes = NUM_CLASSES_DICT[cli_args.dataset]
        classifier_head = MyLinear(inp_dim=512, num_classes=num_classes)

        # Tải trọng số (weights) của mô hình "ngon"
        cli_args.model_path = model_path
        class DummyLogger:
            def info(self, *args, **kwargs): pass
            def warning(self, *args, **kwargs): pass
            def error(self, *args, **kwargs): pass
        simple_logger = DummyLogger()
        load_model(cli_args, simple_logger, model=model, classifier_head=classifier_head)

        model.to(device)
        model.eval()

        # 3. Tạo DataLoader cho phần (chunk) dữ liệu này
        dataset = SimpleImageDataset(file_chunk, preprocess)
        dataloader = DataLoader(dataset, batch_size=cli_args.bsz, shuffle=False,
                                num_workers=0, collate_fn=collate_fn,
                                persistent_workers=False)

        # 4. Chạy trích xuất đặc trưng
        all_img_feats = []
        all_labels = []

        with torch.no_grad():
            for batch_data in tqdm(dataloader, desc=f"Mapper {process_id}", position=process_id):
                if batch_data is None or batch_data[0] is None:
                    logger.warning(f"Bỏ qua batch rỗng hoặc lỗi.")
                    continue
                inputs, labels, paths = batch_data

                inputs = inputs.to(device)
                img_feats = model.encode_image(inputs)
                img_feats /= img_feats.norm(dim=-1, keepdim=True)

                all_img_feats.append(img_feats.cpu())

                # Xử lý labels (đã sửa ở lần trước, giữ nguyên)
                labels_tensor = None
                if isinstance(labels, (list, tuple)):
                    try:
                        labels_tensor = torch.tensor([int(x) for x in labels], dtype=torch.long)
                    except Exception as label_e:
                        logger.warning(f"Lỗi chuyển đổi nhãn list/tuple: {labels} -> {label_e}. Ảnh: {paths}")
                        continue # Bỏ qua cả batch nếu nhãn lỗi
                elif isinstance(labels, (int, str)):
                     try:
                        labels_tensor = torch.tensor([int(labels)], dtype=torch.long)
                     except Exception as label_e:
                        logger.warning(f"Lỗi chuyển đổi nhãn int/str: {labels} -> {label_e}. Ảnh: {paths}")
                        continue # Bỏ qua cả batch nếu nhãn lỗi
                elif isinstance(labels, torch.Tensor):
                     labels_tensor = labels.long()
                else:
                     logger.warning(f"Kiểu dữ liệu nhãn không xác định: {type(labels)}. Ảnh: {paths}")
                     continue # Bỏ qua cả batch nếu nhãn lỗi

                # Chỉ append nếu label hợp lệ
                if labels_tensor is not None:
                     all_labels.append(labels_tensor.cpu())
                else:
                     # Nếu label lỗi, cần xóa feature tương ứng đã thêm
                     if all_img_feats: all_img_feats.pop()


        # 5. Lưu kết quả tạm
        if not all_img_feats or not all_labels:
            logger.info("Không có đặc trưng hoặc nhãn hợp lệ nào được trích xuất.")
            return None

        # Kiểm tra lại số lượng trước khi concat
        if len(all_img_feats) != len(all_labels):
            logger.error(f"Số lượng features ({len(all_img_feats)}) và labels ({len(all_labels)}) không khớp sau khi xử lý!")
            # Cố gắng sửa lỗi bằng cách lấy min_len (có thể mất mát)
            min_len = min(len(all_img_feats), len(all_labels))
            if min_len == 0:
                 logger.error("Không còn dữ liệu nào khớp. Worker thất bại.")
                 return None
            logger.warning(f"Chỉ giữ lại {min_len} cặp feature/label đầu tiên.")
            all_img_feats = all_img_feats[:min_len]
            all_labels = all_labels[:min_len]

        all_features = torch.cat(all_img_feats, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        temp_file_path = os.path.join(temp_dir, f'temp_features_{process_id}.pth')
        torch.save({'image_features': all_features, 'labels': all_labels}, temp_file_path)

        logger.info(f"Hoàn thành. Đã lưu vào {temp_file_path}")
        return temp_file_path

    except Exception as e: # <--- BẮT LỖI TẠI ĐÂY
        # === PHẦN HIỂN THỊ LỖI ===
        logger.error(f"!!! LỖI NGHIÊM TRỌNG trong worker {process_id}: {e}")
        logger.error(traceback.format_exc()) # In chi tiết lỗi (dòng nào, file nào)
        # =========================
        return None # Báo lỗi về tiến trình chính
    # ----- KẾT THÚC TRY...EXCEPT -----


# --- HÀM ĐIỀU KHIỂN MAPREDUCE (SẼ ĐƯỢC GỌI TỪ MAIN.PY) ---
def run_mapreduce_extraction(model_path, target_file_path, num_processes, cli_args, output_file_path):
    """
    Hàm này điều khiển toàn bộ logic MapReduce.
    Nó được gọi từ main.py.
    """
    print(f"\n--- BẮT ĐẦU TRÍCH XUẤT MAPREDUCE (cho file: {target_file_path}) ---")
    cli_args.logger = None
    cli_args.loss_logger = None
    
    # 1. Thiết lập thư mục tạm
    # Tạo tên thư mục tạm duy nhất dựa trên file output
    temp_dir_name = f"temp_features_{os.path.basename(output_file_path)}"
    TEMP_DIR = os.path.join(cli_args.output_dir, temp_dir_name)
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # 2. Đọc và chia Input
    print(f"Đang đọc danh sách ảnh từ: {target_file_path}...")
    all_files_list = []
    
# ----- BẮT ĐẦU SỬA LỖI ĐỌC FILE -----
    print(f"Đang đọc danh sách ảnh từ: {target_file_path}...")
    all_files_list = []
    
    with open(target_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.strip().split()
            if len(parts) < 2:
                print(f"Bỏ qua dòng lỗi: {line}")
                continue

            path = os.path.normpath(parts[0].replace('\\', '/'))
            if not os.path.exists(path):
                print(f"[Cảnh báo] Không tìm thấy file ảnh: {path}")
                continue

            # ép nhãn về số nguyên, bỏ qua phần cờ phụ (cột 3)
            try:
                label = int(parts[1])
            except ValueError:
                print(f"[Lỗi] Nhãn không hợp lệ tại dòng: {line}")
                continue

            all_files_list.append((path, label))


    # Phần còn lại của hàm giữ nguyên...
    print(f"Tổng cộng {len(all_files_list)} ảnh. Chia cho {num_processes} tiến trình.")
    # ...
            
    print(f"Tổng cộng {len(all_files_list)} ảnh. Chia cho {num_processes} tiến trình.")
    
    # Chia danh sách thành N phần (chunks)
    file_chunks = np.array_split(all_files_list, num_processes)
    
    # 3. Chuẩn bị tham số cho các Mapper
    map_tasks_args = []
    for i in range(num_processes):
        if len(file_chunks[i]) > 0: # Chỉ thêm task nếu có dữ liệu
            map_tasks_args.append((i, file_chunks[i], model_path, cli_args, TEMP_DIR))
        else:
            print(f"Tiến trình {i} không có dữ liệu, bỏ qua.")

    # 4. Chạy Giai đoạn MAP
    print(f"--- [MAP] Bắt đầu (với {len(map_tasks_args)} workers) ---")
    with Pool(processes=num_processes) as pool:
        temp_file_paths = pool.starmap(map_task, map_tasks_args)
    
    temp_file_paths = [p for p in temp_file_paths if p is not None]

    # 5. Chạy Giai đoạn REDUCE
    print(f"\n--- [REDUCE] Bắt đầu tổng hợp ---")
    print(f"Đang tổng hợp từ {len(temp_file_paths)} file tạm...")
    
    final_features_list = []
    final_labels_list = []
    
    for temp_file in temp_file_paths:
        try:
            data = torch.load(temp_file)
            final_features_list.append(data['image_features'])
            final_labels_list.append(data['labels'])
            os.remove(temp_file) # Xóa file tạm
        except Exception as e:
            print(f"Lỗi khi đọc file tạm {temp_file}: {e}")
            
    if os.path.exists(TEMP_DIR):
        try:
            os.rmdir(TEMP_DIR) # Xóa thư mục tạm
            print(f"Đã dọn dẹp thư mục tạm: {TEMP_DIR}")
        except OSError as e:
            print(f"Không thể xóa thư mục tạm {TEMP_DIR} (có thể nó không rỗng): {e}")
    
    if not final_features_list:
        print("Không có đặc trưng nào được tổng hợp. Dừng lại.")
        # Trả về False để báo lỗi
        return False
        
    # Nối tất cả lại
    final_features = torch.cat(final_features_list, dim=0)
    final_labels = torch.cat(final_labels_list, dim=0)
    
    print(f"Tổng hợp hoàn tất. Tổng số đặc trưng: {final_features.shape}")
    
    # 6. Lưu kết quả cuối cùng
    torch.save({'image_features': final_features, 'labels': final_labels}, output_file_path)
    print(f"--- HOÀN THÀNH MAPREDUCE ---")
    print(f"Đã lưu kết quả vào: {output_file_path}")
    
    return True # Trả về True để báo thành công