from PIL import Image
import os

# 📁 Các thư mục chứa ảnh
folders = [
    r"C:\Dataset\Mubaohiem\train\Co_doi_mu_bao_hiem",
    r"C:\Dataset\Mubaohiem\train\Khong_doi_mu_bao_hiem",
    r"C:\Dataset\Mubaohiem\test\Co_doi_mu_bao_hiem",
    r"C:\Dataset\Mubaohiem\test\Khong_doi_mu_bao_hiem",
]

# ⚙️ Giới hạn kích thước tối đa (đủ dùng cho model ViT-B/32)
MAX_SIZE = (512,512)

# 🚀 Duyệt và xử lý ảnh
for folder in folders:
    for file in os.listdir(folder):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
            path = os.path.join(folder, file)
            try:
                with Image.open(path) as img:
                    width, height = img.size
                    if width > MAX_SIZE[0] or height > MAX_SIZE[1]:
                        print(f"🔧 Đang resize: {path} ({width}x{height})")
                        # Thu nhỏ theo tỉ lệ, không méo hình
                        img.thumbnail(MAX_SIZE, Image.Resampling.LANCZOS)
                        # Ghi đè chính file cũ (giữ nguyên tên)
                        img.save(path)
                    else:
                        print(f"✅ Ảnh nhỏ sẵn rồi, bỏ qua: {path} ({width}x{height})")
            except Exception as e:
                print(f"⚠️ Lỗi khi xử lý {path}: {e}")

print("🎉 Hoàn thành! Tất cả ảnh lớn đã được thu nhỏ, tên file giữ nguyên.")
