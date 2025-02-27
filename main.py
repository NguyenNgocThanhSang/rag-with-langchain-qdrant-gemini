from underthesea import word_tokenize
from rich import print

text = "Điều 42. Quỹ đất dành cho kết cấu hạ tầng giao thông đường bộ 1. Quỹ đất dành cho kết cấu hạ tầng giao thông đường bộ được xác định tại quy hoạch kết cấu hạ tầng giao thông đường bộ. Ủy ban nhân dân cấp tỉnh xác định và quản lý quỹ đất dành cho dự án xây dựng kết cấu hạ tầng giao thông đường bộ theo quy hoạch đã được phê duyệt. 2. Tỷ lệ quỹ đất giao thông đô thị so với đất xây dựng đô thị phải bảo đảm từ 16% đến 26%. Chính phủ quy định cụ thể tỷ lệ quỹ đất phù hợp với loại đô thị."

tokens = word_tokenize(text)
print(tokens)