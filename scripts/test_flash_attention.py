import torch
import torch.nn.functional as F

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))

# 2.0.1 では hasattr で存在確認するのが安全
print("has enable_flash_sdp:", hasattr(torch.backends.cuda, "enable_flash_sdp"))
print("has flash_sdp_enabled:", hasattr(torch.backends.cuda, "flash_sdp_enabled"))
print("has sdp_kernel:", hasattr(torch.backends.cuda, "sdp_kernel"))

q = torch.randn(2, 8, 1024, 64, device="cuda", dtype=torch.float16)
k = torch.randn(2, 8, 1024, 64, device="cuda", dtype=torch.float16)
v = torch.randn(2, 8, 1024, 64, device="cuda", dtype=torch.float16)

# グローバル設定を明示
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(False)

try:
    y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    torch.cuda.synchronize()
    print("success:", y.shape, y.dtype)
except Exception as e:
    print("failed:", repr(e))
