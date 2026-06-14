
## solution
sudo modprobe nvidia_uvm
sudo nvidia-modprobe -u -c=0


## check

'''

ls -l /dev/nvidia*

python - <<'PY'
import torch
print("torch =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("is_available =", torch.cuda.is_available())
print("device_count =", torch.cuda.device_count())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
    print(torch.zeros((1,), device="cuda:0"))
PY


'''

