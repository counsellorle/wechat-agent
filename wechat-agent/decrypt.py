import os
import ctypes
import ctypes.wintypes
import subprocess
import sys
import struct

from pywxdump.wx_core import batch_decrypt

kernel32 = ctypes.windll.kernel32
psapi = ctypes.windll.psapi

PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010
TH32CS_SNAPMODULE = 0x00000008
TH32CS_SNAPMODULE32 = 0x00000010

class MODULEINFO(ctypes.Structure):
    _fields_ = [
        ("lpBaseOfDll", ctypes.c_void_p),
        ("SizeOfImage", ctypes.c_uint32),
        ("EntryPoint", ctypes.c_void_p),
    ]

def get_module_base_address(pid, module_name):
    h_process = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
    if not h_process:
        return None

    h_module = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, pid)
    if h_module == -1:
        kernel32.CloseHandle(h_process)
        return None

    from ctypes import wintypes

    class MODULEENTRY32(ctypes.Structure):
        _fields_ = [
            ("dwSize", ctypes.c_uint32),
            ("th32ModuleID", ctypes.c_uint32),
            ("th32ProcessID", ctypes.c_uint32),
            ("GlblcntUsage", ctypes.c_uint32),
            ("ProccntUsage", ctypes.c_uint32),
            ("modBaseAddr", ctypes.POINTER(ctypes.c_byte)),
            ("modBaseSize", ctypes.c_uint32),
            ("hModule", ctypes.c_void_p),
            ("szModule", ctypes.c_char * 256),
            ("szExePath", ctypes.c_char * 260),
        ]

    me = MODULEENTRY32()
    me.dwSize = ctypes.sizeof(MODULEENTRY32)

    base_addr = None
    if kernel32.Module32First(h_module, ctypes.byref(me)):
        while True:
            if module_name.lower() in me.szModule.decode('utf-8', errors='ignore').lower():
                base_addr = ctypes.cast(me.modBaseAddr, ctypes.c_void_p).value
                print(f"  找到模块: {me.szModule.decode()}, 基址: {hex(base_addr)}")
                break
            if not kernel32.Module32Next(h_module, ctypes.byref(me)):
                break

    kernel32.CloseHandle(h_module)
    kernel32.CloseHandle(h_process)
    return base_addr

# 获取微信进程PID
result = subprocess.run(["tasklist", "/FI", "IMAGENAME eq WeChat.exe", "/NH"], capture_output=True, text=True)
lines = [l.strip() for l in result.stdout.strip().split("\n") if "WeChat.exe" in l]
if not lines:
    print("微信未运行，请先登录微信")
    sys.exit(1)

pid = int(lines[0].split()[1])
print(f"微信进程PID: {pid}")

# 获取 WeChatWin.dll 基址
print("正在查找 WeChatWin.dll 基址...")
base_addr = get_module_base_address(pid, "WeChatWin.dll")
if not base_addr:
    print("未找到 WeChatWin.dll，请确保微信已登录")
    sys.exit(1)

# 偏移地址
offset = 65204856  # 0x3E5A9B8
target_addr = base_addr + offset
print(f"目标地址: {hex(base_addr)} + {hex(offset)} = {hex(target_addr)}")

# 打开进程读取内存
h_process = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
if not h_process:
    print("无法打开进程，请以管理员身份运行")
    sys.exit(1)

# 读取指针
addr_buf = ctypes.create_string_buffer(8)
if kernel32.ReadProcessMemory(h_process, ctypes.c_void_p(target_addr), addr_buf, 8, None):
    key_ptr = struct.unpack('<Q', addr_buf)[0]
    print(f"密钥指针地址: {hex(key_ptr)}")

    # 读取32字节密钥
    key_buf = ctypes.create_string_buffer(32)
    if kernel32.ReadProcessMemory(h_process, ctypes.c_void_p(key_ptr), key_buf, 32, None):
        key_hex = bytes(key_buf).hex()
        print(f"密钥: {key_hex}")

        # 解密数据库
        wx_dir = r"D:\微信\WeChat Files\wxid_ugwdnifw5flm22"
        msg_dir = os.path.join(wx_dir, "Msg")
        out_dir = r"F:\Ai\wechat-agent\data\decrypted"
        os.makedirs(out_dir, exist_ok=True)
        print(f"数据库目录: {msg_dir}")
        print("开始解密数据库...")
        batch_decrypt(key_hex, msg_dir, out_dir)
        print(f"解密完成！输出目录: {out_dir}")
    else:
        print("无法读取密钥内存")
else:
    print("无法读取指针地址")

kernel32.CloseHandle(h_process)
