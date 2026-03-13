"""
检查 mat 文件版本，并尝试用 h5py 读取 MATLAB table 内容
"""
import sys

MAT_FILE = "data/2023-001_EOPPCC_data/MatlabDataBase/EOP_DB_Campaign.mat"

# 1. 检查文件头，判断是否为 v7.3 (HDF5)
with open(MAT_FILE, "rb") as f:
    header = f.read(128)
print("=== 文件头 (前128字节) ===")
print(header[:116].decode("latin-1"))
print()

is_hdf5 = header[:8] == b'\x89HDF\r\n\x1a\n'
print(f"是否为 HDF5 (v7.3) 格式: {is_hdf5}")
print()

# 2. 尝试 mat73
print("=== 尝试 mat73 ===")
try:
    import mat73
    data = mat73.loadmat(MAT_FILE)
    eop_db = data["EOP_DB"]
    print(f"mat73 读取成功，EOP_DB type: {type(eop_db)}")
    if isinstance(eop_db, dict):
        keys = list(eop_db.keys())
        print(f"顶层 keys (前10): {keys[:10]}")
        # 找第一个 C_ 成员
        for k in keys:
            if k.startswith("C_"):
                member = eop_db[k]
                print(f"\n--- {k} ---")
                print(f"  type: {type(member)}")
                if isinstance(member, dict):
                    sub_keys = list(member.keys())
                    print(f"  sub_keys (前10): {sub_keys[:10]}")
                    mjd_keys = [s for s in sub_keys if s.startswith("MJD_")]
                    print(f"  MJD_* keys: {mjd_keys[:5]}")
                    if mjd_keys:
                        first = member[mjd_keys[0]]
                        print(f"\n  --- {mjd_keys[0]} ---")
                        print(f"    type: {type(first)}")
                        print(f"    repr: {repr(first)[:400]}")
                        if isinstance(first, dict):
                            print(f"    keys: {list(first.keys())}")
                            for col in ["MJD", "x_pole", "y_pole"]:
                                if col in first:
                                    print(f"    {col}: {first[col][:5]}")
                break
except ImportError:
    print("mat73 未安装，运行: pip install mat73")
except Exception as e:
    print(f"mat73 读取失败: {e}")