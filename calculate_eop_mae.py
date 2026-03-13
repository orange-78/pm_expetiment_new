"""
EOP PCC 极移数据误差统计脚本
计算各参与成员的 x_pole / y_pole 预测误差（MAE by day）
并输出 JSON 结果及汇总表格。

前置条件：先在 MATLAB 中运行 convert_tables_to_struct.m
生成 EOP_DB_Campaign_converted.mat，再运行本脚本。
"""

import json
import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path

from error_visualization import calculate_mae_by_step, calculate_mae_of_dataset


# ─────────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────────
MAT_FILE    = Path("data/2023-001_EOPPCC_data/MatlabDataBase/EOP_DB_Campaign_converted.mat")
REF_FILE    = Path("data/eopc04_14_IAU2000.62-250915-pm.csv")
OUTPUT_JSON = Path("data/predicts/eop_mae_results.json")

REPORT_DAYS = [10, 30, 100, 365]


# ─────────────────────────────────────────────
# 1. 载入基准数据
# ─────────────────────────────────────────────
def load_reference(ref_path: Path) -> pd.DataFrame:
    ref = pd.read_csv(ref_path)
    ref = ref[["MJD", "x_pole", "y_pole"]].sort_values("MJD").reset_index(drop=True)
    return ref


# ─────────────────────────────────────────────
# 2. 载入转换后的 .mat 文件
# ─────────────────────────────────────────────
def load_mat_struct(mat_path: Path):
    mat = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    return mat["EOP_DB"]


# ─────────────────────────────────────────────
# 3. 将 MJD_* 字段（已转为 struct）解析为 DataFrame
# ─────────────────────────────────────────────
def extract_prediction_df(struct_obj) -> pd.DataFrame | None:
    try:
        if not hasattr(struct_obj, "_fieldnames"):
            return None

        fields = struct_obj._fieldnames
        if not {"MJD", "x_pole", "y_pole"}.issubset(fields):
            return None

        mjd    = np.atleast_1d(getattr(struct_obj, "MJD")).ravel().astype(float)
        x_pole = np.atleast_1d(getattr(struct_obj, "x_pole")).ravel().astype(float)
        y_pole = np.atleast_1d(getattr(struct_obj, "y_pole")).ravel().astype(float)

        df = pd.DataFrame({"MJD": mjd, "x_pole": x_pole, "y_pole": y_pole})
        df = df.sort_values("MJD").reset_index(drop=True)
        return df

    except Exception as e:
        print(f"    [warn] extract_prediction_df 失败: {e}")
        return None


# ─────────────────────────────────────────────
# 4. 基准对齐
# ─────────────────────────────────────────────
def align_to_reference(pred_df: pd.DataFrame, ref_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
    ref_indexed = ref_df.set_index("MJD")

    valid_mask = pred_df["MJD"].isin(ref_indexed.index)
    if not valid_mask.any():
        return None

    pred_df = pred_df[valid_mask].reset_index(drop=True)
    mjds    = pred_df["MJD"].values

    actual    = ref_indexed.loc[mjds, ["x_pole", "y_pole"]].values.astype(float)
    predicted = pred_df[["x_pole", "y_pole"]].values.astype(float)
    return actual, predicted


# ─────────────────────────────────────────────
# 5. 获取成员的所有预测并裁剪对齐
# ─────────────────────────────────────────────
def get_member_predictions(member_struct) -> list[pd.DataFrame]:
    predictions = []

    if member_struct is None:
        return predictions
    if isinstance(member_struct, np.ndarray) and member_struct.size == 0:
        return predictions
    if not hasattr(member_struct, "_fieldnames"):
        return predictions

    mjd_fields = [f for f in member_struct._fieldnames if f.startswith("MJD_")]

    for field in mjd_fields:
        try:
            sub = getattr(member_struct, field)
            df  = extract_prediction_df(sub)
            if df is not None and len(df) > 0:
                predictions.append(df)
        except Exception as e:
            print(f"    [warn] 读取字段 {field} 失败: {e}")

    return predictions


def clip_to_min_length(predictions: list[pd.DataFrame]) -> list[pd.DataFrame]:
    if not predictions:
        return predictions
    min_len = min(len(df) for df in predictions)
    return [df.iloc[:min_len].reset_index(drop=True) for df in predictions]


# ─────────────────────────────────────────────
# 6. 计算单成员平均每步 MAE
# ─────────────────────────────────────────────
def compute_member_mae(member_struct, ref_df: pd.DataFrame) -> np.ndarray | None:
    predictions = get_member_predictions(member_struct)
    if not predictions:
        return None

    predictions = clip_to_min_length(predictions)

    actuals_list   = []
    predicted_list = []

    for pred_df in predictions:
        result = align_to_reference(pred_df, ref_df)
        if result is None:
            continue
        actual, predicted = result
        actuals_list.append(actual)
        predicted_list.append(predicted)

    if not actuals_list:
        return None

    min_steps     = min(a.shape[0] for a in actuals_list)
    actuals_arr   = np.stack([a[:min_steps] for a in actuals_list],  axis=0)  # (B, steps, 2)
    predicted_arr = np.stack([p[:min_steps] for p in predicted_list], axis=0)

    mae_by_step = calculate_mae_by_step(actuals_arr, predicted_arr)   # (B, steps, 2)
    mean_mae    = calculate_mae_of_dataset(mae_by_step)               # (steps, 2)
    return mean_mae


# ─────────────────────────────────────────────
# 7. 主流程
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("EOP PCC 极移预测误差统计")
    print("=" * 60)

    print(f"\n[1] 载入基准数据: {REF_FILE}")
    ref_df = load_reference(REF_FILE)
    print(f"    共 {len(ref_df)} 行，MJD [{ref_df['MJD'].min()}, {ref_df['MJD'].max()}]")

    print(f"\n[2] 载入 MAT 文件: {MAT_FILE}")
    if not MAT_FILE.exists():
        print(f"\n  [错误] 找不到文件: {MAT_FILE}")
        print("  请先在 MATLAB 中运行 convert_tables_to_struct.m 生成转换后的文件。")
        return

    eop_db = load_mat_struct(MAT_FILE)

    all_fields = eop_db._fieldnames
    member_fields = sorted(
        [f for f in all_fields if f.startswith("C_") and f != "C_200"],
        key=lambda x: int(x.split("_")[1])
    )
    if "C_200" in all_fields:
        member_fields.append("C_200")

    print(f"    共 {len(member_fields)} 个成员字段")

    print("\n[3] 逐成员计算 MAE ...")
    maes_dict: dict[str, list] = {}
    labels: list[str] = []

    for member_id in member_fields:
        print(f"  {member_id} ...", end=" ", flush=True)
        member_struct = getattr(eop_db, member_id)

        if member_struct is None or (isinstance(member_struct, np.ndarray) and member_struct.size == 0):
            print("跳过（空字段）")
            continue

        mae = compute_member_mae(member_struct, ref_df)
        if mae is None:
            print("跳过（无有效数据）")
            continue

        steps = mae.shape[0]
        day10 = min(9, steps - 1)
        print(f"steps={steps}  x@10={mae[day10,0]:.4f}  y@10={mae[day10,1]:.4f}")

        maes_dict[member_id] = mae.tolist()
        labels.append(member_id)

    print(f"\n[4] 保存 JSON: {OUTPUT_JSON}")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"maes_dict": maes_dict, "labels": labels}, f, indent=4, ensure_ascii=False)
    print(f"    已保存 {len(labels)} 个成员")

    print("\n[5] 汇总表格")
    print_summary_table(maes_dict, labels, REPORT_DAYS)

    print("\n完成！")


def print_summary_table(maes_dict: dict, labels: list, report_days: list[int]):
    col_names = ["member"] + [f"@{d}day" for d in report_days]
    rows = []

    for member_id in labels:
        mae_arr = np.array(maes_dict[member_id])
        steps   = mae_arr.shape[0]
        row     = [member_id]
        for d in report_days:
            idx = d - 1
            if idx < steps:
                row.append(f"x={mae_arr[idx,0]:.4f} y={mae_arr[idx,1]:.4f}")
            else:
                row.append("")
        rows.append(row)

    df_table = pd.DataFrame(rows, columns=col_names)
    print(df_table.to_string(index=False))

    csv_path = OUTPUT_JSON.with_suffix(".csv")
    df_table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n    表格已保存: {csv_path}")


if __name__ == "__main__":
    main()