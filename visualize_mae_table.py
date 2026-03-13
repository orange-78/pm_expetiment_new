"""
读取 data/predicts/eop_mae_results.csv，输出格式化表格：
  1. 原始 MAE 表格（x/y 分量分列展示）
  2. 各时间节点 xy 几何平均值表格
并将所有输出保存至 data/predicts/eop_mae_results.txt
"""

import math
import re
import sys
from pathlib import Path

import pandas as pd

INPUT_CSV  = Path("data/predicts/eop_mae_results.csv")
OUTPUT_TXT = Path("data/predicts/eop_mae_results.txt")


# ─────────────────────────────────────────────
# 工具：解析单元格 "x=0.1234 y=0.5678" → (float, float) 或 (None, None)
# ─────────────────────────────────────────────
def parse_cell(cell: str) -> tuple[float | None, float | None]:
    if pd.isna(cell) or str(cell).strip() == "":
        return None, None
    m = re.search(r"x=([\d.eE+\-]+)\s+y=([\d.eE+\-]+)", str(cell))
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


# ─────────────────────────────────────────────
# 格式化表格生成（纯文本，固定宽度列）
# ─────────────────────────────────────────────
def fmt(val: float | None, width: int = 9) -> str:
    if val is None:
        return "-".center(width)
    return f"{val:.4f}".rjust(width)


def build_table(headers: list[str], rows: list[list[str]], col_widths: list[int]) -> str:
    def hline(widths, left="+-", mid="-+-", right="-+"):
        return left + mid.join("-" * w for w in widths) + right

    def row_line(cells, widths, sep=" | "):
        padded = [str(c).ljust(w) if i == 0 else str(c).center(w)
                  for i, (c, w) in enumerate(zip(cells, widths))]
        return "| " + sep.join(padded) + " |"

    lines = []
    lines.append(hline(col_widths))
    lines.append(row_line(headers, col_widths))
    lines.append(hline(col_widths, left="+=", mid="=+=", right="=+"))
    for r in rows:
        lines.append(row_line(r, col_widths))
    lines.append(hline(col_widths))
    return "\n".join(lines)


def main():
    if not INPUT_CSV.exists():
        print(f"[错误] 找不到文件: {INPUT_CSV}")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    day_cols = [c for c in df.columns if c.startswith("@")]  # e.g. @10day

    # ── 解析所有数值 ──────────────────────────────
    # parsed[member][col] = (x, y) or (None, None)
    parsed: dict[str, dict[str, tuple]] = {}
    for _, row in df.iterrows():
        member = row["member"]
        parsed[member] = {}
        for col in day_cols:
            parsed[member][col] = parse_cell(row[col])

    members = df["member"].tolist()

    # ════════════════════════════════════════════
    # 表格 1：原始 MAE（x / y 分量各占一列）
    # ════════════════════════════════════════════
    # 构造表头：member | @10day x | @10day y | @30day x | ...
    h1 = ["member"]
    for col in day_cols:
        label = col.replace("@", "").replace("day", "d")
        h1 += [f"{label} x", f"{label} y"]

    rows1 = []
    for member in members:
        row = [member]
        for col in day_cols:
            x, y = parsed[member][col]
            row += [fmt(x), fmt(y)]
        rows1.append(row)

    # 计算列宽
    w_member = max(len("member"), max(len(m) for m in members))
    w_val    = 9
    cw1 = [w_member] + [w_val] * (len(day_cols) * 2)
    # 列头宽度至少要容纳标题
    for i, h in enumerate(h1):
        cw1[i] = max(cw1[i], len(h))

    table1 = build_table(h1, rows1, cw1)

    # ════════════════════════════════════════════
    # 表格 2：xy 几何平均值  sqrt(x * y)
    # ════════════════════════════════════════════
    h2 = ["member"] + day_cols

    rows2 = []
    for member in members:
        row = [member]
        for col in day_cols:
            x, y = parsed[member][col]
            if x is not None and y is not None and x >= 0 and y >= 0:
                geo = math.sqrt(x* x + y * y)
                row.append(fmt(geo))
            else:
                row.append(fmt(None))
        rows2.append(row)

    w_col2 = [max(len(c), w_val) for c in day_cols]
    cw2    = [w_member] + w_col2
    cw2[0] = max(cw2[0], len("member"))

    table2 = build_table(h2, rows2, cw2)

    # ════════════════════════════════════════════
    # 组合输出
    # ════════════════════════════════════════════
    output_lines = [
        "EOP PCC 极移预测误差统计结果",
        "=" * 60,
        "",
        "【表格 1】各成员每步 MAE（x_pole / y_pole 分量，单位：arcsec）",
        "",
        table1,
        "",
        "【表格 2】各成员每步 MAE 几何平均值 sqrt(x * y)（单位：arcsec）",
        "",
        table2,
        "",
    ]
    output = "\n".join(output_lines)

    # 打印到终端
    print(output)

    # 保存到文件
    OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_TXT.write_text(output, encoding="utf-8")
    print(f"已保存至: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()