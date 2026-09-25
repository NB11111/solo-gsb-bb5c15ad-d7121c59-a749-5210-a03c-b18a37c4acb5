"""已保存配平方案的库存配重补偿(只读, 不改变求解与保存格式)。

问题模型
--------
方案已确定: 各试管质量、槽位角度、转子半径与槽位占用状态全部只读取自方案。
现场有若干**标准配重规格**(至多 6 种), 每种有单件质量 w_j 与库存件数 s_j
(非负整数)。允许在方案的**空闲启用槽**内追加配重, 使补偿后的一阶不平衡量

    U₁' = U₁ + R · Σ w_j · e^{iθ_s}                (g·mm)

不超过给定限值 L(必填, 非负)。约束:

- 配重只放在方案中的**空闲启用槽**, 单槽至多一枚;
- 每种配重的使用件数不得超过其库存件数;
- 最多放置 M 枚(0~3);
- 仅处理至多 16 个空闲启用槽的方案(精确枚举规模上限)。

精确搜索(按枚数分层的槽位 × 质量联合完全枚举)
----------------------------------------------
设空闲启用槽 q ≤ 16 个(按槽号升序), 配重规格 t ≤ 6 种, 有效枚数上限

    M' = min(M, q, Σ_j s_j)

DFS 沿槽号升序逐槽决策(跳过本槽, 或在本槽放一枚库存仍有余量的某规格配重),
每放置一枚即把该组合记入对应枚数层。每个“槽位子集 × 逐槽质量”的联合组合
恰好出现一次, 各层叶子数上界:

    恰好 k 枚 ≤ C(q, k) · t^k
    k = 0..3 合计 ≤ 1 + 16·6 + C(16,2)·6² + C(16,3)·6³ ≈ 12.6 万

目标排序(字典序, 确定性)
------------------------
1. 能达标(补偿后 |U₁| ≤ L)的**最少枚数** k* —— 从 k=0 起逐层枚举,
   首个存在达标组合的 k 即 k*;
2. 在大小为 k* 的达标组合中取补偿后 |U₁| **最低**者;
3. 数值容差(1e-9 相对容差)内并列时, 按**槽号升序的 (槽号, 质量) 序列**
   (即 (s₁, w(s₁)), (s₂, w(s₂)), …, 槽号升序)字典序裁决,
   并统计达到该最优结果的组合数。

若 0..M' 内均无法达标, 返回全部枚举中残余 |U₁| **最低**的组合(并列时先取
枚数少者, 再按槽号/质量序列升序)及超限量, 明确判定**不达标**, 不误报达标。

拒绝条件(ValueError → HTTP 400)
-------------------------------
- 限值不是有限非负数, 或最多枚数不在 0~3;
- 配重规格少于 1 种或多于 6 种、质量非正/非有限、库存为负或非整数、
  存在重复质量;
- 方案缺少槽位状态、槽位/摆放明细自相矛盾(槽位重复、落槽不存在或被禁用、
  占用状态不一致、试管编号或槽位重复、质量非正、缺少有效角度), 或半径非法;
- 空闲启用槽超过 16 个(精确枚举搜索规模随槽位数组合增长)。
"""
from __future__ import annotations

import math
from typing import Any, Optional

# 精确枚举规模上限: 空闲启用槽 16 个、配重规格 6 种、枚数上限 3
MAX_FREE_SLOTS = 16
MAX_WEIGHT_TYPES = 6
MAX_PIECES = 3
# 与 app/evacuation.py / app/reweigh.py 一致的数值容差
_NUM_TOL = 1e-9


def plan_inventory_compensation(plan: dict[str, Any],
                                counterweights: list[dict[str, Any]],
                                u1_limit_g_mm: float,
                                max_pieces: int) -> dict[str, Any]:
    """对已保存方案精确规划库存配重补偿(只读)。

    参数
    ----
    plan: GET /api/plan/{id} 返回结构的方案(含 id/name/request/result),
        仅读取, 不修改。
    counterweights: [{"mass_g": w, "stock": s}, ...], 至多 6 种标准配重;
        质量必须为正的有限数, 库存为非负整数, 质量不得重复。
    u1_limit_g_mm: 一阶不平衡量限值(g·mm), 必填的有限非负数。
    max_pieces: 最多放置枚数, 0~3。

    返回
    ----
    含补偿前/后 |U₁|、放置清单、库存用量与达标判定(或限额内残余最低组合与
    超限量)的字典; ``feasible=False`` 时不误报达标。
    """
    plan_id = plan.get("id")

    # ---- 限值 / 枚数上限 ----
    limit = float(u1_limit_g_mm)
    if not math.isfinite(limit):
        raise ValueError("一阶不平衡量限值必须为有限数值。")
    if limit < 0.0:
        raise ValueError("一阶不平衡量限值不能为负。")
    M = int(max_pieces)
    if not (0 <= M <= MAX_PIECES):
        raise ValueError(f"最多放置枚数必须在 0~{MAX_PIECES} 之间(当前 {M})。")

    # ---- 配重库存 ----
    if not isinstance(counterweights, (list, tuple)) or not counterweights:
        raise ValueError("请至少填写一种标准配重的质量与库存件数。")
    if len(counterweights) > MAX_WEIGHT_TYPES:
        raise ValueError(
            f"标准配重最多 {MAX_WEIGHT_TYPES} 种(当前 {len(counterweights)} 种)。"
        )
    weights: list[dict[str, Any]] = []
    seen_mass: set[float] = set()
    for i, item in enumerate(counterweights):
        mass = float(item["mass_g"])
        if not math.isfinite(mass) or mass <= 0.0:
            raise ValueError(f"第 {i + 1} 种配重的单件质量必须为正数。")
        stock_raw = item["stock"]
        if not isinstance(stock_raw, int) or isinstance(stock_raw, bool):
            raise ValueError(f"第 {i + 1} 种配重的库存件数必须为非负整数。")
        stock = stock_raw
        if stock < 0:
            raise ValueError(f"第 {i + 1} 种配重的库存件数不能为负。")
        if mass in seen_mass:
            raise ValueError(f"配重质量 {mass:g} g 重复填写, 请合并为一种后重试。")
        seen_mass.add(mass)
        weights.append({"mass": mass, "stock": stock})
    total_stock = sum(w["stock"] for w in weights)

    # ---- 半径 / 转速 ----
    result = plan.get("result") or {}
    cfg = result.get("config") or {}
    radius = cfg.get("radius_mm")
    if radius is None:
        radius = (plan.get("request") or {}).get("radius_mm", 100.0)
    radius = float(radius)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("方案中的转子半径必须为正数。")
    rpm = float(cfg.get("rpm", (plan.get("request") or {}).get("rpm", 0.0)))
    omega = 2.0 * math.pi * rpm / 60.0

    # ---- 槽位几何(以方案结果 slot_state 为准) ----
    slot_state = result.get("slot_state") or []
    if not slot_state:
        raise ValueError(
            f"方案 {plan_id} 缺少槽位状态信息, 无法进行库存配重补偿。"
        )
    slot_info: dict[int, dict[str, Any]] = {}
    for item in slot_state:
        idx = int(item["slot"])
        if idx in slot_info:
            raise ValueError(f"方案的槽位状态中槽位 {idx} 重复。")
        ang = float(item["angle"])
        if not math.isfinite(ang):
            raise ValueError(f"槽位 {idx} 的角度必须为有限数值。")
        slot_info[idx] = {
            "angle": ang,
            "enabled": bool(item.get("enabled", True)),
            "tube_id": (int(item["tube_id"]) if item.get("tube_id") is not None
                        else None),
        }

    # ---- 摆放明细: 校验并独立复算补偿前 U₁ ----
    occupied: dict[int, int] = {}
    tube_seen: set[int] = set()
    base_vec = 0j
    tube_count = 0
    for p in (result.get("placements") or []):
        if p.get("slot") is None:
            continue
        tube_count += 1
        tid = int(p["tube_id"])
        if tid in tube_seen:
            raise ValueError(f"摆放明细中试管 {tid} 重复。")
        tube_seen.add(tid)
        slot = int(p["slot"])
        if slot in occupied:
            raise ValueError(f"摆放明细中槽位 {slot} 被重复占用。")
        info = slot_info.get(slot)
        if info is None:
            raise ValueError(f"试管 {tid} 的落槽 {slot} 在方案槽位状态中不存在。")
        if not info["enabled"]:
            raise ValueError(f"试管 {tid} 落在被禁用的槽位 {slot} 上, 方案数据不一致。")
        if info["tube_id"] is not None and info["tube_id"] != tid:
            raise ValueError(
                f"槽位 {slot} 的占用状态(试管 {info['tube_id']})与摆放明细"
                f"(试管 {tid})不一致。"
            )
        mass = float(p["mass"])
        if not math.isfinite(mass) or mass <= 0.0:
            raise ValueError(f"试管 {tid} 的标称质量必须为正数。")
        occupied[slot] = tid
        a = math.radians(info["angle"])
        base_vec += radius * mass * complex(math.cos(a), math.sin(a))
    # slot_state 标记占用但摆放明细缺失的槽位
    for idx, info in slot_info.items():
        if info["enabled"] and info["tube_id"] is not None and idx not in occupied:
            raise ValueError(
                f"槽位 {idx} 在槽位状态中被试管 {info['tube_id']} 占用, "
                "但摆放明细缺少该试管, 方案数据不一致。"
            )

    # ---- 空闲启用槽(配重唯一允许放置的位置) ----
    free_slots = sorted(
        idx for idx, info in slot_info.items()
        if info["enabled"] and info["tube_id"] is None
    )
    if len(free_slots) > MAX_FREE_SLOTS:
        raise ValueError(
            f"方案有 {len(free_slots)} 个空闲启用槽位, 超过库存配重补偿上限 "
            f"{MAX_FREE_SLOTS} 个(精确枚举搜索规模随槽位数组合增长), 明确拒绝。"
        )

    def _block(z: complex) -> dict[str, Any]:
        mag = abs(z)
        return {
            "u1_x_g_mm": z.real,
            "u1_y_g_mm": z.imag,
            "u1_mag_g_mm": mag,
            "u1_angle_deg": math.degrees(math.atan2(z.imag, z.real)) % 360.0,
            "centrifugal_force_n": omega ** 2 * mag * 1e-6,
        }

    baseline_block = _block(base_vec)

    # 有效枚数上限: 不超过设定值、空槽数与库存总件数
    m_eff = min(M, len(free_slots), total_stock)
    # 每槽单位方向上的质径积: R·w·e^{iθ}
    slot_dir = [
        complex(math.cos(math.radians(slot_info[s]["angle"])),
                math.sin(math.radians(slot_info[s]["angle"])))
        for s in free_slots
    ]
    weight_mass = [w["mass"] for w in weights]

    # ---- 按枚数分层的槽位 × 质量联合完全枚举 ----
    # leaves[k] = [(mag, ((slot, mass), ...)), ...]; 组合按槽号升序记录
    leaves: list[list[tuple[float, tuple[tuple[int, float], ...]]]] = [
        [] for _ in range(m_eff + 1)
    ]
    leaves[0].append((abs(base_vec), ()))
    remaining = [w["stock"] for w in weights]

    def rec(pos: int, k: int, v: complex,
            key: tuple[tuple[int, float], ...]) -> None:
        if k >= m_eff or pos >= len(free_slots):
            return
        # 本槽放一枚库存仍有余量的配重, 随后只能在更靠后的槽继续
        for j in range(len(weights)):
            if remaining[j] <= 0:
                continue
            remaining[j] -= 1
            nv = v + radius * weight_mass[j] * slot_dir[pos]
            nkey = key + ((free_slots[pos], weight_mass[j]),)
            leaves[k + 1].append((abs(nv), nkey))
            rec(pos + 1, k + 1, nv, nkey)
            remaining[j] += 1
        # 本槽留空
        rec(pos + 1, k, v, key)

    rec(0, 0, base_vec, ())
    evaluated = sum(len(layer) for layer in leaves)

    def _best_of(layer: list[tuple[float, tuple[tuple[int, float], ...]]]
                 ) -> tuple[float, tuple[tuple[int, float], ...], int]:
        """层内最优: 模长最小; 容差内并列按 (槽号, 质量) 序列升序。"""
        best_mag = min(m for m, _ in layer)
        tol = _NUM_TOL * max(1.0, best_mag)
        tied = [k for m, k in layer if m <= best_mag + tol]
        best_key = min(tied)
        mag = next(m for m, k in layer if k == best_key)
        return mag, best_key, len(set(tied))

    limit_tol = _NUM_TOL * max(1.0, limit)
    winner: Optional[dict[str, Any]] = None       # 首个达标 k 上的最优组合
    best_effort: Optional[dict[str, Any]] = None  # 全部层内残余最低组合
    for k in range(0, m_eff + 1):
        layer = leaves[k]
        if not layer:
            continue
        k_mag, k_key, k_count = _best_of(layer)
        if best_effort is None or k_mag < best_effort["mag"] - _NUM_TOL:
            best_effort = {"k": k, "mag": k_mag, "key": k_key, "count": k_count}
        feasible_layer = [(m, key) for m, key in layer
                          if m <= limit + limit_tol]
        if feasible_layer and winner is None:
            f_mag, f_key, f_count = _best_of(feasible_layer)
            winner = {"k": k, "mag": f_mag, "key": f_key, "count": f_count}

    # k=0 恒有一个叶子, 故 best_effort 必存在
    assert best_effort is not None
    chosen = winner if winner is not None else best_effort
    chosen_key: tuple[tuple[int, float], ...] = chosen["key"]

    # 由放置清单独立复算补偿后 U₁(不依赖 DFS 的浮点累加)
    final_vec = base_vec
    placement_rows: list[dict[str, Any]] = []
    used_by_mass = {w["mass"]: 0 for w in weights}
    added_mass = 0.0
    for slot, mass in chosen_key:
        a = math.radians(slot_info[slot]["angle"])
        uv = radius * mass * complex(math.cos(a), math.sin(a))
        final_vec += uv
        used_by_mass[mass] += 1
        added_mass += mass
        placement_rows.append({
            "slot": slot,
            "angle_deg": slot_info[slot]["angle"],
            "mass_g": mass,
            "unbalance_g_mm": radius * mass,
            "u1": {
                "u1_x_g_mm": uv.real,
                "u1_y_g_mm": uv.imag,
                "u1_mag_g_mm": abs(uv),
                "u1_angle_deg": slot_info[slot]["angle"] % 360.0,
            },
        })
    final_block = _block(final_vec)
    final_mag = final_block["u1_mag_g_mm"]

    inventory_rows = []
    for w in weights:
        used_count = used_by_mass[w["mass"]]
        inventory_rows.append({
            "mass_g": w["mass"],
            "stock": w["stock"],
            "used": used_count,
            "remaining": w["stock"] - used_count,
        })

    feasible = winner is not None
    within = final_mag <= limit + limit_tol
    if within:
        excess = 0.0
        margin = max(0.0, limit - final_mag)
    else:
        excess = max(0.0, final_mag - limit)
        margin = 0.0
    pieces = chosen["k"]
    ids_txt = ", ".join(f"槽 {s} 放 {m:g} g" for s, m in chosen_key)

    if feasible and pieces == 0:
        reason = (
            f"补偿前 |U₁|={baseline_block['u1_mag_g_mm']:.6g} g·mm "
            f"已不超过限值 {limit:g} g·mm, 无需放置配重(0 枚)。"
        )
    elif feasible:
        reason = (
            f"放置 {pieces} 枚配重({ids_txt})即可达标: "
            f"补偿后 |U₁|={final_mag:.6g} g·mm ≤ 限值 {limit:g} g·mm, "
            f"余量 {margin:.6g} g·mm(补偿前 "
            f"{baseline_block['u1_mag_g_mm']:.6g} g·mm); "
            "这是能达标的最少枚数。"
        )
    else:
        reason = (
            f"在最多放置 {M} 枚的限制内共精确评估 {evaluated} 种槽位 × 质量"
            f"组合, 没有使补偿后 |U₁| ≤ 限值 {limit:g} g·mm 的组合; "
            f"残余最低的组合放置 {pieces} 枚"
            f"({ids_txt or '无'}), 补偿后 |U₁|={final_mag:.6g} g·mm, "
            f"仍超出 {excess:.6g} g·mm, 判定不达标。"
        )
    if total_stock == 0:
        reason += "注意: 全部配重规格库存均为 0, 实际上无法放置任何配重。"

    return {
        "method": (
            "按枚数分层的槽位 × 质量联合完全枚举"
            f"(空闲启用槽≤{MAX_FREE_SLOTS}、配重规格≤{MAX_WEIGHT_TYPES}、"
            f"枚数上限 0~{MAX_PIECES}): 配重只放空槽、单槽至多一枚且每种用量"
            "不超库存; 先求能达标的最少枚数, 再在该枚数内取残余 |U₁| 最低"
            "的组合, 并列按槽号升序的 (槽号, 质量) 序列裁决; 限额内无法达标"
            "时给出残余最低的组合与超限量, 不误报达标"
        ),
        "feasible": feasible,
        "plan": {
            "id": plan.get("id"),
            "name": plan.get("name"),
            "created_at": plan.get("created_at"),
        },
        "radius_mm": radius,
        "rpm": rpm,
        "tube_count": tube_count,
        "occupied_slot_count": len(occupied),
        "free_enabled_slots": [
            {"slot": s, "angle_deg": slot_info[s]["angle"]} for s in free_slots
        ],
        "free_enabled_slot_count": len(free_slots),
        "counterweights": inventory_rows,
        "u1_limit_g_mm": limit,
        "max_pieces": M,
        "effective_max_pieces": m_eff,
        "evaluated_combinations": evaluated,
        "baseline_u1": baseline_block,
        "pieces_required": pieces,
        "placements": placement_rows,
        "optimal_combination_count": chosen["count"],
        "tie_broken": chosen["count"] > 1,
        "final_u1": final_block,
        "within_limit": within,
        "excess_g_mm": excess,
        "margin_g_mm": margin,
        "improvement_g_mm": baseline_block["u1_mag_g_mm"] - final_mag,
        "added_mass_g": added_mass,
        "reason": reason,
        "note": (
            "库存配重补偿为只读规划结果, 未修改任何已保存方案; "
            "配重只放在方案的空闲启用槽, 单槽至多一枚, 每种用量不超过库存。"
        ),
    }
