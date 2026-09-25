"""库存配重补偿测试: python -m unittest discover -s tests

核心断言: 按枚数分层的槽位 × 质量联合完全枚举结果(最少达标枚数、补偿后
|U₁|、放置 (槽号, 质量) 序列与最优组合条数)必须与独立的全枚举参考实现
一致; 配重只放空闲启用槽、单槽至多一枚、每种用量不超库存; 限额内无达标
组合时返回残余最低组合与超限量, 不误报达标; 空闲启用槽超过 16 个、配重
规格超过 6 种、重复质量、非整数库存或枚数 0~3 之外必须拒绝; 规划过程不
修改入参(只读)。
"""
import copy
import itertools
import math
import random
import unittest

from app.compensation import plan_inventory_compensation


def _plan(pid, placements, radius=100.0, slots_n=12, enabled=None,
          rpm=4000.0, name=None, slot_state=None):
    """placements: {槽号: (试管编号, 质量, 是否锁定)}; 角度按槽号均布。"""
    if enabled is None:
        enabled = set(range(slots_n))
    if slot_state is None:
        slot_state = [
            {
                "slot": i,
                "angle": 360.0 * i / slots_n,
                "enabled": i in enabled,
                "tube_id": next(
                    (t for s, (t, m, lk) in placements.items() if s == i), None
                ),
                "locked": next(
                    (lk for s, (t, m, lk) in placements.items() if s == i), False
                ),
            }
            for i in range(slots_n)
        ]
    pls = [
        {
            "tube_id": t,
            "mass": m,
            "slot": s,
            "angle": next(x["angle"] for x in slot_state if x["slot"] == s),
            "locked": lk,
        }
        for s, (t, m, lk) in placements.items()
    ]
    return {
        "id": pid,
        "name": name or f"方案{pid}",
        "created_at": "2026-09-25T00:00:00+00:00",
        "request": {"radius_mm": radius, "rpm": rpm},
        "result": {
            "config": {"radius_mm": radius, "rpm": rpm},
            "slot_state": slot_state,
            "placements": pls,
            "metrics": {"u1_mag_g_mm": 0.0},
        },
    }


def _brute(plan, counterweights, limit, max_pieces):
    """独立全枚举参考(itertools)。

    返回 dict: feasible / pieces / mag / key / count / evaluated /
    effective_max_pieces。
    """
    res = plan["result"]
    radius = res["config"]["radius_mm"]
    info = {s["slot"]: s for s in res["slot_state"]}
    occupied = {p["slot"] for p in res["placements"] if p["slot"] is not None}
    free = sorted(s for s, inf in info.items()
                  if inf["enabled"] and s not in occupied)
    masses = [float(c["mass_g"]) for c in counterweights]
    stock = [int(c["stock"]) for c in counterweights]

    def vmass(m, slot):
        a = math.radians(info[slot]["angle"])
        return radius * m * complex(math.cos(a), math.sin(a))

    base = sum(
        (vmass(float(p["mass"]), p["slot"])
         for p in res["placements"] if p["slot"] is not None),
        0j,
    )
    base_mag = abs(base)
    m_eff = min(max_pieces, len(free), sum(stock))
    layers = {0: [(base_mag, ())]}
    evaluated = 1
    for k in range(1, m_eff + 1):
        leaves = []
        for subset in itertools.combinations(free, k):
            for types in itertools.product(range(len(masses)), repeat=k):
                usage = [0] * len(masses)
                for t in types:
                    usage[t] += 1
                if any(usage[j] > stock[j] for j in range(len(masses))):
                    continue
                u = base
                key_items = []
                for slot, t in zip(subset, types):
                    u += vmass(masses[t], slot)
                    key_items.append((slot, masses[t]))
                key = tuple(sorted(key_items))
                leaves.append((abs(u), key))
        layers[k] = leaves
        evaluated += len(leaves)

    tol = 1e-9
    limit_tol = tol * max(1.0, limit)

    def best_of(leaves):
        bm = min(m for m, _ in leaves)
        tt = tol * max(1.0, bm)
        keys = [k for m, k in leaves if m <= bm + tt]
        key = min(keys)
        mag = next(m for m, k in leaves if k == key)
        return mag, key, len(set(keys))

    winner = None
    effort = None
    for k in range(0, m_eff + 1):
        leaves = layers[k]
        if not leaves:
            continue
        km, kk, kc = best_of(leaves)
        if effort is None or km < effort[2] - tol:
            effort = (k, kk, km, kc)
        feas = [(m, key) for m, key in leaves if m <= limit + limit_tol]
        if feas and winner is None:
            fm, fk, fc = best_of(feas)
            winner = (k, fk, fm, fc)
    chosen = winner if winner is not None else effort
    return {
        "feasible": winner is not None,
        "pieces": chosen[0],
        "key": chosen[1],
        "mag": chosen[2],
        "count": chosen[3],
        "evaluated": evaluated,
        "effective_max_pieces": m_eff,
    }


class StockCompensationTests(unittest.TestCase):
    def assertMatchBrute(self, plan, counterweights, limit, max_pieces):
        out = plan_inventory_compensation(
            copy.deepcopy(plan), counterweights, limit, max_pieces
        )
        ref = _brute(plan, counterweights, limit, max_pieces)
        self.assertIs(out["feasible"], ref["feasible"])
        self.assertEqual(out["pieces_required"], ref["pieces"])
        self.assertEqual(
            tuple((r["slot"], float(r["mass_g"])) for r in out["placements"]),
            ref["key"],
        )
        self.assertAlmostEqual(
            out["final_u1"]["u1_mag_g_mm"], ref["mag"],
            delta=1e-6 * max(1.0, ref["mag"]),
        )
        self.assertEqual(out["optimal_combination_count"], ref["count"])
        self.assertEqual(out["evaluated_combinations"], ref["evaluated"])
        self.assertEqual(out["effective_max_pieces"], ref["effective_max_pieces"])
        # 放置清单: 空槽、单槽一枚、库存约束; 且独立复算残余一致
        info = {s["slot"]: s for s in plan["result"]["slot_state"]}
        occupied = {p["slot"] for p in plan["result"]["placements"]
                    if p["slot"] is not None}
        slots_used = [r["slot"] for r in out["placements"]]
        self.assertEqual(len(slots_used), len(set(slots_used)))
        for s in slots_used:
            self.assertNotIn(s, occupied)
            self.assertTrue(info[s]["enabled"])
        for row in out["counterweights"]:
            self.assertLessEqual(row["used"], row["stock"])
            self.assertEqual(row["remaining"], row["stock"] - row["used"])
        return out

    def test_exact_single_counterweight(self):
        """单管 50g@槽0: 槽 6 放 50g 完全对消。"""
        plan = _plan(1, {0: (1, 50.0, False)})
        out = self.assertMatchBrute(
            plan, [{"mass_g": 50.0, "stock": 3}], 0.0, 3
        )
        self.assertTrue(out["feasible"])
        self.assertEqual(out["pieces_required"], 1)
        self.assertEqual(
            [(r["slot"], r["mass_g"]) for r in out["placements"]],
            [(6, 50.0)],
        )
        self.assertLess(out["final_u1"]["u1_mag_g_mm"], 1e-7)
        self.assertAlmostEqual(out["baseline_u1"]["u1_mag_g_mm"], 5000.0)
        self.assertEqual(out["within_limit"], True)
        self.assertEqual(out["excess_g_mm"], 0.0)
        self.assertEqual(out["counterweights"][0]["used"], 1)
        self.assertEqual(out["counterweights"][0]["remaining"], 2)
        # 独立复算离心力 F = ω²|U|×1e-6 (R=100, rpm=4000)
        omega = 2.0 * math.pi * 4000.0 / 60.0
        self.assertAlmostEqual(
            out["final_u1"]["centrifugal_force_n"],
            omega ** 2 * out["final_u1"]["u1_mag_g_mm"] * 1e-6,
        )

    def test_already_balanced_zero_pieces(self):
        """补偿前已达标时必须返回 0 枚, 不放置任何配重。"""
        plan = _plan(2, {0: (1, 50.0, True), 6: (2, 50.0, False)})
        self.assertLess(
            plan_inventory_compensation(
                plan, [{"mass_g": 50.0, "stock": 2}], 0.0, 3
            )["final_u1"]["u1_mag_g_mm"],
            1e-7,
        )
        out = self.assertMatchBrute(
            plan, [{"mass_g": 50.0, "stock": 2}], 1000.0, 3
        )
        self.assertTrue(out["feasible"])
        self.assertEqual(out["pieces_required"], 0)
        self.assertEqual(out["placements"], [])
        self.assertEqual(out["added_mass_g"], 0.0)
        self.assertAlmostEqual(
            out["final_u1"]["u1_mag_g_mm"],
            out["baseline_u1"]["u1_mag_g_mm"],
        )

    def test_max_pieces_zero(self):
        """最多 0 枚时只评估空组合; 基线超限则不达标并给出超限量。"""
        plan = _plan(3, {0: (1, 50.0, False)})
        out = plan_inventory_compensation(
            plan, [{"mass_g": 50.0, "stock": 3}], 1000.0, 0
        )
        self.assertFalse(out["feasible"])
        self.assertFalse(out["within_limit"])
        self.assertEqual(out["pieces_required"], 0)
        self.assertEqual(out["evaluated_combinations"], 1)
        self.assertAlmostEqual(out["excess_g_mm"], 4000.0)
        self.assertIn("不达标", out["reason"])

    def test_tie_break_slot_and_mass_sequence(self):
        """单管 50g@槽0、槽6禁用: 25g 配重在对径邻槽 5/7(±150°)残余相同。

        限值取 3200 g·mm: 不放(5000)超限, 一枚 25g 在槽 5 或槽 7 的残余
        ≈3098 均达标, 枚数层内并列, 按槽号升序取槽 5。
        """
        enabled = set(range(12)) - {6}
        plan = _plan(4, {0: (1, 50.0, True)}, enabled=enabled)
        out = self.assertMatchBrute(
            plan, [{"mass_g": 25.0, "stock": 1}], 3200.0, 1
        )
        self.assertTrue(out["feasible"])
        self.assertEqual(out["pieces_required"], 1)
        self.assertEqual(out["placements"][0]["slot"], 5)
        self.assertEqual(out["placements"][0]["mass_g"], 25.0)
        self.assertEqual(out["optimal_combination_count"], 2)
        self.assertTrue(out["tie_broken"])
        # 独立复算残余: |50 + 25·e^{i150°}|·R
        expect = 100.0 * math.hypot(50 + 25 * math.cos(math.radians(150)),
                                    25 * math.sin(math.radians(150)))
        self.assertAlmostEqual(out["final_u1"]["u1_mag_g_mm"], expect, places=6)

    def test_tie_break_mass_sequence(self):
        """两枚不同质量互换槽位残余相同时, 按 (槽号, 质量) 序列升序裁决。

        6 槽转子槽 3(对径位)禁用, 50g 管锁定在槽 0; 10g、20g 各 1 枚。
        两枚放在关于槽 0 对称的槽 2/4(120°/240°)时, 交换质量的两种放置
        残余完全相同(≈3605.5), 取槽号升序序列 ((2,10),(4,20))。
        限值 4000 下 k=0(5000)与 k=1(最佳≈4359)均超限, k=2 才达标。
        """
        enabled = set(range(6)) - {3}
        plan = _plan(5, {0: (1, 50.0, True)}, slots_n=6, enabled=enabled)
        cw = [{"mass_g": 10.0, "stock": 1}, {"mass_g": 20.0, "stock": 1}]
        out = self.assertMatchBrute(plan, cw, 4000.0, 2)
        self.assertTrue(out["feasible"])
        self.assertEqual(out["pieces_required"], 2)
        self.assertEqual(
            [(r["slot"], r["mass_g"]) for r in out["placements"]],
            [(2, 10.0), (4, 20.0)],
        )
        self.assertEqual(out["optimal_combination_count"], 2)
        self.assertTrue(out["tie_broken"])
        # 交换质量的另一放置残余相同
        x = 50 + 10 * math.cos(math.radians(120)) + 20 * math.cos(math.radians(240))
        y = 10 * math.sin(math.radians(120)) + 20 * math.sin(math.radians(240))
        self.assertAlmostEqual(out["final_u1"]["u1_mag_g_mm"],
                               100.0 * math.hypot(x, y), places=6)

    def test_infeasible_returns_best_effort_and_excess(self):
        """只有 49g 配重时对 50g 单管无法完全对消: 残余 100 g·mm 并报超限。"""
        plan = _plan(7, {0: (1, 50.0, False)})
        out = self.assertMatchBrute(
            plan, [{"mass_g": 49.0, "stock": 1}], 0.0, 1
        )
        self.assertFalse(out["feasible"])
        self.assertFalse(out["within_limit"])
        self.assertEqual(out["pieces_required"], 1)
        self.assertEqual(out["placements"][0]["slot"], 6)
        self.assertAlmostEqual(out["final_u1"]["u1_mag_g_mm"], 100.0)
        self.assertAlmostEqual(out["excess_g_mm"], 100.0)
        self.assertEqual(out["margin_g_mm"], 0.0)

    def test_stock_limit_and_effective_max_pieces(self):
        """库存总件数限制有效枚数上限; 缺货规格用量恒为 0。"""
        plan = _plan(8, {0: (1, 50.0, False)})
        cw = [
            {"mass_g": 50.0, "stock": 0},
            {"mass_g": 10.0, "stock": 2},
        ]
        out = self.assertMatchBrute(plan, cw, 0.0, 3)
        self.assertEqual(out["effective_max_pieces"], 2)
        self.assertEqual(out["counterweights"][0]["used"], 0)
        self.assertEqual(out["counterweights"][0]["remaining"], 0)
        # 两枚 10g 不能同槽; 用量不超库存
        self.assertLessEqual(out["counterweights"][1]["used"], 2)

    def test_random_cross_check(self):
        """随机小算例与 itertools 全枚举参考逐项对照。"""
        rng = random.Random(20260925)
        mass_menu = (5.0, 10.0, 25.0, 49.0, 50.0, 51.0)
        for trial in range(60):
            slots_n = rng.choice((4, 6, 8, 12))
            enabled = set(range(slots_n))
            for s in range(slots_n):
                if rng.random() < 0.15:
                    enabled.discard(s)
            enabled = sorted(enabled)
            n_tubes = rng.randint(0, min(3, len(enabled)))
            chosen = rng.sample(enabled, n_tubes)
            placements = {
                s: (i + 1, float(rng.choice((20.0, 49.0, 50.0, 51.0))), False)
                for i, s in enumerate(chosen)
            }
            plan = _plan(100 + trial, placements, slots_n=slots_n,
                         enabled=set(enabled))
            t_types = rng.randint(1, 4)
            cw = [
                {"mass_g": float(m), "stock": rng.randint(0, 2)}
                for m in rng.sample(mass_menu, t_types)
            ]
            limit = rng.choice((0.0, 100.0, 1000.0, 5000.0, 1e12))
            m_max = rng.randint(0, 3)
            self.assertMatchBrute(plan, cw, limit, m_max)

    def test_no_more_than_16_free_slots(self):
        n = 24
        enabled = set(range(n))
        # 0 管 → 24 个空闲启用槽, 拒绝
        plan = _plan(20, {}, slots_n=n, enabled=enabled)
        with self.assertRaises(ValueError) as ctx:
            plan_inventory_compensation(
                plan, [{"mass_g": 10.0, "stock": 1}], 1.0, 3
            )
        self.assertIn("16", str(ctx.exception))
        # 占用 8 个 → 恰 16 个空闲启用槽, 接受
        placements = {i: (i + 1, 10.0, False) for i in range(8)}
        plan_ok = _plan(21, placements, slots_n=n, enabled=enabled)
        out = plan_inventory_compensation(
            plan_ok, [{"mass_g": 10.0, "stock": 1}], 0.0, 0
        )
        self.assertEqual(out["free_enabled_slot_count"], 16)
        self.assertEqual(out["evaluated_combinations"], 1)

    def test_invalid_inputs(self):
        plan = _plan(30, {0: (1, 50.0, False)})
        good = [{"mass_g": 10.0, "stock": 1}]
        # 配重规格为空 / 超过 6 种
        with self.assertRaises(ValueError):
            plan_inventory_compensation(plan, [], 1.0, 3)
        with self.assertRaises(ValueError):
            plan_inventory_compensation(
                plan, [{"mass_g": float(i + 1), "stock": 1} for i in range(7)],
                1.0, 3,
            )
        # 质量非正 / 库存为负 / 非整数库存 / 重复质量
        with self.assertRaises(ValueError):
            plan_inventory_compensation(plan, [{"mass_g": 0.0, "stock": 1}], 1.0, 3)
        with self.assertRaises(ValueError):
            plan_inventory_compensation(plan, [{"mass_g": 10.0, "stock": -1}], 1.0, 3)
        with self.assertRaises(ValueError):
            plan_inventory_compensation(
                plan, [{"mass_g": 10.0, "stock": 1.5}], 1.0, 3
            )
        with self.assertRaises(ValueError):
            plan_inventory_compensation(
                plan, [{"mass_g": 10.0, "stock": 2},
                       {"mass_g": 10.0, "stock": 1}], 1.0, 3
            )
        # 负限值 / 枚数越界
        with self.assertRaises(ValueError):
            plan_inventory_compensation(plan, good, -1.0, 3)
        with self.assertRaises(ValueError):
            plan_inventory_compensation(plan, good, 1.0, 4)
        # 非有限值
        with self.assertRaises(ValueError):
            plan_inventory_compensation(plan, good, float("nan"), 3)
        with self.assertRaises(ValueError):
            plan_inventory_compensation(
                plan, [{"mass_g": float("inf"), "stock": 1}], 1.0, 3
            )

    def test_plan_data_consistency(self):
        # 缺少槽位状态
        plan = _plan(40, {0: (1, 50.0, False)})
        plan["result"]["slot_state"] = []
        with self.assertRaises(ValueError):
            plan_inventory_compensation(
                plan, [{"mass_g": 10.0, "stock": 1}], 1.0, 3
            )
        # 落槽不存在
        plan = _plan(41, {0: (1, 50.0, False)})
        plan["result"]["placements"][0]["slot"] = 99
        with self.assertRaises(ValueError):
            plan_inventory_compensation(
                plan, [{"mass_g": 10.0, "stock": 1}], 1.0, 3
            )
        # 落在禁用槽
        plan = _plan(42, {0: (1, 50.0, False)}, enabled=set(range(1, 12)))
        with self.assertRaises(ValueError):
            plan_inventory_compensation(
                plan, [{"mass_g": 10.0, "stock": 1}], 1.0, 3
            )
        # 占用状态与摆放明细不一致
        plan = _plan(43, {0: (1, 50.0, False)})
        plan["result"]["slot_state"][1]["tube_id"] = 7
        with self.assertRaises(ValueError):
            plan_inventory_compensation(
                plan, [{"mass_g": 10.0, "stock": 1}], 1.0, 3
            )
        # 半径非法
        plan = _plan(44, {0: (1, 50.0, False)})
        plan["result"]["config"]["radius_mm"] = -1.0
        with self.assertRaises(ValueError):
            plan_inventory_compensation(
                plan, [{"mass_g": 10.0, "stock": 1}], 1.0, 3
            )

    def test_read_only_input_not_modified(self):
        plan = _plan(50, {0: (1, 50.0, False)})
        snapshot = copy.deepcopy(plan)
        cw = [{"mass_g": 50.0, "stock": 3}, {"mass_g": 10.0, "stock": 2}]
        cw_snapshot = copy.deepcopy(cw)
        plan_inventory_compensation(plan, cw, 0.0, 3)
        self.assertEqual(plan, snapshot)
        self.assertEqual(cw, cw_snapshot)

    def test_disabled_and_occupied_slots_excluded(self):
        """禁用槽与占用槽都不得放置配重; 只用空闲启用槽。"""
        # 12 槽中槽 2/11 禁用, 槽 0 被占用 → 可选空槽为其余 9 个
        enabled = set(range(12)) - {2, 11}
        plan = _plan(60, {0: (1, 50.0, True)}, enabled=enabled)
        out = plan_inventory_compensation(
            plan, [{"mass_g": 50.0, "stock": 3}], 0.0, 3
        )
        free = {s["slot"] for s in out["free_enabled_slots"]}
        self.assertEqual(free, enabled - {0})
        for r in out["placements"]:
            self.assertIn(r["slot"], free)


if __name__ == "__main__":
    unittest.main()
