################################################################################
#                   LLM GOVERNOR EXPERIMENT — vLLM OPTIMIZED
#
#  Key changes from previous version:
#    - vLLM offline inference engine (~3-5x faster than HuggingFace)
#    - max_new_tokens raised to 4096 for ALL LLM conditions (was 1024)
#    - Truncation tracking on every decision
#    - Full 3×2 design (thinking + nonthinking + rulebased)
#
#  Paste into ONE Colab cell. Requires L4 GPU (24GB).
#  Estimated runtime: ~1.5-2 hours with vLLM speedup
################################################################################

import os, sys, time, json, re, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

OUTPUT_DIR = "/content/experiment_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("PHASE 1: ENVIRONMENT SETUP")
print("=" * 70)

# ── Install vLLM + dependencies ──
print("\n[1/2] Installing vLLM and packages (this takes ~3-5 min)...")
os.system("pip install -q vllm scipy statsmodels matplotlib seaborn tqdm 2>&1 | tail -5")

print("\n[2/2] Loading Qwen3-8B via vLLM...")
import torch
from vllm import LLM, SamplingParams

MODEL_NAME = "Qwen/Qwen3-8B"

# vLLM handles model loading, quantization, and GPU placement automatically
# enforce_eager=True avoids CUDA graph overhead for single-request inference
llm = LLM(
    model=MODEL_NAME,
    dtype="bfloat16",
    max_model_len=8192,      # Total context window
    gpu_memory_utilization=0.85,
    enforce_eager=True,       # Faster for single-request sequential inference
    trust_remote_code=True,
)
tokenizer = llm.get_tokenizer()
print(f"  ✓ vLLM loaded {MODEL_NAME}")


################################################################################
# PHASE 2: CONFIGURATION — THINKING MODE ONLY RE-RUN
################################################################################

print("\n" + "=" * 70)
print("PHASE 2: CONFIGURING EXPERIMENT (Thinking re-run only)")
print("=" * 70)

CONFIG = {
    "conditions": [
        {"strategy": "thinking",    "cycle": "expansionary"},
        {"strategy": "thinking",    "cycle": "recessionary"},
        {"strategy": "nonthinking", "cycle": "expansionary"},
        {"strategy": "nonthinking", "cycle": "recessionary"},
        {"strategy": "rulebased",   "cycle": "expansionary"},
        {"strategy": "rulebased",   "cycle": "recessionary"},
    ],
    "replications": 3,
    "total_periods": 120,
    "calibration_periods": 8,
    "decision_interval": 8,
    "pi_target": (14.0, 30.0),
    "cu_target": (0.74, 0.86),
    "temperature": 0.0,
    "max_new_tokens_thinking": 4096,   # KEY CHANGE: was 1024
    "max_new_tokens_nonthinking": 4096, # Same limit for fair comparison
    "model_name": MODEL_NAME,
    "output_dir": OUTPUT_DIR,
}

PI_MID = (CONFIG["pi_target"][0] + CONFIG["pi_target"][1]) / 2
CU_MID = (CONFIG["cu_target"][0] + CONFIG["cu_target"][1]) / 2
total_runs = len(CONFIG["conditions"]) * CONFIG["replications"]
calls_per_rep = (CONFIG["total_periods"] - CONFIG["calibration_periods"]) // CONFIG["decision_interval"]

print(f"  Full 3×2 design: {len(CONFIG['conditions'])} conditions × {CONFIG['replications']} reps = {total_runs} runs")
print(f"  Token limit: {CONFIG['max_new_tokens_thinking']} for ALL LLM conditions (was 1024)")
print(f"  LLM calls/rep: ~{calls_per_rep}")


################################################################################
# PHASE 2b: PROMPTS, LLM INTERFACE, ABM
################################################################################

SYSTEM_PROMPT = """You are the macroeconomic governor of a multi-agent economic simulation.
Your mandate is to stabilize the EUROZONE (EUR).

TARGETS:
- Price Index (PI): [14.0, 30.0]
- Credit Utilization (CU): [0.74, 0.86]

You govern three zones: EUR, USD, YEN. Cross-zone spillovers exist.

LEVERS (per zone):
1. GROWTH — factory productivity +10% per pull
2. CONTRACTION — factory productivity -10% per pull
3. DEFICIT — inject money into household accounts (demand stimulus, large)
4. POPULATION — add 200 household agents (labor + eventual demand)

Respond with ONLY valid JSON:
{"reasoning": "...", "levers": [{"zone": "EUR", "lever": "GROWTH", "pulls": 2}]}

No action needed: {"reasoning": "No intervention required", "levers": []}

Max 5 pulls per lever. Inaction is often optimal."""


def build_prompt(history, state, cfg):
    pi_lo, pi_hi = cfg["pi_target"]
    cu_lo, cu_hi = cfg["cu_target"]
    p = f"""PERIOD {state['period']}:
EUR: PI={state['EUR_PI']:.4f} [target {pi_lo}-{pi_hi}], CU={state['EUR_CU']:.4f} [target {cu_lo}-{cu_hi}], M1={state['EUR_M1']:.0f}
USD: PI={state['USD_PI']:.4f}, CU={state['USD_CU']:.4f}, M1={state['USD_M1']:.0f}
YEN: PI={state['YEN_PI']:.4f}, CU={state['YEN_CU']:.4f}, M1={state['YEN_M1']:.0f}"""
    if len(history) >= 2:
        p += "\nTREND:"
        for s in history[-4:]:
            p += f"\n  P{s['period']}: PI={s['EUR_PI']:.4f} CU={s['EUR_CU']:.4f}"
    p += "\nJSON response:"
    return p


def query_llm_vllm(prompt, thinking=True):
    """Query using vLLM offline inference — much faster than HF generate."""
    t0 = time.time()

    tag = "/think" if thinking else "/no_think"
    max_tokens = CONFIG["max_new_tokens_thinking"] if thinking else CONFIG["max_new_tokens_nonthinking"]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt + f"\n{tag}"},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    sampling_params = SamplingParams(
        temperature=CONFIG["temperature"],
        top_p=1.0,
        max_tokens=max_tokens,
    )

    outputs = llm.generate([text], sampling_params)
    resp = outputs[0].outputs[0].text
    tok_count = len(outputs[0].outputs[0].token_ids)
    latency = time.time() - t0

    return resp, latency, tok_count


def parse_response(text):
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    m = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if not m:
        return [], "No JSON found"
    try:
        d = json.loads(m.group())
        cmds = []
        for lv in d.get("levers", []):
            z = lv.get("zone", "").upper()
            l = lv.get("lever", "").upper()
            n = min(int(lv.get("pulls", 1)), 5)
            if z in ["EUR","USD","YEN"] and l in ["GROWTH","CONTRACTION","DEFICIT","POPULATION"]:
                cmds.append(f"{z}_{l}_{n}")
        return cmds, d.get("reasoning", "")
    except json.JSONDecodeError as e:
        return [], f"Parse error: {e}"


# ── Python ABM (identical to previous version) ──

def init_economy(cycle):
    zones = {}
    for z in ["EUR", "USD", "YEN"]:
        if cycle == "expansionary":
            zones[z] = {
                "price_index": 9.0 + np.random.uniform(-0.5, 0.5),
                "credit_utilization": 0.95 + np.random.uniform(-0.02, 0.02),
                "m1": 50000 + np.random.uniform(-2000, 2000),
                "productivity": 1.0,
                "n_households": 500, "n_factories": 50,
                "inventory": 10000, "demand": 9000,
                "wages": 10.0, "interest_rate": 0.05,
                "savings": 5000, "debt": 47500, "bank_cap": 50000,
            }
        else:
            zones[z] = {
                "price_index": (55.0 if z == "EUR" else 20.0) + np.random.uniform(-5, 5),
                "credit_utilization": 0.95 + np.random.uniform(-0.02, 0.02),
                "m1": 80000 + np.random.uniform(-5000, 5000),
                "productivity": 0.5,
                "n_households": 500, "n_factories": 50,
                "inventory": 15000, "demand": 5000,
                "wages": 15.0, "interest_rate": 0.08,
                "savings": 2000, "debt": 76000, "bank_cap": 80000,
            }
    return {"zones": zones, "period": 0}


def step(econ):
    zones = econ["zones"]
    for z in zones.values():
        output = z["n_factories"] * z["productivity"] * 20
        z["inventory"] += output
        cprop = 0.7 + 0.1 * (z["interest_rate"] < 0.05)
        pot_demand = z["n_households"] * z["wages"] * cprop / max(z["price_index"], 0.01)
        act_demand = min(pot_demand, z["inventory"] * 0.8)
        z["demand"] = act_demand
        z["inventory"] = max(z["inventory"] - act_demand, 0)
        sd_ratio = output / max(act_demand, 1)
        adj = 0.02 * (1 / max(sd_ratio, 0.01) - 1)
        z["price_index"] *= (1 + adj * 0.3)
        z["price_index"] = max(z["price_index"], 0.01)
        new_borrow = max(act_demand * z["price_index"] * 0.1 - z["savings"] * 0.05, 0)
        repay = z["debt"] * z["interest_rate"] / 52
        z["debt"] = max(z["debt"] + new_borrow - repay, 0)
        z["bank_cap"] += z["savings"] * 0.001
        z["credit_utilization"] = min(z["debt"] / max(z["bank_cap"], 1), 1.0)
        z["m1"] = max(z["m1"] + new_borrow - repay, 0)
        ld = z["n_factories"] * 10
        ls = z["n_households"]
        if ld > ls * 0.9: z["wages"] *= 1.005
        elif ld < ls * 0.7: z["wages"] *= 0.995
        z["interest_rate"] = 0.02 + 0.08 * z["credit_utilization"]
        income = z["n_households"] * z["wages"]
        spending = act_demand * z["price_index"]
        z["savings"] += (income - spending) * 0.1
    names = list(zones.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            z1, z2 = zones[names[i]], zones[names[j]]
            pr = z1["price_index"] / max(z2["price_index"], 0.01)
            if pr < 0.8:
                f = z1["inventory"] * 0.02; z1["inventory"] -= f; z2["inventory"] += f
            elif pr > 1.2:
                f = z2["inventory"] * 0.02; z2["inventory"] -= f; z1["inventory"] += f
    econ["period"] += 1
    return econ


def get_state(econ, period):
    s = {"period": period}
    for n, z in econ["zones"].items():
        s[f"{n}_PI"] = z["price_index"]
        s[f"{n}_CU"] = z["credit_utilization"]
        s[f"{n}_M1"] = z["m1"]
    return s


def apply_lever(econ, cmd):
    parts = cmd.split("_")
    if len(parts) < 3: return econ
    zone, lever = parts[0], parts[1]
    try: n = int(parts[2])
    except: n = 1
    if zone not in econ["zones"]: return econ
    z = econ["zones"][zone]
    for _ in range(n):
        if lever == "GROWTH": z["productivity"] *= 1.10
        elif lever == "CONTRACTION": z["productivity"] *= 0.90
        elif lever == "DEFICIT":
            z["m1"] += 10000; z["savings"] += 3000; z["debt"] += 7000; z["bank_cap"] += 5000
        elif lever == "POPULATION": z["n_households"] += 200
    return econ


# ── Rule-based governor ──

def rule_governor(state, cfg):
    cmds, reasons = [], []
    pi, cu = state["EUR_PI"], state["EUR_CU"]
    pi_lo, pi_hi = cfg["pi_target"]
    cu_lo, cu_hi = cfg["cu_target"]
    if pi < pi_lo:
        n = min(int((pi_lo - pi) / 4) + 1, 3)
        cmds.append(f"EUR_CONTRACTION_{n}"); reasons.append(f"PI low")
    elif pi > pi_hi:
        n = min(int((pi - pi_hi) / 4) + 1, 3)
        cmds.append(f"EUR_GROWTH_{n}"); reasons.append(f"PI high")
    if cu < cu_lo:
        n = min(int((cu_lo - cu) / 0.05) + 1, 2)
        cmds.append(f"EUR_DEFICIT_{n}"); reasons.append(f"CU low")
    elif cu > cu_hi:
        n = min(int((cu - cu_hi) / 0.05) + 1, 2)
        cmds.append(f"EUR_GROWTH_{n}"); reasons.append(f"CU high")
    return cmds, "; ".join(reasons) or "No action"


# ── Single replication ──

def run_replication(cond, rep, cfg):
    strat, cycle = cond["strategy"], cond["cycle"]
    cname = f"{strat}_{cycle}"

    hist, logs = [], []
    tok_total = 0; lat_total = 0.0; dec_ct = 0; branches = 0
    truncated_count = 0
    econ = init_economy(cycle)

    max_tok = cfg["max_new_tokens_thinking"] if strat == "thinking" else cfg["max_new_tokens_nonthinking"]

    for p in range(cfg["total_periods"]):
        econ = step(econ)
        state = get_state(econ, p)
        hist.append(state)

        if p < cfg["calibration_periods"]:
            continue
        if (p - cfg["calibration_periods"]) % cfg["decision_interval"] != 0:
            continue

        dec_num = (p - cfg["calibration_periods"]) // cfg["decision_interval"] + 1
        total_decs = (cfg["total_periods"] - cfg["calibration_periods"]) // cfg["decision_interval"]

        if strat == "rulebased":
            cmds, reason = rule_governor(state, cfg)
            lat, tok = 0.0, 0
            was_truncated = False
            print(f"\r    {cname} rep {rep}: decision {dec_num}/{total_decs} (rule) PI={state['EUR_PI']:.2f} CU={state['EUR_CU']:.4f}", end="", flush=True)
        else:
            print(f"\r    {cname} rep {rep}: decision {dec_num}/{total_decs} (vLLM calling...)", end="", flush=True)
            prompt = build_prompt(hist, state, cfg)
            resp, lat, tok = query_llm_vllm(prompt, thinking=(strat == "thinking"))
            cmds, reason = parse_response(resp)
            tok_total += tok; lat_total += lat; dec_ct += 1
            if strat == "thinking" and tok > 200: branches += 1

            was_truncated = (tok >= max_tok - 5)
            if was_truncated: truncated_count += 1
            print(f"\r    {cname} rep {rep}: decision {dec_num}/{total_decs} ({lat:.1f}s, {tok}tok{'⚠TRUNC' if was_truncated else ''}) PI={state['EUR_PI']:.2f} CU={state['EUR_CU']:.4f}  ", end="", flush=True)

        for c in cmds:
            econ = apply_lever(econ, c)

        logs.append({"period": p, "commands": cmds, "reasoning": reason[:500],
                      "latency": lat, "tokens": tok, "truncated": was_truncated,
                      "EUR_PI": state["EUR_PI"], "EUR_CU": state["EUR_CU"]})

    final = hist[-1] if hist else {}
    active = hist[cfg["calibration_periods"]:]
    na = max(len(active), 1)
    cu_in = sum(1 for s in active if cfg["cu_target"][0] <= s["EUR_CU"] <= cfg["cu_target"][1])

    result = {
        "condition": cname, "strategy": strat, "cycle": cycle, "replication": rep,
        "final_EUR_PI": final.get("EUR_PI"),
        "final_EUR_CU": final.get("EUR_CU"),
        "pi_distance_from_target": abs(final.get("EUR_PI", 0) - PI_MID),
        "cu_distance_from_target": abs(final.get("EUR_CU", 0) - CU_MID),
        "cu_time_in_target_pct": 100 * cu_in / na,
        "total_lever_activations": sum(len(l["commands"]) for l in logs),
        "reasoning_branches": branches,
        "mean_decision_latency": lat_total / max(dec_ct, 1),
        "total_tokens": tok_total,
        "decision_count": dec_ct,
        "truncated_decisions": truncated_count,
        "truncation_rate": 100 * truncated_count / max(dec_ct, 1),
    }

    print(f"\n    ✓ {cname} rep {rep} | PI={final.get('EUR_PI',0):.2f} (dist={result['pi_distance_from_target']:.2f}) CU={final.get('EUR_CU',0):.4f} | {result['total_lever_activations']} levers, {tok_total} tok, {truncated_count}/{dec_ct} truncated")
    return result, hist, logs


################################################################################
# PHASE 3: RUN THINKING RE-EXPERIMENT
################################################################################

print("\n" + "=" * 70)
print("PHASE 3: RUNNING FULL EXPERIMENT (vLLM, 4096 tokens)")
print("=" * 70)

thinking_results = []
t_start = time.time()

for cond in CONFIG["conditions"]:
    cname = f"{cond['strategy']}_{cond['cycle']}"
    print(f"\n  === {cname} ===")

    for rep in range(CONFIG["replications"]):
        try:
            result, hist, logs = run_replication(cond, rep, CONFIG)
            thinking_results.append(result)

            with open(f"{OUTPUT_DIR}/history_{cname}_rep{rep}_4096.json", "w") as f:
                json.dump(hist, f)
            with open(f"{OUTPUT_DIR}/levers_{cname}_rep{rep}_4096.json", "w") as f:
                json.dump(logs, f)

        except Exception as e:
            print(f"\n    ✗ ERROR: {e}")
            import traceback; traceback.print_exc()

    print()

elapsed = time.time() - t_start
results_df = pd.DataFrame(thinking_results)
results_df.to_csv(f"{OUTPUT_DIR}/final_results_4096.csv", index=False)

print(f"\n{'='*70}")
print(f"EXPERIMENT COMPLETE in {elapsed/3600:.1f} hours")
print(f"{'='*70}")
print(results_df.groupby(["strategy", "cycle"]).agg({
    "final_EUR_PI": ["mean", "std"],
    "pi_distance_from_target": ["mean", "std"],
    "total_lever_activations": ["mean", "std"],
    "mean_decision_latency": ["mean"],
    "total_tokens": ["mean"],
    "truncated_decisions": ["mean"],
    "truncation_rate": ["mean"],
}).round(3))


################################################################################
# PHASE 4: STATISTICAL ANALYSIS & FIGURES
################################################################################

print("\n" + "=" * 70)
print("PHASE 4: STATISTICAL ANALYSIS")
print("=" * 70)

import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({"figure.dpi": 150, "font.size": 10})
df = results_df.copy()

# ── Descriptive stats ──
print("\nTABLE 1: DESCRIPTIVE STATISTICS")
print("-" * 60)
desc = df.groupby(["strategy","cycle"]).agg({
    "final_EUR_PI": ["mean","std","min","max"],
    "final_EUR_CU": ["mean","std"],
    "pi_distance_from_target": ["mean","std"],
    "cu_time_in_target_pct": ["mean","std"],
    "total_lever_activations": ["mean","std"],
    "mean_decision_latency": ["mean"],
    "total_tokens": ["mean"],
    "truncated_decisions": ["mean"],
    "truncation_rate": ["mean"],
}).round(3)
print(desc)
desc.to_csv(f"{OUTPUT_DIR}/table1_descriptive_4096.csv")

# ── Kruskal-Wallis tests ──
print("\nTABLE 2: INFERENTIAL TESTS")
print("-" * 60)
dvs = ["final_EUR_PI", "final_EUR_CU", "pi_distance_from_target",
       "cu_time_in_target_pct", "total_lever_activations"]
for dv in dvs:
    groups = [g[dv].dropna().values for _, g in df.groupby("condition")]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2: continue
    try:
        if all(g.std() == 0 for g in groups if len(g) > 1):
            print(f"  {dv}: No variance — skipped")
            continue
        h, p = stats.kruskal(*groups)
        print(f"  {dv} (Kruskal-Wallis): H={h:.4f} p={p:.4f}")
    except Exception as e:
        print(f"  {dv}: {e}")

# ── Tukey HSD ──
print("\nTABLE 3: POST-HOC (Tukey HSD)")
print("-" * 60)
for dv in dvs:
    clean = df.dropna(subset=[dv])
    if clean["condition"].nunique() < 2 or clean[dv].std() == 0: continue
    try:
        tk = pairwise_tukeyhsd(clean[dv], clean["condition"], alpha=0.05)
        print(f"\n  {dv}:")
        print(tk.summary())
    except Exception as e:
        print(f"  {dv}: {e}")

# ── Figures ──
COLORS = {
    "thinking_expansionary": "#2196F3", "thinking_recessionary": "#F44336",
    "nonthinking_expansionary": "#4CAF50", "nonthinking_recessionary": "#FF9800",
    "rulebased_expansionary": "#9C27B0", "rulebased_recessionary": "#795548",
}
ORDER = ["thinking_expansionary", "thinking_recessionary",
         "nonthinking_expansionary", "nonthinking_recessionary",
         "rulebased_expansionary", "rulebased_recessionary"]

def plot_timeseries(key, ylabel, title, filename, target=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for i, cycle in enumerate(["expansionary", "recessionary"]):
        ax = axes[i]
        for strat in ["thinking", "nonthinking", "rulebased"]:
            cond = f"{strat}_{cycle}"
            traces = []
            for r in range(CONFIG["replications"]):
                hf = f"{OUTPUT_DIR}/history_{cond}_rep{r}_4096.json"
                if os.path.exists(hf):
                    with open(hf) as f: h = json.load(f)
                    traces.append([s.get(key, np.nan) for s in h])
            if traces:
                ml = max(len(t) for t in traces)
                arr = np.array([t + [np.nan]*(ml-len(t)) for t in traces])
                mn = np.nanmean(arr, axis=0)
                ci = 1.96 * np.nanstd(arr, axis=0) / np.sqrt(np.sum(~np.isnan(arr), axis=0))
                x = np.arange(len(mn))
                c = COLORS.get(cond, "gray")
                ax.plot(x, mn, label=strat, color=c, linewidth=1.5)
                ax.fill_between(x, mn-ci, mn+ci, alpha=0.12, color=c)
        if target:
            ax.axhspan(*target, alpha=0.08, color="green", label="Target")
        ax.set_title(f"{title} — {cycle.title()}")
        ax.set_xlabel("Period (Weeks)"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}", dpi=300, bbox_inches="tight")
    plt.show()

plot_timeseries("EUR_PI", "EUR Price Index", "Price Index", "fig1_price_index_4096.png", CONFIG["pi_target"])
plot_timeseries("EUR_CU", "EUR Credit Utilization", "Credit Utilization", "fig2_credit_util_4096.png", CONFIG["cu_target"])

# PI distance bar chart
fig, ax = plt.subplots(figsize=(10, 5))
palette = [COLORS.get(c, "gray") for c in ORDER]
sns.barplot(data=df, x="condition", y="pi_distance_from_target", order=ORDER,
            palette=palette, ax=ax, errorbar=("ci", 95), capsize=0.1)
ax.set_ylabel(f"Distance from PI Target Midpoint ({PI_MID})")
ax.set_title("Price Index: Distance from Target (4096 token limit) — Lower is Better")
ax.set_xticklabels([c.replace("_", "\n") for c in ORDER], fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig3_pi_distance_4096.png", dpi=300, bbox_inches="tight")
plt.show()

# Lever activations
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=df, x="condition", y="total_lever_activations", order=ORDER,
            palette=palette, ax=ax, errorbar=("ci", 95), capsize=0.1)
ax.set_ylabel("Total Lever Activations")
ax.set_title("Lever Usage (4096 token limit)")
ax.set_xticklabels([c.replace("_", "\n") for c in ORDER], fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig4_levers_4096.png", dpi=300, bbox_inches="tight")
plt.show()

# Latency
llm_df = df[df["strategy"] != "rulebased"].copy()
if len(llm_df) > 0:
    llm_order = [c for c in ORDER if "rulebased" not in c]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=llm_df, x="condition", y="mean_decision_latency", order=llm_order,
                palette=[COLORS.get(c,"gray") for c in llm_order], ax=ax, errorbar=("ci", 95), capsize=0.1)
    ax.set_ylabel("Mean Decision Latency (seconds)")
    ax.set_title("Inference Cost (4096 token limit)")
    ax.set_xticklabels([c.replace("_", "\n") for c in llm_order], fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig5_latency_4096.png", dpi=300, bbox_inches="tight")
    plt.show()


################################################################################
# DONE
################################################################################

print("\n" + "=" * 70)
print("ALL DONE!")
print("=" * 70)
print(f"\nResults: {OUTPUT_DIR}/")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if "4096" in f:
        sz = os.path.getsize(f"{OUTPUT_DIR}/{f}")
        print(f"  {f}  ({sz/1024:.1f} KB)")
print(f"\nModel: {MODEL_NAME} via vLLM")
print(f"Token limit: 4096 for all LLM conditions")
print(f"\nKey finding: Check truncation_rate in the results.")
print(f"If thinking mode still truncates at 4096 → need even more tokens.")
print(f"If no truncation → deliberative paralysis is genuine.")
print(f"If more levers are pulled → truncation was the cause of inaction.")