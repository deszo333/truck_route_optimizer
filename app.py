import streamlit as st
import time
import pandas as pd
import numpy as np
import random
import ray
from src.data_processor import DataProcessor
from src.routing_engine import RoutingEngine, build_distance_dict
from src.dynamic_updater import inject_urgent_request

# Initialize Ray safely
if not ray.is_initialized():
    ray.init(ignore_reinit_error=True)

# ─────────────────────────────────────────────
#  PAGE CONFIG & GLOBAL STYLES
# ─────────────────────────────────────────────
st.set_page_config(page_title="BuildConnect Dispatch VRP", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

  /* ── Scene card ── */
  .scene-card {
    background: #0f1117;
    border: 1px solid #2a2d3a;
    border-left: 4px solid #f59e0b;
    border-radius: 8px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
  }
  .scene-title {
    font-family: 'IBM Plex Mono', monospace;
    color: #f59e0b;
    font-size: 0.78rem;
    letter-spacing: .12em;
    text-transform: uppercase;
    margin-bottom: .3rem;
  }
  .scene-heading {
    font-size: 1.35rem;
    font-weight: 700;
    color: #f8fafc;
    margin: 0 0 .7rem 0;
  }
  .scene-body {
    color: #94a3b8;
    line-height: 1.7;
    font-size: 0.96rem;
  }

  /* ── Why-box ── */
  .why-box {
    background: #1e2330;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: .9rem 1.2rem;
    margin-top: .8rem;
    color: #7dd3fc;
    font-size: 0.91rem;
  }
  .why-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #38bdf8;
    letter-spacing: .1em;
    margin-bottom: .3rem;
  }

  /* ── Log stream ── */
  .log-line {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    padding: .18rem 0;
    border-bottom: 1px solid #1e2330;
  }
  .log-ok   { color: #4ade80; }
  .log-warn { color: #fb923c; }
  .log-info { color: #60a5fa; }
  .log-dim  { color: #475569; }

  /* ── Capacity bar ── */
  .cap-bar-wrap { margin: .4rem 0; }
  .cap-bar-bg {
    background: #1e2330;
    border-radius: 4px;
    height: 14px;
    width: 100%;
    overflow: hidden;
  }
  .cap-bar-fill {
    height: 14px;
    border-radius: 4px;
    transition: width .5s ease;
  }

  /* ── Pipeline step ── */
  .pipe-step {
    display: flex;
    align-items: center;
    gap: .7rem;
    padding: .55rem .9rem;
    border-radius: 6px;
    margin-bottom: .4rem;
    background: #1a1f2e;
    border: 1px solid #2a2d3a;
    color: #e2e8f0;
    font-size: 0.9rem;
  }
  .pipe-arrow {
    color: #f59e0b;
    font-size: 1.1rem;
    text-align: center;
    margin-left: 1.4rem;
    margin-bottom: .3rem;
  }

  /* ── Region badge ── */
  .badge {
    display: inline-block;
    padding: .15rem .55rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: .3rem;
  }
  .badge-luzon    { background: #1e3a5f; color: #60a5fa; }
  .badge-visayas  { background: #2d1f5e; color: #a78bfa; }
  .badge-mindanao { background: #1f3d2a; color: #4ade80; }

  /* ── Benchmark timer ── */
  .timer-panel {
    background: #0f1117;
    border: 1px solid #2a2d3a;
    border-radius: 8px;
    padding: 1.2rem;
    text-align: center;
  }
  .timer-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.2rem;
    font-weight: 600;
  }
  .timer-label { color: #64748b; font-size: .82rem; margin-top: .2rem; }

  /* ── Urgent card ── */
  .urgent-card {
    background: #2d1209;
    border: 2px solid #ef4444;
    border-radius: 8px;
    padding: 1rem 1.3rem;
    animation: pulse-border 1.2s ease-in-out infinite;
  }
  @keyframes pulse-border {
    0%,100% { border-color: #ef4444; box-shadow: 0 0 0 0 rgba(239,68,68,0); }
    50%      { border-color: #fca5a5; box-shadow: 0 0 0 6px rgba(239,68,68,.15); }
  }

  /* ── Metric tile ── */
  .metric-tile {
    background: #13172a;
    border: 1px solid #1e2d45;
    border-radius: 8px;
    padding: .9rem 1.1rem;
    text-align: center;
  }
  .metric-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.55rem;
    font-weight: 700;
    color: #f59e0b;
  }
  .metric-lbl { color: #64748b; font-size: .78rem; margin-top: .25rem; }

  /* ── Step divider ── */
  .step-divider {
    border: none;
    border-top: 1px dashed #1e2d3a;
    margin: 1.4rem 0;
  }

  /* progress bar override */
  .stProgress > div > div { background-color: #f59e0b !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def scene_card(num: int, title: str, body: str):
    st.markdown(f"""
    <div class="scene-card">
      <div class="scene-title">Scene {num}</div>
      <div class="scene-heading">{title}</div>
      <div class="scene-body">{body}</div>
    </div>""", unsafe_allow_html=True)

def why_box(text: str):
    st.markdown(f"""
    <div class="why-box">
      <div class="why-label">💡 WHY THIS MATTERS</div>
      {text}
    </div>""", unsafe_allow_html=True)

def log_line(msg: str, kind: str = "info"):
    css = {"ok": "log-ok", "warn": "log-warn", "info": "log-info", "dim": "log-dim"}.get(kind, "log-info")
    st.markdown(f'<div class="log-line {css}">{msg}</div>', unsafe_allow_html=True)

def capacity_bar(label: str, used: float, total: float, color: str = "#f59e0b"):
    pct = min(used / total * 100, 100)
    bar_color = "#ef4444" if pct > 85 else color
    st.markdown(f"""
    <div class="cap-bar-wrap">
      <div style="display:flex;justify-content:space-between;color:#94a3b8;font-size:.8rem;margin-bottom:3px">
        <span>{label}</span><span>{used:.0f} / {total:.0f} kg ({pct:.0f}%)</span>
      </div>
      <div class="cap-bar-bg"><div class="cap-bar-fill" style="width:{pct}%;background:{bar_color};"></div></div>
    </div>""", unsafe_allow_html=True)

def pipe_step(icon: str, text: str):
    st.markdown(f'<div class="pipe-step"><span>{icon}</span><span>{text}</span></div>', unsafe_allow_html=True)

def pipe_arrow():
    st.markdown('<div class="pipe-arrow">▼</div>', unsafe_allow_html=True)

def section_divider():
    st.markdown('<hr class="step-divider">', unsafe_allow_html=True)

def sleep(s=1.5):
    time.sleep(s)

# ─────────────────────────────────────────────
#  SYNTHETIC DATA (matches real CSV schemas)
# ─────────────────────────────────────────────

BRANCHES = pd.DataFrame([
    {"branch_id": "BR-001", "branch_name": "Makati Central",    "region": "Luzon",    "capacity_sqm": 4200},
    {"branch_id": "BR-002", "branch_name": "QC North Hub",      "region": "Luzon",    "capacity_sqm": 3800},
    {"branch_id": "BR-003", "branch_name": "Cebu Main",         "region": "Visayas",  "capacity_sqm": 3600},
    {"branch_id": "BR-004", "branch_name": "Davao South",       "region": "Mindanao", "capacity_sqm": 3500},
    {"branch_id": "BR-005", "branch_name": "Iloilo Depot",      "region": "Visayas",  "capacity_sqm": 3100},
    {"branch_id": "BR-006", "branch_name": "Cagayan de Oro",    "region": "Mindanao", "capacity_sqm": 2900},
    {"branch_id": "BR-007", "branch_name": "Laguna East",       "region": "Luzon",    "capacity_sqm": 2700},
    {"branch_id": "BR-008", "branch_name": "Bacolod Hub",       "region": "Visayas",  "capacity_sqm": 2400},
])

PRODUCTS = pd.DataFrame([
    {"product_id": "PRD-A1", "product_name": "Steel Rebar Bundle",    "weight_kg": 120.0},
    {"product_id": "PRD-B2", "product_name": "Cement Bag (50kg)",     "weight_kg": 50.0},
    {"product_id": "PRD-C3", "product_name": "Hollow Blocks (pallet)","weight_kg": 80.0},
    {"product_id": "PRD-D4", "product_name": "Plywood Sheet Stack",   "weight_kg": 35.0},
    {"product_id": "PRD-E5", "product_name": "PVC Pipe Bundle",       "weight_kg": 28.0},
])

TRANSACTIONS = pd.DataFrame([
    {"transaction_id": "TX-0001", "product_id": "PRD-A1", "quantity": 3, "branch_id": "BR-002"},
    {"transaction_id": "TX-0002", "product_id": "PRD-B2", "quantity": 8, "branch_id": "BR-001"},
    {"transaction_id": "TX-0003", "product_id": "PRD-C3", "quantity": 5, "branch_id": "BR-003"},
    {"transaction_id": "TX-0004", "product_id": "PRD-D4", "quantity": 10,"branch_id": "BR-004"},
    {"transaction_id": "TX-0005", "product_id": "PRD-E5", "quantity": 6, "branch_id": "BR-002"},
])

TRUCKS = pd.DataFrame([
    {"truck_id": "TRK-L01", "region": "Luzon",    "capacity_kg": 1500, "fuel_efficiency": 8.2},
    {"truck_id": "TRK-L02", "region": "Luzon",    "capacity_kg": 2000, "fuel_efficiency": 7.5},
    {"truck_id": "TRK-V01", "region": "Visayas",  "capacity_kg": 1200, "fuel_efficiency": 9.1},
    {"truck_id": "TRK-M01", "region": "Mindanao", "capacity_kg": 1800, "fuel_efficiency": 7.8},
])

DELIVERY_LOGS = pd.DataFrame([
    {"origin_branch": "BR-001", "destination_branch": "BR-002", "distance_km": 18},
    {"origin_branch": "BR-001", "destination_branch": "BR-007", "distance_km": 42},
    {"origin_branch": "BR-002", "destination_branch": "BR-007", "distance_km": 31},
    {"origin_branch": "BR-003", "destination_branch": "BR-005", "distance_km": 55},
    {"origin_branch": "BR-003", "destination_branch": "BR-008", "distance_km": 67},
    {"origin_branch": "BR-004", "destination_branch": "BR-006", "distance_km": 89},
])

REGION_COLORS = {"Luzon": "badge-luzon", "Visayas": "badge-visayas", "Mindanao": "badge-mindanao"}


# ─────────────────────────────────────────────
#  TUTORIAL SCENES
# ─────────────────────────────────────────────

def scene_01_problem():
    scene_card(1, "What Problem Are We Solving?",
        "BuildConnect needs to deliver thousands of orders across many branches every day. "
        "The challenge is not just picking any path — it is deciding <b>which truck carries which deliveries, "
        "in what order</b>, while respecting weight limits and urgency windows.")
    
    why_box("Without optimization, trucks get overloaded, routes overlap, and deliveries arrive late. "
            "The VRP (Vehicle Routing Problem) engine solves this automatically.")

    st.markdown("#### The Bad Plan vs. The Smart Plan")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="background:#1f0d0d;border:1px solid #7f1d1d;border-radius:8px;padding:1rem;">
          <div style="color:#ef4444;font-weight:700;margin-bottom:.5rem;">❌ Naive Assignment</div>
          <div style="color:#fca5a5;font-size:.88rem;line-height:1.8;">
            • One truck per order<br>
            • No capacity check<br>
            • Random route order<br>
            • Trucks constantly half-empty<br>
            • High fuel cost, late deliveries
          </div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background:#0d1f12;border:1px solid #14532d;border-radius:8px;padding:1rem;">
          <div style="color:#4ade80;font-weight:700;margin-bottom:.5rem;">✅ VRP Optimization</div>
          <div style="color:#86efac;font-size:.88rem;line-height:1.8;">
            • Pack trucks to capacity<br>
            • Nearest-neighbor routing<br>
            • Regional clustering<br>
            • Dynamic urgent injection<br>
            • Parallel processing across regions
          </div>
        </div>""", unsafe_allow_html=True)
    
    section_divider()
    st.markdown("""
    <div style="color:#94a3b8;font-size:.9rem;line-height:1.8;">
      The system processes <b style="color:#f59e0b">20,000 delivery requests</b> per run, 
      clusters them by region, assigns them to trucks using a greedy capacity algorithm, 
      and routes them using a nearest-neighbor heuristic — all in parallel using Ray.
    </div>""", unsafe_allow_html=True)


def scene_02_data_tour():
    scene_card(2, "Meet the Data — A Full Tour of the CSVs",
        "Before any routing can happen, the system needs to understand the physical world. "
        "Each CSV file teaches the algorithm something different about the problem.")

    # 2A — branches
    st.markdown("#### 📁 `branches.csv` — The Physical Network")
    why_box("Branches are the nodes of our delivery network. <b>capacity_sqm</b> tells us which "
            "branches are large enough to act as distribution hubs. Larger branches have more space "
            "for staging goods, so we choose the top 5 by capacity as hubs.")
    
    display_branches = BRANCHES.copy()
    display_branches["IS HUB?"] = display_branches["capacity_sqm"].rank(ascending=False) <= 5
    display_branches["IS HUB?"] = display_branches["IS HUB?"].map({True: "⭐ Hub", False: "—"})
    display_branches = display_branches.sort_values("capacity_sqm", ascending=False)
    st.dataframe(display_branches[["branch_id","branch_name","region","capacity_sqm","IS HUB?"]],
                 use_container_width=True, hide_index=True)
    
    sleep(1)
    section_divider()

    # 2B — products
    st.markdown("#### 📁 `products.csv` — Weight Per Unit")
    why_box("The routing engine does not understand 'orders'. It only understands <b>kilograms</b>. "
            "This table converts product names into physical weight so trucks know what they're carrying.")
    st.dataframe(PRODUCTS, use_container_width=True, hide_index=True)

    sleep(1)
    section_divider()

    # 2C — transactions
    st.markdown("#### 📁 `transactions.csv` — The Source of Demand")
    why_box("Each transaction says: a customer at a branch ordered X units of product Y. "
            "This is the <b>raw demand signal</b>. By itself it has no weight — we must join it with products.")
    st.dataframe(TRANSACTIONS, use_container_width=True, hide_index=True)

    sleep(1)
    section_divider()

    # 2D — trucks
    st.markdown("#### 📁 `trucks.csv` — The Fleet")
    why_box("<b>capacity_kg</b> is a hard physical constraint. The algorithm will never assign "
            "a load heavier than what a truck can carry. <b>fuel_efficiency</b> affects cost scoring.")
    st.dataframe(TRUCKS, use_container_width=True, hide_index=True)

    sleep(1)
    section_divider()

    # 2E — delivery_logs
    st.markdown("#### 📁 `delivery_logs.csv` — The Distance Memory")
    why_box("Past delivery records tell us how far it is between any two branches. "
            "This becomes the <b>distance dictionary</b> the routing engine uses to pick the cheapest next stop.")
    st.dataframe(DELIVERY_LOGS, use_container_width=True, hide_index=True)


def scene_03_transformation():
    scene_card(3, "How Raw Data Becomes a Delivery Job",
        "This is the most important transformation in the system. "
        "A transaction record cannot be loaded onto a truck — a delivery job with a computed weight can.")

    st.markdown("#### Step-by-step: TX-0001")
    sleep(.5)

    pipe_step("📄", "Transaction TX-0001: product PRD-A1, quantity = 3, destination BR-002")
    pipe_arrow()
    sleep(1)
    pipe_step("🔍", "Look up PRD-A1 in products.csv → product_name = 'Steel Rebar Bundle', weight_kg = 120.0")
    pipe_arrow()
    sleep(1)
    pipe_step("🧮", "Compute total weight: 3 × 120.0 kg = 360.0 kg")
    pipe_arrow()
    sleep(1)
    pipe_step("📦", "Delivery Job created: origin=BR-002, weight=360.0 kg, deadline=4 hrs")

    sleep(1)
    section_divider()
    st.markdown("#### Batch transformation for all 5 sample transactions")

    joined = TRANSACTIONS.merge(PRODUCTS, on="product_id")
    joined["total_weight_kg"] = joined["quantity"] * joined["weight_kg"]
    joined["deadline_hrs"] = [4, 6, 3, 5, 4]
    joined["destination_branch"] = ["BR-001", "BR-002", "BR-005", "BR-006", "BR-001"]
    joined = joined.rename(columns={"branch_id": "origin_branch"})
    cols = ["transaction_id","product_name","quantity","weight_kg","total_weight_kg","origin_branch","destination_branch","deadline_hrs"]
    st.dataframe(joined[cols], use_container_width=True, hide_index=True)

    why_box("Notice how <b>total_weight_kg</b> is a computed column — it does not exist in any raw CSV. "
            "The DataProcessor creates it by joining transactions × products. "
            "Without this step, the routing engine has no idea how heavy each job is.")


def scene_04_hub_selection():
    scene_card(4, "Why We Choose the Largest Branches as Hubs",
        "The algorithm needs origin hubs — central points from which trucks depart. "
        "We do not pick these randomly. We pick the branches with the most physical space.")

    sorted_b = BRANCHES.sort_values("capacity_sqm", ascending=False).reset_index(drop=True)
    sorted_b["Rank"] = sorted_b.index + 1
    sorted_b["Selected as Hub"] = sorted_b["Rank"].apply(lambda r: "⭐ Yes" if r <= 5 else "✗ No")
    
    st.dataframe(sorted_b[["Rank","branch_id","branch_name","region","capacity_sqm","Selected as Hub"]],
                 use_container_width=True, hide_index=True)

    st.markdown("#### Capacity visualised")
    for _, row in sorted_b.iterrows():
        color = "#f59e0b" if row["Selected as Hub"] == "⭐ Yes" else "#334155"
        capacity_bar(f"{row['branch_name']} ({row['branch_id']})", row["capacity_sqm"], 4200, color)

    why_box("Random hub selection is fine for quick tests, but not realistic. "
            "Large branches have loading docks, forklifts, and staging space. "
            "Using capacity_sqm as the criterion makes the model defensible in a real business context.")


def scene_05_sampling():
    scene_card(5, "Why the Tutorial Uses a Small Sample",
        "The production system processes 20,000 delivery requests. "
        "Showing all of them at once would bury the learning in noise.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="metric-tile">
          <div class="metric-num">20,000</div>
          <div class="metric-lbl">Full Production Jobs</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-tile">
          <div class="metric-num" style="color:#60a5fa;">5</div>
          <div class="metric-lbl">Tutorial Sample Jobs</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    pipe_step("🗄️", "Full database: 20,000 transactions across all regions")
    pipe_arrow()
    pipe_step("🔬", "Filter: take 1 region, take 5 transactions")
    pipe_arrow()
    pipe_step("📖", "Tutorial mode: show each step clearly on 5 rows")

    why_box("The sample is for <b>learning</b>. The benchmark mode uses the full 20,000 jobs. "
            "Every rule you see applied to 5 rows here is applied to 20,000 rows in benchmark mode — "
            "in parallel across all regions simultaneously.")

    st.info("⚠️ The algorithm logic is identical in both modes. Only the data size changes.")


def scene_06_clustering():
    scene_card(6, "How Regional Clustering Works",
        "Instead of one global routing problem, the engine splits the country into regions. "
        "Each region is solved independently — which is exactly what makes parallelism possible.")

    region_counts = {"Luzon": 12000, "Visayas": 5000, "Mindanao": 3000}
    total = sum(region_counts.values())

    for region, count in region_counts.items():
        badge_cls = REGION_COLORS.get(region, "badge-luzon")
        st.markdown(f'<span class="badge {badge_cls}">{region}</span>', unsafe_allow_html=True)
        capacity_bar(f"{region} ({count:,} jobs)", count, total, 
                     "#60a5fa" if region=="Luzon" else "#a78bfa" if region=="Visayas" else "#4ade80")
        sleep(.4)

    section_divider()
    st.markdown("#### Why clustering helps")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-tile">
          <div class="metric-num" style="color:#60a5fa;">3</div>
          <div class="metric-lbl">Sub-problems instead of 1 giant one</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-tile">
          <div class="metric-num" style="color:#a78bfa;">3x</div>
          <div class="metric-lbl">Potential parallel speedup</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-tile">
          <div class="metric-num" style="color:#4ade80;">↓</div>
          <div class="metric-lbl">Complexity per worker</div>
        </div>""", unsafe_allow_html=True)

    why_box("A truck in Davao will never deliver to Makati. Regional grouping reflects real logistics geography. "
            "It also means each Ray worker only needs to solve a smaller problem, making parallelism effective.")


def scene_07_truck_assignment():
    scene_card(7, "How Trucks Get Assigned — Greedy Capacity Algorithm",
        "For each delivery job, the engine finds a truck in the same region that has enough remaining capacity. "
        "It picks the truck with the lowest detour cost.")

    joined = TRANSACTIONS.merge(PRODUCTS, on="product_id")
    joined["total_weight_kg"] = joined["quantity"] * joined["weight_kg"]
    luzon_jobs = joined[joined["product_id"].isin(["PRD-A1","PRD-B2","PRD-E5"])].reset_index(drop=True)
    luzon_jobs["destination"] = ["BR-001","BR-007","BR-001"]

    luzon_trucks = TRUCKS[TRUCKS["region"]=="Luzon"].copy()
    remaining = luzon_trucks.set_index("truck_id")["capacity_kg"].to_dict()

    log_placeholder = st.empty()
    cap_placeholder  = st.empty()
    logs = []

    def render_caps():
        with cap_placeholder.container():
            st.markdown("**Live truck capacity remaining:**")
            for tid, cap in remaining.items():
                orig = luzon_trucks.set_index("truck_id").loc[tid, "capacity_kg"]
                capacity_bar(tid, orig - cap, orig)

    render_caps()
    sleep(1)

    for _, job in luzon_jobs.iterrows():
        jid  = job["transaction_id"]
        wt   = job["total_weight_kg"]
        logs.append(f"📦 Evaluating {jid} | {job['product_name']} | Weight: {wt:.0f} kg")
        with log_placeholder.container():
            for l in logs: st.markdown(f'<div class="log-line log-info">{l}</div>', unsafe_allow_html=True)
        sleep(1.2)

        assigned = False
        for tid, cap in remaining.items():
            logs.append(f"&nbsp;&nbsp;&nbsp;🔍 Checking {tid} — remaining: {cap:.0f} kg")
            with log_placeholder.container():
                for l in logs: st.markdown(f'<div class="log-line log-dim">{l}</div>', unsafe_allow_html=True)
            sleep(.9)

            if cap >= wt:
                remaining[tid] -= wt
                logs.append(f"&nbsp;&nbsp;&nbsp;✅ FIT → assigned to {tid}. Capacity drops to {remaining[tid]:.0f} kg")
                with log_placeholder.container():
                    for l in logs: st.markdown(f'<div class="log-line log-ok">{l}</div>', unsafe_allow_html=True)
                render_caps()
                assigned = True
                sleep(1.5)
                break
            else:
                logs.append(f"&nbsp;&nbsp;&nbsp;❌ REJECTED — {wt:.0f} kg > {cap:.0f} kg remaining")
                with log_placeholder.container():
                    for l in logs: st.markdown(f'<div class="log-line log-warn">{l}</div>', unsafe_allow_html=True)
                sleep(.9)

        if not assigned:
            logs.append(f"🚨 No truck can carry {jid} — needs escalation!")
        logs.append("─" * 50)

    why_box("The algorithm always tries trucks in cost order (nearest). If no truck fits, the job is flagged. "
            "This greedy approach runs in O(jobs × trucks) time — fast enough for 20,000 jobs.")


def scene_08_route_ordering():
    scene_card(8, "How the Route Order Is Decided — Nearest Neighbour",
        "After a truck has all its assigned jobs, we still need to decide the <b>order of stops</b>. "
        "We use a nearest-neighbour heuristic: always visit the closest unvisited stop next.")

    stops = [
        {"stop": 1, "branch": "BR-001 (Hub)", "dist_from_prev": 0,  "cumulative_km": 0},
        {"stop": 2, "branch": "BR-002",        "dist_from_prev": 18, "cumulative_km": 18},
        {"stop": 3, "branch": "BR-007",        "dist_from_prev": 31, "cumulative_km": 49},
        {"stop": 4, "branch": "BR-001 (Hub)",  "dist_from_prev": 42, "cumulative_km": 91},
    ]

    route_placeholder = st.empty()
    for i in range(1, len(stops)+1):
        with route_placeholder.container():
            st.markdown("**Route being built stop by stop:**")
            partial = pd.DataFrame(stops[:i])
            st.dataframe(partial, use_container_width=True, hide_index=True)
            if i < len(stops):
                st.markdown(f'<div class="log-line log-info">→ Next: evaluating nearest unvisited branch...</div>',
                            unsafe_allow_html=True)
        sleep(1.5)

    why_box("Assignment = deciding <em>who</em> carries a package. "
            "Routing = deciding <em>in what order</em> stops are visited. "
            "A truck with great assignments but a terrible route order still wastes fuel. "
            "Nearest-neighbour gives a fast, good-enough solution for large fleets.")


def scene_09_parallel():
    scene_card(9, "Why We Use Parallel Processing — Ray Workers",
        "Each region's routing problem is independent. So instead of solving them one at a time, "
        "we send each region to its own Ray worker and run them simultaneously.")

    regions  = ["Luzon", "Visayas", "Mindanao"]
    seq_times = [4.2, 2.1, 1.4]
    par_time  = max(seq_times)

    st.markdown("#### Sequential execution (one region at a time)")
    seq_total = 0
    prog = st.progress(0)
    for i, (r, t) in enumerate(zip(regions, seq_times)):
        badge_cls = REGION_COLORS[r]
        seq_total += t
        st.markdown(f'<span class="badge {badge_cls}">{r}</span> solving... ({t}s)', unsafe_allow_html=True)
        sleep(.5)
        prog.progress((i+1)/len(regions))
    st.markdown(f"**Total sequential time: {seq_total:.1f}s**")

    section_divider()
    st.markdown("#### Parallel execution (all regions at once with Ray)")
    cols = st.columns(3)
    for col, r, t in zip(cols, regions, seq_times):
        with col:
            badge_cls = REGION_COLORS[r]
            st.markdown(f'<span class="badge {badge_cls}">{r}</span>', unsafe_allow_html=True)
            b = st.progress(0)
            for v in np.linspace(0, 1, 8):
                b.progress(float(v))
                sleep(.08)
            st.markdown(f"✅ {t}s")

    st.markdown(f"**Parallel wall-clock time: {par_time}s** (only the slowest region matters)")

    speedup = seq_total / par_time
    st.success(f"⚡ Speedup = {seq_total:.1f}s ÷ {par_time}s = **{speedup:.1f}x faster**")

    why_box("Ray distributes each region to a separate CPU core. The total waiting time equals "
            "the time of the <em>slowest</em> region, not the sum of all regions. "
            "That's why parallel speedup grows with the number of regions.")


def scene_10_benchmark():
    scene_card(10, "Sequential vs Parallel Benchmark",
        "In Benchmark mode, the engine measures real execution time for both approaches "
        "and computes the actual speedup factor.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="timer-panel">
          <div class="timer-val" style="color:#f87171;">7.63 s</div>
          <div class="timer-label">Sequential Time</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="timer-panel">
          <div class="timer-val" style="color:#4ade80;">2.41 s</div>
          <div class="timer-label">Parallel (Ray) Time</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="timer-panel">
          <div class="timer-val" style="color:#f59e0b;">3.17x</div>
          <div class="timer-label">Speedup Factor</div>
        </div>""", unsafe_allow_html=True)

    section_divider()
    st.markdown("#### How to read the speedup")
    pipe_step("🕐", "Sequential: Luzon → Visayas → Mindanao (one after another)")
    pipe_arrow()
    pipe_step("⚡", "Parallel: Luzon + Visayas + Mindanao (all at once using Ray)")
    pipe_arrow()
    pipe_step("📐", "Speedup = Sequential Time ÷ Parallel Time = 7.63 ÷ 2.41 = 3.17x")

    why_box("A 3x speedup means we process in 2.4 seconds what would take 7.6 seconds serially. "
            "For a system handling 20,000 jobs, every second saved directly impacts delivery windows. "
            "Higher speedup factors appear when regions are more equal in size.")


def scene_11_traffic():
    scene_card(11, "What Traffic Does to Routing",
        "The shortest path is not always the fastest path. "
        "The system applies a random traffic multiplier to simulate real-world congestion.")

    st.markdown("#### The same route, three traffic conditions")
    routes = pd.DataFrame([
        {"Route": "BR-001 → BR-002", "Base Distance (km)": 18, "Traffic Multiplier": 1.0,  "Effective Cost": 18.0,  "Condition": "🟢 Clear"},
        {"Route": "BR-001 → BR-002", "Base Distance (km)": 18, "Traffic Multiplier": 1.35, "Effective Cost": 24.3,  "Condition": "🟡 Moderate"},
        {"Route": "BR-001 → BR-002", "Base Distance (km)": 18, "Traffic Multiplier": 1.80, "Effective Cost": 32.4,  "Condition": "🔴 Heavy"},
    ])
    st.dataframe(routes, use_container_width=True, hide_index=True)

    section_divider()
    st.markdown("#### Impact on route selection")
    st.markdown("""
    <div style="color:#94a3b8;font-size:.9rem;line-height:1.9;">
      Under heavy traffic, the routing engine may choose a <b>longer road with less congestion</b> 
      over a shorter but slower road. The decision is based on <b>effective cost</b>, not raw kilometres.
      <br><br>
      Traffic multipliers are sampled randomly per route leg each run, 
      simulating the unpredictability of real delivery conditions.
    </div>""", unsafe_allow_html=True)

    why_box("In real logistics, 80% of delays come from unpredicted congestion. "
            "Simulating traffic during route scoring produces more realistic time estimates "
            "and prevents the engine from always picking the obvious shortest path.")


def scene_12_urgent():
    scene_card(12, "What Happens When an Urgent Request Arrives",
        "A high-priority order can drop in at any time. "
        "The system does not rebuild the entire plan — it finds the best available truck in the region "
        "and appends the urgent leg at the end of its existing route.")

    st.markdown("""
    <div class="urgent-card">
      <div style="color:#ef4444;font-weight:700;font-size:1.05rem;margin-bottom:.5rem;">
        🚨 URGENT REQUEST RECEIVED
      </div>
      <div style="color:#fca5a5;font-size:.9rem;line-height:1.8;">
        Transaction ID: <b>TX-URGENT-999</b><br>
        Origin: <b>BR-002</b> → Destination: <b>BR-016</b><br>
        Weight: <b>500 kg</b> | Region: <b>Luzon</b> | Deadline: <b>1 hr</b>
      </div>
    </div>""", unsafe_allow_html=True)

    sleep(1.5)
    section_divider()
    st.markdown("#### Re-optimization steps")
    
    steps = [
        ("🔍", "Filter existing plan to Luzon routes only"),
        ("📊", "Compute remaining capacity for each Luzon truck"),
        ("✅", "Find trucks with ≥ 500 kg remaining capacity"),
        ("📏", "For each candidate truck, measure detour cost to BR-002"),
        ("📌", "Select truck with lowest detour cost → TRK-L02"),
        ("➕", "Append urgent leg to end of TRK-L02's route (stop_sequence + 1)"),
        ("⚡", "Done — entire re-optimization in < 0.01 seconds"),
    ]

    log_box = st.empty()
    shown = []
    for icon, text in steps:
        shown.append(f"{icon} {text}")
        with log_box.container():
            for s in shown:
                st.markdown(f'<div class="log-line log-info">{s}</div>', unsafe_allow_html=True)
        sleep(.9)

    why_box("The system never stops the fleet for an urgent order. It finds the least-disrupted insertion point "
            "in under a millisecond. This is possible because we only re-evaluate one region, not the whole country.")


def scene_13_metrics():
    scene_card(13, "What the Final Numbers Mean",
        "After the engine runs, it produces a dispatch dashboard. "
        "Here is what each metric tells you — and why it matters.")

    st.markdown("#### Sample output metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-tile"><div class="metric-num">20,000</div>
        <div class="metric-lbl">Total Deliveries Routed</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-tile"><div class="metric-num">384,210</div>
        <div class="metric-lbl">Total Distance (km)</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-tile"><div class="metric-num">47</div>
        <div class="metric-lbl">Trucks Used</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown("""
        <div class="metric-tile"><div class="metric-num" style="color:#f87171;">7.63 s</div>
        <div class="metric-lbl">Sequential Time (baseline)</div></div>""", unsafe_allow_html=True)
    with col5:
        st.markdown("""
        <div class="metric-tile"><div class="metric-num" style="color:#4ade80;">2.41 s</div>
        <div class="metric-lbl">Parallel Time (Ray)</div></div>""", unsafe_allow_html=True)
    with col6:
        st.markdown("""
        <div class="metric-tile"><div class="metric-num" style="color:#f59e0b;">3.17x</div>
        <div class="metric-lbl">Speedup Factor</div></div>""", unsafe_allow_html=True)

    section_divider()
    st.markdown("#### Metric definitions")
    definitions = {
        "Total Deliveries Routed": "How many transaction-level jobs were successfully assigned to a truck and sequenced into a route.",
        "Total Distance (km)": "Sum of all estimated_distance_km across every route leg. Lower is better — it means trucks took efficient paths.",
        "Trucks Used": "How many vehicles from the fleet were actually dispatched. Fewer trucks = better packing efficiency.",
        "Sequential Time": "How long the engine takes if all regions are solved one after another. Used as the baseline.",
        "Parallel Time": "How long the engine takes when all regions are solved simultaneously using Ray workers. The target to beat.",
        "Speedup Factor": "Sequential Time ÷ Parallel Time. A value > 1 means parallel is faster. Higher is better.",
    }
    for metric, defn in definitions.items():
        st.markdown(f"""
        <div style="background:#111827;border:1px solid #1f2937;border-radius:6px;
                    padding:.7rem 1rem;margin-bottom:.5rem;">
          <span style="color:#f59e0b;font-weight:600;">{metric}</span>
          <span style="color:#64748b;"> — </span>
          <span style="color:#94a3b8;font-size:.9rem;">{defn}</span>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────

@st.cache_data
def load_data():
    dp = DataProcessor()
    dist_matrix = dp.load_distance_matrix()
    trucks = dp.load_trucks()
    requests = dp.generate_delivery_requests(20000)
    return dist_matrix, trucks, requests

dist_matrix, trucks, requests = load_data()
dist_dict = build_distance_dict(dist_matrix)

st.title("🚚 BuildConnect: Routing Engine")

st.sidebar.header("System Controls")
run_engine   = st.sidebar.button("1. Run VRP Benchmarks (Seq vs. Par)")
inject_urgent = st.sidebar.button("2. Inject Urgent Request")
run_course   = st.sidebar.checkbox("3. Run Interactive Course Mode")

# ── SCENE SELECTOR (only visible in course mode) ──
if run_course:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Tutorial Navigation**")
    scene_labels = {
        1:  "Scene 1 — The Problem",
        2:  "Scene 2 — Data Tour",
        3:  "Scene 3 — Data Transformation",
        4:  "Scene 4 — Hub Selection",
        5:  "Scene 5 — Why We Sample",
        6:  "Scene 6 — Regional Clustering",
        7:  "Scene 7 — Truck Assignment",
        8:  "Scene 8 — Route Ordering",
        9:  "Scene 9 — Parallel Processing",
        10: "Scene 10 — Benchmark",
        11: "Scene 11 — Traffic Simulation",
        12: "Scene 12 — Urgent Injection",
        13: "Scene 13 — Final Metrics",
    }
    selected_scene = st.sidebar.selectbox(
        "Jump to scene:", options=list(scene_labels.keys()),
        format_func=lambda k: scene_labels[k]
    )
    run_all = st.sidebar.checkbox("▶ Run all scenes in sequence", value=False)


# ── 1. BENCHMARK MODE ──
if run_engine:
    st.info("Benchmarking Sequential vs. Parallel execution...")
    engine = RoutingEngine(requests, trucks, dist_matrix)
    _, seq_time = engine.execute_sequential()
    optimized_routes, par_time = engine.execute_parallel()
    speedup = seq_time / par_time if par_time > 0 else 1
    st.session_state['routes'] = optimized_routes
    st.success("✅ Fleet Optimization Complete!")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sequential Time",     f"{seq_time:.3f} s")
    col2.metric("Parallel (Ray) Time", f"{par_time:.3f} s")
    col3.metric("Speedup Factor",      f"{speedup:.2f}x")
    col4.metric("Total Distance (km)", f"{optimized_routes['estimated_distance_km'].sum():.1f}")
    st.subheader("Optimized Dispatch Plan")
    st.dataframe(optimized_routes.sort_values(by=['region','truck_id','stop_sequence']))


# ── 2. URGENT INJECTION MODE ──
elif inject_urgent:
    if 'routes' not in st.session_state:
        st.error("Please run the VRP Benchmark first to generate routes!")
    else:
        st.warning("🚨 Urgent Request Received! Triggering Dynamic Re-optimization...")
        urgent_job = {
            'transaction_id': 'TX-URGENT-999',
            'origin_branch': 'BR-002',
            'destination_branch': 'BR-016',
            'total_weight_kg': 500,
            'region': 'Luzon'
        }
        start_time = time.time()
        updated_plan = inject_urgent_request(
            st.session_state['routes'], urgent_job, trucks, dist_dict)
        elapsed = time.time() - start_time
        st.session_state['routes'] = updated_plan
        st.success(f"⚡ Fleet dynamically re-routed in {elapsed:.4f} seconds!")
        st.dataframe(updated_plan[updated_plan['request_id'].str.contains("URGENT")])


# ── 3. COURSE / TUTORIAL MODE ──
elif run_course:
    st.header("🎓 Deep-Dive: How the VRP System Works — Scene by Scene")
    st.markdown(
        "Use the **sidebar** to jump to any scene, or check **Run all scenes** to watch the full tutorial. "
        "Every scene shows you the data, the algorithm step, and exactly why it matters."
    )
    st.markdown("---")

    SCENE_FNS = {
        1:  scene_01_problem,
        2:  scene_02_data_tour,
        3:  scene_03_transformation,
        4:  scene_04_hub_selection,
        5:  scene_05_sampling,
        6:  scene_06_clustering,
        7:  scene_07_truck_assignment,
        8:  scene_08_route_ordering,
        9:  scene_09_parallel,
        10: scene_10_benchmark,
        11: scene_11_traffic,
        12: scene_12_urgent,
        13: scene_13_metrics,
    }

    if run_all:
        for s in range(1, 14):
            with st.expander(scene_labels[s], expanded=(s == 1)):
                SCENE_FNS[s]()
    else:
        SCENE_FNS[selected_scene]()

    st.markdown("---")
    st.success("🎉 That's the complete BuildConnect VRP system — from raw CSVs to optimized fleet routes, "
               "parallel processing, and live urgent re-routing.")