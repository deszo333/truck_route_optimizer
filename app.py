import json
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.data_processor import DataProcessor
from src.dynamic_updater import inject_urgent_request
from src.routing_engine import RoutingEngine, build_distance_dict


try:
    import ray

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    USE_RAY = True
except Exception:
    USE_RAY = False


st.set_page_config(page_title="BuildConnect | Logistics Control Tower", layout="wide")

st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  code {
    font-family: 'JetBrains Mono', monospace;
  }

  [data-testid="stMetricValue"] {
    font-weight: 800;
    font-size: 2rem !important;
  }

  [data-testid="stSidebar"], [data-testid="collapsedControl"] {
    display: none;
  }

  .dataframe {
    font-size: 12px !important;
  }

  h1 {
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #1e293b;
  }

  h2, h3 {
    font-weight: 650;
    color: #334155;
  }

  .topbar-card {
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 14px;
    padding: 0.9rem 1rem;
    background: rgba(15, 23, 42, 0.25);
    color: #e2e8f0;
  }

  .ready-chip {
    display: inline-block;
    padding: 0.45rem 0.8rem;
    border-radius: 999px;
    background: rgba(34, 197, 94, 0.12);
    color: #86efac;
    border: 1px solid rgba(34, 197, 94, 0.28);
    font-size: 0.85rem;
    font-weight: 700;
    animation: pulseGlow 1.5s infinite;
  }

  .waiting-chip {
    display: inline-block;
    padding: 0.45rem 0.8rem;
    border-radius: 999px;
    background: rgba(148, 163, 184, 0.10);
    color: #cbd5e1;
    border: 1px solid rgba(148, 163, 184, 0.20);
    font-size: 0.85rem;
    font-weight: 700;
  }

  @keyframes pulseGlow {
    0%   { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.55); transform: translateY(0px); }
    70%  { box-shadow: 0 0 0 10px rgba(34, 197, 94, 0.00); transform: translateY(-1px); }
    100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.00); transform: translateY(0px); }
  }

  div[data-testid="stButton"] > button[kind="primary"] {
    animation: pulseGlow 1.5s infinite;
  }
</style>
""",
    unsafe_allow_html=True,
)


class _StatusFallback:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        st.info(self.label)
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def write(self, msg):
        st.write(msg)

    def update(self, **kwargs):
        pass


def open_status(label: str, state: str = "running", expanded: bool = True):
    if hasattr(st, "status"):
        return st.status(label, state=state, expanded=expanded)
    return _StatusFallback(label)


def safe_numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if df is None or df.empty or column not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(0)


def make_area_chart(chart_data: pd.DataFrame, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.5))
    x = np.arange(len(chart_data))

    for col in chart_data.columns:
        values = pd.to_numeric(chart_data[col], errors="coerce").fillna(0).to_numpy()
        ax.plot(x, values, linewidth=1.8, label=col)
        ax.fill_between(x, values, alpha=0.12)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Time Slice")
    ax.set_ylabel("Relative Load")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", ncol=min(3, len(chart_data.columns)), frameon=False)
    st.pyplot(fig, clear_figure=True, use_container_width=True)


def make_bar_chart(series: pd.Series, title: str, xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    if series is None or series.empty:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
    else:
        series.plot(kind="bar", ax=ax)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(axis="x", rotation=30)
    st.pyplot(fig, clear_figure=True, use_container_width=True)


def make_line_chart(series: pd.Series, title: str, xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    if series is None or series.empty:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
    else:
        series.sort_index().plot(kind="line", ax=ax, linewidth=2)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
    st.pyplot(fig, clear_figure=True, use_container_width=True)


def normalize_route_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    normalized = df.copy()
    rename_map = {}

    if "origin_branch" in normalized.columns and "origin" not in normalized.columns:
        rename_map["origin_branch"] = "origin"
    if "destination_branch" in normalized.columns and "destination" not in normalized.columns:
        rename_map["destination_branch"] = "destination"

    if rename_map:
        normalized = normalized.rename(columns=rename_map)

    return normalized


def get_available_regions(requests_df: pd.DataFrame, routes_df: pd.DataFrame | None = None):
    regions = []
    if routes_df is not None and isinstance(routes_df, pd.DataFrame) and "region" in routes_df.columns:
        regions = [str(x) for x in routes_df["region"].dropna().unique().tolist()]
    if not regions and "region" in requests_df.columns:
        regions = [str(x) for x in requests_df["region"].dropna().unique().tolist()]
    return regions or ["Luzon"]


def get_branch_candidates(distance_matrix: pd.DataFrame, requests_df: pd.DataFrame):
    branch_candidates = []

    if isinstance(distance_matrix, pd.DataFrame):
        cols = set(distance_matrix.columns)
        if {"origin_branch", "destination_branch"}.issubset(cols):
            branch_candidates = sorted(
                set(distance_matrix["origin_branch"].astype(str).dropna().tolist())
                | set(distance_matrix["destination_branch"].astype(str).dropna().tolist())
            )

    if not branch_candidates and "origin_branch" in requests_df.columns:
        branch_candidates = list(
            pd.unique(
                pd.concat(
                    [
                        requests_df["origin_branch"].astype(str),
                        requests_df["destination_branch"].astype(str),
                    ],
                    ignore_index=True,
                )
            )
        )

    if not branch_candidates:
        branch_candidates = [f"BR-{i:03}" for i in range(1, 21)]

    return branch_candidates


@st.cache_data(show_spinner=False)
def load_data():
    dp = DataProcessor()
    dist_matrix = dp.load_distance_matrix()
    trucks = dp.load_trucks()
    requests = dp.generate_delivery_requests(12000)
    return dist_matrix, trucks, requests


dist_matrix, trucks, requests = load_data()
dist_dict = build_distance_dict(dist_matrix)

routes_ready = (
    "routes" in st.session_state
    and isinstance(st.session_state["routes"], pd.DataFrame)
    and not st.session_state["routes"].empty
)

st.markdown("# LYKA HAYOP")

col_run, col_urgent, col_reset, col_state = st.columns([1, 1, 1, 5])

with col_run:
    run_engine = st.button("Optimize Fleet", use_container_width=True)

with col_urgent:
    if routes_ready:
        inject_urgent = st.button("Urgent Dispatch", type="primary", use_container_width=True)
        st.markdown(
            '<div class="ready-chip" style="margin-top:0.35rem;">Urgent dispatch ready</div>',
            unsafe_allow_html=True,
        )
    else:
        inject_urgent = False

with col_reset:
    reset_clicked = st.button("Reset Session", type="secondary", use_container_width=True)

with col_state:
    st.markdown(
        '<div class="topbar-card">'
        f'<strong>Compute Engine:</strong> {"Ray Parallel" if USE_RAY else "Serial"}'
        f' &nbsp; | &nbsp; <strong>Routes Ready:</strong> {"Yes" if routes_ready else "No"}'
        "</div>",
        unsafe_allow_html=True,
    )

if reset_clicked:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


if run_engine:
    with open_status("Running VRP Solvers...", expanded=True) as status:
        engine = RoutingEngine(requests, trucks, dist_matrix)

        st.write("Measuring Sequential Performance...")
        _, seq_time = engine.execute_sequential()

        st.write("Executing Parallel Optimization...")
        if USE_RAY:
            optimized_routes, par_time = engine.execute_parallel()
        else:
            optimized_routes, par_time = engine.execute_sequential()

        if not isinstance(optimized_routes, pd.DataFrame):
            optimized_routes = pd.DataFrame(optimized_routes)

        st.session_state["routes"] = optimized_routes
        st.session_state["benchmarks"] = {
            "seq_time": seq_time,
            "par_time": par_time,
            "speedup": seq_time / par_time if par_time > 0 else 1,
        }

        status.update(label="Optimization Complete", state="complete", expanded=False)

    st.rerun()


if inject_urgent:
    if "routes" not in st.session_state:
        st.warning("Run Optimize Fleet first.")
    else:
        routes_state = normalize_route_schema(st.session_state["routes"])
        region_choices = get_available_regions(requests, routes_state)
        branch_candidates = get_branch_candidates(dist_matrix, requests)

        origin_branch = random.choice(branch_candidates)
        destination_pool = [b for b in branch_candidates if b != origin_branch] or branch_candidates
        destination_branch = random.choice(destination_pool)
        urgent_region = random.choice(region_choices)

        urgent_job = {
            "transaction_id": f"URGENT-{random.randint(1000, 9999)}",
            "origin_branch": origin_branch,
            "destination_branch": destination_branch,
            "total_weight_kg": 750,
            "region": urgent_region,
        }

        with open_status("Injecting urgent request...", expanded=False) as status:
            start_time = time.time()
            routes_before = len(routes_state)
            updated_plan = inject_urgent_request(routes_state, urgent_job, trucks, dist_dict)
            routes_after = len(updated_plan)
            elapsed = time.time() - start_time

            urgent_mask = pd.Series(False, index=updated_plan.index)
            if "request_id" in updated_plan.columns:
                rid = updated_plan["request_id"].astype(str)
                urgent_mask = rid.str.contains("URGENT", case=False, na=False) | rid.str.contains(
                    urgent_job["transaction_id"], case=False, na=False
                )

            urgent_truck_id = None
            if urgent_mask.any() and "truck_id" in updated_plan.columns:
                urgent_truck_id = str(updated_plan.loc[urgent_mask, "truck_id"].iloc[-1])

            st.session_state["routes"] = updated_plan
            st.session_state["last_urgent_time"] = elapsed
            st.session_state["urgent_flash_until"] = time.time() + 8
            st.session_state["urgent_request_id"] = urgent_job["transaction_id"]
            st.session_state["last_urgent_job"] = urgent_job
            st.session_state["urgent_route_delta"] = (routes_before, routes_after)
            st.session_state["urgent_truck_id"] = urgent_truck_id
            status.update(label="Urgent request injected", state="complete", expanded=False)

        if st.session_state.get("urgent_truck_id"):
            st.success(
                f'Urgent job inserted into truck {st.session_state["urgent_truck_id"]}. '
                f"Routes: {routes_before} → {routes_after}"
            )
        else:
            st.success(f"Urgent job inserted. Routes: {routes_before} → {routes_after}")

        st.info("Urgent route highlight is active for 8 seconds.")


def render_neural_simulation(df: pd.DataFrame, urgent_until=None, urgent_truck_id=None):
    if df is None or df.empty:
        st.info("No optimized routes are available for simulation yet.")
        return

    normalized = normalize_route_schema(df)
    required = {"origin", "destination", "truck_id", "stop_sequence"}
    missing = required - set(normalized.columns)
    if missing:
        st.warning(f"Simulation data is missing required columns: {', '.join(sorted(missing))}.")
        st.write("Available columns:", list(normalized.columns))
        return

    if "request_id" in normalized.columns:
        urgent_mask_all = normalized["request_id"].astype(str).str.contains("URGENT", case=False, na=False)
    else:
        urgent_mask_all = pd.Series(False, index=normalized.index)

    urgent_rows = normalized[urgent_mask_all].copy()

    sample_limit = 800
    non_urgent_rows = normalized[~urgent_mask_all].head(max(0, sample_limit - len(urgent_rows))).copy()

    sim_data = pd.concat([non_urgent_rows, urgent_rows], ignore_index=True)
    if sim_data.empty:
        sim_data = normalized.head(sample_limit).copy()

    if "request_id" in sim_data.columns:
        sim_data["is_urgent"] = sim_data["request_id"].astype(str).str.contains("URGENT", case=False, na=False)
    else:
        sim_data["is_urgent"] = False

    if urgent_truck_id is not None and "truck_id" in sim_data.columns:
        sim_data["is_urgent"] = sim_data["is_urgent"] | (sim_data["truck_id"].astype(str) == str(urgent_truck_id))

    if "weight_kg" in sim_data.columns:
        sim_data["weight_kg"] = pd.to_numeric(sim_data["weight_kg"], errors="coerce").fillna(0)
    else:
        sim_data["weight_kg"] = 0.0

    active_trucks = sim_data["truck_id"].nunique()
    active_routes = len(sim_data)
    urgent_routes = int(sim_data["is_urgent"].sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Active Trucks", f"{active_trucks}")
    m2.metric("Active Routes", f"{active_routes}")
    m3.metric("Urgent Routes", f"{urgent_routes}")
    if urgent_until is not None:
        remaining = max(0.0, urgent_until - time.time())
        m4.metric("Urgent Highlight", f"{remaining:.1f}s")
    else:
        m4.metric("Urgent Highlight", "Idle")

    unique_branches = sorted(
        set(sim_data["origin"].astype(str)) | set(sim_data["destination"].astype(str))
    )

    rng = random.Random(42)
    coords = {b: {"x": rng.randint(60, 920), "y": rng.randint(60, 480)} for b in unique_branches}

    routes_json = []
    for truck_id, group in sim_data.groupby("truck_id"):
        path = []

        if "request_id" in group.columns:
            ordered = group.sort_values(["stop_sequence", "request_id"])
        else:
            ordered = group.sort_values(["stop_sequence"])

        truck_is_urgent = False
        if urgent_truck_id is not None and str(truck_id) == str(urgent_truck_id):
            truck_is_urgent = True

        for _, row in ordered.iterrows():
            origin = str(row["origin"])
            dest = str(row["destination"])

            if origin not in coords:
                coords[origin] = {"x": rng.randint(60, 920), "y": rng.randint(60, 480)}
            if dest not in coords:
                coords[dest] = {"x": rng.randint(60, 920), "y": rng.randint(60, 480)}

            route_is_urgent = bool(row.get("is_urgent", False)) or truck_is_urgent

            path.append(
                {
                    "origin": origin,
                    "dest": dest,
                    "ox": coords[origin]["x"],
                    "oy": coords[origin]["y"],
                    "dx": coords[dest]["x"],
                    "dy": coords[dest]["y"],
                    "urgent": route_is_urgent,
                }
            )

        routes_json.append(
            {
                "truck_id": str(truck_id),
                "path": path,
                "stops": len(path),
                "urgent_stops": int(sum(1 for p in path if p["urgent"])),
                "is_urgent_truck": truck_is_urgent,
            }
        )

    urgent_until_ms = int((urgent_until or 0) * 1000)
    urgent_truck_json = json.dumps("" if urgent_truck_id is None else str(urgent_truck_id))

    html_template = """
    <div style="background: #0f172a; border-radius: 16px; padding: 16px; border: 1px solid #334155; overflow: hidden;">
        <div style="display:flex; gap:12px; flex-wrap:wrap; align-items:center; margin-bottom:12px; color:#e2e8f0; font-family: Inter, sans-serif;">
            <div style="padding:6px 10px; background: rgba(30,41,59,0.7); border-radius:999px;">Active trucks: __ACTIVE_TRUCKS__</div>
            <div style="padding:6px 10px; background: rgba(30,41,59,0.7); border-radius:999px;">Routes: __ACTIVE_ROUTES__</div>
            <div style="padding:6px 10px; background: rgba(30,41,59,0.7); border-radius:999px;">Urgent routes: __URGENT_ROUTES__</div>
            <div style="padding:6px 10px; background: rgba(30,41,59,0.7); border-radius:999px;">Urgent truck: __URGENT_TRUCK__</div>
        </div>
        <canvas id="simCanvas" style="width:100%; display:block; border-radius: 12px; background: radial-gradient(circle at top, #13203a, #0b1020);"></canvas>
    </div>
    <script>
        const canvas = document.getElementById('simCanvas');
        const ctx = canvas.getContext('2d');
        const routes = __ROUTES_JSON__;
        const urgentUntilMs = __URGENT_UNTIL__;
        const urgentTruckId = __URGENT_TRUCK_ID__;

        function resizeCanvas() {
            const width = canvas.parentElement.clientWidth - 2;
            const height = 560;
            const dpr = window.devicePixelRatio || 1;
            canvas.width = Math.floor(width * dpr);
            canvas.height = Math.floor(height * dpr);
            canvas.style.width = '100%';
            canvas.style.height = height + 'px';
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        }

        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);

        let trucks = routes.map(r => ({
            truckId: r.truck_id,
            path: r.path,
            segmentIndex: 0,
            progress: Math.random(),
            speed: 0.004 + (Math.random() * 0.006),
            color: 'hsl(' + (Math.random() * 360) + ', 75%, 62%)',
            urgent: r.is_urgent_truck || r.urgent_stops > 0,
            urgentStops: r.urgent_stops,
            stops: r.stops
        }));

        function drawGrid(w, h) {
            ctx.strokeStyle = 'rgba(148, 163, 184, 0.10)';
            ctx.lineWidth = 1;
            const step = 40;
            for (let x = 0; x < w; x += step) {
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, h);
                ctx.stroke();
            }
            for (let y = 0; y < h; y += step) {
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(w, y);
                ctx.stroke();
            }
        }

        function drawNodes() {
            const nodes = {};
            routes.forEach(r => {
                r.path.forEach(p => {
                    nodes[p.origin] = {x: p.ox, y: p.oy};
                    nodes[p.dest] = {x: p.dx, y: p.dy};
                });
            });

            ctx.font = '12px Inter, sans-serif';
            Object.entries(nodes).forEach(([name, pos], idx) => {
                const radius = 4;
                ctx.beginPath();
                ctx.fillStyle = idx % 3 === 0 ? '#60a5fa' : (idx % 3 === 1 ? '#4ade80' : '#a855f7');
                ctx.shadowBlur = 12;
                ctx.shadowColor = ctx.fillStyle;
                ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
                ctx.fill();
                ctx.shadowBlur = 0;

                ctx.fillStyle = 'rgba(226, 232, 240, 0.85)';
                ctx.fillText(name, pos.x + 8, pos.y - 8);
            });
        }

        function drawRoutes() {
            const urgentActive = Date.now() < urgentUntilMs;
            routes.forEach(r => {
                r.path.forEach((p) => {
                    const isUrgent = p.urgent || urgentActive || r.is_urgent_truck;
                    ctx.beginPath();
                    ctx.setLineDash(isUrgent ? [] : [6, 7]);
                    ctx.lineWidth = isUrgent ? 3.2 : 1.2;
                    ctx.strokeStyle = isUrgent
                        ? 'rgba(251, 191, 36, 0.95)'
                        : 'rgba(148, 163, 184, 0.18)';
                    ctx.shadowBlur = isUrgent ? 16 : 0;
                    ctx.shadowColor = isUrgent ? 'rgba(251, 191, 36, 0.85)' : 'transparent';
                    ctx.moveTo(p.ox, p.oy);
                    ctx.lineTo(p.dx, p.dy);
                    ctx.stroke();
                });
            });
            ctx.setLineDash([]);
            ctx.shadowBlur = 0;
        }

        function drawTrucks() {
            const urgentActive = Date.now() < urgentUntilMs;
            trucks.forEach(t => {
                const seg = t.path[t.segmentIndex];
                if (!seg) return;

                const x = seg.ox + (seg.dx - seg.ox) * t.progress;
                const y = seg.oy + (seg.dy - seg.oy) * t.progress;
                const urgentNow = urgentActive && (seg.urgent || t.urgent || (urgentTruckId && String(t.truckId) === String(urgentTruckId)));
                const radius = urgentNow ? 6.5 : 4.2;

                ctx.beginPath();
                ctx.shadowBlur = urgentNow ? 18 : 10;
                ctx.shadowColor = urgentNow ? 'rgba(251, 191, 36, 0.95)' : t.color;
                ctx.fillStyle = urgentNow ? '#fbbf24' : t.color;
                ctx.arc(x, y, radius, 0, Math.PI * 2);
                ctx.fill();

                if (urgentNow) {
                    ctx.beginPath();
                    ctx.strokeStyle = 'rgba(251, 191, 36, 0.6)';
                    ctx.lineWidth = 1.5;
                    ctx.arc(x, y, radius + 8 + Math.sin(Date.now() / 140) * 1.5, 0, Math.PI * 2);
                    ctx.stroke();
                }

                ctx.shadowBlur = 0;

                t.progress += t.speed;
                if (t.progress >= 1) {
                    t.progress = 0;
                    if (t.path.length > 0) {
                        t.segmentIndex = (t.segmentIndex + 1) % t.path.length;
                    }
                }
            });
        }

        function drawLegend(w, h) {
            const urgentActive = Date.now() < urgentUntilMs;
            const x = 16;
            const y = h - 58;
            ctx.fillStyle = 'rgba(15, 23, 42, 0.72)';
            ctx.strokeStyle = 'rgba(148, 163, 184, 0.20)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.rect(x, y, 290, 40);
            ctx.fill();
            ctx.stroke();

            ctx.font = '12px Inter, sans-serif';
            ctx.fillStyle = 'rgba(226, 232, 240, 0.88)';
            ctx.fillText('Urgent highlight: ' + (urgentActive ? 'active' : 'idle'), x + 12, y + 16);
            ctx.fillText('Moving trucks: ' + trucks.length, x + 12, y + 30);
            ctx.fillText('Urgent truck: ' + (urgentTruckId ? urgentTruckId : 'none'), x + 150, y + 30);
        }

        function animate() {
            const w = canvas.width / (window.devicePixelRatio || 1);
            const h = canvas.height / (window.devicePixelRatio || 1);
            ctx.clearRect(0, 0, w, h);
            drawGrid(w, h);
            drawRoutes();
            drawNodes();
            drawTrucks();
            drawLegend(w, h);
            requestAnimationFrame(animate);
        }

        animate();
    </script>
    """

    canvas_html = (
        html_template
        .replace("__ROUTES_JSON__", json.dumps(routes_json))
        .replace("__URGENT_UNTIL__", str(urgent_until_ms))
        .replace("__URGENT_TRUCK_ID__", urgent_truck_json)
        .replace("__ACTIVE_TRUCKS__", str(active_trucks))
        .replace("__ACTIVE_ROUTES__", str(active_routes))
        .replace("__URGENT_ROUTES__", str(urgent_routes))
        .replace("__URGENT_TRUCK__", urgent_truck_id if urgent_truck_id else "none")
    )

    st.components.v1.html(canvas_html, height=650, scrolling=False)


st.caption("Optimize the fleet first, then inject an urgent dispatch into the active plan.")

if "routes" not in st.session_state:
    st.markdown("### Operational Readiness")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Active Fleet", len(trucks), "Ready")
    m2.metric("Queue Depth", len(requests), "Pending")
    m3.metric("Service Regions", "Luzon, Visayas, Mindanao")

    st.info("System initialized. Awaiting fleet optimization command.")
    st.divider()

    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader("Network Load Distribution")
        chart_data = pd.DataFrame(
            {
                "Luzon": np.linspace(12, 15, 20) + np.sin(np.linspace(0, 3, 20)),
                "Visayas": np.linspace(11, 14, 20) + np.cos(np.linspace(0, 3, 20)),
                "Mindanao": np.linspace(9, 7, 20) + np.sin(np.linspace(0, 4, 20)) * 0.8,
            }
        )
        make_area_chart(chart_data, "Network Load Distribution")

    with c2:
        st.subheader("Fleet Composition")
        total_trucks = max(len(trucks), 1)
        base_count = total_trucks // 3
        remainder = total_trucks % 3
        counts = [base_count + (1 if i < remainder else 0) for i in range(3)]

        fleet_df = pd.DataFrame(
            {
                "Type": ["Light", "Medium", "Heavy"],
                "Count": counts,
                "Availability": ["100%", "100%", "100%"],
            }
        )
        st.table(fleet_df)

else:
    routes_df = normalize_route_schema(st.session_state["routes"].copy())

    if "benchmarks" in st.session_state:
        b = st.session_state["benchmarks"]
        total_payload = safe_numeric_series(routes_df, "weight_kg").sum() / 1000.0
        network_coverage = safe_numeric_series(routes_df, "estimated_distance_km").sum()
        used_trucks = routes_df["truck_id"].nunique() if "truck_id" in routes_df.columns else 0
        fleet_efficiency = min(100, int((used_trucks / max(len(trucks), 1)) * 100))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Compute Speedup", f"{b['speedup']:.2f}x", f"{b['par_time']:.3f}s Parallel")
        m2.metric("Total Payload", f"{total_payload:.1f}T")
        m3.metric("Fleet Efficiency", f"{fleet_efficiency}%", "Optimized")
        m4.metric("Network Coverage", f"{network_coverage:,.0f} km")

    if "last_urgent_time" in st.session_state:
        st.warning(
            f"DYNAMIC RE-ROUTE ACTIVE: job synchronized in {st.session_state['last_urgent_time']:.4f}s"
        )

    if st.session_state.get("urgent_truck_id"):
        st.info(f'Urgent job assigned to truck {st.session_state["urgent_truck_id"]}')

    csv_bytes = routes_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download optimized routes CSV",
        data=csv_bytes,
        file_name="optimized_routes.csv",
        mime="text/csv",
        use_container_width=False,
    )

    tab1, tab2, tab3 = st.tabs(["Dispatch Schedule", "Live Simulation", "Route Analytics"])

    with tab1:
        sort_cols = [c for c in ["region", "truck_id", "stop_sequence"] if c in routes_df.columns]
        display_df = routes_df.sort_values(by=sort_cols) if sort_cols else routes_df
        st.dataframe(display_df, use_container_width=True, height=500)

    with tab2:
        st.subheader("Neural Network Simulation")
        st.caption("Active fleet nodes navigating optimized paths in real-time.")
        render_neural_simulation(
            routes_df,
            st.session_state.get("urgent_flash_until"),
            st.session_state.get("urgent_truck_id"),
        )

    with tab3:
        col_a, col_b = st.columns(2)

        with col_a:
            st.write("Route Distance by Region")
            if "region" in routes_df.columns:
                regional_dist = (
                    routes_df.assign(
                        estimated_distance_km=safe_numeric_series(routes_df, "estimated_distance_km")
                    )
                    .groupby("region")["estimated_distance_km"]
                    .sum()
                    .sort_values(ascending=False)
                )
            else:
                regional_dist = safe_numeric_series(routes_df, "estimated_distance_km")

            make_bar_chart(regional_dist, "Route Distance by Region", "Region", "Estimated Distance (km)")

        with col_b:
            st.write("Truck Utilization")
            if "truck_id" in routes_df.columns:
                util = routes_df.groupby("truck_id").size()
            else:
                util = pd.Series(dtype=float)
            make_line_chart(util, "Truck Utilization", "Truck ID", "Stops")