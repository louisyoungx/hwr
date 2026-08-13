"""Read-only snapshots for the local foundation-training dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


DASHBOARD_SCHEMA = "hwr.foundation-dashboard/v1"


def load_dashboard_snapshot(run_path: Path) -> dict[str, object]:
    """Read only bounded JSON artifacts; never touch replay shards or weights."""
    run_path = run_path.resolve()
    manifest = _optional_json(run_path / "run-manifest.json")
    metrics = _optional_json(run_path / "metrics/latest.json")
    latest = _optional_json(run_path / "latest.json")
    episodes = _episode_summary(run_path / "episodes.jsonl")
    return {
        "schema_version": DASHBOARD_SCHEMA,
        "run_id": run_path.name,
        "manifest": manifest,
        "metrics": metrics,
        "latest_checkpoint": latest,
        "episodes": episodes,
    }


def _optional_json(path: Path) -> Mapping[str, object] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"dashboard artifact is not an object: {path}")
    return dict(value)


def _episode_summary(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"count": 0, "success_count": 0, "recent": []}
    recent: list[dict[str, object]] = []
    count = 0
    successes = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            count += 1
            successes += int(bool(value.get("success")))
            recent.append(value)
            recent = recent[-12:]
    return {"count": count, "success_count": successes, "recent": recent}


DASHBOARD_HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>HWR 训练监控</title><style>
:root{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:#e8edf2;background:#0d1117}
body{max-width:1180px;margin:auto;padding:24px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}
.card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px;margin-bottom:12px}h1{font-size:22px}h2{font-size:15px;color:#8b949e}
.value{font-size:24px}.ok{color:#3fb950}.bad{color:#f85149}table{width:100%;border-collapse:collapse;font-size:12px}td,th{padding:7px;border-bottom:1px solid #30363d;text-align:left}
pre{white-space:pre-wrap;max-height:360px;overflow:auto;color:#b1bac4}.muted{color:#8b949e}</style></head>
<body><h1>HWR Foundation 训练监控</h1><p id="stamp" class="muted"></p><div id="root"></div>
<script>
const n=(v,d=3)=>typeof v==='number'?v.toFixed(d):(v??'—');
const card=(t,v,c='')=>`<div class="card"><h2>${t}</h2><div class="value ${c}">${v}</div></div>`;
function render(x){const m=x.metrics||{},t=m.training||m.metrics||{},c=m.action_causality||{};
 const eps=x.episodes||{}, stage=m.stage||m.record_type||'等待指标';
 const ratio=c.shuffled_to_true_ratio??t['world/action_causality_ratio'];
 const rows=(eps.recent||[]).map(e=>`<tr><td>${e.episode_index}</td><td>${e.task_id}</td><td>${e.action_source}</td><td>${n(e.episode_return)}</td><td>${e.success?'是':'否'}</td><td>${n(e.safety_intervention_rate)}</td></tr>`).join('');
 document.getElementById('root').innerHTML=`<div class="grid">${card('状态',stage)}${card('Episode',`${eps.count||0} / ${x.manifest?.training_config?.episodes??'—'}`)}${card('更新',m.update_count??'—')}${card('成功',eps.success_count||0,(eps.success_count||0)>0?'ok':'')}${card('动作因果比',n(ratio),ratio>=1.05?'ok':'bad')}${card('世界模型损失',n(t['world/total']))}</div><div class="card"><h2>最近 Episode</h2><table><thead><tr><th>#</th><th>任务</th><th>动作源</th><th>回报</th><th>成功</th><th>安全干预</th></tr></thead><tbody>${rows}</tbody></table></div><div class="card"><h2>最近完整指标</h2><pre>${JSON.stringify(m,null,2)}</pre></div>`;
 document.getElementById('stamp').textContent=`运行：${x.run_id} · 本页每 5 秒读取小型 JSON，不读取 replay/权重 · ${new Date().toLocaleTimeString()}`;}
async function poll(){try{const r=await fetch('/api/snapshot',{cache:'no-store'});render(await r.json())}catch(e){document.getElementById('stamp').textContent=e}setTimeout(poll,5000)}poll();
</script></body></html>"""
