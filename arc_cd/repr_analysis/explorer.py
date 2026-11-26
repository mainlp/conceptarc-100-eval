"""Portable, offline task-map explorer with embedded ARC support grids."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from arc_cd.data_prep import Grid
from arc_cd.eval.visualize import ARC_COLORS
from arc_cd.repr_analysis.explorer_pdf import EXPLORER_PDF_SCRIPT
from arc_cd.repr_analysis.metadata_explorer_ui import (
    COMPOSITION_HTML,
    METADATA_CONTROLS,
    METADATA_CSS,
    METADATA_SCRIPT,
)
from arc_cd.repr_analysis.population import POPULATIONS
from arc_cd.repr_analysis.spec_metadata import load_taxonomy, validate_saved_metadata


def pack_support_grid(grid: Grid, task_id: str) -> dict[str, int | str]:
    """Encode a support grid in the explorer's lossless row-major format.

    Args:
        grid (Grid): Support input or output.
        task_id (str): Task identifier used in validation errors.

    Returns:
        dict[str, int | str]: Height, width, and cell colors.
    """
    values = np.asarray(grid)
    if values.ndim != 2 or not values.size or not np.isin(values, np.arange(10)).all():
        raise ValueError(f"Invalid ARC grid in {task_id}")
    return {
        "rows": values.shape[0],
        "cols": values.shape[1],
        "cells": "".join(str(int(cell)) for cell in values.flat),
    }


def save_explorer_supports(directory: Path, records: list[dict[str, Any]]) -> None:
    """Save aligned support provenance with compact, lossless grid encoding.

    Args:
        directory (Path): Staging directory.
        records (list[dict[str, Any]]): Validated version manifest in archive order.
    """
    packed = []
    for record in records:
        supports = [
            {
                name: pack_support_grid(pair[name], record["case_id"])
                for name in ("input", "output")
            }
            for pair in record["support_examples"]
        ]
        packed.append(
            {
                "task_id": record["task_id"],
                "case_id": record["case_id"],
                "source_path": record["source_path"],
                "parent_task_id": record["metadata"]["parent_task_id"],
                "support_set_id": record["metadata"]["support_set_id"],
                "selected_token_indices": record["selected_token_indices"],
                "supports": supports,
            }
        )
    (directory / "supports.json").write_text(
        json.dumps(packed, separators=(",", ":"), ensure_ascii=False), encoding="utf-8"
    )


def load_explorer_data(directory: Path) -> dict[str, Any]:
    """Load portable explorer artifacts and validate their shared identity axes.

    Args:
        directory (Path): Completed or staged embedding/metadata explorer.

    Returns:
        dict[str, Any]: Aligned display data, including optional task metadata.
    """
    config = json.loads((directory / "run_config.json").read_text())
    if (
        config["analysis"] not in ("dimensionality_reduction", "metadata_explorer")
        or config["schema_version"] != 1
    ):
        raise ValueError("Expected a dimensionality-reduction analysis.")
    axes = json.loads((directory / "axes.json").read_text())
    supports = json.loads((directory / "supports.json").read_text())
    n_tasks = len(axes["task_ids"])
    if (
        not n_tasks
        or len(set(axes["task_ids"])) != n_tasks
        or len(axes["labels"]) != n_tasks
    ):
        raise ValueError("Invalid parent-task axes.")
    for key in ("version_rows", "version_task_ids", "support_set_ids", "case_ids"):
        if len(axes[key]) != n_tasks or any(len(row) != 8 for row in axes[key]):
            raise ValueError(f"Invalid eight-variant axis: {key}")
    for key in ("positions", "layers", "methods", "concepts"):
        if not axes[key] or len(set(axes[key])) != len(axes[key]):
            raise ValueError(f"Invalid or duplicate {key} axis.")
    if (
        not set(axes["positions"]) <= {"S1", "S2", "S3"}
        or not set(axes["methods"]) <= {"umap", "pca"}
        or any(type(layer) is not int or layer < 0 for layer in axes["layers"])
    ):
        raise ValueError("Unsupported position/method/layer axis.")
    if any(
        type(label) is not int or not 0 <= label < len(axes["concepts"])
        for label in axes["labels"]
    ):
        raise ValueError("Invalid concept label axis.")
    population = config.get("population")
    if population != axes.get("population"):
        raise ValueError("Explorer population differs between configuration and axes.")
    if population is not None and (
        not isinstance(population, dict)
        or population.get("name") not in POPULATIONS
        or population.get("concepts") != axes["concepts"]
        or population.get("num_parents") != n_tasks
        or population.get("num_versions") != len(supports)
    ):
        raise ValueError("Explorer population does not match the saved task axes.")
    flattened = [row for rows in axes["version_rows"] for row in rows]
    if any(type(row) is not int for row in flattened) or sorted(flattened) != list(
        range(len(supports))
    ):
        raise ValueError("Support rows must form an aligned one-to-one version axis.")
    for task, rows in enumerate(axes["version_rows"]):
        for version, row in enumerate(rows):
            if (
                supports[row]["task_id"] != axes["version_task_ids"][task][version]
                or supports[row]["parent_task_id"] != axes["task_ids"][task]
                or supports[row]["case_id"] != axes["case_ids"][task][version]
                or supports[row]["support_set_id"]
                != axes["support_set_ids"][task][version]
            ):
                raise ValueError(
                    "Explorer support rows do not align with task identities."
                )
    frames = []
    for position in axes["positions"]:
        for layer in axes["layers"]:
            with np.load(
                directory / position / f"embedding_layer_{layer}.npz",
                allow_pickle=False,
            ) as archive:
                coordinates = archive["coordinates"]
                if coordinates.shape != (
                    len(axes["methods"]),
                    len(axes["task_ids"]),
                    2,
                ):
                    raise ValueError(
                        "Embedding coordinates do not align with saved axes."
                    )
                for key in (
                    "cluster_labels",
                    "cluster_strengths",
                    "representative_versions",
                    "within_task_dispersion",
                    "valid_vectors",
                ):
                    if archive[key].shape != (n_tasks,):
                        raise ValueError(
                            f"Embedding {key} does not align with task axes."
                        )
                if (
                    np.isinf(coordinates).any()
                    or not np.equal(
                        np.isfinite(coordinates).all(axis=2),
                        np.isfinite(coordinates).any(axis=2),
                    ).all()
                ):
                    raise ValueError(
                        "Embedding coordinates must be finite pairs or missing pairs."
                    )
                for key, lower, upper in (
                    ("representative_versions", -1, 7),
                    ("cluster_labels", -2, n_tasks - 1),
                ):
                    if (
                        archive[key].dtype.kind not in "iu"
                        or ((archive[key] < lower) | (archive[key] > upper)).any()
                    ):
                        raise ValueError(f"Invalid embedding {key}.")
                frames.append(
                    {
                        "position": position,
                        "layer": layer,
                        **{key: archive[key].tolist() for key in archive.files},
                    }
                )
    data = {
        "dataset": config["extraction_config"]["dataset"]["id"],
        "model": config["extraction_config"]["extraction"]["model_name"],
        "axes": axes,
        "supports": supports,
        "frames": frames,
        "settings": config.get("embedding_arguments", config["arguments"]),
        "palette": ["#%02x%02x%02x" % ARC_COLORS[i] for i in range(10)],
    }
    if config["analysis"] == "metadata_explorer":
        metadata = json.loads((directory / "task_metadata.json").read_text())
        taxonomy = load_taxonomy(directory / "taxonomy.json")
        validate_saved_metadata(metadata, taxonomy)
        if config.get("taxonomy_version") != taxonomy["version"]:
            raise ValueError("Run configuration conflicts with the taxonomy version.")
        if [row["task_id"] for row in metadata["records"]] != axes["task_ids"]:
            raise ValueError("Metadata parents do not align with embedding axes.")
        if any(
            row["concept_name"] != axes["concepts"][axes["labels"][i]]
            for i, row in enumerate(metadata["records"])
        ):
            raise ValueError("Metadata concepts do not align with embedding labels.")
        data["metadata"] = metadata
    return data


def write_explorer(directory: Path) -> Path:
    """Regenerate a standalone offline explorer from saved numerical artifacts.

    Args:
        directory (Path): Completed or staged embedding/metadata explorer.

    Returns:
        Path: HTML containing scripts, coordinates, metadata, and support grids.
    """
    data = load_explorer_data(directory)
    # Escape HTML-significant characters so IDs cannot end the embedded script.
    payload = json.dumps(
        data, separators=(",", ":"), ensure_ascii=False, allow_nan=True
    )
    # Numerical missingness must be JSON null, including inside coordinate lists.
    payload = json.dumps(
        json.loads(payload, parse_constant=lambda _: None),
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    payload = (
        payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    )
    path = directory / "explorer.html"
    path.write_text(
        HTML_TEMPLATE.replace("__METADATA_CSS__", METADATA_CSS)
        .replace("__METADATA_CONTROLS__", METADATA_CONTROLS)
        .replace("__COMPOSITION_HTML__", COMPOSITION_HTML)
        .replace("__METADATA_SCRIPT__", METADATA_SCRIPT)
        .replace("__EXPLORER_PDF_SCRIPT__", EXPLORER_PDF_SCRIPT)
        .replace("__EXPLORER_DATA__", payload),
        encoding="utf-8",
    )
    return path


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ConceptARC · Task map</title>
<style>
:root{color-scheme:light;--ink:#162737;--muted:#596b79;--line:#dce4e9;--accent:#176c70}
*{box-sizing:border-box}body{margin:0;background:#f5f7f9;color:var(--ink);font:15px system-ui,sans-serif}
header{background:#152e40;color:white;padding:24px 32px}h1{font-size:25px;margin:0 0 7px}h2{font-size:19px;margin:0 0 12px}h3{font-size:15px;margin:0 0 8px;overflow-wrap:anywhere}
header p{margin:0;color:#c6d6e1}.layout{display:grid;grid-template-columns:235px minmax(0,1fr);gap:22px;max-width:1900px;margin:auto;padding:24px;align-items:start}
main{display:grid;grid-template-columns:minmax(0,1fr) minmax(340px,.65fr);gap:20px;min-width:0;align-items:start;position:sticky;top:20px}
#gallery{display:flex;flex-direction:column;min-width:0;max-height:calc(100vh - 155px)}#gallery>.toolbar{flex-shrink:0}#gallery>.toolbar>div:first-child{flex-basis:100%;min-width:0}#gallery h2{overflow-wrap:anywhere}#gallery .cards{min-height:0;overflow-y:auto;scrollbar-gutter:stable;padding-right:6px;align-content:start}.card .small{overflow-wrap:anywhere}
aside{min-width:0}.panel{background:white;border:1px solid var(--line);border-radius:12px;padding:18px;margin-bottom:16px}
.controls{display:grid;grid-template-columns:1fr 1fr;gap:12px}label{display:block;font-size:13px;color:var(--muted)}select{display:block;width:100%;margin-top:5px;padding:8px;border:1px solid #b9c9d4;border-radius:6px;background:white;color:var(--ink)}
button{font:inherit;cursor:pointer;border:1px solid #c4d2db;border-radius:6px;background:white;color:var(--ink);padding:7px 10px}button:hover{background:#eaf2f4}button:focus-visible,select:focus-visible{outline:3px solid #e8a624;outline-offset:2px}button:disabled{opacity:.4;cursor:default}
.small{font-size:12px;color:var(--muted);line-height:1.6}.note{font-size:13px;line-height:1.55;color:var(--muted)}.group{display:flex;align-items:center;gap:8px;width:100%;margin:5px 0;text-align:left}.group span:last-child{margin-left:auto}.swatch{display:inline-block;width:10px;height:10px;border-radius:50%;flex-shrink:0}
#groups{max-height:410px;overflow:auto}#legend{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}#legend button{font-size:12px;padding:4px 8px}
#plot{display:block;width:100%;height:auto;max-height:650px;overflow:hidden;touch-action:none;user-select:none;border:1px solid #edf0f4;border-radius:8px;background:#fcfdff}circle{cursor:pointer;vector-effect:non-scaling-stroke}circle:focus{stroke:#111;stroke-width:3}#plot[data-pan=true],#plot[data-pan=true] circle{cursor:grab}#plot.panning,#plot.panning circle{cursor:grabbing}#selection-box{fill:#157f9a22;stroke:#176c70;stroke-dasharray:5;pointer-events:none;vector-effect:non-scaling-stroke}.view-controls{display:flex;flex-wrap:wrap;align-items:center;gap:7px;margin-bottom:8px}.view-controls button[aria-pressed=true]{background:#e1eff2;border-color:#176c70}#zoom-level{min-width:3em;text-align:center}
.toolbar{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:12px}.toolbar p{margin:0}.cards{display:grid;grid-template-columns:minmax(0,1fr);gap:16px}.card{padding:16px;background:white;border:1px solid var(--line);border-radius:10px;min-width:0}.variant{margin-top:12px}.support{border-top:1px solid var(--line);margin-top:12px;padding-top:10px}.future{opacity:.45}.support-title{font-size:12px;font-weight:600;color:var(--accent);margin:0 0 8px}.future .support-title{color:var(--muted)}
.grids{display:grid;grid-template-columns:1fr 1fr;gap:12px}.grid-label{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);margin:0 0 6px}canvas{display:block;image-rendering:pixelated;max-width:100%;height:auto;border:1px solid #8896a1}
.variant-id{font-size:12px;font-family:ui-monospace,monospace;overflow-wrap:anywhere;margin:8px 0}.variant-picker{margin:12px 0 8px}.pagination{display:flex;align-items:center;gap:12px}.all-toggle{margin-top:10px}.empty{padding:30px;color:var(--muted)}#hover{font-size:13px;color:var(--muted);min-height:22px;margin:8px 0 0}#frame-info{margin-top:8px}
@media(max-width:1100px){.layout{grid-template-columns:1fr;padding:16px}aside{display:grid;grid-template-columns:1fr 1fr;gap:12px}#groups{max-height:170px}}
@media(max-width:850px){.layout{padding:12px}main{grid-template-columns:1fr;position:static}#gallery{max-height:none}#gallery .cards{overflow:visible;padding-right:0;scrollbar-gutter:auto}header{padding:20px}}@media(max-width:520px){aside{display:block}}
__METADATA_CSS__
</style>
</head><body>
<header><h1>ConceptARC · Task map</h1><p id="subtitle"></p></header>
<div class="layout"><aside>
<section class="panel"><h2>Embedding</h2><div class="controls">
<label>Method<select id="method"></select></label><label>Position<select id="position"></select></label>
<label>Layer<select id="layer"></select></label><label>Color by<select id="color"><option value="concept">Concept</option><option value="cluster">Cluster</option></select></label>
</div><p class="note">Each point is a parent task, averaged over eight support-set variants. Click a point or drag a box to inspect tasks.</p>
<p class="small">Maps are fitted independently at each layer and position. Their axes and cluster numbers do not align across views.</p><p id="settings" class="small"></p></section>
__METADATA_CONTROLS__
<section class="panel"><h2>Groups</h2><p class="small">HDBSCAN groups use original cosine distances. They are independent of the 2D map and concept labels.</p><div id="groups"></div></section>
</aside><main>
<section class="panel"><div class="toolbar"><h2 id="map-title">Task representations</h2><button id="clear">Show all tasks</button></div>
<div class="view-controls" role="group" aria-label="Feature space view controls"><button id="zoom-out" aria-label="Zoom out">−</button><span id="zoom-level" class="small" aria-live="polite">1.0×</span><button id="zoom-in" aria-label="Zoom in">+</button><button id="pan-mode" aria-pressed="false">Pan</button><button id="reset-view">Reset view</button><button id="export-pdf" title="Download the current plot as a vector PDF">Export PDF</button><span id="pdf-status" class="small" role="status"></span></div>
<p class="small">Scroll to zoom. Drag to select; Shift-drag or enable Pan to move the view.</p>
<svg id="plot" viewBox="0 0 900 570" role="group" aria-label="Interactive task embedding; scroll to zoom, drag to select, Shift-drag to pan"><defs id="metadata-color-defs"></defs><g id="points"></g><rect id="selection-box" visibility="hidden"/></svg>
<p id="hover" aria-live="polite">Click a point or choose a group to view its support examples.</p><input type="search" id="legend-search" placeholder="Search legend" aria-label="Search legend" hidden><div id="legend"></div><p id="color-note" class="small" hidden></p><p id="frame-info" class="small"></p>__COMPOSITION_HTML__</section>
<section id="gallery"><div class="toolbar"><div><h2 id="selection-title">Support examples</h2><p class="small" id="selection-info"></p></div><button id="export">Export selected IDs</button><div class="pagination"><button id="previous">Previous</button><span id="page-info" class="small"></span><button id="next">Next</button></div></div><div id="cards" class="cards"></div></section>
</main></div>
<script type="application/json" id="explorer-data">__EXPLORER_DATA__</script>
<script>
'use strict';
const data = JSON.parse(document.getElementById('explorer-data').textContent);
const axes = data.axes, byId = id => document.getElementById(id), ns = 'http://www.w3.org/2000/svg';
const colors = ['#0072b2','#d55e00','#009e73','#cc79a7','#7662a3','#a27916','#00a6ba','#e46263','#5a7f2f','#9b5f41','#5577bb','#bc5090','#332288','#aa4499','#44aa99','#999933'];
let frame, coordinates, selected = [], selectionName = 'All tasks', page = 0, plotted = [], drag = null;
const pageSize = 6;
let plotView = {x:0,y:0,width:900,height:570}, panMode = false;
__METADATA_SCRIPT__
function element(tag, text, cls) {const e = document.createElement(tag); if(text !== undefined) e.textContent = text; if(cls) e.className = cls; return e;}
function fillSelect(id, values) {for(const value of values) {const o = element('option', String(value)); o.value = value; byId(id).append(o);}}
function concept(task) {return axes.concepts[axes.labels[task]];}
function clusterName(label) {return label === -2 ? 'Undefined cosine' : label === -1 ? 'Unassigned' : `Cluster ${label + 1}`;}
function clusterColor(label) {return label < 0 ? '#8c9ba5' : colors[label % colors.length];}
function validPoint(xy) {return xy.every(v => typeof v === 'number' && Number.isFinite(v));}
function number(value) {return value === null || value === undefined ? 'undefined' : value.toFixed(4);}
function pointColor(task) {return metadataPointColor(task);}
function setSelection(tasks, name) {selected = tasks.filter(taskVisible); selectionName = name; page = 0; updateHighlights(); renderCards();}
function allTasks() {return axes.task_ids.map((_, i) => i);}
function renderGroups() {
  const root = byId('groups'); root.replaceChildren();
  const labels = [...new Set(frame.cluster_labels)].sort((a,b) => a < 0 || b < 0 ? (a < 0 ? 1 : -1) - (b < 0 ? 1 : -1) || b-a : a-b);
  for(const label of labels) {
    const tasks = visibleTasks().filter(i => frame.cluster_labels[i] === label);
    if(!tasks.length) continue;
    const button = element('button', undefined, 'group'); button.dataset.cluster = label;
    const swatch = element('span', undefined, 'swatch'); swatch.style.background = clusterColor(label);
    button.append(swatch, element('span', clusterName(label)), element('span', tasks.length));
    button.addEventListener('click', () => setSelection(tasks, clusterName(label))); root.append(button);
  }
}
function renderLegend() {
  const root = byId('legend'); root.replaceChildren();
  const mode = byId('color').value, query = byId('legend-search').value.toLowerCase(), familyMode = mode.startsWith('family.');
  byId('color-note').hidden = !familyMode;
  byId('color-note').textContent = familyMode ? 'Striped points belong to multiple families; gray means unspecified or unmapped. Legend counts include each task in every matching family. Click a family to inspect its tasks.' : '';
  if(mode === 'cluster') return;
  let entries;
  if(mode === 'metadata_tag') {
    if(!highlightedTag) return;
    const {field,value} = highlightedTag;
    entries = [{name:valueLabel(field,value),color:'#b44818',tasks:visibleTasks().filter(task => hasValue(task,field,value))}, {name:'Other / unspecified',color:'#bdc5cc',tasks:visibleTasks().filter(task => !hasValue(task,field,value))}];
  } else if(mode.startsWith('op.') || familyMode) {
    entries = (metadataFields.get(mode)?.values || []).map(entry => ({value:entry.value,name:entry.label,color:categoricalColor(mode,entry.value),tasks:visibleTasks().filter(task => hasValue(task,mode,entry.value))}));
    if(familyMode) {
      const unspecified = visibleTasks().filter(task => !(metadata.records[task].values[mode] || []).length);
      if(unspecified.length) entries.push({value:'__missing__',name:'Unspecified / unmapped',color:'#bdc5cc',tasks:unspecified});
    }
  } else entries = axes.concepts.map((name,label) => ({name,color:colors[label % colors.length],tasks:visibleTasks().filter(task => axes.labels[task] === label)}));
  for(const entry of entries) {
    if(!entry.name.toLowerCase().includes(query)) continue;
    const button = element('button'); const swatch = element('span',undefined,'swatch'); swatch.style.background = entry.color;
    if(entry.value !== undefined) {button.dataset.field = mode; button.dataset.value = entry.value;}
    button.append(swatch,document.createTextNode(` ${entry.name} (${entry.tasks.length})`)); button.addEventListener('click',() => setSelection(entry.tasks,entry.name)); root.append(button);
  }
}
function updatePointTabStop(circle) {
  const x = circle.cx.baseVal.value, y = circle.cy.baseVal.value;
  const visible = taskVisible(Number(circle.dataset.task)) && x >= plotView.x && x <= plotView.x+plotView.width && y >= plotView.y && y <= plotView.y+plotView.height;
  circle.setAttribute('tabindex',visible ? '0' : '-1');
  if(!visible && document.activeElement === circle) circle.blur();
}
function updateHighlights() {
  const ids = new Set(selected);
  for(const circle of byId('points').children) {
    circle.style.display = taskVisible(Number(circle.dataset.task)) ? '' : 'none';
    const task = Number(circle.dataset.task), description = pointDescription(task); circle.setAttribute('aria-label',description); circle.querySelector('title').textContent = description;
    const chosen = ids.has(task); circle.setAttribute('fill', pointColor(task));
    circle.setAttribute('opacity', chosen ? '.9' : '.13'); circle.setAttribute('stroke', chosen && selected.length < axes.task_ids.length ? '#152e40' : 'white');
    circle.setAttribute('stroke-width', chosen && selected.length < axes.task_ids.length ? '1.5' : '.7');
    updatePointTabStop(circle);
  }
}
function renderPlot() {
  const root = byId('points'); root.replaceChildren();
  const visible = coordinates.filter(validPoint); plotted = [];
  if(!visible.length) return;
  const min = [0,1].map(j => Math.min(...visible.map(xy => xy[j]))), max = [0,1].map(j => Math.max(...visible.map(xy => xy[j])));
  const scale = Math.min(820 / (max[0]-min[0] || 1), 490 / (max[1]-min[1] || 1));
  coordinates.forEach((xy, task) => {
    if(!validPoint(xy)) return;
    const x = 450 + (xy[0] - (max[0]+min[0])/2) * scale, y = 285 - (xy[1] - (max[1]+min[1])/2) * scale;
    plotted.push({task,x,y}); const circle = document.createElementNS(ns,'circle');
    for(const [key,value] of Object.entries({cx:x,cy:y,r:4.7,tabindex:0,role:'button','aria-label':`${axes.task_ids[task]}, ${clusterName(frame.cluster_labels[task])}`})) circle.setAttribute(key,value);
    circle.dataset.task = task;
    const title = document.createElementNS(ns,'title'); title.textContent = `${axes.task_ids[task]} · ${clusterName(frame.cluster_labels[task])}`; circle.append(title);
    circle.addEventListener('pointerenter', () => {byId('hover').textContent = title.textContent;});
    circle.addEventListener('keydown', event => {if(event.key === 'Enter' || event.key === ' ') {event.preventDefault(); setSelection([task], axes.task_ids[task]);}});
    root.append(circle);
  }); updateHighlights();
}
function renderGrid(grid) {
  const canvas = element('canvas'); const cell = 9, gap = 1;
  canvas.width = grid.cols * cell; canvas.height = grid.rows * cell;
  canvas.style.width = `${Math.min(canvas.width, 250)}px`;
  canvas.setAttribute('role', 'img'); canvas.setAttribute('aria-label', `${grid.rows} by ${grid.cols} ARC grid`);
  const context = canvas.getContext('2d'); context.fillStyle = '#4a5158'; context.fillRect(0,0,canvas.width,canvas.height);
  for(let row=0; row<grid.rows; row++) for(let col=0; col<grid.cols; col++) {context.fillStyle = data.palette[Number(grid.cells[row*grid.cols+col])]; context.fillRect(col*cell,row*cell,cell-gap,cell-gap);}
  return canvas;
}
function renderVariant(task, version) {
  const record = data.supports[axes.version_rows[task][version]], root = element('div', undefined, 'variant');
  root.dataset.version = version; root.dataset.caseId = record.case_id;
  root.append(element('p', `${record.task_id} · token ${record.selected_token_indices[frame.position]}`, 'variant-id'));
  const source = element('p', record.source_path, 'small'); source.title = record.case_id; root.append(source);
  record.supports.forEach((pair, index) => {
    const included = index < Number(frame.position.slice(1)), row = element('div', undefined, `support${included ? '' : ' future'}`);
    row.append(element('p', `Support ${index+1} · ${included ? 'processed at ' + frame.position : 'not yet processed'}`, 'support-title'));
    const grids = element('div', undefined, 'grids'); for(const kind of ['input','output']) {const column = element('div'); column.append(element('p',kind,'grid-label'),renderGrid(pair[kind])); grids.append(column);} row.append(grids); root.append(row);
  }); return root;
}
function taskCard(task) {
  const card = element('article', undefined, 'card'); card.dataset.task = task;
  card.append(element('h3', axes.task_ids[task]), element('p', `${concept(task)} · ${clusterName(frame.cluster_labels[task])} · within-task dispersion ${number(frame.within_task_dispersion[task])}`, 'small'));
  renderTaskMetadata(task, card);
  const representative = frame.representative_versions[task], initial = representative >= 0 ? representative : 0;
  const label = element('label', 'Support-set variant', 'variant-picker'), picker = element('select'); picker.setAttribute('aria-label', `Support-set variant for ${axes.task_ids[task]}`);
  axes.support_set_ids[task].forEach((id, version) => {const option = element('option', `${id}${version === representative ? ' · closest to task mean' : ''}`); option.value = version; picker.append(option);}); picker.value = initial; label.append(picker); card.append(label);
  if(representative < 0) card.append(element('p', 'Representative cosine is undefined; showing the first variant.', 'small'));
  const variants = element('div'), button = element('button', 'Show all 8 variants', 'all-toggle'); let expanded = false;
  function showVariants() {variants.replaceChildren(); if(expanded) axes.version_rows[task].forEach((_,version) => variants.append(renderVariant(task,version))); else variants.append(renderVariant(task,Number(picker.value))); button.textContent = expanded ? 'Show selected variant' : 'Show all 8 variants'; button.setAttribute('aria-expanded', expanded);}
  picker.addEventListener('change', () => {expanded = false; showVariants();}); button.addEventListener('click', () => {expanded = !expanded; showVariants();});
  card.append(button, variants); showVariants(); return card;
}
function renderCards() {
  const root = byId('cards'); root.replaceChildren();
  byId('selection-title').textContent = selectionName;
  byId('selection-info').textContent = `${selected.length} of ${axes.task_ids.length} tasks · ${frame.position} · layer ${frame.layer}${metadata ? ' · '+visibleSet.size+' visible' : ''}`;
  const pages = Math.max(1, Math.ceil(selected.length / pageSize)); page = Math.min(page, pages-1);
  byId('page-info').textContent = `Page ${page+1} / ${pages}`; byId('previous').disabled = page === 0; byId('next').disabled = page+1 >= pages;
  for(const task of selected.slice(page*pageSize,(page+1)*pageSize)) root.append(taskCard(task));
  if(!selected.length) root.append(element('p', 'No tasks in this selection. Drag another box or choose a group.', 'empty'));
  renderComposition();
}
function changeFrame() {
  resetPlotView();
  frame = data.frames.find(f => f.position === byId('position').value && String(f.layer) === byId('layer').value);
  const method = byId('method').value; coordinates = frame.coordinates[axes.methods.indexOf(method)];
  byId('map-title').textContent = `${method.toUpperCase()} · ${frame.position} · layer ${frame.layer}`;
  const projected = coordinates.filter(validPoint).length, groups = new Set(frame.cluster_labels.filter(x => x >= 0)).size;
  const explained = frame.pca_explained_variance_ratio;
  byId('frame-info').textContent = `${projected} / ${axes.task_ids.length} tasks plotted · ${groups} automatic groups. ` + (method === 'pca' && explained.every(v => v !== null) ? `PCA variance explained: ${(100*explained[0]).toFixed(1)}% + ${(100*explained[1]).toFixed(1)}%. ` : '') + (projected < axes.task_ids.length ? 'Unplotted tasks remain available in the gallery. ' : '') + 'Apparent 2D separation is exploratory; it does not establish distinct task families.';
  renderPlot(); renderGroups(); renderLegend(); renderFilterStatus(); setSelection(visibleTasks(), metadataFilters.size || conceptFilters.size ? 'Filtered tasks' : 'All tasks');
}
function svgPoint(event, inverse = byId('plot').getScreenCTM().inverse()) {const point = byId('plot').createSVGPoint(); point.x = event.clientX; point.y = event.clientY; return point.matrixTransform(inverse);}
function setPlotView(x, y, width) {
  width = Math.min(900,Math.max(45,width)); const height = width * 570 / 900;
  plotView = {x:Math.max(0,Math.min(900-width,x)),y:Math.max(0,Math.min(570-height,y)),width,height};
  byId('plot').setAttribute('viewBox',`${plotView.x} ${plotView.y} ${width} ${height}`);
  const scale = 900 / width;
  for(const circle of byId('points').children) {circle.setAttribute('r',4.7 / scale); updatePointTabStop(circle);}
  byId('zoom-level').textContent = `${scale.toFixed(1)}×`; byId('zoom-out').disabled = scale <= 1; byId('zoom-in').disabled = scale >= 20;
}
function zoomPlot(factor, anchor = {x:plotView.x+plotView.width/2,y:plotView.y+plotView.height/2}) {
  if(drag) return;
  const width = Math.min(900,Math.max(45,plotView.width/factor)), ratio = width / plotView.width;
  setPlotView(anchor.x-(anchor.x-plotView.x)*ratio,anchor.y-(anchor.y-plotView.y)*ratio,width);
}
function tasksInBox(start, end) {
  const left = Math.max(plotView.x,Math.min(start.x,end.x)), right = Math.min(plotView.x+plotView.width,Math.max(start.x,end.x));
  const top = Math.max(plotView.y,Math.min(start.y,end.y)), bottom = Math.min(plotView.y+plotView.height,Math.max(start.y,end.y));
  return plotted.filter(p => p.x >= left && p.x <= right && p.y >= top && p.y <= bottom).map(p => p.task);
}
function stopPlotDrag(event) {
  if(event && drag && event.pointerId !== drag.pointerId) return;
  const active = drag; drag = null;
  if(active && byId('plot').hasPointerCapture(active.pointerId)) byId('plot').releasePointerCapture(active.pointerId);
  byId('selection-box').setAttribute('visibility','hidden'); byId('plot').classList.remove('panning');
}
function resetPlotView() {stopPlotDrag(); setPlotView(0,0,900);}
byId('zoom-in').addEventListener('click', () => zoomPlot(1.5));
byId('zoom-out').addEventListener('click', () => zoomPlot(1/1.5));
byId('reset-view').addEventListener('click',resetPlotView);
__EXPLORER_PDF_SCRIPT__
byId('pan-mode').addEventListener('click', () => {panMode = !panMode; byId('pan-mode').setAttribute('aria-pressed',panMode); byId('plot').dataset.pan = panMode;});
byId('plot').addEventListener('wheel', event => {
  event.preventDefault();
  const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? 570 : 1;
  zoomPlot(Math.exp(-Math.max(-100,Math.min(100,event.deltaY*unit))*0.002),svgPoint(event));
},{passive:false});
byId('plot').addEventListener('pointerdown', event => {
  if(event.button !== 0 || !event.isPrimary || drag) return;
  event.preventDefault(); const inverse = byId('plot').getScreenCTM().inverse();
  drag = {start:svgPoint(event,inverse),clientX:event.clientX,clientY:event.clientY,inverse,view:{...plotView},pan:panMode || event.shiftKey,pointerId:event.pointerId,task:event.target.closest('circle')?.dataset.task};
  byId('plot').setPointerCapture(event.pointerId); if(drag.pan) byId('plot').classList.add('panning');
});
byId('plot').addEventListener('pointermove', event => {
  if(!drag || event.pointerId !== drag.pointerId) return;
  const end = svgPoint(event,drag.inverse);
  if(drag.pan) {setPlotView(drag.view.x+drag.start.x-end.x,drag.view.y+drag.start.y-end.y,drag.view.width); return;}
  const box = byId('selection-box');
  for(const [key,value] of Object.entries({x:Math.min(drag.start.x,end.x),y:Math.min(drag.start.y,end.y),width:Math.abs(end.x-drag.start.x),height:Math.abs(end.y-drag.start.y),visibility:'visible'})) box.setAttribute(key,value);
});
byId('plot').addEventListener('pointerup', event => {
  if(!drag || event.pointerId !== drag.pointerId) return;
  if(!drag.pan) {
    const end = svgPoint(event), start = drag.start;
    if(Math.hypot(event.clientX-drag.clientX,event.clientY-drag.clientY) > 4) setSelection(tasksInBox(start,end), 'Box selection');
    else if(drag.task !== undefined) setSelection([Number(drag.task)], axes.task_ids[Number(drag.task)]);
  }
  stopPlotDrag();
});
byId('plot').addEventListener('pointercancel',stopPlotDrag);
byId('plot').addEventListener('lostpointercapture',stopPlotDrag);
fillSelect('method', axes.methods); fillSelect('position', axes.positions); fillSelect('layer', axes.layers);
for(const id of ['method','position','layer']) byId(id).addEventListener('change',changeFrame);
byId('color').addEventListener('change', () => {updateHighlights(); renderLegend();});
byId('clear').addEventListener('click', () => setSelection(visibleTasks(),metadataFilters.size || conceptFilters.size ? 'Filtered tasks' : 'All tasks'));
byId('previous').addEventListener('click', () => {page--; renderCards();}); byId('next').addEventListener('click', () => {page++; renderCards();});
byId('export').addEventListener('click', () => {
  const result = {dataset:data.dataset,model:data.model,position:frame.position,layer:frame.layer,method:byId('method').value,selection:selectionName,tasks:selected.map(task => ({task_id:axes.task_ids[task],concept:concept(task),cluster:frame.cluster_labels[task],representative_version:frame.representative_versions[task] < 0 ? null : axes.version_task_ids[task][frame.representative_versions[task]],version_ids:axes.version_task_ids[task]}))};
  if(axes.population) result.population = axes.population;
  if(metadata) {result.taxonomy_version = metadata.taxonomy_version; result.filters = [...metadataFilters].map(([field,values]) => ({field,values:[...values]})); result.concept_filters = [...conceptFilters].map(index => axes.concepts[index]); result.highlight = highlightedTag; selected.forEach((task,index) => {result.tasks[index].metadata = metadata.records[task];});}
  const url = URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'})), link = element('a'); link.href=url; link.download=`tasks_${frame.position}_layer_${frame.layer}.json`; link.click(); setTimeout(() => URL.revokeObjectURL(url),1000);
});
byId('subtitle').textContent = `${data.dataset} · ${data.model}${axes.population ? " · " + axes.population.name + " population" : ""} · ${axes.task_ids.length} parent tasks · ${axes.concepts.length} concepts`;
byId('settings').textContent = `UMAP neighbors: ${data.settings.n_neighbors}; min distance: ${data.settings.min_dist}. HDBSCAN minimum group: ${data.settings.min_cluster_size}; min samples: ${data.settings.min_samples}; selection: ${data.settings.cluster_selection_method}.`;
initializeMetadata();
changeFrame();
</script></body></html>"""
