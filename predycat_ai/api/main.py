"""FastAPI Backend for PredycatAI Universal Optimizer"""

import os
import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("PredycatAI API starting up...")
    yield

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PredycatAI Universal Optimizer</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #07070d; --surface: #0f0f1a; --surface2: #141420; --border: #1e1e35;
    --green: #00ff88; --green-dim: #00cc6a; --blue: #4f8ef7; --red: #ff4d6a;
    --yellow: #ffd166; --purple: #9b5de5; --text: #d4d4e8; --muted: #6b6b8a;
    --radius: 10px;
  }
  html, body { height: 100%; }
  html { pointer-events: auto; }
  body { background: var(--bg); color: var(--text); font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; font-size: 14px; display: flex; flex-direction: column; pointer-events: auto; }

  /* ── TOP BAR ── */
  .topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 1.5rem; height: 54px;
    background: var(--surface); border-bottom: 1px solid var(--border);
    position: sticky; top: 0; z-index: 100;
    pointer-events: none;
  }
  .topbar > * { pointer-events: auto; }
  .logo { display: flex; align-items: center; gap: 0.5rem; font-size: 1.1rem; font-weight: 700; color: var(--green); }
  .logo svg { width: 22px; height: 22px; }
  .topbar-right { display: flex; align-items: center; gap: 1rem; }
  .status-pill { display: flex; align-items: center; gap: 0.4rem; font-size: 12px; color: var(--muted); }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
  .version { font-size: 11px; color: var(--muted); background: var(--surface2); padding: 2px 8px; border-radius: 20px; border: 1px solid var(--border); }

  /* ── LAYOUT ── */
  .layout { display: flex; flex: 1; overflow: hidden; pointer-events: none; }
  .layout > * { pointer-events: auto; }
  .sidebar { width: 200px; background: var(--surface); border-right: 1px solid var(--border); padding: 1rem 0; flex-shrink: 0; display: flex; flex-direction: column; gap: 2px; overflow: visible; pointer-events: auto; position: relative; z-index: 10; }
  .nav-item { display: flex; align-items: center; gap: 0.6rem; padding: 0.6rem 1rem; cursor: pointer; color: var(--muted); border-radius: 0; transition: all .15s; font-size: 13px; border-left: 3px solid transparent; pointer-events: auto; }
  .nav-item:hover { color: var(--text); background: var(--surface2); }
  .nav-item.active { color: var(--green); background: rgba(0,255,136,.06); border-left-color: var(--green); }
  .nav-icon { font-size: 15px; width: 18px; text-align: center; }
  .main { flex: 1; overflow-y: auto; padding: 1.5rem; display: flex; flex-direction: column; gap: 1.25rem; pointer-events: auto; }

  /* ── PANELS ── */
  .panel { display: none; flex-direction: column; gap: 1.25rem; pointer-events: auto; }
  .panel.active { display: flex; }

  /* ── CARDS ── */
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.25rem; }
  .card-title { font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin-bottom: 1rem; display: flex; align-items: center; gap: .5rem; }
  .card-title span { color: var(--green); font-size: 14px; }

  /* ── SYSTEM STATS ── */
  .stats-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 1rem; }
  .stat-box { background: var(--surface2); border: 1px solid var(--border); border-radius: var(--radius); padding: 1rem; text-align: center; }
  .stat-val { font-size: 1.6rem; font-weight: 700; color: var(--green); line-height: 1; }
  .stat-sub { font-size: 11px; color: var(--muted); margin-top: .35rem; }
  .bar-wrap { background: var(--bg); border-radius: 4px; height: 6px; margin-top: .5rem; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 4px; transition: width .6s; }
  .bar-green { background: var(--green); }
  .bar-blue { background: var(--blue); }
  .bar-yellow { background: var(--yellow); }
  .bar-red { background: var(--red); }

  /* ── TABS ── */
  .tabs { display: flex; gap: .25rem; border-bottom: 1px solid var(--border); margin-bottom: 1.25rem; }
  .tab { padding: .5rem 1rem; cursor: pointer; color: var(--muted); font-size: 13px; border-bottom: 2px solid transparent; margin-bottom: -1px; transition: all .15s; pointer-events: auto; }
  .tab:hover { color: var(--text); }
  .tab.active { color: var(--green); border-bottom-color: var(--green); }
  .tab-panel { display: none; }
  .tab-panel.active { display: block; }

  /* ── FORM ── */
  .form-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1rem; }
  .field { display: flex; flex-direction: column; gap: .4rem; }
  .field label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }
  input[type=text], input[type=number], select, textarea {
    background: var(--surface2); border: 1px solid var(--border); color: var(--text);
    padding: .6rem .75rem; border-radius: 8px; font-size: 13px; width: 100%;
    transition: border-color .15s;
  }
  input:focus, select:focus, textarea:focus { outline: none; border-color: var(--green); }
  .toggles { display: flex; flex-wrap: wrap; gap: .75rem; margin: .75rem 0; }
  .toggle-wrap { display: flex; align-items: center; gap: .4rem; cursor: pointer; user-select: none; pointer-events: auto; }
  .toggle-wrap input { display: none; }
  .toggle { width: 34px; height: 18px; background: var(--border); border-radius: 9px; position: relative; transition: background .2s; flex-shrink: 0; }
  .toggle::after { content:''; position: absolute; left: 2px; top: 2px; width: 14px; height: 14px; border-radius: 50%; background: #fff; transition: left .2s; }
  .toggle-wrap input:checked + .toggle { background: var(--green); }
  .toggle-wrap input:checked + .toggle::after { left: 18px; }
  .toggle-label { font-size: 12px; color: var(--muted); }

  /* ── BUTTONS ── */
  .btn { display: inline-flex; align-items: center; gap: .4rem; padding: .6rem 1.25rem; border: none; border-radius: 8px; font-size: 13px; font-weight: 600; cursor: pointer; transition: all .15s; pointer-events: auto; }
  .btn-primary { background: linear-gradient(135deg, var(--green), var(--green-dim)); color: #000; }
  .btn-primary:hover { box-shadow: 0 0 20px rgba(0,255,136,.35); transform: translateY(-1px); }
  .btn-primary:disabled { opacity: .5; cursor: not-allowed; transform: none; box-shadow: none; }
  .btn-ghost { background: transparent; color: var(--muted); border: 1px solid var(--border); }
  .btn-ghost:hover { color: var(--text); border-color: var(--muted); }
  .btn-danger { background: rgba(255,77,106,.12); color: var(--red); border: 1px solid rgba(255,77,106,.3); }

  /* ── FILE DROP ── */
  .drop-zone { border: 2px dashed var(--border); border-radius: var(--radius); padding: 2rem; text-align: center; cursor: pointer; transition: all .2s; color: var(--muted); pointer-events: auto; }
  .drop-zone:hover, .drop-zone.dragging { border-color: var(--green); color: var(--green); background: rgba(0,255,136,.04); }
  .drop-zone .drop-icon { font-size: 2rem; margin-bottom: .5rem; }

  /* ── PROGRESS ── */
  .job-progress { background: var(--surface2); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.25rem; }
  .job-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: .75rem; }
  .job-id { font-family: monospace; font-size: 12px; color: var(--muted); }
  .badge { display: inline-flex; align-items: center; gap: .3rem; padding: .2rem .6rem; border-radius: 20px; font-size: 11px; font-weight: 600; }
  .badge-running { background: rgba(79,142,247,.15); color: var(--blue); }
  .badge-done { background: rgba(0,255,136,.12); color: var(--green); }
  .badge-failed { background: rgba(255,77,106,.12); color: var(--red); }
  .badge-queued { background: rgba(255,209,102,.12); color: var(--yellow); }
  .prog-track { background: var(--bg); border-radius: 4px; height: 8px; overflow: hidden; margin: .5rem 0; }
  .prog-bar { height: 100%; border-radius: 4px; background: linear-gradient(90deg, var(--green), var(--blue)); transition: width .4s; }
  .prog-bar.indeterminate { width: 40% !important; animation: slide 1.5s ease-in-out infinite; }
  @keyframes slide { 0%{transform:translateX(-100%)} 100%{transform:translateX(300%)} }

  /* ── RESULTS ── */
  .results-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: .75rem; }
  .result-box { background: var(--surface2); border: 1px solid var(--border); border-radius: 8px; padding: .85rem; text-align: center; }
  .result-val { font-size: 1.4rem; font-weight: 700; color: var(--green); }
  .result-sub { font-size: 11px; color: var(--muted); margin-top: .25rem; }
  .result-box.accent-blue .result-val { color: var(--blue); }
  .result-box.accent-yellow .result-val { color: var(--yellow); }
  .result-box.accent-red .result-val { color: var(--red); }

  /* ── JOBS TABLE ── */
  .table-wrap { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; padding: .6rem 1rem; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .05em; border-bottom: 1px solid var(--border); }
  td { padding: .65rem 1rem; border-bottom: 1px solid rgba(30,30,53,.7); }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: var(--surface2); }
  .mono { font-family: monospace; font-size: 11px; color: var(--muted); }

  /* ── STRATEGY PANEL ── */
  .strategy-list { display: flex; flex-direction: column; gap: .5rem; }
  .strategy-step { display: flex; align-items: center; gap: .75rem; padding: .6rem .85rem; background: var(--surface2); border-radius: 8px; border: 1px solid var(--border); }
  .step-num { width: 22px; height: 22px; border-radius: 50%; background: var(--green); color: #000; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700; flex-shrink: 0; }
  .step-name { flex: 1; font-size: 13px; }
  .step-chip { font-size: 11px; background: rgba(79,142,247,.15); color: var(--blue); padding: .15rem .5rem; border-radius: 20px; }

  /* ── TOAST ── */
  .toasts { position: fixed; bottom: 1rem; right: 1rem; display: flex; flex-direction: column; gap: .5rem; z-index: 999; pointer-events: none; }
  .toast { pointer-events: auto; }
  .toast { padding: .65rem 1rem; border-radius: 8px; font-size: 13px; display: flex; align-items: center; gap: .5rem; animation: toastIn .2s ease; max-width: 320px; }
  .toast-ok { background: rgba(0,255,136,.12); border: 1px solid rgba(0,255,136,.3); color: var(--green); }
  .toast-err { background: rgba(255,77,106,.12); border: 1px solid rgba(255,77,106,.3); color: var(--red); }
  .toast-info { background: rgba(79,142,247,.12); border: 1px solid rgba(79,142,247,.3); color: var(--blue); }
  @keyframes toastIn { from{opacity:0;transform:translateX(20px)} to{opacity:1;transform:none} }

  /* ── SCROLLBAR ── */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

  /* ── LOG ── */
  .log-box { background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: .75rem; max-height: 200px; overflow-y: auto; font-family: 'Courier New', monospace; font-size: 12px; line-height: 1.6; }
  .log-entry { color: var(--muted); }
  .log-entry.ok { color: var(--green); }
  .log-entry.err { color: var(--red); }
  .log-entry.info { color: var(--blue); }

  /* ── EMPTY STATE ── */
  .empty { text-align: center; padding: 3rem 1rem; color: var(--muted); }
  .empty-icon { font-size: 2.5rem; margin-bottom: .75rem; opacity: .4; }

  /* ── MODEL INFO CARD ── */
  .model-info-card { background: linear-gradient(135deg, rgba(0,255,136,.06), rgba(79,142,247,.06)); border: 1px solid rgba(0,255,136,.2); border-radius: var(--radius); padding: 1rem; margin-top: .75rem; }
  .model-info-header { display: flex; align-items: center; gap: .75rem; margin-bottom: .75rem; }
  .model-info-id { font-size: 14px; font-weight: 600; color: var(--text); }
  .model-info-task { font-size: 11px; background: rgba(79,142,247,.15); color: var(--blue); padding: .15rem .5rem; border-radius: 20px; }
  .model-meta-row { display: flex; flex-wrap: wrap; gap: .5rem 1.5rem; font-size: 12px; color: var(--muted); margin-bottom: .75rem; }
  .model-meta-row strong { color: var(--text); }
  .size-warn { color: var(--yellow); font-weight: 600; }
  .size-ok { color: var(--green); font-weight: 600; }
  .model-tags { display: flex; flex-wrap: wrap; gap: .3rem; margin-bottom: .75rem; }
  .model-tag { font-size: 10px; background: var(--surface2); color: var(--muted); padding: .1rem .45rem; border-radius: 4px; border: 1px solid var(--border); }
  .model-files { font-size: 11px; color: var(--muted); margin-bottom: .75rem; }
  .model-files summary { cursor: pointer; color: var(--blue); }

  /* ── INFERENCE PANEL ── */
  .infer-panel { background: var(--surface2); border: 1px solid var(--border); border-radius: var(--radius); padding: 1rem; margin-top: .75rem; }
  .infer-panel textarea { width: 100%; min-height: 72px; resize: vertical; font-family: monospace; font-size: 12px; }
  .infer-response { background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: .75rem; font-size: 12px; font-family: monospace; white-space: pre-wrap; max-height: 200px; overflow-y: auto; color: var(--green); margin-top: .5rem; }
  .infer-response.loading { color: var(--muted); animation: pulse 1s infinite; }

  /* ── DATASET CARD ── */
  .dataset-row { padding: .6rem .85rem; cursor: pointer; display: grid; grid-template-columns: 1fr auto; gap: .5rem; align-items: center; border-bottom: 1px solid rgba(30,30,53,.5); font-size: 12px; }
  .dataset-row:hover { background: var(--surface2); }
  .dataset-row.selected { background: rgba(0,255,136,.07); border-left: 3px solid var(--green); }
  .sel-dataset-badge { display: inline-flex; align-items: center; gap: .35rem; padding: .3rem .75rem; background: rgba(0,255,136,.1); border: 1px solid rgba(0,255,136,.25); border-radius: 6px; font-size: 12px; color: var(--green); }

  /* ── FILTER SELECTS (model picker) ── */
  select.filter-sel { width: auto !important; font-size: 11px !important; padding: .25rem .5rem !important; border-radius: 6px !important; cursor: pointer; flex-shrink: 0; }

  /* ── PLAYGROUND ── */
  .pg-status { display:flex; align-items:center; gap:.6rem; padding:.6rem .85rem; background:var(--surface2); border:1px solid var(--border); border-radius:8px; font-size:12px; color:var(--muted); }
  .pg-status.loaded { border-color:rgba(0,255,136,.3); color:var(--green); background:rgba(0,255,136,.06); }
  .pg-task-ui { display:none; }
  .pg-task-ui.active { display:flex; flex-direction:column; gap:1rem; }
  .pg-output { background:var(--bg); border:1px solid var(--border); border-radius:8px; padding:.85rem; font-size:12px; font-family:monospace; white-space:pre-wrap; max-height:240px; overflow-y:auto; color:var(--green); min-height:48px; }
  .pg-output.empty { color:var(--muted); font-family:inherit; }
  .pg-label-row { display:flex; justify-content:space-between; align-items:center; padding:.45rem .7rem; background:var(--surface2); border-radius:6px; font-size:12px; margin-bottom:.35rem; }
  .pg-bar { height:6px; background:var(--border); border-radius:3px; overflow:hidden; width:120px; }
  .pg-bar-fill { height:100%; background:var(--green); border-radius:3px; transition:width .4s; }
  .pg-img-preview { max-width:100%; max-height:300px; border-radius:8px; border:1px solid var(--border); display:none; margin-top:.5rem; }
</style>
</head>
<body>

<!-- TOP BAR -->
<header class="topbar">
  <div class="logo">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
    PredycatAI
  </div>
  <div class="topbar-right">
    <div class="status-pill"><div class="dot"></div><span id="api-status">Connected</span></div>
    <div class="version">v1.0.0</div>
  </div>
</header>

<div class="layout">

  <!-- SIDEBAR -->
  <nav class="sidebar">
    <div class="nav-item active" onclick="showPanel('optimize', this)">
      <span class="nav-icon">⚡</span> Optimize
    </div>
    <div class="nav-item" onclick="showPanel('jobs', this)">
      <span class="nav-icon">📋</span> Jobs
    </div>
    <div class="nav-item" onclick="showPanel('strategy', this)">
      <span class="nav-icon">🗺</span> Strategy
    </div>
    <div class="nav-item" onclick="showPanel('system', this)">
      <span class="nav-icon">🖥</span> System
    </div>
    <div class="nav-item" onclick="showPanel('datasets', this)">
      <span class="nav-icon">🗂</span> Datasets
    </div>
    <div class="nav-item" onclick="showPanel('api', this)">
      <span class="nav-icon">📖</span> API Docs
    </div>
    <div class="nav-item" onclick="showPanel('playground', this)">
      <span class="nav-icon">🎮</span> Playground
    </div>
  </nav>

  <!-- MAIN -->
  <main class="main">

    <!-- ── OPTIMIZE PANEL ── -->
    <div class="panel active" id="panel-optimize">

      <div class="card">
        <div class="card-title"><span>⚡</span> Optimize Model</div>

        <div class="tabs">
          <div class="tab active" onclick="switchTab('by-id', this)">By HuggingFace ID</div>
          <div class="tab" onclick="switchTab('by-file', this)">Upload File</div>
        </div>

        <!-- BY ID -->
        <div class="tab-panel active" id="tab-by-id">
          <div class="form-grid">
            <div class="field" style="grid-column: 1/-1">
              <label>Model ID — type any HuggingFace model or pick below</label>
              <div style="position:relative">
                <input type="text" id="model-id" placeholder="e.g. gpt2, microsoft/phi-2, Qwen/Qwen2-1.5B-Instruct" oninput="filterModels(this.value)" onfocus="showModelPicker()" onblur="setTimeout(hideModelPicker,200)" autocomplete="off">
                <div id="model-picker" onmousedown="event.preventDefault()" style="display:none;position:absolute;top:100%;left:0;right:0;z-index:200;background:var(--surface);border:1px solid var(--border);border-radius:8px;max-height:440px;overflow-y:auto;box-shadow:0 8px 32px rgba(0,0,0,.4)">
                  <div id="category-chips" style="display:flex;flex-wrap:wrap;gap:.3rem;padding:.55rem .75rem;border-bottom:1px solid var(--border);position:sticky;top:0;background:var(--surface);z-index:2"></div>
                  <div style="padding:.4rem .75rem;border-bottom:1px solid var(--border);background:var(--surface);position:sticky;top:0;z-index:2">
                    <div style="display:flex;align-items:center;gap:.3rem;flex-wrap:wrap;margin-bottom:.3rem">
                      <span style="font-size:10px;color:var(--muted);white-space:nowrap;margin-right:.15rem">💾 Size:</span>
                      <div id="filter-size-chips" style="display:flex;gap:.25rem;flex-wrap:wrap"></div>
                    </div>
                    <div style="display:flex;align-items:center;gap:.3rem;flex-wrap:wrap">
                      <span style="font-size:10px;color:var(--muted);white-space:nowrap;margin-right:.15rem">⚙ Params:</span>
                      <div id="filter-params-chips" style="display:flex;gap:.25rem;flex-wrap:wrap"></div>
                    </div>
                    <span id="filter-hint" style="font-size:10px;color:var(--muted);display:block;margin-top:.3rem">🔍 HuggingFace live</span>
                  </div>
                  <div id="model-list"></div>
                </div>
              </div>
            </div>
            <div class="field">
              <label>Target Platform</label>
              <select id="target-platform">
                <option value="mobile">📱 Mobile</option>
                <option value="laptop" selected>💻 Laptop</option>
                <option value="cloud">☁️ Cloud</option>
                <option value="edge">🔌 Edge</option>
              </select>
            </div>
            <div class="field">
              <label>Optimization Preset</label>
              <select id="preset">
                <option value="balanced" selected>⚖️ Balanced</option>
                <option value="ultra_compression">🗜 Ultra Compression</option>
                <option value="max_accuracy">🎯 Max Accuracy</option>
                <option value="mobile_safe">🛡 Mobile Safe</option>
              </select>
            </div>
            <div class="field">
              <label>Max Accuracy Drop (%)</label>
              <input type="number" id="max-acc-drop" value="2.0" min="0" max="20" step="0.5">
            </div>
          </div>

          <div class="toggles">
            <label class="toggle-wrap">
              <input type="checkbox" id="use-quant" checked>
              <div class="toggle"></div>
              <span class="toggle-label">Quantization</span>
            </label>
            <label class="toggle-wrap">
              <input type="checkbox" id="use-prune" checked>
              <div class="toggle"></div>
              <span class="toggle-label">Pruning</span>
            </label>
            <label class="toggle-wrap">
              <input type="checkbox" id="use-distill" checked>
              <div class="toggle"></div>
              <span class="toggle-label">Distillation</span>
            </label>
            <label class="toggle-wrap">
              <input type="checkbox" id="use-reason" checked>
              <div class="toggle"></div>
              <span class="toggle-label">Reasoning Check</span>
            </label>
          </div>

          <div style="display:flex;gap:.75rem;flex-wrap:wrap;align-items:center">
            <button class="btn btn-primary" id="run-btn" onclick="runOptimize()">⚡ Run Optimization</button>
            <button class="btn btn-ghost" id="infer-btn" onclick="toggleInferPanel()" style="display:none">🧪 Test Inference</button>
          </div>

          <!-- MODEL INFO CARD -->
          <div id="model-info-card" class="model-info-card" style="display:none"></div>

          <!-- INFERENCE PANEL -->
          <div id="infer-panel" class="infer-panel" style="display:none">
            <div style="font-size:12px;font-weight:600;color:var(--text);margin-bottom:.5rem">🧪 Quick Inference Test</div>
            <div style="font-size:11px;color:var(--muted);margin-bottom:.5rem">Requires HF_TOKEN in .env · tests model before/after optimization</div>
            <textarea id="infer-prompt" placeholder="Enter prompt to test...">Hello, my name is</textarea>
            <div style="display:flex;gap:.5rem;margin-top:.5rem">
              <button class="btn btn-primary" style="flex:1" onclick="runInferenceTest()">▶ Run</button>
              <button class="btn btn-ghost" onclick="document.getElementById('infer-panel').style.display=\\'none\\'">✕</button>
            </div>
            <div id="infer-response" style="display:none" class="infer-response"></div>
          </div>
        </div>

        <!-- BY FILE -->
        <div class="tab-panel" id="tab-by-file">
          <div class="drop-zone" id="drop-zone" onclick="document.getElementById('file-input').click()"
            ondragover="event.preventDefault();this.classList.add('dragging')"
            ondragleave="this.classList.remove('dragging')"
            ondrop="handleDrop(event)">
            <div class="drop-icon">📦</div>
            <div>Drop model file here or <strong style="color:var(--green)">browse</strong></div>
            <div style="font-size:11px;margin-top:.35rem">.pt, .pth, .bin, .onnx, .safetensors</div>
            <div id="file-name" style="margin-top:.5rem;color:var(--green);font-size:12px"></div>
          </div>
          <input type="file" id="file-input" style="display:none" accept=".pt,.pth,.bin,.onnx,.safetensors" onchange="fileSelected(this)">

          <div class="form-grid" style="margin-top:1rem">
            <div class="field">
              <label>Target Platform</label>
              <select id="file-target">
                <option value="mobile">📱 Mobile</option>
                <option value="laptop" selected>💻 Laptop</option>
                <option value="cloud">☁️ Cloud</option>
                <option value="edge">🔌 Edge</option>
              </select>
            </div>
            <div class="field">
              <label>Preset</label>
              <select id="file-preset">
                <option value="balanced" selected>⚖️ Balanced</option>
                <option value="ultra_compression">🗜 Ultra Compression</option>
                <option value="max_accuracy">🎯 Max Accuracy</option>
                <option value="mobile_safe">🛡 Mobile Safe</option>
              </select>
            </div>
          </div>
          <button class="btn btn-primary" id="file-run-btn" onclick="runFileOptimize()">⚡ Run Optimization</button>
        </div>
      </div>

      <!-- ACTIVE JOB PROGRESS -->
      <div id="active-job-card" class="card" style="display:none">
        <div class="card-title"><span>⏳</span> Active Job</div>
        <div class="job-progress">
          <div class="job-header">
            <span class="job-id" id="active-job-id">-</span>
            <span class="badge badge-running" id="active-job-badge">Running</span>
          </div>
          <div class="prog-track"><div class="prog-bar indeterminate" id="active-prog-bar"></div></div>
          <div style="font-size:12px;color:var(--muted)" id="active-job-msg">Processing...</div>
        </div>

        <!-- RESULTS (shown on complete) -->
        <div id="job-results" style="display:none;margin-top:1rem">
          <div class="results-grid">
            <div class="result-box"><div class="result-val" id="res-compression">-</div><div class="result-sub">Compression Ratio</div></div>
            <div class="result-box accent-blue"><div class="result-val" id="res-latency">-</div><div class="result-sub">Latency Improvement</div></div>
            <div class="result-box accent-yellow"><div class="result-val" id="res-orig-size">-</div><div class="result-sub">Original Size (MB)</div></div>
            <div class="result-box accent-yellow"><div class="result-val" id="res-opt-size">-</div><div class="result-sub">Optimized Size (MB)</div></div>
            <div class="result-box accent-red"><div class="result-val" id="res-acc-drop">-</div><div class="result-sub">Accuracy Drop (%)</div></div>
          </div>
          <div id="res-outputs" style="margin-top:.75rem;font-size:12px;color:var(--muted)"></div>
        </div>

        <!-- LOG -->
        <div class="log-box" id="job-log" style="margin-top:1rem"></div>
      </div>

    </div>

    <!-- ── JOBS PANEL ── -->
    <div class="panel" id="panel-jobs">
      <div class="card">
        <div class="card-title" style="justify-content:space-between">
          <span><span>📋</span> Job History</span>
          <button class="btn btn-ghost" style="font-size:12px;padding:.3rem .75rem" onclick="loadJobs()">↻ Refresh</button>
        </div>
        <div class="table-wrap" id="jobs-table-wrap">
          <div class="empty"><div class="empty-icon">📭</div>No jobs yet</div>
        </div>
      </div>
    </div>

    <!-- ── STRATEGY PANEL ── -->
    <div class="panel" id="panel-strategy">
      <div class="card">
        <div class="card-title"><span>🗺</span> Optimization Strategy Planner</div>
        <div class="form-grid">
          <div class="field">
            <label>Target Platform</label>
            <select id="strat-platform">
              <option value="mobile">📱 Mobile</option>
              <option value="laptop">💻 Laptop</option>
              <option value="cloud">☁️ Cloud</option>
              <option value="edge">🔌 Edge</option>
            </select>
          </div>
          <div class="field">
            <label>Model Type</label>
            <select id="strat-model-type">
              <option value="llm">LLM</option>
              <option value="transformer">Transformer</option>
              <option value="cnn">CNN</option>
              <option value="hybrid">Hybrid</option>
            </select>
          </div>
          <div class="field">
            <label>Preset</label>
            <select id="strat-preset">
              <option value="balanced">⚖️ Balanced</option>
              <option value="ultra_compression">🗜 Ultra Compression</option>
              <option value="max_accuracy">🎯 Max Accuracy</option>
              <option value="mobile_safe">🛡 Mobile Safe</option>
            </select>
          </div>
        </div>
        <button class="btn btn-primary" onclick="loadStrategy()">Get Strategy</button>
      </div>

      <div class="card" id="strategy-result" style="display:none">
        <div class="card-title"><span>📐</span> Recommended Strategy</div>
        <div class="stats-row" id="strat-meta" style="margin-bottom:1rem"></div>
        <div class="strategy-list" id="strategy-steps"></div>
      </div>
    </div>

    <!-- ── SYSTEM PANEL ── -->
    <div class="panel" id="panel-system">
      <div class="card">
        <div class="card-title" style="justify-content:space-between">
          <span><span>🖥</span> System Resources</span>
          <button class="btn btn-ghost" style="font-size:12px;padding:.3rem .75rem" onclick="loadSystem()">↻ Refresh</button>
        </div>
        <div class="stats-row" id="system-stats">
          <div class="stat-box"><div class="stat-val" id="sys-cpu">-</div><div class="stat-sub">CPU Usage</div><div class="bar-wrap"><div class="bar-fill bar-blue" id="bar-cpu" style="width:0%"></div></div></div>
          <div class="stat-box"><div class="stat-val" id="sys-ram">-</div><div class="stat-sub">RAM Available</div><div class="bar-wrap"><div class="bar-fill bar-green" id="bar-ram" style="width:0%"></div></div></div>
          <div class="stat-box"><div class="stat-val" id="sys-ram-total">-</div><div class="stat-sub">RAM Total</div></div>
          <div class="stat-box"><div class="stat-val" id="sys-cuda">-</div><div class="stat-sub">CUDA</div></div>
          <div class="stat-box" id="sys-gpu-box" style="display:none"><div class="stat-val" id="sys-vram">-</div><div class="stat-sub">VRAM Available</div><div class="bar-wrap"><div class="bar-fill bar-yellow" id="bar-vram" style="width:0%"></div></div></div>
        </div>
        <div id="sys-device" style="margin-top:1rem;font-size:12px;color:var(--muted)"></div>
      </div>
      <div class="card">
        <div class="card-title"><span>🔌</span> Available Platforms &amp; Presets</div>
        <div id="platforms-list" style="display:flex;flex-wrap:wrap;gap:.5rem"></div>
      </div>
    </div>

    <!-- ── DATASETS PANEL ── -->
    <div class="panel" id="panel-datasets">
      <div class="card">
        <div class="card-title" style="justify-content:space-between">
          <span><span>🗂</span> HuggingFace Datasets</span>
          <div style="display:flex;gap:.5rem;align-items:center">
            <div id="sel-dataset-display" style="display:none"></div>
            <button class="btn btn-ghost" style="font-size:12px;padding:.3rem .75rem" onclick="loadDatasets()">↻ Refresh</button>
          </div>
        </div>
        <div class="form-grid" style="margin-bottom:1rem">
          <div class="field" style="grid-column:1/-1">
            <input type="text" id="dataset-search" placeholder="Search datasets..." oninput="debounceDatasets(this.value)">
          </div>
          <div class="field">
            <label>Task Category</label>
            <select id="dataset-task" onchange="loadDatasets()">
              <option value="">All Tasks</option>
              <option value="text-classification">Text Classification</option>
              <option value="token-classification">Token Classification</option>
              <option value="question-answering">Question Answering</option>
              <option value="text-generation">Text Generation</option>
              <option value="summarization">Summarization</option>
              <option value="translation">Translation</option>
              <option value="image-classification">Image Classification</option>
              <option value="automatic-speech-recognition">Speech Recognition</option>
            </select>
          </div>
          <div class="field">
            <label>Sort By</label>
            <select id="dataset-sort" onchange="loadDatasets()">
              <option value="downloads">Downloads</option>
              <option value="trendingScore">Trending</option>
              <option value="likes">Likes</option>
            </select>
          </div>
        </div>
        <div id="dataset-list">
          <div class="empty"><div class="empty-icon">🗂</div>Click Refresh to load datasets</div>
        </div>
      </div>
      <div class="card" id="dataset-eval-card" style="display:none">
        <div class="card-title"><span>✅</span> Selected Eval Dataset</div>
        <div id="dataset-eval-info"></div>
        <button class="btn btn-ghost" style="margin-top:.75rem" onclick="clearDataset()">✕ Clear</button>
      </div>
    </div>

    <!-- ── API DOCS PANEL ── -->
    <div class="panel" id="panel-api">
      <div class="card">
        <div class="card-title"><span>📖</span> API Reference</div>
        <div style="display:flex;flex-direction:column;gap:.75rem" id="api-endpoints"></div>
      </div>
    </div>

    <!-- ── PLAYGROUND PANEL ── -->
    <div class="panel" id="panel-playground">

      <div class="card">
        <div class="card-title"><span>🎮</span> Model Playground</div>
        <div class="form-grid" style="margin-bottom:1rem">
          <div class="field" style="grid-column:1/-1">
            <label>Local Optimized Model</label>
            <div style="display:flex;gap:.5rem">
              <select id="pg-model-select" style="flex:1">
                <option value="">— select a model from outputs/ —</option>
              </select>
              <button class="btn btn-ghost" style="padding:.6rem .85rem;font-size:12px" onclick="loadPlaygroundModels()">↻</button>
              <button class="btn btn-primary" id="pg-load-btn" onclick="playgroundLoad()" disabled>Load</button>
            </div>
          </div>
        </div>
        <div id="pg-status" class="pg-status">⬜ No model loaded</div>
      </div>

      <!-- TEXT-GEN UI -->
      <div class="card pg-task-ui" id="pg-ui-text-generation">
        <div class="card-title"><span>✍️</span> Text Generation</div>
        <div class="field" style="margin-bottom:.75rem">
          <label>Prompt</label>
          <textarea id="pg-tg-prompt" rows="3" placeholder="Enter your prompt..."></textarea>
        </div>
        <div class="form-grid" style="margin-bottom:.75rem">
          <div class="field">
            <label>Max Tokens</label>
            <input type="number" id="pg-tg-tokens" value="100" min="1" max="2048">
          </div>
          <div class="field" style="justify-content:flex-end">
            <button class="btn btn-primary" id="pg-tg-btn" onclick="playgroundRun('text-generation')">▶ Generate</button>
          </div>
        </div>
        <div class="field">
          <label>Output</label>
          <div class="pg-output empty" id="pg-tg-output">Output will appear here...</div>
        </div>
        <div id="pg-tg-stats" style="display:none;margin-top:.5rem;font-size:11px;color:var(--muted);display:flex;gap:1rem"></div>
      </div>

      <!-- TEXT-CLASSIFICATION UI -->
      <div class="card pg-task-ui" id="pg-ui-text-classification">
        <div class="card-title"><span>🏷</span> Text Classification</div>
        <div class="field" style="margin-bottom:.75rem">
          <label>Input Text</label>
          <textarea id="pg-tc-text" rows="3" placeholder="Enter text to classify..."></textarea>
        </div>
        <button class="btn btn-primary" id="pg-tc-btn" onclick="playgroundRun('text-classification')" style="margin-bottom:.75rem">▶ Classify</button>
        <div id="pg-tc-results"></div>
      </div>

      <!-- IMAGE-CLASSIFICATION UI -->
      <div class="card pg-task-ui" id="pg-ui-image-classification">
        <div class="card-title"><span>🖼</span> Image Classification</div>
        <div class="drop-zone" id="pg-img-drop" onclick="document.getElementById('pg-img-input').click()" ondragover="event.preventDefault();this.classList.add('dragging')" ondragleave="this.classList.remove('dragging')" ondrop="pgImgDrop(event)">
          <div class="drop-icon">🖼</div>
          <div>Drop image or click to browse</div>
          <div style="font-size:11px;margin-top:.25rem">JPG · PNG · WEBP</div>
        </div>
        <input type="file" id="pg-img-input" accept="image/*" style="display:none" onchange="pgImgSelected(this)">
        <img id="pg-img-preview" class="pg-img-preview">
        <button class="btn btn-primary" id="pg-ic-btn" onclick="playgroundRun('image-classification')" style="margin-top:.75rem" disabled>▶ Predict</button>
        <div id="pg-ic-results" style="margin-top:.75rem"></div>
      </div>

      <!-- TEXT-TO-IMAGE UI -->
      <div class="card pg-task-ui" id="pg-ui-text-to-image">
        <div class="card-title"><span>🎨</span> Text → Image</div>
        <div class="field" style="margin-bottom:.75rem">
          <label>Prompt</label>
          <input type="text" id="pg-ti-prompt" placeholder="A photo of a cat on a beach...">
        </div>
        <div class="form-grid" style="margin-bottom:.75rem">
          <div class="field">
            <label>Steps</label>
            <input type="number" id="pg-ti-steps" value="20" min="1" max="100">
          </div>
          <div class="field" style="justify-content:flex-end">
            <button class="btn btn-primary" id="pg-ti-btn" onclick="playgroundRun('text-to-image')">▶ Generate</button>
          </div>
        </div>
        <img id="pg-ti-img" class="pg-img-preview">
      </div>

      <!-- ASR UI -->
      <div class="card pg-task-ui" id="pg-ui-automatic-speech-recognition">
        <div class="card-title"><span>🎙</span> Speech → Text</div>
        <div class="drop-zone" onclick="document.getElementById('pg-audio-input').click()">
          <div class="drop-icon">🎙</div>
          <div id="pg-audio-name">Drop audio or click to browse</div>
          <div style="font-size:11px;margin-top:.25rem">WAV · MP3 · FLAC</div>
        </div>
        <input type="file" id="pg-audio-input" accept="audio/*" style="display:none" onchange="pgAudioSelected(this)">
        <button class="btn btn-primary" id="pg-asr-btn" onclick="playgroundRun('automatic-speech-recognition')" style="margin-top:.75rem" disabled>▶ Transcribe</button>
        <div class="field" style="margin-top:.75rem">
          <label>Transcript</label>
          <div class="pg-output empty" id="pg-asr-output">Transcript will appear here...</div>
        </div>
      </div>

      <!-- SEQ2SEQ UI -->
      <div class="card pg-task-ui" id="pg-ui-translation-or-summarization">
        <div class="card-title"><span>🔄</span> Translation / Summarization</div>
        <div class="field" style="margin-bottom:.75rem">
          <label>Input Text</label>
          <textarea id="pg-s2s-input" rows="4" placeholder="Enter text to translate or summarize..."></textarea>
        </div>
        <button class="btn btn-primary" id="pg-s2s-btn" onclick="playgroundRun('translation-or-summarization')" style="margin-bottom:.75rem">▶ Run</button>
        <div class="field">
          <label>Output</label>
          <div class="pg-output empty" id="pg-s2s-output">Output will appear here...</div>
        </div>
      </div>

      <!-- SUMMARIZATION UI -->
      <div class="card pg-task-ui" id="pg-ui-summarization">
        <div class="card-title"><span>🔄</span> Summarization</div>
        <div class="field" style="margin-bottom:.75rem">
          <label>Input Text</label>
          <textarea id="pg-sum-input" rows="4" placeholder="Enter text to summarize..."></textarea>
        </div>
        <button class="btn btn-primary" id="pg-sum-btn" onclick="playgroundRun('summarization')" style="margin-bottom:.75rem">▶ Summarize</button>
        <div class="field">
          <label>Output</label>
          <div class="pg-output empty" id="pg-sum-output">Output will appear here...</div>
        </div>
      </div>

      <!-- PERF STATS -->
      <div class="card" id="pg-perf-card" style="display:none">
        <div class="card-title"><span>⚡</span> Performance</div>
        <div class="stats-row" id="pg-perf-stats"></div>
      </div>

    </div>

  </main>
</div>

<!-- TOASTS -->
<div class="toasts" id="toasts"></div>

<script>
// ── MODEL CATALOGUE ──
const MODEL_CATALOGUE = [
  // LLMs
  {id:'gpt2', label:'GPT-2 (124M)', cat:'LLM', tags:'small fast text'},
  {id:'gpt2-medium', label:'GPT-2 Medium (345M)', cat:'LLM', tags:'text medium'},
  {id:'gpt2-large', label:'GPT-2 Large (774M)', cat:'LLM', tags:'text large'},
  {id:'gpt2-xl', label:'GPT-2 XL (1.5B)', cat:'LLM', tags:'text xl'},
  {id:'microsoft/phi-2', label:'Phi-2 (2.7B)', cat:'LLM', tags:'best small reasoning phi'},
  {id:'microsoft/phi-1_5', label:'Phi-1.5 (1.3B)', cat:'LLM', tags:'small phi code'},
  {id:'microsoft/Phi-3-mini-4k-instruct', label:'Phi-3 Mini 4K (3.8B)', cat:'LLM', tags:'instruct phi3'},
  {id:'Qwen/Qwen2-1.5B-Instruct', label:'Qwen2 1.5B Instruct', cat:'LLM', tags:'instruct chat qwen'},
  {id:'Qwen/Qwen2-7B-Instruct', label:'Qwen2 7B Instruct', cat:'LLM', tags:'instruct chat qwen large'},
  {id:'Qwen/Qwen2.5-1.5B-Instruct', label:'Qwen2.5 1.5B Instruct', cat:'LLM', tags:'instruct qwen2.5'},
  {id:'Qwen/Qwen2.5-7B-Instruct', label:'Qwen2.5 7B Instruct', cat:'LLM', tags:'instruct qwen2.5 large'},
  {id:'TinyLlama/TinyLlama-1.1B-Chat-v1.0', label:'TinyLlama 1.1B Chat', cat:'LLM', tags:'tiny fast chat instruct'},
  {id:'facebook/opt-125m', label:'OPT 125M', cat:'LLM', tags:'small fast meta'},
  {id:'facebook/opt-350m', label:'OPT 350M', cat:'LLM', tags:'medium meta'},
  {id:'facebook/opt-1.3b', label:'OPT 1.3B', cat:'LLM', tags:'1b meta'},
  {id:'facebook/opt-2.7b', label:'OPT 2.7B', cat:'LLM', tags:'2b meta'},
  {id:'facebook/opt-6.7b', label:'OPT 6.7B', cat:'LLM', tags:'large meta'},
  {id:'EleutherAI/gpt-neo-125M', label:'GPT-Neo 125M', cat:'LLM', tags:'small neox'},
  {id:'EleutherAI/gpt-neo-1.3B', label:'GPT-Neo 1.3B', cat:'LLM', tags:'1b neox'},
  {id:'EleutherAI/gpt-neo-2.7B', label:'GPT-Neo 2.7B', cat:'LLM', tags:'2b neox'},
  {id:'EleutherAI/gpt-j-6b', label:'GPT-J 6B', cat:'LLM', tags:'6b strong neox'},
  {id:'EleutherAI/gpt-neox-20b', label:'GPT-NeoX 20B', cat:'LLM', tags:'20b large'},
  {id:'bigscience/bloom-560m', label:'BLOOM 560M', cat:'LLM', tags:'multilingual small'},
  {id:'bigscience/bloom-1b7', label:'BLOOM 1.7B', cat:'LLM', tags:'multilingual 1b'},
  {id:'bigscience/bloom-7b1', label:'BLOOM 7B', cat:'LLM', tags:'multilingual large'},
  {id:'tiiuae/falcon-7b', label:'Falcon 7B', cat:'LLM', tags:'falcon 7b'},
  {id:'tiiuae/falcon-7b-instruct', label:'Falcon 7B Instruct', cat:'LLM', tags:'falcon instruct chat'},
  {id:'mistralai/Mistral-7B-v0.1', label:'Mistral 7B 🔒', cat:'LLM', tags:'mistral gated 7b'},
  {id:'mistralai/Mistral-7B-Instruct-v0.2', label:'Mistral 7B Instruct 🔒', cat:'LLM', tags:'mistral instruct gated'},
  {id:'meta-llama/Llama-3.2-1B-Instruct', label:'Llama 3.2 1B Instruct 🔒', cat:'LLM', tags:'llama meta gated instruct'},
  {id:'meta-llama/Llama-3.2-3B-Instruct', label:'Llama 3.2 3B Instruct 🔒', cat:'LLM', tags:'llama meta gated instruct'},
  {id:'meta-llama/Meta-Llama-3-8B-Instruct', label:'Llama 3 8B Instruct 🔒', cat:'LLM', tags:'llama3 meta gated instruct'},
  {id:'google/gemma-2b', label:'Gemma 2B 🔒', cat:'LLM', tags:'google gemma gated'},
  {id:'google/gemma-7b', label:'Gemma 7B 🔒', cat:'LLM', tags:'google gemma gated large'},
  {id:'stabilityai/stablelm-2-1_6b', label:'StableLM 2 1.6B', cat:'LLM', tags:'stablelm stability'},
  {id:'stabilityai/stablelm-zephyr-3b', label:'StableLM Zephyr 3B', cat:'LLM', tags:'stablelm instruct zephyr'},
  {id:'HuggingFaceTB/SmolLM-1.7B-Instruct', label:'SmolLM 1.7B Instruct', cat:'LLM', tags:'smol tiny instruct fast'},
  {id:'HuggingFaceTB/SmolLM-360M-Instruct', label:'SmolLM 360M Instruct', cat:'LLM', tags:'smol tiny fast instruct'},
  // BERT / Encoder
  {id:'bert-base-uncased', label:'BERT Base Uncased', cat:'Encoder', tags:'bert classification embedding'},
  {id:'bert-large-uncased', label:'BERT Large Uncased', cat:'Encoder', tags:'bert large classification'},
  {id:'roberta-base', label:'RoBERTa Base', cat:'Encoder', tags:'roberta classification'},
  {id:'roberta-large', label:'RoBERTa Large', cat:'Encoder', tags:'roberta large classification'},
  {id:'distilbert-base-uncased', label:'DistilBERT Base', cat:'Encoder', tags:'distil bert small fast'},
  {id:'albert-base-v2', label:'ALBERT Base v2', cat:'Encoder', tags:'albert small efficient'},
  {id:'sentence-transformers/all-MiniLM-L6-v2', label:'MiniLM L6 (embeddings)', cat:'Encoder', tags:'embedding sentence similarity fast'},
  {id:'sentence-transformers/all-mpnet-base-v2', label:'MPNet Base (embeddings)', cat:'Encoder', tags:'embedding sentence similarity'},
  // Seq2Seq
  {id:'t5-small', label:'T5 Small', cat:'Seq2Seq', tags:'t5 summarize translate small'},
  {id:'t5-base', label:'T5 Base', cat:'Seq2Seq', tags:'t5 summarize translate'},
  {id:'t5-large', label:'T5 Large', cat:'Seq2Seq', tags:'t5 summarize translate large'},
  {id:'google/flan-t5-small', label:'Flan-T5 Small', cat:'Seq2Seq', tags:'flan t5 instruct small'},
  {id:'google/flan-t5-base', label:'Flan-T5 Base', cat:'Seq2Seq', tags:'flan t5 instruct'},
  {id:'google/flan-t5-large', label:'Flan-T5 Large', cat:'Seq2Seq', tags:'flan t5 instruct large'},
  {id:'google/flan-t5-xl', label:'Flan-T5 XL (3B)', cat:'Seq2Seq', tags:'flan t5 instruct xl'},
  {id:'facebook/bart-base', label:'BART Base', cat:'Seq2Seq', tags:'bart summarize'},
  {id:'facebook/bart-large', label:'BART Large', cat:'Seq2Seq', tags:'bart summarize large'},
  {id:'facebook/bart-large-cnn', label:'BART Large CNN', cat:'Seq2Seq', tags:'bart summarize news'},
  // Vision
  {id:'google/vit-base-patch16-224', label:'ViT Base (image class)', cat:'Vision', tags:'vit image classification vision'},
  {id:'microsoft/resnet-50', label:'ResNet-50', cat:'Vision', tags:'resnet image classification cnn'},
  {id:'openai/clip-vit-base-patch32', label:'CLIP ViT Base', cat:'Vision', tags:'clip vision text multimodal'},
  // Code
  {id:'Salesforce/codegen-350M-mono', label:'CodeGen 350M', cat:'Code', tags:'code python generation'},
  {id:'Salesforce/codegen-2B-mono', label:'CodeGen 2B', cat:'Code', tags:'code python generation'},
  {id:'microsoft/codebert-base', label:'CodeBERT Base', cat:'Code', tags:'code bert understanding'},
  {id:'bigcode/starcoder2-3b', label:'StarCoder2 3B', cat:'Code', tags:'code starcoder generation'},
  // Reasoning
  {id:'Qwen/QwQ-32B', label:'QwQ 32B', cat:'Reasoning', tags:'reasoning qwen thinking chain-of-thought'},
  {id:'deepseek-ai/DeepSeek-R1', label:'DeepSeek R1 671B 🔒', cat:'Reasoning', tags:'reasoning deepseek gated large'},
  {id:'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', label:'DeepSeek R1 Distill 1.5B', cat:'Reasoning', tags:'reasoning distill small fast'},
  {id:'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', label:'DeepSeek R1 Distill 7B', cat:'Reasoning', tags:'reasoning distill 7b'},
  {id:'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', label:'DeepSeek R1 Distill 8B', cat:'Reasoning', tags:'reasoning distill 8b llama'},
  {id:'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B', label:'DeepSeek R1 Distill 14B', cat:'Reasoning', tags:'reasoning distill 14b'},
  {id:'Qwen/Qwen3-1.7B', label:'Qwen3 1.7B', cat:'Reasoning', tags:'reasoning qwen3 thinking'},
  {id:'Qwen/Qwen3-4B', label:'Qwen3 4B', cat:'Reasoning', tags:'reasoning qwen3 thinking'},
  {id:'Qwen/Qwen3-8B', label:'Qwen3 8B', cat:'Reasoning', tags:'reasoning qwen3 thinking'},
  // Image Generation
  {id:'stabilityai/stable-diffusion-v1-5', label:'SD 1.5', cat:'Image Gen', tags:'image generation diffusion stable'},
  {id:'stabilityai/stable-diffusion-2-1', label:'SD 2.1', cat:'Image Gen', tags:'image generation diffusion stable'},
  {id:'stabilityai/stable-diffusion-xl-base-1.0', label:'SDXL Base 1.0', cat:'Image Gen', tags:'image generation xl diffusion'},
  {id:'black-forest-labs/FLUX.1-schnell', label:'FLUX.1 Schnell (fast)', cat:'Image Gen', tags:'image generation flux fast'},
  {id:'black-forest-labs/FLUX.1-dev', label:'FLUX.1 Dev 🔒', cat:'Image Gen', tags:'image generation flux quality gated'},
  {id:'CompVis/stable-diffusion-v1-4', label:'SD 1.4', cat:'Image Gen', tags:'image generation diffusion'},
  // Audio
  {id:'openai/whisper-tiny', label:'Whisper Tiny', cat:'Audio', tags:'speech asr tiny fast whisper'},
  {id:'openai/whisper-base', label:'Whisper Base', cat:'Audio', tags:'speech asr base whisper'},
  {id:'openai/whisper-small', label:'Whisper Small', cat:'Audio', tags:'speech asr small whisper'},
  {id:'openai/whisper-medium', label:'Whisper Medium', cat:'Audio', tags:'speech asr medium whisper'},
  {id:'openai/whisper-large-v3', label:'Whisper Large v3', cat:'Audio', tags:'speech asr large best whisper'},
  {id:'facebook/wav2vec2-base-960h', label:'Wav2Vec2 Base', cat:'Audio', tags:'speech wav2vec recognition'},
];

const TASK_ICONS = {'text-generation':'✍️','text2text-generation':'🔄','fill-mask':'🎭','token-classification':'🏷','question-answering':'❓','summarization':'📝','translation':'🌐','sentence-similarity':'🔗','image-classification':'🖼','object-detection':'🔍','image-to-text':'👁','text-to-image':'🎨','automatic-speech-recognition':'🎙','audio-classification':'🎵','zero-shot-classification':'🎯','feature-extraction':'📊','text-ranking':'📈',''  :'🤖'};
let _searchTimer = null;
let _lastQuery = null;
let _activeCategories = new Set(['all']);

const TASK_CATEGORIES = [
  {id:'all',       label:'🌐 All',         hf:'',                             local:''},
  {id:'trending',  label:'🔥 Trending',    hf:'',                             local:'', trending:true},
  {id:'Reasoning', label:'🧠 Reasoning',   hf:'',                             local:'Reasoning'},
  {id:'LLM',       label:'✍️ LLM',          hf:'text-generation',              local:'LLM'},
  {id:'Image Gen', label:'🎨 Image Gen',   hf:'text-to-image',                local:'Image Gen'},
  {id:'Code',      label:'💻 Code',         hf:'',                             local:'Code'},
  {id:'Encoder',   label:'📊 Encoder',     hf:'fill-mask',                    local:'Encoder'},
  {id:'Seq2Seq',   label:'🔄 Seq2Seq',     hf:'text2text-generation',         local:'Seq2Seq'},
  {id:'Vision',    label:'🖼 Vision',       hf:'image-classification',         local:'Vision'},
  {id:'Audio',     label:'🎙 Audio',        hf:'automatic-speech-recognition', local:'Audio'},
];

const MODEL_AUTOCONFIG = {
  Reasoning:   {platform:'laptop', preset:'max_accuracy',  quant:true,  prune:false, distill:true,  reason:true},
  LLM:         {platform:'laptop', preset:'balanced',       quant:true,  prune:true,  distill:false, reason:true},
  'Image Gen': {platform:'cloud',  preset:'balanced',       quant:true,  prune:false, distill:false, reason:false},
  Code:        {platform:'laptop', preset:'max_accuracy',  quant:true,  prune:false, distill:true,  reason:false},
  Encoder:     {platform:'mobile', preset:'mobile_safe',   quant:true,  prune:true,  distill:true,  reason:false},
  Seq2Seq:     {platform:'laptop', preset:'balanced',       quant:true,  prune:true,  distill:true,  reason:false},
  Vision:      {platform:'laptop', preset:'balanced',       quant:true,  prune:true,  distill:false, reason:false},
  Audio:       {platform:'mobile', preset:'mobile_safe',   quant:true,  prune:true,  distill:false, reason:false},
};

function fmtNum(n) {
  if (!n) return '0';
  if (n >= 1e9) return (n/1e9).toFixed(1)+'B';
  if (n >= 1e6) return (n/1e6).toFixed(1)+'M';
  if (n >= 1e3) return (n/1e3).toFixed(0)+'K';
  return n;
}

function _modelRow(id, subtitle, badge) {
  return '<div data-mid="' + id.replace(/"/g,'&quot;') + '" class="mrow" style="padding:.5rem .75rem;cursor:pointer;display:grid;grid-template-columns:1fr auto;gap:.5rem;align-items:center;border-bottom:1px solid rgba(30,30,53,.5);font-size:12px">'
    + '<div><span style="color:var(--text)">📦 ' + id + '</span>'
    + '<div style="font-size:10px;color:var(--muted);margin-top:.1rem">' + subtitle + '</div></div>'
    + '<div style="font-size:10px;color:var(--green)">' + badge + '</div></div>';
}

function _fmtParams(n) {
  if (!n) return '';
  if (n >= 1e12) return (n/1e12).toFixed(1) + 'T';
  if (n >= 1e9)  return (n/1e9).toFixed(1) + 'B';
  if (n >= 1e6)  return (n/1e6).toFixed(0) + 'M';
  return n + '';
}
function _fmtSize(bytes) {
  if (!bytes) return '';
  if (bytes >= 1e9) return (bytes/1e9).toFixed(1) + ' GB';
  if (bytes >= 1e6) return (bytes/1e6).toFixed(0) + ' MB';
  return (bytes/1e3).toFixed(0) + ' KB';
}
function _hfRow(m) {
  const icon = TASK_ICONS[m.task] || '🤖';
  const gated = m.gated ? ' 🔒' : '';
  const paramsBadge = m.params ? '<span style="background:rgba(79,142,247,.15);color:var(--blue);padding:.1rem .35rem;border-radius:4px;font-size:10px;margin-right:.3rem">⚙ ' + _fmtParams(m.params) + '</span>' : '';
  const sizeBadge  = m.size_bytes ? '<span style="background:rgba(0,255,136,.1);color:var(--green);padding:.1rem .35rem;border-radius:4px;font-size:10px">💾 ' + _fmtSize(m.size_bytes) + '</span>' : '';
  return '<div data-mid="' + m.id.replace(/"/g,'&quot;') + '" class="mrow" style="padding:.5rem .75rem;cursor:pointer;display:grid;grid-template-columns:1fr auto;gap:.5rem;align-items:center;border-bottom:1px solid rgba(30,30,53,.5);font-size:12px">'
    + '<div><span style="color:var(--text)">' + icon + ' ' + m.id + gated + '</span>'
    + '<div style="font-size:10px;color:var(--muted);margin-top:.25rem">' + (m.task||'general') + (paramsBadge || sizeBadge ? '&nbsp;&nbsp;' + paramsBadge + sizeBadge : '') + '</div></div>'
    + '<div style="text-align:right;font-size:10px;color:var(--muted)">⬇ ' + fmtNum(m.downloads) + '</div></div>';
}

function _attachRows(list) {
  list.querySelectorAll('.mrow').forEach(function(el) {
    el.onmouseover = function() { this.style.background = 'var(--surface2)'; };
    el.onmouseout  = function() { this.style.background = ''; };
    el.onclick = function() { selectModel(this.getAttribute('data-mid')); };
  });
}

async function fetchModels(q) {
  const list = document.getElementById('model-list');
  const isAll = _activeCategories.has('all');

  // Filter local catalogue — OR logic across selected categories
  let local = MODEL_CATALOGUE;
  if (!isAll) {
    local = local.filter(function(m) { return _activeCategories.has(m.cat); });
  }
  if (q) {
    const ql = q.toLowerCase();
    local = local.filter(function(m) {
      return m.id.toLowerCase().includes(ql) || m.tags.toLowerCase().includes(ql) || m.label.toLowerCase().includes(ql);
    });
  }

  // Show local results immediately
  list.innerHTML = local.map(function(m) {
    return _modelRow(m.id, m.cat + ' · ' + m.label, 'Local');
  }).join('') || '<div style="padding:.75rem;text-align:center;font-size:12px;color:var(--muted)">⏳ Searching HuggingFace...</div>';
  _attachRows(list);

  // Collect unique HF task tags + trending flags from active categories
  const hfTasks = [];
  let isTrending = false;
  _activeCategories.forEach(function(cat) {
    const c = TASK_CATEGORIES.find(function(tc) { return tc.id === cat; });
    if (!c) return;
    if (c.trending) isTrending = true;
    if (c.hf && !hfTasks.includes(c.hf)) hfTasks.push(c.hf);
  });

  // Read active filters from chip state vars
  const maxSizeGb  = _filterSizeGb  || 0;
  const maxParamsB = _filterParamsB || 0;
  const hint = document.getElementById('filter-hint');
  if (hint) hint.textContent = (maxSizeGb || maxParamsB) ? '⏳ filtering…' : '🔍 HuggingFace live';

  // Fetch from HuggingFace — one request per HF task, merge + dedup by id
  const seen = new Set();
  const fetches = (hfTasks.length ? hfTasks : ['']).map(function(task) {
    const params = new URLSearchParams({limit: 60, q: q || ''});
    if (task) params.set('task', task);
    if (isTrending) params.set('sort', 'trendingScore');
    if (maxSizeGb)  params.set('max_size_gb',  maxSizeGb);
    if (maxParamsB) params.set('max_params_b', maxParamsB);
    return fetch('/api/models/search?' + params).then(function(r) { return r.json(); }).catch(function() { return []; });
  });

  Promise.all(fetches).then(function(results) {
    const merged = [];
    results.forEach(function(models) {
      models.forEach(function(m) {
        if (!seen.has(m.id)) { seen.add(m.id); merged.push(m); }
      });
    });
    const hint2 = document.getElementById('filter-hint');
    if (merged.length) {
      list.innerHTML += merged.map(_hfRow).join('');
      _attachRows(list);
      if (hint2) hint2.textContent = merged.length + ' results';
    } else if (!local.length) {
      const filterMsg = (maxSizeGb || maxParamsB)
        ? '⚠ No matches — try relaxing filters or search a specific model'
        : '⚠ HF error — type model ID directly';
      list.innerHTML = '<div style="padding:.75rem;text-align:center;font-size:12px;color:var(--red)">' + filterMsg + '</div>';
      if (hint2) hint2.textContent = '⚠ No results';
    } else {
      if (hint2) hint2.textContent = '🔍 HuggingFace live';
    }
  });
}

function renderCategoryChips() {
  const container = document.getElementById('category-chips');
  container.innerHTML = TASK_CATEGORIES.map(function(c) {
    const active = _activeCategories.has(c.id);
    return '<span data-cat="' + c.id + '" style="display:inline-flex;align-items:center;padding:.2rem .55rem;border-radius:20px;font-size:11px;cursor:pointer;white-space:nowrap;'
      + (active ? 'background:var(--green);color:#000;font-weight:600' : 'background:var(--surface2);color:var(--muted);border:1px solid var(--border)')
      + '">' + c.label + '</span>';
  }).join('');
  container.querySelectorAll('[data-cat]').forEach(function(el) {
    el.onclick = function() { toggleCategory(this.getAttribute('data-cat')); };
  });
}

function toggleCategory(cat) {
  if (cat === 'all') {
    _activeCategories = new Set(['all']);
  } else {
    _activeCategories.delete('all');
    if (_activeCategories.has(cat)) {
      _activeCategories.delete(cat);
      if (_activeCategories.size === 0) _activeCategories.add('all');
    } else {
      _activeCategories.add(cat);
    }
  }
  renderCategoryChips();
  _lastQuery = null;
  fetchModels(document.getElementById('model-id').value);
}

function selectModel(id) {
  document.getElementById('model-id').value = id;
  hideModelPicker();
  const meta = MODEL_CATALOGUE.find(function(m) { return m.id === id; });
  if (meta) {
    const cfg = MODEL_AUTOCONFIG[meta.cat];
    if (cfg) {
      document.getElementById('target-platform').value = cfg.platform;
      document.getElementById('preset').value = cfg.preset;
      document.getElementById('use-quant').checked = cfg.quant;
      document.getElementById('use-prune').checked = cfg.prune;
      document.getElementById('use-distill').checked = cfg.distill;
      document.getElementById('use-reason').checked = cfg.reason;
      toast('Auto-configured: ' + cfg.platform + ' / ' + cfg.preset, 'success');
    }
  }
  fetchModelDetails(id);
}

async function fetchModelDetails(id) {
  const card = document.getElementById('model-info-card');
  card.style.display = 'block';
  card.innerHTML = '<div style="color:var(--muted);font-size:12px;animation:pulse 1s infinite">⏳ Loading model details...</div>';
  try {
    const res = await fetch('/api/model-info/' + encodeURIComponent(id));
    const d = await res.json();
    if (d.error) { card.innerHTML = '<div style="color:var(--red);font-size:12px">⚠ ' + d.error + '</div>'; return; }
    showModelInfoCard(d);
    document.getElementById('infer-btn').style.display = '';
  } catch(e) {
    card.innerHTML = '<div style="color:var(--muted);font-size:12px">Could not load model details</div>';
  }
}

function showModelInfoCard(d) {
  const card = document.getElementById('model-info-card');
  const totalMB = d.total_size_bytes ? (d.total_size_bytes / 1024 / 1024) : 0;
  const totalGB = (totalMB / 1024).toFixed(2);
  const sizeStr = totalMB > 1024
    ? '<span class="size-warn">⚠ ~' + totalGB + ' GB download</span>'
    : totalMB > 0 ? '<span class="size-ok">~' + totalMB.toFixed(0) + ' MB</span>' : '<span style="color:var(--muted)">size unknown</span>';

  const license = d.license ? d.license.replace('license:', '') : 'unknown';
  const params = d.safetensors && d.safetensors.total ? fmtNum(d.safetensors.total) + ' params' : '';
  const taskBadge = d.task ? '<span class="model-info-task">' + d.task + '</span>' : '';
  const gatedWarn = d.gated ? '<span style="color:var(--yellow);font-size:11px">🔒 Gated — requires HF token</span>' : '';

  const topTags = (d.tags || []).filter(function(t) {
    return !t.startsWith('license:') && !t.startsWith('arxiv:') && t.length < 30;
  }).slice(0, 10).map(function(t) {
    return '<span class="model-tag">' + t + '</span>';
  }).join('');

  const filesHtml = d.files && d.files.length
    ? '<details class="model-files"><summary>' + d.files.length + ' files</summary><div style="margin-top:.35rem">'
      + d.files.slice(0, 15).map(function(f) {
          const mb = f.size ? (f.size / 1024 / 1024).toFixed(1) + ' MB' : '';
          return '<div style="display:flex;justify-content:space-between;padding:.2rem 0;border-bottom:1px solid rgba(255,255,255,.04)">'
            + '<span style="color:var(--text);font-family:monospace;font-size:11px">' + f.name + '</span>'
            + '<span style="color:var(--muted);font-size:11px">' + mb + '</span></div>';
        }).join('')
      + '</div></details>'
    : '';

  card.innerHTML = '<div class="model-info-header">'
    + '<div class="model-info-id">📦 ' + d.id + '</div>' + taskBadge
    + '</div>'
    + '<div class="model-meta-row">'
    + '<span>⬇ <strong>' + fmtNum(d.downloads) + '</strong> downloads</span>'
    + '<span>♥ <strong>' + fmtNum(d.likes) + '</strong></span>'
    + '<span>📜 <strong>' + license + '</strong></span>'
    + (params ? '<span>🔢 <strong>' + params + '</strong></span>' : '')
    + '<span>' + sizeStr + '</span>'
    + (gatedWarn ? '<span>' + gatedWarn + '</span>' : '')
    + '</div>'
    + (topTags ? '<div class="model-tags">' + topTags + '</div>' : '')
    + filesHtml;
}

function toggleInferPanel() {
  const p = document.getElementById('infer-panel');
  p.style.display = p.style.display === 'none' ? 'block' : 'none';
}

async function runInferenceTest() {
  const modelId = document.getElementById('model-id').value.trim();
  const prompt = document.getElementById('infer-prompt').value.trim();
  if (!modelId || !prompt) { toast('Need model ID and prompt', 'error'); return; }

  const respEl = document.getElementById('infer-response');
  respEl.style.display = 'block';
  respEl.className = 'infer-response loading';
  respEl.textContent = '⏳ Running inference...';

  try {
    const res = await fetch('/api/inference/' + encodeURIComponent(modelId), {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({inputs: prompt, parameters: {max_new_tokens: 100}})
    });
    const data = await res.json();
    respEl.className = 'infer-response';
    if (Array.isArray(data) && data[0] && data[0].generated_text) {
      respEl.textContent = data[0].generated_text;
    } else if (data.error) {
      respEl.style.color = 'var(--red)';
      respEl.textContent = '⚠ ' + data.error;
    } else {
      respEl.textContent = JSON.stringify(data, null, 2);
    }
  } catch(e) {
    respEl.className = 'infer-response';
    respEl.style.color = 'var(--red)';
    respEl.textContent = '⚠ ' + e.message;
  }
}

let _selectedDataset = null;
let _datasetTimer = null;

function debounceDatasets(val) {
  clearTimeout(_datasetTimer);
  _datasetTimer = setTimeout(loadDatasets, 350);
}

async function loadDatasets() {
  const wrap = document.getElementById('dataset-list');
  wrap.innerHTML = '<div style="padding:.75rem;text-align:center;font-size:12px;color:var(--muted)">⏳ Loading datasets...</div>';
  const q = (document.getElementById('dataset-search') || {value:''}).value || '';
  const task = (document.getElementById('dataset-task') || {value:''}).value || '';
  const sort = (document.getElementById('dataset-sort') || {value:'downloads'}).value || 'downloads';
  try {
    const params = new URLSearchParams({q: q, limit: 50, sort: sort});
    if (task) params.set('task', task);
    const res = await fetch('/api/datasets/search?' + params);
    const datasets = await res.json();
    if (!datasets.length) {
      wrap.innerHTML = '<div class="empty"><div class="empty-icon">📭</div>No datasets found</div>';
      return;
    }
    wrap.innerHTML = datasets.map(function(d) {
      const sel = _selectedDataset && _selectedDataset.id === d.id ? ' selected' : '';
      return '<div class="dataset-row' + sel + '" data-dsid="' + d.id.replace(/"/g,'&quot;') + '">'
        + '<div><span style="color:var(--text);font-size:13px">🗂 ' + d.id + '</span>'
        + '<div style="font-size:10px;color:var(--muted);margin-top:.1rem">' + (d.task || 'general') + '</div></div>'
        + '<div style="text-align:right">'
        + '<div style="font-size:10px;color:var(--muted)">⬇ ' + fmtNum(d.downloads) + '</div>'
        + '<button style="font-size:10px;margin-top:.25rem;padding:.1rem .4rem" class="btn btn-ghost">Use</button>'
        + '</div></div>';
    }).join('');
    wrap.querySelectorAll('[data-dsid]').forEach(function(el) {
      el.querySelector('button').onclick = function(e) {
        e.stopPropagation();
        const did = el.getAttribute('data-dsid');
        const dt = datasets.find(function(d) { return d.id === did; });
        selectDataset(dt || {id: did});
      };
    });
  } catch(e) {
    wrap.innerHTML = '<div class="empty"><div class="empty-icon">⚠️</div>Failed to load datasets</div>';
  }
}

function selectDataset(d) {
  _selectedDataset = d;
  toast('Eval dataset: ' + d.id, 'success');
  const badge = document.getElementById('sel-dataset-display');
  badge.style.display = '';
  badge.innerHTML = '<span class="sel-dataset-badge">📊 ' + d.id + '</span>';
  document.getElementById('dataset-eval-card').style.display = 'block';
  document.getElementById('dataset-eval-info').innerHTML =
    '<div style="font-size:13px;color:var(--text)">🗂 ' + d.id + '</div>'
    + '<div style="font-size:11px;color:var(--muted);margin-top:.35rem">Task: ' + (d.task || 'general') + ' · ⬇ ' + fmtNum(d.downloads) + ' downloads</div>';
  loadDatasets();
}

function clearDataset() {
  _selectedDataset = null;
  document.getElementById('dataset-eval-card').style.display = 'none';
  document.getElementById('sel-dataset-display').style.display = 'none';
  loadDatasets();
}

function showModelPicker() {
  document.getElementById('model-picker').style.display = 'block';
  renderCategoryChips();
  _renderFilterChips();
  const cur = document.getElementById('model-id').value;
  fetchModels(cur);
  _lastQuery = cur;
}
function hideModelPicker() {
  document.getElementById('model-picker').style.display = 'none';
}
// ── MODEL FILTERS (chip-based, no native select) ──
const _SIZE_OPTS   = [{v:0,l:'Any'},{v:0.5,l:'500MB'},{v:1,l:'1GB'},{v:3,l:'3GB'},{v:7,l:'7GB'},{v:13,l:'13GB'},{v:30,l:'30GB'},{v:70,l:'70GB'}];
const _PARAMS_OPTS = [{v:0,l:'Any'},{v:0.125,l:'125M'},{v:0.5,l:'500M'},{v:1,l:'1B'},{v:3,l:'3B'},{v:7,l:'7B'},{v:13,l:'13B'},{v:70,l:'70B'}];
let _filterSizeGb   = 0;
let _filterParamsB  = 0;

function _renderFilterChips() {
  function chips(opts, selected, setter, containerId) {
    const c = document.getElementById(containerId);
    if (!c) return;
    c.innerHTML = opts.map(function(o) {
      const active = o.v === selected;
      return '<div onmousedown="event.preventDefault()" onclick="' + setter + '(' + o.v + ')" style="padding:.15rem .5rem;border-radius:20px;font-size:10px;cursor:pointer;border:1px solid ' + (active ? 'var(--green)' : 'var(--border)') + ';color:' + (active ? 'var(--green)' : 'var(--muted)') + ';background:' + (active ? 'rgba(0,255,136,.1)' : 'transparent') + '">' + o.l + '</div>';
    }).join('');
  }
  chips(_SIZE_OPTS,   _filterSizeGb,  'setFilterSize',   'filter-size-chips');
  chips(_PARAMS_OPTS, _filterParamsB, 'setFilterParams', 'filter-params-chips');
}

function setFilterSize(v) {
  _filterSizeGb = v;
  _renderFilterChips();
  applyModelFilters();
}
function setFilterParams(v) {
  _filterParamsB = v;
  _renderFilterChips();
  applyModelFilters();
}

function applyModelFilters() {
  const q = (document.getElementById('model-id') || {value:''}).value;
  _lastQuery = null;
  document.getElementById('model-picker').style.display = 'block';
  fetchModels(q);
}

function filterModels(val) {
  document.getElementById('model-picker').style.display = 'block';
  if (!document.getElementById('category-chips').innerHTML) renderCategoryChips();
  clearTimeout(_searchTimer);
  _searchTimer = setTimeout(function() { if (val !== _lastQuery) { _lastQuery = val; fetchModels(val); } }, 300);
}

// ── STATE ──
let activeJobId = null;
let pollTimer = null;
let selectedFile = null;
let logLines = [];

// ── NAVIGATION ──
function showPanel(name, el) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('panel-' + name).classList.add('active');
  if (el) {
    el.classList.add('active');
  } else if (event && event.currentTarget) {
    event.currentTarget.classList.add('active');
  }
  if (name === 'system') loadSystem();
  if (name === 'jobs') loadJobs();
  if (name === 'api') renderApiDocs();
  if (name === 'playground') loadPlaygroundModels();
}

// ── TABS ──
function switchTab(id, el) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.getElementById('tab-' + id).classList.add('active');
  if (el) {
    el.classList.add('active');
  } else if (event && event.currentTarget) {
    event.currentTarget.classList.add('active');
  }
}

// ── TOAST ──
function toast(msg, type='info') {
  const c = document.getElementById('toasts');
  const t = document.createElement('div');
  t.className = 'toast toast-' + (type === 'error' ? 'err' : type === 'success' ? 'ok' : 'info');
  t.innerHTML = (type==='success'?'✓':type==='error'?'✗':'ℹ') + ' ' + msg;
  c.appendChild(t);
  setTimeout(() => t.remove(), 3500);
}

// ── LOG ──
function addLog(msg, type='') {
  logLines.push({msg, type, time: new Date().toLocaleTimeString()});
  const box = document.getElementById('job-log');
  const el = document.createElement('div');
  el.className = 'log-entry ' + type;
  el.textContent = '[' + new Date().toLocaleTimeString() + '] ' + msg;
  box.appendChild(el);
  box.scrollTop = box.scrollHeight;
}

// ── FILE UPLOAD ──
function fileSelected(inp) {
  if (inp.files[0]) {
    selectedFile = inp.files[0];
    document.getElementById('file-name').textContent = '📄 ' + selectedFile.name + ' (' + (selectedFile.size/1024/1024).toFixed(2) + ' MB)';
  }
}
function handleDrop(e) {
  e.preventDefault();
  document.getElementById('drop-zone').classList.remove('dragging');
  if (e.dataTransfer.files[0]) {
    selectedFile = e.dataTransfer.files[0];
    document.getElementById('file-name').textContent = '📄 ' + selectedFile.name + ' (' + (selectedFile.size/1024/1024).toFixed(2) + ' MB)';
  }
}

// ── OPTIMIZE BY ID ──
async function runOptimize() {
  const modelId = document.getElementById('model-id').value.trim();
  if (!modelId) { toast('Enter a model ID', 'error'); return; }
  const btn = document.getElementById('run-btn');
  btn.disabled = true; btn.textContent = '⏳ Submitting...';
  resetJobUI();
  addLog('Submitting optimization job for ' + modelId + '...', 'info');
  try {
    const res = await fetch('/api/optimize', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        model_id: modelId,
        target_platform: document.getElementById('target-platform').value,
        preset: document.getElementById('preset').value,
        max_accuracy_drop: parseFloat(document.getElementById('max-acc-drop').value),
        use_quantization: document.getElementById('use-quant').checked,
        use_pruning: document.getElementById('use-prune').checked,
        use_distillation: document.getElementById('use-distill').checked,
        enable_reasoning_check: document.getElementById('use-reason').checked,
      })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Request failed');
    startPolling(data.job_id);
    toast('Job started: ' + data.job_id, 'success');
  } catch(e) {
    addLog('Error: ' + e.message, 'err');
    toast(e.message, 'error');
  } finally {
    btn.disabled = false; btn.textContent = '⚡ Run Optimization';
  }
}

// ── OPTIMIZE BY FILE ──
async function runFileOptimize() {
  if (!selectedFile) { toast('Select a model file first', 'error'); return; }
  const btn = document.getElementById('file-run-btn');
  btn.disabled = true; btn.textContent = '⏳ Uploading...';
  resetJobUI();
  addLog('Uploading ' + selectedFile.name + '...', 'info');
  showPanel('optimize', null);
  try {
    const fd = new FormData();
    fd.append('model', selectedFile);
    fd.append('target_platform', document.getElementById('file-target').value);
    fd.append('preset', document.getElementById('file-preset').value);
    const res = await fetch('/optimize', {method: 'POST', body: fd});
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Upload failed');
    startPolling(data.job_id);
    toast('Job started: ' + data.job_id, 'success');
  } catch(e) {
    addLog('Error: ' + e.message, 'err');
    toast(e.message, 'error');
  } finally {
    btn.disabled = false; btn.textContent = '⚡ Run Optimization';
  }
}

// ── JOB POLLING ──
function resetJobUI() {
  document.getElementById('active-job-card').style.display = 'block';
  document.getElementById('job-results').style.display = 'none';
  document.getElementById('job-log').innerHTML = '';
  logLines = [];
  document.getElementById('active-prog-bar').className = 'prog-bar indeterminate';
  document.getElementById('active-job-badge').className = 'badge badge-running';
  document.getElementById('active-job-badge').textContent = 'Running';
  document.getElementById('active-job-msg').textContent = 'Processing...';
}

function startPolling(jobId) {
  activeJobId = jobId;
  document.getElementById('active-job-id').textContent = jobId;
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(() => pollJob(jobId), 2000);
  addLog('Polling job ' + jobId, 'info');
}

async function pollJob(jobId) {
  try {
    const res = await fetch('/jobs/' + jobId);
    const data = await res.json();
    if (!res.ok) { clearInterval(pollTimer); return; }

    const badge = document.getElementById('active-job-badge');
    const msg = document.getElementById('active-job-msg');

    if (data.status === 'processing') {
      badge.className = 'badge badge-running';
      badge.textContent = 'Running';
      msg.textContent = 'Optimization in progress...';
    } else if (data.status === 'completed') {
      clearInterval(pollTimer);
      badge.className = 'badge badge-done';
      badge.textContent = 'Completed';
      msg.textContent = 'Done!';
      document.getElementById('active-prog-bar').className = 'prog-bar';
      document.getElementById('active-prog-bar').style.width = '100%';
      addLog('Job completed successfully', 'ok');
      showResults(data.result);
      loadJobs();
    } else if (data.status === 'failed') {
      clearInterval(pollTimer);
      badge.className = 'badge badge-failed';
      badge.textContent = 'Failed';
      msg.textContent = data.error || 'Job failed';
      document.getElementById('active-prog-bar').style.background = 'var(--red)';
      document.getElementById('active-prog-bar').className = 'prog-bar';
      document.getElementById('active-prog-bar').style.width = '100%';
      addLog('Job failed: ' + (data.error || 'unknown error'), 'err');
      toast('Job failed', 'error');
    }
  } catch(e) {
    addLog('Poll error: ' + e.message, 'err');
  }
}

function showResults(result) {
  if (!result) return;
  document.getElementById('job-results').style.display = 'block';
  document.getElementById('res-compression').textContent = result.compression_ratio ? result.compression_ratio.toFixed(2) + 'x' : '-';
  document.getElementById('res-latency').textContent = result.latency_improvement ? result.latency_improvement.toFixed(1) + '%' : '-';
  document.getElementById('res-orig-size').textContent = result.original_size_mb ? result.original_size_mb.toFixed(1) : '-';
  document.getElementById('res-opt-size').textContent = result.optimized_size_mb ? result.optimized_size_mb.toFixed(1) : '-';
  document.getElementById('res-acc-drop').textContent = result.accuracy_drop !== undefined ? result.accuracy_drop.toFixed(2) + '%' : '-';
  if (result.output_paths && result.output_paths.length) {
    document.getElementById('res-outputs').innerHTML = '📁 Outputs: ' + result.output_paths.map(p => '<code style="color:var(--green)">' + p + '</code>').join(', ');
  }
  toast('Optimization complete!', 'success');
}

// ── LOAD JOBS ──
async function loadJobs() {
  const wrap = document.getElementById('jobs-table-wrap');
  try {
    const res = await fetch('/jobs');
    const jobs = await res.json();
    if (!jobs.length) {
      wrap.innerHTML = '<div class="empty"><div class="empty-icon">📭</div>No jobs yet</div>';
      return;
    }
    const rows = jobs.slice().reverse().map(j => {
      const statusCls = j.status === 'completed' ? 'badge-done' : j.status === 'failed' ? 'badge-failed' : j.status === 'processing' ? 'badge-running' : 'badge-queued';
      const comp = j.result?.compression_ratio ? j.result.compression_ratio.toFixed(2) + 'x' : '-';
      const size = j.result?.optimized_size_mb ? j.result.optimized_size_mb.toFixed(1) + ' MB' : '-';
      return '<tr>' +
        '<td class="mono">' + j.job_id + '</td>' +
        '<td>' + (j.model_id || j.model_filename || '-') + '</td>' +
        '<td><span class="badge ' + statusCls + '">' + j.status + '</span></td>' +
        '<td>' + comp + '</td>' +
        '<td>' + size + '</td>' +
        '<td class="mono">' + (j.created_at ? j.created_at.slice(0,19).replace('T',' ') : '-') + '</td>' +
        '<td><button class="btn btn-ghost" style="font-size:11px;padding:.2rem .6rem" onclick="viewJob(\\'' + j.job_id + '\\')">View</button></td>' +
        '</tr>';
    }).join('');
    wrap.innerHTML = '<table><thead><tr><th>Job ID</th><th>Model</th><th>Status</th><th>Compression</th><th>Output Size</th><th>Created</th><th></th></tr></thead><tbody>' + rows + '</tbody></table>';
  } catch(e) {
    wrap.innerHTML = '<div class="empty"><div class="empty-icon">⚠️</div>Failed to load jobs</div>';
  }
}

async function viewJob(jobId) {
  document.querySelectorAll('.nav-item')[0].click();
  resetJobUI();
  document.getElementById('active-job-id').textContent = jobId;
  try {
    const res = await fetch('/jobs/' + jobId);
    const data = await res.json();
    const badge = document.getElementById('active-job-badge');
    const msg = document.getElementById('active-job-msg');
    if (data.status === 'completed') {
      badge.className = 'badge badge-done'; badge.textContent = 'Completed';
      msg.textContent = 'Done'; showResults(data.result);
      document.getElementById('active-prog-bar').className = 'prog-bar';
      document.getElementById('active-prog-bar').style.width = '100%';
    } else if (data.status === 'failed') {
      badge.className = 'badge badge-failed'; badge.textContent = 'Failed';
      msg.textContent = data.error || 'Failed';
    } else if (data.status === 'processing') {
      startPolling(jobId);
    }
  } catch(e) { toast('Failed to load job', 'error'); }
}

// ── STRATEGY ──
async function loadStrategy() {
  const platform = document.getElementById('strat-platform').value;
  const modelType = document.getElementById('strat-model-type').value;
  const preset = document.getElementById('strat-preset').value;
  try {
    const res = await fetch('/strategies?target_platform=' + platform + '&model_type=' + modelType + '&preset=' + preset, {method:'POST'});
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Failed');

    document.getElementById('strategy-result').style.display = 'block';
    document.getElementById('strat-meta').innerHTML =
      '<div class="stat-box"><div class="stat-val">' + (data.quantize_type||'-') + '</div><div class="stat-sub">Quantization</div></div>' +
      '<div class="stat-box"><div class="stat-val">' + ((data.prune_ratio||0)*100).toFixed(0) + '%</div><div class="stat-sub">Prune Ratio</div></div>' +
      '<div class="stat-box"><div class="stat-val">' + (data.batch_size||'-') + '</div><div class="stat-sub">Batch Size</div></div>' +
      '<div class="stat-box"><div class="stat-val">' + (data.max_accuracy_drop||'-') + '%</div><div class="stat-sub">Max Acc Drop</div></div>';

    const steps = (data.steps||[]).map((s, i) =>
      '<div class="strategy-step"><div class="step-num">' + (i+1) + '</div><div class="step-name">' + s + '</div><div class="step-chip">' + platform + '</div></div>'
    ).join('');
    document.getElementById('strategy-steps').innerHTML = steps || '<div class="empty">No steps returned</div>';
    toast('Strategy loaded', 'success');
  } catch(e) { toast(e.message, 'error'); }
}

// ── SYSTEM ──
async function loadSystem() {
  try {
    const res = await fetch('/system');
    const d = await res.json();
    if (d.error) { toast('System info: ' + d.error, 'error'); return; }

    document.getElementById('sys-cpu').textContent = (d.cpu_percent||0).toFixed(0) + '%';
    document.getElementById('bar-cpu').style.width = (d.cpu_percent||0) + '%';
    document.getElementById('bar-cpu').className = 'bar-fill ' + (d.cpu_percent > 80 ? 'bar-red' : d.cpu_percent > 50 ? 'bar-yellow' : 'bar-blue');

    document.getElementById('sys-ram').textContent = (d.ram_available_gb||0).toFixed(1) + ' GB';
    document.getElementById('sys-ram-total').textContent = (d.ram_total_gb||0).toFixed(1) + ' GB';
    const ramPct = d.ram_total_gb ? (d.ram_available_gb/d.ram_total_gb*100) : 0;
    document.getElementById('bar-ram').style.width = ramPct + '%';

    if (d.has_cuda) {
      document.getElementById('sys-cuda').textContent = '✓ CUDA ' + (d.cuda_version||'');
      document.getElementById('sys-gpu-box').style.display = 'block';
      document.getElementById('sys-vram').textContent = (d.vram_available_gb||0).toFixed(1) + ' GB';
      const vramPct = d.vram_total_gb ? (d.vram_available_gb/d.vram_total_gb*100) : 0;
      document.getElementById('bar-vram').style.width = vramPct + '%';
    } else if (d.has_mps) {
      document.getElementById('sys-cuda').textContent = '✓ MPS (Apple Silicon)';
    } else {
      document.getElementById('sys-cuda').textContent = '✗ None (CPU only)';
    }
    if (d.device_name) document.getElementById('sys-device').textContent = '🎮 GPU: ' + d.device_name;

    // Load platforms
    const platformIcons = {laptop:'💻', cloud:'☁️', mobile:'📱', edge:'🔌'};
    const pr = await fetch('/platforms');
    const pd = await pr.json();
    const chips = [...(pd.platforms||[]).map(p => '<span style="background:rgba(79,142,247,.12);color:var(--blue);padding:.2rem .6rem;border-radius:20px;font-size:12px">'+(platformIcons[p]||'🖥')+' '+p+'</span>'),
                   ...(pd.presets||[]).map(p => '<span style="background:rgba(155,93,229,.12);color:var(--purple);padding:.2rem .6rem;border-radius:20px;font-size:12px">⚙️ '+p+'</span>')];
    document.getElementById('platforms-list').innerHTML = chips.join('');
  } catch(e) { toast('Failed to load system info', 'error'); }
}

// ── API DOCS ──
function renderApiDocs() {
  const endpoints = [
    {method:'POST', path:'/api/optimize', desc:'Optimize model by HuggingFace ID', body:'{ model_id, target_platform, preset, max_accuracy_drop, use_quantization, use_pruning, use_distillation }'},
    {method:'POST', path:'/optimize', desc:'Optimize uploaded model file', body:'multipart/form-data: model (file), target_platform, preset'},
    {method:'GET', path:'/jobs', desc:'List all optimization jobs', body:null},
    {method:'GET', path:'/jobs/{job_id}', desc:'Get job status and results', body:null},
    {method:'POST', path:'/strategies', desc:'Get recommended strategy', body:'Query params: target_platform, model_type, preset'},
    {method:'GET', path:'/platforms', desc:'List available platforms and presets', body:null},
    {method:'GET', path:'/system', desc:'Get system hardware info', body:null},
    {method:'GET', path:'/health', desc:'API health check', body:null},
  ];
  const methodColor = {GET:'var(--green)', POST:'var(--blue)', PUT:'var(--yellow)', DELETE:'var(--red)'};
  document.getElementById('api-endpoints').innerHTML = endpoints.map(e =>
    '<div style="background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:.85rem">' +
    '<div style="display:flex;align-items:center;gap:.75rem;margin-bottom:.35rem">' +
    '<span style="font-family:monospace;font-size:11px;font-weight:700;color:' + (methodColor[e.method]||'var(--muted)') + ';background:rgba(0,0,0,.3);padding:.15rem .5rem;border-radius:4px">' + e.method + '</span>' +
    '<code style="color:var(--text);font-size:13px">' + e.path + '</code></div>' +
    '<div style="font-size:12px;color:var(--muted)">' + e.desc + '</div>' +
    (e.body ? '<div style="font-size:11px;color:var(--muted);margin-top:.35rem;font-family:monospace">' + e.body + '</div>' : '') +
    '</div>'
  ).join('');
}

// ── PLAYGROUND ──
let pgSession = null;
let pgCurrentTask = null;
let pgImgData = null;
let pgAudioData = null;

async function loadPlaygroundModels() {
  const sel = document.getElementById('pg-model-select');
  try {
    const res = await fetch('/api/local-models');
    const models = await res.json();
    sel.innerHTML = '<option value="">— select a model from outputs/ —</option>' +
      models.map(m => '<option value="' + m.path + '">' + m.name + ' (' + m.size_mb.toFixed(1) + ' MB)</option>').join('');
    document.getElementById('pg-load-btn').disabled = sel.value === '';
    sel.onchange = () => { document.getElementById('pg-load-btn').disabled = sel.value === ''; };
    if (!models.length) toast('No models in outputs/ directory', 'info');
  } catch(e) { toast('Failed to scan local models', 'error'); }
}

async function playgroundLoad() {
  const path = document.getElementById('pg-model-select').value;
  if (!path) return;
  const status = document.getElementById('pg-status');
  status.className = 'pg-status';
  status.textContent = '⏳ Loading model...';
  document.getElementById('pg-load-btn').disabled = true;
  document.querySelectorAll('.pg-task-ui').forEach(el => el.classList.remove('active'));
  document.getElementById('pg-perf-card').style.display = 'none';
  try {
    const res = await fetch('/api/playground/load', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model_path: path})
    });
    const data = await res.json();
    if (data.error || data.detail) throw new Error(data.error || data.detail);
    pgSession = data.session_id;
    pgCurrentTask = data.task;
    status.className = 'pg-status loaded';
    status.textContent = '✅ ' + data.model_name + ' · ' + data.task;
    showTaskUI(data.task);
    toast('Model loaded: ' + data.model_name, 'success');
  } catch(e) {
    status.textContent = '❌ ' + e.message;
    toast('Load failed: ' + e.message, 'error');
  }
  document.getElementById('pg-load-btn').disabled = false;
}

function showTaskUI(task) {
  document.querySelectorAll('.pg-task-ui').forEach(el => el.classList.remove('active'));
  const ui = document.getElementById('pg-ui-' + task);
  if (ui) ui.classList.add('active');
}

async function playgroundRun(task) {
  if (!pgSession) { toast('Load a model first', 'error'); return; }
  let inputs = {};
  let btn = null, outputEl = null;
  if (task === 'text-generation') {
    inputs = {prompt: document.getElementById('pg-tg-prompt').value, max_tokens: parseInt(document.getElementById('pg-tg-tokens').value) || 100};
    btn = document.getElementById('pg-tg-btn');
    outputEl = document.getElementById('pg-tg-output');
  } else if (task === 'text-classification') {
    inputs = {text: document.getElementById('pg-tc-text').value};
    btn = document.getElementById('pg-tc-btn');
  } else if (task === 'image-classification') {
    if (!pgImgData) { toast('Upload an image first', 'error'); return; }
    inputs = {image_b64: pgImgData};
    btn = document.getElementById('pg-ic-btn');
  } else if (task === 'text-to-image') {
    inputs = {prompt: document.getElementById('pg-ti-prompt').value, steps: parseInt(document.getElementById('pg-ti-steps').value) || 20};
    btn = document.getElementById('pg-ti-btn');
  } else if (task === 'automatic-speech-recognition') {
    if (!pgAudioData) { toast('Upload audio first', 'error'); return; }
    inputs = {audio_b64: pgAudioData};
    btn = document.getElementById('pg-asr-btn');
    outputEl = document.getElementById('pg-asr-output');
  } else if (task === 'translation-or-summarization') {
    inputs = {input_text: document.getElementById('pg-s2s-input').value};
    btn = document.getElementById('pg-s2s-btn');
    outputEl = document.getElementById('pg-s2s-output');
  } else if (task === 'summarization') {
    inputs = {input_text: document.getElementById('pg-sum-input').value};
    btn = document.getElementById('pg-sum-btn');
    outputEl = document.getElementById('pg-sum-output');
  }
  const origText = btn ? btn.textContent : '';
  if (btn) { btn.disabled = true; btn.textContent = '⏳ Running...'; }
  if (outputEl) { outputEl.className = 'pg-output'; outputEl.textContent = '⏳ Running...'; }
  try {
    const res = await fetch('/api/playground/run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({session_id: pgSession, task, inputs})
    });
    const data = await res.json();
    if (data.error || data.detail) throw new Error(data.error || data.detail);
    renderPlaygroundOutput(task, data);
    showPerfStats(data);
  } catch(e) {
    if (outputEl) { outputEl.className = 'pg-output'; outputEl.textContent = '❌ ' + e.message; }
    toast('Inference failed: ' + e.message, 'error');
  }
  if (btn) { btn.disabled = false; btn.textContent = origText; }
}

function renderPlaygroundOutput(task, data) {
  const medals = ['🥇','🥈','🥉'];
  function labelRows(output) {
    return (output || []).slice(0,5).map(function(l, i) {
      const pct = (l.score * 100).toFixed(1);
      return '<div class="pg-label-row"><span>' + (medals[i]||'·') + ' ' + l.label + '</span>' +
        '<div style="display:flex;align-items:center;gap:.5rem">' +
        '<div class="pg-bar"><div class="pg-bar-fill" style="width:' + pct + '%"></div></div>' +
        '<span style="font-size:11px;color:var(--muted);min-width:42px;text-align:right">' + pct + '%</span>' +
        '</div></div>';
    }).join('');
  }
  if (task === 'text-generation') {
    const el = document.getElementById('pg-tg-output');
    el.className = 'pg-output'; el.textContent = data.output || '';
    const stats = document.getElementById('pg-tg-stats');
    if (data.tokens_per_sec) {
      stats.style.display = 'flex';
      stats.innerHTML = '<span>⚡ ' + data.tokens_per_sec.toFixed(1) + ' tok/s</span><span>💾 ' + (data.memory_mb||0).toFixed(0) + ' MB</span><span>⏱ ' + data.latency_ms + 'ms</span>';
    }
  } else if (task === 'text-classification') {
    document.getElementById('pg-tc-results').innerHTML = labelRows(data.output);
  } else if (task === 'image-classification') {
    document.getElementById('pg-ic-results').innerHTML = labelRows(data.output);
  } else if (task === 'text-to-image') {
    const img = document.getElementById('pg-ti-img');
    img.src = 'data:image/png;base64,' + data.output;
    img.style.display = 'block';
  } else if (task === 'automatic-speech-recognition') {
    const el = document.getElementById('pg-asr-output');
    el.className = 'pg-output'; el.textContent = data.output || '';
  } else if (task === 'translation-or-summarization') {
    const el = document.getElementById('pg-s2s-output');
    el.className = 'pg-output'; el.textContent = data.output || '';
  } else if (task === 'summarization') {
    const el = document.getElementById('pg-sum-output');
    el.className = 'pg-output'; el.textContent = data.output || '';
  }
}

function showPerfStats(data) {
  const card = document.getElementById('pg-perf-card');
  const stats = document.getElementById('pg-perf-stats');
  card.style.display = 'block';
  const items = [
    {label:'Latency', val:(data.latency_ms||0)+'ms', cls:''},
    {label:'Memory', val:(data.memory_mb||0).toFixed(0)+' MB', cls:'accent-blue'},
  ];
  if (data.tokens_per_sec) items.push({label:'Throughput', val:data.tokens_per_sec.toFixed(1)+' tok/s', cls:'accent-yellow'});
  stats.innerHTML = items.map(function(s) {
    return '<div class="result-box ' + s.cls + '"><div class="result-val">' + s.val + '</div><div class="result-sub">' + s.label + '</div></div>';
  }).join('');
}

function pgImgDrop(e) {
  e.preventDefault();
  document.getElementById('pg-img-drop').classList.remove('dragging');
  const file = e.dataTransfer.files[0];
  if (file) pgReadImgFile(file);
}
function pgImgSelected(inp) { if (inp.files[0]) pgReadImgFile(inp.files[0]); }
function pgReadImgFile(file) {
  const reader = new FileReader();
  reader.onload = function(e) {
    pgImgData = e.target.result.split(',')[1];
    const prev = document.getElementById('pg-img-preview');
    prev.src = e.target.result; prev.style.display = 'block';
    document.getElementById('pg-ic-btn').disabled = false;
  };
  reader.readAsDataURL(file);
}
function pgAudioSelected(inp) {
  if (!inp.files[0]) return;
  const file = inp.files[0];
  document.getElementById('pg-audio-name').textContent = '🎙 ' + file.name;
  const reader = new FileReader();
  reader.onload = function(e) {
    pgAudioData = e.target.result.split(',')[1];
    document.getElementById('pg-asr-btn').disabled = false;
  };
  reader.readAsDataURL(file);
}

// ── INIT ──
(async function init() {
  try {
    const res = await fetch('/health');
    if (!res.ok) throw new Error();
    document.getElementById('api-status').textContent = 'Connected';
  } catch {
    document.getElementById('api-status').textContent = 'Offline';
    document.querySelector('.dot').style.background = 'var(--red)';
  }
  loadSystem();
  setInterval(loadSystem, 30000);
})();
</script>
</body>
</html>"""

app = FastAPI(
    title="PredycatAI Universal Optimizer API",
    description="Production-grade AI optimization system for any neural network",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass
class OptimizationRequest:
    model_id: Optional[str] = None
    model_url: Optional[str] = None
    target_platform: str = "mobile"
    preset: str = "balanced"
    max_accuracy_drop: float = 2.0
    max_size_mb: Optional[float] = None
    max_ram_mb: Optional[float] = None
    export_formats: List[str] = Field(default_factory=lambda: ["pt"])
    use_quantization: bool = True
    use_pruning: bool = True
    use_distillation: bool = True
    enable_reasoning_check: bool = True


class OptimizeByIdRequest(BaseModel):
    model_id: str
    target_platform: str = "mobile"
    preset: str = "balanced"
    max_accuracy_drop: float = 2.0
    use_quantization: bool = True
    use_pruning: bool = True
    use_distillation: bool = True
    enable_reasoning_check: bool = True


@dataclass
class OptimizationJob:
    job_id: str
    status: str
    created_at: str
    result: Optional[dict] = None


optimizer_instance = None
jobs_storage = {}


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.get("/")
async def root():
    return {
        "name": "PredycatAI Universal Optimizer",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/optimize")
async def optimize_model(
    background_tasks: BackgroundTasks,
    model: UploadFile = File(...),
    target_platform: str = "mobile",
    preset: str = "balanced",
    max_accuracy_drop: float = 2.0,
    max_size_mb: Optional[float] = None,
    max_ram_mb: Optional[float] = None,
    use_quantization: bool = True,
    use_pruning: bool = True,
    use_distillation: bool = True,
    enable_reasoning_check: bool = True
):
    import json
    import time
    from datetime import datetime
    
    job_id = f"job_{int(time.time())}"
    
    save_path = Path(f"./uploads/{job_id}")
    save_path.mkdir(parents=True, exist_ok=True)
    
    model_path = save_path / model.filename
    content = await model.read()
    
    with open(model_path, "wb") as f:
        f.write(content)
    
    export_formats = ["pt"]
    
    try:
        from predycat_ai.core.optimizer import PredycatOptimizer, OptimizationConfig
        
        config = OptimizationConfig(
            target_platform=target_platform,
            preset=preset,
            max_accuracy_drop=max_accuracy_drop,
            max_size_mb=max_size_mb,
            max_ram_mb=max_ram_mb,
            export_formats=export_formats,
            use_quantization=use_quantization,
            use_pruning=use_pruning,
            use_distillation=use_distillation,
            enable_reasoning_check=enable_reasoning_check
        )
        
        optimizer = PredycatOptimizer(config=config)
        
        background_tasks.add_task(run_optimization, job_id, optimizer, str(model_path))
        
        jobs_storage[job_id] = {
            "job_id": job_id,
            "status": "processing",
            "created_at": datetime.now().isoformat(),
            "model_filename": model.filename
        }
        
        return {
            "job_id": job_id,
            "status": "processing",
            "message": "Optimization started"
        }
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/optimize")
async def optimize_by_id(request: OptimizeByIdRequest, background_tasks: BackgroundTasks):
    import time
    from datetime import datetime

    job_id = f"job_{int(time.time())}"

    try:
        from predycat_ai.core.optimizer import PredycatOptimizer, OptimizationConfig

        config = OptimizationConfig(
            target_platform=request.target_platform,
            preset=request.preset,
            max_accuracy_drop=request.max_accuracy_drop,
            use_quantization=request.use_quantization,
            use_pruning=request.use_pruning,
            use_distillation=request.use_distillation,
            enable_reasoning_check=request.enable_reasoning_check,
        )

        optimizer = PredycatOptimizer(config=config)

        background_tasks.add_task(run_optimization, job_id, optimizer, request.model_id)

        jobs_storage[job_id] = {
            "job_id": job_id,
            "status": "processing",
            "created_at": datetime.now().isoformat(),
            "model_id": request.model_id,
        }

        return {"job_id": job_id, "status": "processing", "message": "Optimization started"}

    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def run_optimization(job_id: str, optimizer, model_path: str):
    try:
        # If model_path is a HuggingFace model id or local path, load it first
        from predycat_ai.core.loader import UniversalLoader
        loader = UniversalLoader()
        model, model_info = loader.load_model(model_path)
        result = optimizer.optimize(model, model_path=model_path)
        
        jobs_storage[job_id]["status"] = "completed"
        jobs_storage[job_id]["result"] = {
            "success": result.success,
            "compression_ratio": result.compression_ratio,
            "original_size_mb": result.original_size_mb,
            "optimized_size_mb": result.optimized_size_mb,
            "latency_improvement": result.latency_improvement,
            "accuracy_drop": result.accuracy_drop,
            "output_paths": result.output_paths
        }
        
    except Exception as e:
        jobs_storage[job_id]["status"] = "failed"
        jobs_storage[job_id]["error"] = str(e)


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in jobs_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs_storage[job_id]


@app.get("/jobs")
async def list_jobs():
    return list(jobs_storage.values())


@app.post("/strategies")
async def get_strategy(
    target_platform: str = "mobile",
    model_type: str = "llm",
    preset: str = "balanced"
):
    from predycat_ai.core.strategy import StrategySelector, TargetPlatform, ModelType, OptimizationPreset

    selector = StrategySelector()
    strategy = selector.select_strategy(
        platform=TargetPlatform(target_platform.lower()),
        model_type=ModelType(model_type.lower()),
        preset=OptimizationPreset(preset.lower()),
    )
    
    return {
        "steps": [s.value for s in strategy.steps],
        "quantize_type": strategy.quantize_type,
        "prune_ratio": strategy.prune_ratio,
        "batch_size": strategy.batch_size,
        "max_accuracy_drop": strategy.max_accuracy_drop
    }


@app.get("/platforms")
async def list_platforms():
    return {
        "platforms": ["laptop", "cloud", "mobile", "edge"],
        "presets": ["balanced", "ultra_compression", "max_accuracy", "mobile_safe"]
    }


@app.get("/dashboard")
async def dashboard():
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=DASHBOARD_HTML)


@app.get("/system")
async def system_info():
    try:
        import torch
        import psutil
        
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        info = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "ram_available_gb": psutil.virtual_memory().available / (1024**3),
            "ram_total_gb": psutil.virtual_memory().total / (1024**3),
            "has_cuda": torch.cuda.is_available(),
            "has_mps": has_mps,
        }

        if torch.cuda.is_available():
            info["vram_available_gb"] = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / (1024**3)
            info["vram_total_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            info["cuda_version"] = torch.version.cuda
            info["device_name"] = torch.cuda.get_device_name(0)
        elif has_mps:
            info["device_name"] = "Apple Silicon (MPS)"
            info["cuda_version"] = "MPS"

        return info
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/models/search")
async def search_hf_models(
    q: str = "",
    limit: int = 50,
    sort: str = "downloads",
    task: str = "",
    offset: int = 0,
    max_size_gb: float = 0.0,   # 0 = no filter
    max_params_b: float = 0.0,  # 0 = no filter (billions)
):
    import httpx, os, re
    token = os.environ.get("HF_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    filtering = max_size_gb > 0 or max_params_b > 0
    base_params = {
        "sort": sort,
        "direction": "-1",
        "limit": 100,
        "search": q,
        "full": "False",
    }
    if task:
        base_params["pipeline_tag"] = task

    def _passes(m: dict) -> Optional[dict]:
        if not (isinstance(m, dict) and m.get("id")):
            return None
        st = m.get("safetensors") or {}
        params_count: Optional[int] = st.get("total")
        size_bytes: Optional[int] = (params_count * 2) if params_count else None
        if max_params_b > 0 and params_count is not None:
            if params_count > max_params_b * 1_000_000_000:
                return None
        if max_size_gb > 0 and size_bytes is not None:
            if size_bytes > max_size_gb * 1_000_000_000:
                return None
        return {
            "id": m.get("id", ""),
            "downloads": m.get("downloads", 0),
            "likes": m.get("likes", 0),
            "task": m.get("pipeline_tag", ""),
            "gated": m.get("gated", False),
            "private": m.get("private", False),
            "params": params_count,
            "size_bytes": size_bytes,
        }

    try:
        result = []
        next_url = "https://huggingface.co/api/models"
        next_params: Optional[dict] = dict(base_params)
        # When filtering, paginate up to 5 pages (500 models) to find small models.
        # Top downloads are all multi-billion → need to scan deeper.
        max_pages = 5 if filtering else 1
        page = 0

        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            while next_url and page < max_pages and len(result) < limit:
                r = await client.get(next_url, params=next_params, headers=headers)
                r.raise_for_status()
                models = r.json()
                if not isinstance(models, list):
                    break

                for m in models:
                    row = _passes(m)
                    if row:
                        result.append(row)
                    if len(result) >= limit:
                        break

                # Follow HF Link header for next page
                next_url = None
                next_params = None
                link_hdr = r.headers.get("Link", "")
                m = re.search(r'<([^>]+)>;\s*rel="next"', link_hdr)
                if m and len(result) < limit:
                    next_url = m.group(1)
                page += 1

        return result[:limit]
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"HF API error: {e}")


@app.get("/api/model-info/{model_id:path}")
async def get_model_info(model_id: str):
    import httpx, os
    token = os.environ.get("HF_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            r = await client.get(f"https://huggingface.co/api/models/{model_id}?blobs=true", headers=headers)
        if r.status_code == 401:
            return {"error": "Model gated — add HF_TOKEN to .env", "id": model_id}
        data = r.json()
        siblings = data.get("siblings", [])
        total_size = sum(s.get("size", 0) or 0 for s in siblings)
        return {
            "id": data.get("id", model_id),
            "task": data.get("pipeline_tag", ""),
            "downloads": data.get("downloads", 0),
            "likes": data.get("likes", 0),
            "tags": data.get("tags", []),
            "license": next((t for t in data.get("tags", []) if t.startswith("license:")), ""),
            "total_size_bytes": total_size,
            "files": [{"name": s.get("rfilename", ""), "size": s.get("size", 0) or 0} for s in siblings],
            "safetensors": data.get("safetensors", {}),
            "gated": bool(data.get("gated", False)),
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"HF API error: {e}")


@app.post("/api/inference/{model_id:path}")
async def run_inference(model_id: str, request: dict):
    import httpx, os
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise HTTPException(status_code=400, detail="HF_TOKEN required for inference — add to .env")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                f"https://api-inference.huggingface.co/models/{model_id}",
                json=request, headers=headers
            )
        return r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Inference error: {e}")


@app.get("/api/datasets/search")
async def search_hf_datasets(q: str = "", limit: int = 50, sort: str = "downloads", task: str = ""):
    import httpx, os
    token = os.environ.get("HF_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    params = {"sort": sort, "direction": "-1", "limit": min(limit, 100), "search": q, "full": "False"}
    if task:
        params["task_categories"] = task
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            r = await client.get("https://huggingface.co/api/datasets", params=params, headers=headers)
        datasets = r.json()
        return [
            {
                "id": d.get("id", ""),
                "downloads": d.get("downloads", 0),
                "likes": d.get("likes", 0),
                "task": (d.get("task_categories") or [""])[0],
            }
            for d in datasets
            if isinstance(d, dict) and d.get("id")
        ]
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"HF API error: {e}")


# ── PLAYGROUND SESSION STATE ──
import uuid as _uuid

_pg_sessions: dict = {}

ARCH_TO_TASK = {
    "GPT2LMHeadModel": "text-generation",
    "LlamaForCausalLM": "text-generation",
    "MistralForCausalLM": "text-generation",
    "FalconForCausalLM": "text-generation",
    "PhiForCausalLM": "text-generation",
    "Qwen2ForCausalLM": "text-generation",
    "BertForSequenceClassification": "text-classification",
    "RobertaForSequenceClassification": "text-classification",
    "DistilBertForSequenceClassification": "text-classification",
    "ViTForImageClassification": "image-classification",
    "ResNetForImageClassification": "image-classification",
    "SwinForImageClassification": "image-classification",
    "UNet2DConditionModel": "text-to-image",
    "WhisperForConditionalGeneration": "automatic-speech-recognition",
    "Wav2Vec2ForCTC": "automatic-speech-recognition",
    "T5ForConditionalGeneration": "translation-or-summarization",
    "BartForConditionalGeneration": "summarization",
    "CLIPModel": "image-text-matching",
}


@app.get("/api/local-models")
async def list_local_models():
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        return []
    models = []
    for ext in ("*.pt", "*.onnx", "*.safetensors", "*.bin"):
        for f in outputs_dir.glob(ext):
            models.append({
                "path": str(f),
                "name": f.stem,
                "size_mb": f.stat().st_size / (1024 * 1024),
                "modified": f.stat().st_mtime,
            })
    return sorted(models, key=lambda x: x["modified"], reverse=True)


class PlaygroundLoadRequest(BaseModel):
    model_path: str


def _detect_hf_model_id_from_state_dict(state_dict: dict) -> Optional[str]:
    """Detect HuggingFace model ID from state dict key patterns + weight shapes."""
    keys = set(state_dict.keys())
    # GPT-2 family
    if "transformer.wte.weight" in keys:
        hidden = state_dict["transformer.wte.weight"].shape[1]
        return {768: "gpt2", 1024: "gpt2-medium", 1280: "gpt2-large", 1600: "gpt2-xl"}.get(hidden, "gpt2")
    # LLaMA / Mistral
    if "model.embed_tokens.weight" in keys and any("self_attn.q_proj" in k for k in keys):
        return None  # needs auth; return None, caller handles
    # BERT family
    if "embeddings.word_embeddings.weight" in keys:
        hidden = state_dict["embeddings.word_embeddings.weight"].shape[1]
        return {768: "bert-base-uncased", 1024: "bert-large-uncased"}.get(hidden, "bert-base-uncased")
    # T5
    if "encoder.embed_tokens.weight" in keys and "decoder.embed_tokens.weight" in keys:
        return "t5-small"
    # BART
    if "model.encoder.embed_tokens.weight" in keys and "model.decoder.embed_tokens.weight" in keys:
        return "facebook/bart-base"
    # Whisper
    if "encoder.conv1.weight" in keys and "decoder.embed_tokens.weight" in keys:
        return "openai/whisper-tiny"
    # DistilBERT
    if "embeddings.word_embeddings.weight" in keys and any("distilbert" in k for k in keys):
        return "distilbert-base-uncased"
    return None


def _task_for_model_id(model_id: str) -> str:
    mid = model_id.lower()
    if any(x in mid for x in ["gpt", "llama", "mistral", "falcon", "phi", "qwen", "bloom", "opt"]):
        return "text-generation"
    if any(x in mid for x in ["bert", "roberta", "distilbert", "albert", "electra"]):
        return "text-classification"
    if "t5" in mid or "marian" in mid or "pegasus" in mid:
        return "translation-or-summarization"
    if "bart" in mid:
        return "summarization"
    if "whisper" in mid or "wav2vec" in mid:
        return "automatic-speech-recognition"
    if any(x in mid for x in ["vit", "resnet", "swin"]):
        return "image-classification"
    return "text-generation"


@app.post("/api/playground/load")
async def playground_load(request: PlaygroundLoadRequest):
    import torch, json as _json
    path = Path(request.model_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    session_id = str(_uuid.uuid4())[:8]
    model_name = path.stem
    task = "unknown"
    model = None
    tokenizer = None
    labels: list = []

    try:
        # ── 1. Check for sidecar metadata (saved by exporter) ──
        meta_path = path.with_suffix(".meta.json")
        hf_model_id: Optional[str] = None
        if meta_path.exists():
            with open(meta_path) as f:
                meta = _json.load(f)
            hf_model_id = meta.get("model_id")
            task = meta.get("task", "unknown")

        # ── 2. Try HuggingFace config in same directory ──
        if not hf_model_id:
            config_dir = str(path.parent)
            try:
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(config_dir)
                archs = config.architectures or []
                for arch in archs:
                    if arch in ARCH_TO_TASK:
                        task = ARCH_TO_TASK[arch]
                        break
                if task == "unknown" and config.model_type:
                    mt = config.model_type.lower()
                    if any(x in mt for x in ["gpt", "llama", "mistral", "falcon", "phi", "qwen", "bloom", "opt"]):
                        task = "text-generation"
                    elif any(x in mt for x in ["bert", "roberta", "distilbert"]):
                        task = "text-classification"
                    elif any(x in mt for x in ["t5", "marian", "pegasus"]):
                        task = "translation-or-summarization"
                    elif "bart" in mt:
                        task = "summarization"
                    elif any(x in mt for x in ["whisper", "wav2vec"]):
                        task = "automatic-speech-recognition"
                    elif any(x in mt for x in ["vit", "resnet", "swin"]):
                        task = "image-classification"
                # If HF config found, use that dir for loading
                if task != "unknown":
                    hf_model_id = config_dir
            except Exception:
                pass

        # ── 3. Load checkpoint and detect from state dict key patterns ──
        checkpoint = None
        if path.suffix == ".pt":
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)

        if checkpoint is not None and isinstance(checkpoint, dict) and not hf_model_id:
            detected_id = _detect_hf_model_id_from_state_dict(checkpoint)
            if detected_id:
                hf_model_id = detected_id
                task = _task_for_model_id(detected_id)

        if task == "unknown":
            task = "text-generation"
        if not hf_model_id:
            hf_model_id = "gpt2"  # safe fallback

        # ── 4. Reconstruct model from HF + load state dict (or load from HF pretrained) ──
        if task == "text-generation":
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig as AC
            try:
                if checkpoint is not None and isinstance(checkpoint, dict):
                    # State dict path: build model from config, inject weights
                    try:
                        cfg = AC.from_pretrained(hf_model_id)
                    except Exception:
                        cfg = AC.from_pretrained("gpt2")
                    model = AutoModelForCausalLM.from_config(cfg)
                    sd = checkpoint.get("model_state_dict", checkpoint)
                    # Convert FP16 → FP32 for CPU inference
                    sd_f32 = {k: v.float() if v.is_floating_point() else v for k, v in sd.items()}
                    missing, unexpected = model.load_state_dict(sd_f32, strict=False)
                    if missing:
                        logger.warning(f"Missing keys: {len(missing)}")
                else:
                    model = AutoModelForCausalLM.from_pretrained(hf_model_id, torch_dtype=torch.float32)
                tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"text-generation load failed: {e}")

        elif task == "text-classification":
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            try:
                if checkpoint is not None and isinstance(checkpoint, dict):
                    from transformers import AutoConfig as AC
                    cfg = AC.from_pretrained(hf_model_id)
                    model = AutoModelForSequenceClassification.from_config(cfg)
                    sd = checkpoint.get("model_state_dict", checkpoint)
                    sd_f32 = {k: v.float() if v.is_floating_point() else v for k, v in sd.items()}
                    model.load_state_dict(sd_f32, strict=False)
                else:
                    model = AutoModelForSequenceClassification.from_pretrained(hf_model_id)
                tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
                labels = list(model.config.id2label.values()) if hasattr(model.config, "id2label") else []
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"text-classification load failed: {e}")

        elif task == "image-classification":
            from transformers import AutoModelForImageClassification
            try:
                if checkpoint is not None and isinstance(checkpoint, dict):
                    from transformers import AutoConfig as AC
                    cfg = AC.from_pretrained(hf_model_id)
                    model = AutoModelForImageClassification.from_config(cfg)
                    sd = checkpoint.get("model_state_dict", checkpoint)
                    sd_f32 = {k: v.float() if v.is_floating_point() else v for k, v in sd.items()}
                    model.load_state_dict(sd_f32, strict=False)
                else:
                    model = AutoModelForImageClassification.from_pretrained(hf_model_id)
                labels = list(model.config.id2label.values()) if hasattr(model.config, "id2label") else []
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"image-classification load failed: {e}")

        elif task in ("translation-or-summarization", "summarization"):
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            try:
                if checkpoint is not None and isinstance(checkpoint, dict):
                    from transformers import AutoConfig as AC
                    cfg = AC.from_pretrained(hf_model_id)
                    model = AutoModelForSeq2SeqLM.from_config(cfg)
                    sd = checkpoint.get("model_state_dict", checkpoint)
                    sd_f32 = {k: v.float() if v.is_floating_point() else v for k, v in sd.items()}
                    model.load_state_dict(sd_f32, strict=False)
                else:
                    model = AutoModelForSeq2SeqLM.from_pretrained(hf_model_id)
                tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"seq2seq load failed: {e}")

        elif task == "automatic-speech-recognition":
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            try:
                model = WhisperForConditionalGeneration.from_pretrained(hf_model_id)
                tokenizer = WhisperProcessor.from_pretrained(hf_model_id)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"ASR load failed: {e}")

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported task: {task}")

        if hasattr(model, "eval"):
            model.eval()

        _pg_sessions[session_id] = {
            "model": model,
            "tokenizer": tokenizer,
            "task": task,
            "model_name": model_name,
            "labels": labels,
            "path": str(path),
            "hf_model_id": hf_model_id,
        }
        return {"session_id": session_id, "task": task, "model_name": model_name, "labels": labels}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")


class PlaygroundRunRequest(BaseModel):
    session_id: str
    task: str
    inputs: dict


@app.post("/api/playground/run")
async def playground_run(request: PlaygroundRunRequest):
    import torch
    import time
    import tracemalloc

    session = _pg_sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found — reload model")

    model = session["model"]
    tokenizer = session["tokenizer"]
    task = request.task
    inputs = request.inputs

    tracemalloc.start()
    t0 = time.time()

    try:
        output = None
        tokens_per_sec = None

        if task == "text-generation":
            prompt = inputs.get("prompt", "")
            max_tokens = min(int(inputs.get("max_tokens", 100)), 512)
            enc = tokenizer(prompt, return_tensors="pt")
            t1 = time.time()
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    temperature=1.0,
                    repetition_penalty=1.3,
                    pad_token_id=tokenizer.eos_token_id or 0,
                )
            n_new = out.shape[1] - enc["input_ids"].shape[1]
            gen_time = max(time.time() - t1, 0.001)
            tokens_per_sec = n_new / gen_time
            output = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

        elif task == "text-classification":
            import torch.nn.functional as F
            text = inputs.get("text", "")
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                logits = model(**enc).logits
            probs = F.softmax(logits[0], dim=-1).tolist()
            id2label = model.config.id2label if hasattr(model.config, "id2label") else {i: str(i) for i in range(len(probs))}
            output = sorted(
                [{"label": id2label[i], "score": probs[i]} for i in range(len(probs))],
                key=lambda x: -x["score"],
            )

        elif task == "image-classification":
            import base64, io
            import torch.nn.functional as F
            from PIL import Image
            img = Image.open(io.BytesIO(base64.b64decode(inputs.get("image_b64", "")))).convert("RGB")
            try:
                from transformers import AutoFeatureExtractor
                feat_ext = AutoFeatureExtractor.from_pretrained(str(Path(session["path"]).parent))
                enc = feat_ext(images=img, return_tensors="pt")
            except Exception:
                import torchvision.transforms as T
                transform = T.Compose([
                    T.Resize((224, 224)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
                enc = {"pixel_values": transform(img).unsqueeze(0)}
            with torch.no_grad():
                logits = model(**enc).logits
            probs = F.softmax(logits[0], dim=-1).tolist()
            id2label = model.config.id2label if hasattr(model.config, "id2label") else {i: str(i) for i in range(len(probs))}
            output = sorted(
                [{"label": id2label[i], "score": probs[i]} for i in range(len(probs))],
                key=lambda x: -x["score"],
            )[:10]

        elif task == "text-to-image":
            prompt = inputs.get("prompt", "")
            steps = min(int(inputs.get("steps", 20)), 50)
            result = model(prompt, num_inference_steps=steps)
            img = result.images[0]
            import io, base64
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            output = base64.b64encode(buf.getvalue()).decode()

        elif task == "automatic-speech-recognition":
            import base64, io
            audio_bytes = base64.b64decode(inputs.get("audio_b64", ""))
            try:
                import soundfile as sf
                import numpy as np
                audio_arr, sr = sf.read(io.BytesIO(audio_bytes))
                if audio_arr.ndim > 1:
                    audio_arr = audio_arr.mean(axis=1)
                if sr != 16000:
                    import librosa
                    audio_arr = librosa.resample(audio_arr, orig_sr=sr, target_sr=16000)
                proc_inputs = tokenizer(audio_arr, sampling_rate=16000, return_tensors="pt")
                with torch.no_grad():
                    predicted_ids = model.generate(**proc_inputs)
                output = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            except ImportError:
                raise HTTPException(status_code=400, detail="soundfile/librosa not installed — run: pip install soundfile librosa")

        elif task in ("translation-or-summarization", "summarization"):
            input_text = inputs.get("input_text", "")
            enc = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=256)
            output = tokenizer.decode(out[0], skip_special_tokens=True)

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported task: {task}")

        latency_ms = round((time.time() - t0) * 1000)
        _, mem_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result: dict = {"output": output, "latency_ms": latency_ms, "memory_mb": mem_peak / (1024 * 1024)}
        if tokens_per_sec is not None:
            result["tokens_per_sec"] = tokens_per_sec
        return result

    except HTTPException:
        tracemalloc.stop()
        raise
    except Exception as e:
        tracemalloc.stop()
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")


@app.get("/api/playground/session/{session_id}")
async def playground_session_status(session_id: str):
    s = _pg_sessions.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "task": s["task"], "model_name": s["model_name"], "status": "loaded"}


@app.delete("/api/playground/session/{session_id}")
async def playground_unload(session_id: str):
    if session_id not in _pg_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del _pg_sessions[session_id]
    return {"unloaded": session_id}


def start_server(host: str = "0.0.0.0", port: int = 8080):
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()