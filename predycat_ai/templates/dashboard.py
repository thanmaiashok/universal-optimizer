"""Simple Web Dashboard"""

import os
import json

html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PredycatAI Universal Optimizer</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', system-ui, sans-serif; }
        body { background: #0a0a0f; color: #e0e0e0; min-height: 100vh; }
        
        .header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 1.5rem 2rem;
            border-bottom: 1px solid #2a2a4a;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .logo { font-size: 1.5rem; font-weight: 700; color: #00ff88; }
        .logo span { color: #fff; }
        
        .container { max-width: 1200px; margin: 0 auto; padding: 2rem; }
        
        .card {
            background: #12121a;
            border: 1px solid #2a2a4a;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }
        
        .card h2 { 
            color: #00ff88; 
            font-size: 1.1rem; 
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .form-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 1rem;
        }
        
        .form-group { display: flex; flex-direction: column; gap: 0.5rem; }
        .form-group label { color: #888; font-size: 0.85rem; }
        
        input, select {
            background: #1a1a2e;
            border: 1px solid #3a3a5a;
            color: #fff;
            padding: 0.75rem 1rem;
            border-radius: 8px;
            font-size: 1rem;
        }
        input:focus, select:focus { outline: none; border-color: #00ff88; }
        
        .btn {
            background: linear-gradient(135deg, #00ff88, #00cc6a);
            color: #000;
            border: none;
            padding: 0.75rem 2rem;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 4px 20px rgba(0,255,136,0.3); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        
        .btn-secondary { background: #2a2a4a; color: #fff; }
        .btn-secondary:hover { background: #3a3a5a; }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
        }
        
        .stat { 
            background: #1a1a2e; 
            padding: 1rem; 
            border-radius: 8px;
            text-align: center;
        }
        .stat-value { font-size: 1.5rem; font-weight: 700; color: #00ff88; }
        .stat-label { color: #888; font-size: 0.8rem; margin-top: 0.25rem; }
        
        .logs {
            background: #0a0a0f;
            border: 1px solid #2a2a4a;
            border-radius: 8px;
            padding: 1rem;
            max-height: 300px;
            overflow-y: auto;
            font-family: 'Consolas', monospace;
            font-size: 0.85rem;
        }
        .log-entry { margin-bottom: 0.25rem; color: #888; }
        .log-entry.success { color: #00ff88; }
        .log-entry.error { color: #ff4444; }
        .log-entry.info { color: #4488ff; }
        
        .progress-bar {
            background: #1a1a2e;
            height: 8px;
            border-radius: 4px;
            overflow: hidden;
            margin: 1rem 0;
        }
        .progress-fill {
            background: linear-gradient(90deg, #00ff88, #00cc6a);
            height: 100%;
            width: 0%;
            transition: width 0.3s;
        }
        
        .status-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .status-idle { background: #2a2a4a; color: #888; }
        .status-running { background: #224422; color: #00ff88; }
        .status-done { background: #003322; color: #00ff88; }
        .status-error { background: #442222; color: #ff4444; }
        
        .history-item {
            background: #1a1a2e;
            padding: 0.75rem;
            border-radius: 8px;
            margin-bottom: 0.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .loading { animation: pulse 1s infinite; }
    </style>
</head>
<body>
    <header class="header">
        <div class="logo">🦊 <span>PredycatAI</span> Universal Optimizer</div>
        <div>
            <span id="serverStatus" class="status-badge status-idle">IDLE</span>
        </div>
    </header>
    
    <div class="container">
        <div class="card">
            <h2>⚡ Optimize Model</h2>
            <div class="form-row">
                <div class="form-group">
                    <label>Model (HuggingFace ID or path)</label>
                    <input type="text" id="modelInput" placeholder="gpt2, facebook/opt-125m, /path/to/model...">
                </div>
                <div class="form-group">
                    <label>Target Platform</label>
                    <select id="platformSelect">
                        <option value="mobile">Mobile</option>
                        <option value="laptop">Laptop</option>
                        <option value="cloud">Cloud</option>
                        <option value="edge">Edge</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Preset</label>
                    <select id="presetSelect">
                        <option value="balanced">Balanced</option>
                        <option value="ultra_compression">Ultra Compression</option>
                        <option value="max_accuracy">Max Accuracy</option>
                        <option value="mobile_safe">Mobile Safe</option>
                    </select>
                </div>
            </div>
            <div class="form-row">
                <div class="form-group">
                    <label>Max Accuracy Drop (%)</label>
                    <input type="number" id="maxDrop" value="2" min="0" max="10" step="0.5">
                </div>
                <div class="form-group">
                    <label>Export Formats</label>
                    <select id="formatSelect">
                        <option value="pt">PyTorch (.pt)</option>
                        <option value="onnx">ONNX</option>
                        <option value="gguf">GGUF (llama.cpp)</option>
                    </select>
                </div>
            </div>
            <button class="btn" id="optimizeBtn" onclick="startOptimization()">🚀 Start Optimization</button>
            <button class="btn btn-secondary" onclick="stopOptimization()" style="display:none" id="stopBtn">⏹ Stop</button>
        </div>
        
        <div class="card" id="progressCard" style="display:none">
            <h2>📊 Progress</h2>
            <div class="form-row">
                <div class="stat">
                    <div class="stat-value" id="compressionStat">-</div>
                    <div class="stat-label">Compression</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="latencyStat">-</div>
                    <div class="stat-label">Latency ↓</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="sizeStat">-</div>
                    <div class="stat-label">Size MB</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="accuracyStat">-</div>
                    <div class="stat-label">Accuracy</div>
                </div>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
            <p id="progressText" style="text-align:center;color:#888">Initializing...</p>
        </div>
        
        <div class="card">
            <h2>📝 Logs</h2>
            <div class="logs" id="logContainer">
                <div class="log-entry info">Ready. Enter model and click Optimize.</div>
            </div>
        </div>
        
        <div class="card">
            <h2>📈 System Info</h2>
            <div class="stats-grid" id="systemStats">
                <div class="stat">
                    <div class="stat-value" id="cpuStat">-</div>
                    <div class="stat-label">CPU Usage</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="ramStat">-</div>
                    <div class="stat-label">RAM Available</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="gpuStat">-</div>
                    <div class="stat-label">GPU</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="vramStat">-</div>
                    <div class="stat-label">VRAM</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let optimizing = false;
        
        function log(msg, type = 'info') {
            const container = document.getElementById('logContainer');
            const entry = document.createElement('div');
            entry.className = `log-entry ${type}`;
            entry.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
            container.insertBefore(entry, container.firstChild);
        }
        
        async function startOptimization() {
            const model = document.getElementById('modelInput').value;
            if (!model) {
                log('Please enter a model', 'error');
                return;
            }
            
            optimizing = true;
            document.getElementById('optimizeBtn').disabled = true;
            document.getElementById('stopBtn').style.display = 'inline-block';
            document.getElementById('progressCard').style.display = 'block';
            document.getElementById('serverStatus').className = 'status-badge status-running';
            document.getElementById('serverStatus').textContent = 'RUNNING';
            
            log(`Starting optimization: ${model}`, 'info');
            
            try {
                const response = await fetch('/optimize', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model_id: model,
                        target_platform: document.getElementById('platformSelect').value,
                        preset: document.getElementById('presetSelect').value,
                        max_accuracy_drop: parseFloat(document.getElementById('maxDrop').value),
                        export_formats: [document.getElementById('formatSelect').value]
                    })
                });
                
                const data = await response.json();
                
                if (data.job_id) {
                    log(`Job started: ${data.job_id}`, 'success');
                    monitorJob(data.job_id);
                } else if (data.error) {
                    log(`Error: ${data.error}`, 'error');
                }
            } catch (e) {
                log(`Request failed: ${e}`, 'error');
            }
        }
        
        async function monitorJob(jobId) {
            const steps = ['Loading model...', 'Profiling...', 'Quantizing...', 'Optimizing...', 'Exporting...'];
            let step = 0;
            
            while (optimizing && step < steps.length) {
                document.getElementById('progressText').textContent = steps[step];
                document.getElementById('progressFill').style.width = ((step + 1) / steps.length * 100) + '%';
                log(steps[step], 'info');
                
                await new Promise(r => setTimeout(r, 1500));
                step++;
                
                try {
                    const resp = await fetch(`/jobs/${jobId}`);
                    const job = await resp.json();
                    
                    if (job.status === 'completed') {
                        document.getElementById('compressionStat').textContent = (job.result?.compression_ratio || 0).toFixed(1) + 'x';
                        document.getElementById('latencyStat').textContent = (job.result?.latency_improvement || 0).toFixed(1) + 'x';
                        document.getElementById('sizeStat').textContent = (job.result?.optimized_size_mb || 0).toFixed(0) + 'MB';
                        document.getElementById('accuracyStat').textContent = '-' + (job.result?.accuracy_drop || 0).toFixed(1) + '%';
                        log(`Done! ${job.result.compression_ratio}x compression`, 'success');
                        break;
                    } else if (job.status === 'failed') {
                        log(`Failed: ${job.error}`, 'error');
                        break;
                    }
                } catch {}
            }
            
            finishOptimization();
        }
        
        function stopOptimization() {
            optimizing = false;
            log('Stopped by user', 'error');
            finishOptimization();
        }
        
        function finishOptimization() {
            optimizing = false;
            document.getElementById('optimizeBtn').disabled = false;
            document.getElementById('stopBtn').style.display = 'none';
            document.getElementById('serverStatus').className = 'status-badge status-done';
            document.getElementById('serverStatus').textContent = 'DONE';
        }
        
        async function updateSystemInfo() {
            try {
                const resp = await fetch('/system');
                const data = await resp.json();
                
                document.getElementById('cpuStat').textContent = data.cpu_percent?.toFixed(0) + '%' || '-';
                document.getElementById('ramStat').textContent = (data.ram_available_gb || 0).toFixed(1) + 'GB';
                document.getElementById('gpuStat').textContent = data.has_cuda ? 'Yes' : 'No';
                document.getElementById('vramStat').textContent = (data.vram_available_gb || 0).toFixed(1) + 'GB';
            } catch {}
        }
        
        updateSystemInfo();
        setInterval(updateSystemInfo, 5000);
    </script>
</body>
</html>
'''

with open('predycat_ai/templates/dashboard.html', 'w') as f:
    f.write(html_content)

print("Dashboard created!")