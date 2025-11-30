#!/usr/bin/env python3
"""
Defensio Miner - Web Dashboard
==============================
Web-accessible dashboard with real-time updates via WebSocket.
"""

import os
import sys
import json
import time
import threading
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass, field, asdict

# Try to import Flask
try:
    from flask import Flask, render_template_string, jsonify
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available. Install with: pip install flask flask-cors")

# Try to import Flask-SocketIO for real-time updates
try:
    from flask_socketio import SocketIO, emit
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False


# HTML template for the dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Defensio Miner Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
            background: #0d1117;
            color: #c9d1d9;
            padding: 20px;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 20px;
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        
        .header-title {
            font-size: 18px;
            color: #58a6ff;
        }
        
        .header-stats {
            display: flex;
            gap: 20px;
            font-size: 14px;
        }
        
        .stat {
            display: flex;
            gap: 5px;
        }
        
        .stat-label {
            color: #8b949e;
        }
        
        .stat-value {
            color: #3fb950;
            font-weight: bold;
        }
        
        .stat-value.yellow {
            color: #d29922;
        }
        
        .stat-value.cyan {
            color: #58a6ff;
        }
        
        .wallets-table {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            overflow: hidden;
        }
        
        .table-header {
            display: grid;
            grid-template-columns: 40px 60px 180px 1fr 60px 100px 80px;
            padding: 12px 15px;
            background: #21262d;
            border-bottom: 1px solid #30363d;
            font-size: 12px;
            color: #8b949e;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .wallet-row {
            display: grid;
            grid-template-columns: 40px 60px 180px 1fr 60px 100px 80px;
            padding: 10px 15px;
            border-bottom: 1px solid #21262d;
            font-size: 13px;
            transition: background 0.2s;
        }
        
        .wallet-row:hover {
            background: #21262d;
        }
        
        .wallet-row:last-child {
            border-bottom: none;
        }
        
        .indicator {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
        }
        
        .indicator.solved { background: #3fb950; }
        .indicator.mining { background: #58a6ff; animation: pulse 1s infinite; }
        .indicator.waiting { background: #d29922; }
        .indicator.idle { background: #484f58; }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .address {
            font-family: monospace;
            color: #8b949e;
        }
        
        .progress-bar {
            display: flex;
            gap: 2px;
            align-items: center;
        }
        
        .progress-block {
            width: 16px;
            height: 16px;
            border-radius: 2px;
            display: inline-block;
        }
        
        .progress-block.solved {
            background: #238636;
        }
        
        .progress-block.unsolved {
            background: #21262d;
            border: 1px solid #30363d;
        }
        
        .progress-group {
            display: flex;
            gap: 2px;
            margin-right: 8px;
        }
        
        .status {
            font-weight: 500;
        }
        
        .status.solved { color: #3fb950; }
        .status.mining { color: #58a6ff; }
        .status.waiting { color: #d29922; }
        
        .time {
            color: #3fb950;
            text-align: right;
        }
        
        .footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 20px;
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            margin-top: 20px;
            font-size: 14px;
        }
        
        .footer-stats {
            display: flex;
            gap: 30px;
        }
        
        .pagination {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        
        .pagination button {
            background: #21262d;
            border: 1px solid #30363d;
            color: #c9d1d9;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-family: inherit;
            transition: all 0.2s;
        }
        
        .pagination button:hover {
            background: #30363d;
            border-color: #58a6ff;
        }
        
        .pagination button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .page-info {
            color: #8b949e;
        }
        
        .loading {
            text-align: center;
            padding: 50px;
            color: #8b949e;
        }
        
        .connection-status {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
        }
        
        .connection-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }
        
        .connection-dot.connected { background: #3fb950; }
        .connection-dot.disconnected { background: #f85149; }
        
        /* Scrollable table body */
        .table-body {
            max-height: calc(100vh - 280px);
            overflow-y: auto;
        }
        
        /* Custom scrollbar */
        .table-body::-webkit-scrollbar {
            width: 8px;
        }
        
        .table-body::-webkit-scrollbar-track {
            background: #21262d;
        }
        
        .table-body::-webkit-scrollbar-thumb {
            background: #30363d;
            border-radius: 4px;
        }
        
        .table-body::-webkit-scrollbar-thumb:hover {
            background: #484f58;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-title">
                Defensio <span id="version">V1.0</span> | 
                <span style="color: #58a6ff;">Hybrid</span> | 
                <span id="workers">0</span> Workers | 
                <span id="cpu-gpu" style="color: #d29922;">0 CPU + 0 GPU</span>
            </div>
            <div class="header-stats">
                <div class="stat">
                    <span class="stat-label">Tokens:</span>
                    <span class="stat-value" id="tokens">0</span>
                </div>
                <div class="connection-status">
                    <span class="connection-dot" id="connection-dot"></span>
                    <span id="connection-text">Connecting...</span>
                </div>
            </div>
        </div>
        
        <div class="wallets-table">
            <div class="table-header">
                <div></div>
                <div>#</div>
                <div>Address</div>
                <div>Progress</div>
                <div>Sol</div>
                <div>Status</div>
                <div>Time</div>
            </div>
            <div class="table-body" id="wallets-container">
                <div class="loading">Loading wallets...</div>
            </div>
        </div>
        
        <div class="footer">
            <div class="footer-stats">
                <div class="stat">
                    <span class="stat-label">Today:</span>
                    <span class="stat-value" id="total-solved">0</span>
                    <span class="stat-label">Solved</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Rate:</span>
                    <span class="stat-value cyan" id="rate">0/h</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Hash:</span>
                    <span class="stat-value" id="hashrate">0 H/s</span>
                </div>
                <div class="stat">
                    <span class="stat-label">CPU:</span>
                    <span class="stat-value yellow" id="cpu">0%</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Task:</span>
                    <span class="stat-value cyan" id="task">-</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Diff:</span>
                    <span class="stat-value cyan" id="difficulty">-</span>
                </div>
            </div>
            <div class="pagination">
                <button id="prev-btn" onclick="prevPage()" disabled>← Prev</button>
                <span class="page-info" id="page-info">Page 1</span>
                <button id="next-btn" onclick="nextPage()">Next →</button>
            </div>
        </div>
    </div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.5.4/socket.io.min.js"></script>
    <script>
        let wallets = {};
        let stats = {};
        let config = {};
        let currentPage = 0;
        const walletsPerPage = 50;
        let socket = null;
        let connected = false;
        
        // Initialize Socket.IO connection
        function initSocket() {
            socket = io();
            
            socket.on('connect', () => {
                connected = true;
                updateConnectionStatus();
                console.log('Connected to server');
            });
            
            socket.on('disconnect', () => {
                connected = false;
                updateConnectionStatus();
                console.log('Disconnected from server');
            });
            
            socket.on('update', (data) => {
                if (data.wallets) {
                    wallets = data.wallets;
                }
                if (data.stats) {
                    stats = data.stats;
                }
                if (data.config) {
                    config = data.config;
                }
                render();
            });
            
            socket.on('wallet_update', (data) => {
                wallets[data.id] = data;
                render();
            });
            
            socket.on('stats_update', (data) => {
                stats = {...stats, ...data};
                updateStats();
            });
        }
        
        // Fallback to polling if WebSocket not available
        function initPolling() {
            setInterval(fetchData, 1000);
            fetchData();
        }
        
        async function fetchData() {
            try {
                const response = await fetch('/api/data');
                const data = await response.json();
                wallets = data.wallets || {};
                stats = data.stats || {};
                config = data.config || {};
                connected = true;
                updateConnectionStatus();
                render();
            } catch (e) {
                connected = false;
                updateConnectionStatus();
            }
        }
        
        function updateConnectionStatus() {
            const dot = document.getElementById('connection-dot');
            const text = document.getElementById('connection-text');
            
            if (connected) {
                dot.className = 'connection-dot connected';
                text.textContent = 'Connected';
            } else {
                dot.className = 'connection-dot disconnected';
                text.textContent = 'Disconnected';
            }
        }
        
        function formatAddress(addr) {
            if (!addr) return '-';
            return addr.slice(0, 10) + '...' + addr.slice(-4);
        }
        
        function formatTime(seconds) {
            if (!seconds) return '';
            if (seconds < 60) return Math.round(seconds) + 's';
            return (seconds / 60).toFixed(1) + 'm';
        }
        
        function formatHashrate(rate) {
            if (rate >= 1000000) return (rate / 1000000).toFixed(1) + 'M/s';
            if (rate >= 1000) return (rate / 1000).toFixed(1) + 'K/s';
            return rate.toFixed(0) + ' H/s';
        }
        
        function createProgressBar(challenges) {
            if (!challenges || !Array.isArray(challenges)) {
                challenges = Array(24).fill(false);
            }
            
            let html = '<div class="progress-bar">';
            for (let i = 0; i < 24; i += 4) {
                html += '<div class="progress-group">';
                for (let j = 0; j < 4 && (i + j) < 24; j++) {
                    const solved = challenges[i + j];
                    html += `<div class="progress-block ${solved ? 'solved' : 'unsolved'}"></div>`;
                }
                html += '</div>';
            }
            html += '</div>';
            return html;
        }
        
        function render() {
            // Update header
            document.getElementById('version').textContent = 'V' + (config.version || '1.0');
            document.getElementById('workers').textContent = config.num_workers || 0;
            document.getElementById('cpu-gpu').textContent = 
                `${config.num_cpu || 0} CPU + ${config.num_gpu || 0} GPU`;
            document.getElementById('tokens').textContent = 
                ((stats.tokens_earned || 0) / 1000).toFixed(1) + 'K';
            
            // Render wallets
            const container = document.getElementById('wallets-container');
            const walletArray = Object.values(wallets).sort((a, b) => a.id - b.id);
            const totalPages = Math.ceil(walletArray.length / walletsPerPage);
            
            const start = currentPage * walletsPerPage;
            const end = Math.min(start + walletsPerPage, walletArray.length);
            const pageWallets = walletArray.slice(start, end);
            
            if (pageWallets.length === 0) {
                container.innerHTML = '<div class="loading">No wallets found</div>';
            } else {
                container.innerHTML = pageWallets.map(wallet => {
                    const statusClass = (wallet.status || 'idle').toLowerCase();
                    return `
                        <div class="wallet-row">
                            <div><span class="indicator ${statusClass}"></span></div>
                            <div>${wallet.id}</div>
                            <div class="address">${formatAddress(wallet.address)}</div>
                            <div>${createProgressBar(wallet.challenges_solved)}</div>
                            <div>${wallet.total_solved || 0}</div>
                            <div class="status ${statusClass}">${wallet.status || 'Idle'}</div>
                            <div class="time">${formatTime(wallet.solve_time)}</div>
                        </div>
                    `;
                }).join('');
            }
            
            // Update pagination
            document.getElementById('prev-btn').disabled = currentPage === 0;
            document.getElementById('next-btn').disabled = currentPage >= totalPages - 1;
            document.getElementById('page-info').textContent = 
                `Page ${currentPage + 1} of ${totalPages || 1} (${walletArray.length} wallets)`;
            
            updateStats();
        }
        
        function updateStats() {
            document.getElementById('total-solved').textContent = stats.total_solved || 0;
            document.getElementById('rate').textContent = (stats.rate_per_hour || 0).toFixed(0) + '/h';
            document.getElementById('hashrate').textContent = formatHashrate(stats.hash_rate || 0);
            document.getElementById('cpu').textContent = (stats.cpu_usage || 0).toFixed(0) + '%';
            document.getElementById('task').textContent = stats.current_task || '-';
            document.getElementById('difficulty').textContent = stats.difficulty || '-';
        }
        
        function prevPage() {
            if (currentPage > 0) {
                currentPage--;
                render();
            }
        }
        
        function nextPage() {
            const totalPages = Math.ceil(Object.keys(wallets).length / walletsPerPage);
            if (currentPage < totalPages - 1) {
                currentPage++;
                render();
            }
        }
        
        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowDown' || e.key === 'j') nextPage();
            if (e.key === 'ArrowUp' || e.key === 'k') prevPage();
            if (e.key === 'Home') { currentPage = 0; render(); }
            if (e.key === 'End') { 
                currentPage = Math.max(0, Math.ceil(Object.keys(wallets).length / walletsPerPage) - 1);
                render();
            }
        });
        
        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            // Try WebSocket first, fall back to polling
            if (typeof io !== 'undefined') {
                initSocket();
            } else {
                initPolling();
            }
        });
    </script>
</body>
</html>
"""


class WebDashboard:
    """Web-accessible dashboard with real-time updates."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self.wallets: Dict[int, dict] = {}
        self.stats: dict = {
            'total_solved': 0,
            'rate_per_hour': 0,
            'hash_rate': 0,
            'cpu_usage': 0,
            'current_task': '',
            'difficulty': '',
            'tokens_earned': 0
        }
        self.config: dict = {
            'version': '1.0',
            'num_workers': 0,
            'num_cpu': 0,
            'num_gpu': 0
        }
        self.lock = threading.Lock()
        self.app = None
        self.socketio = None
        self.running = False
        
        if FLASK_AVAILABLE:
            self._setup_flask()
    
    def _setup_flask(self):
        """Setup Flask application."""
        self.app = Flask(__name__)
        CORS(self.app)
        
        if SOCKETIO_AVAILABLE:
            self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Routes
        @self.app.route('/')
        def index():
            return render_template_string(DASHBOARD_HTML)
        
        @self.app.route('/api/data')
        def get_data():
            with self.lock:
                return jsonify({
                    'wallets': self.wallets,
                    'stats': self.stats,
                    'config': self.config
                })
        
        @self.app.route('/api/wallets')
        def get_wallets():
            with self.lock:
                return jsonify(self.wallets)
        
        @self.app.route('/api/stats')
        def get_stats():
            with self.lock:
                return jsonify(self.stats)
    
    def set_config(self, num_workers: int, num_cpu: int, num_gpu: int, version: str = "1.0"):
        """Set miner configuration."""
        with self.lock:
            self.config = {
                'version': version,
                'num_workers': num_workers,
                'num_cpu': num_cpu,
                'num_gpu': num_gpu
            }
        self._emit_update()
    
    def update_wallet(self, wallet_id: int, address: str,
                      challenge_idx: Optional[int] = None,
                      solved: bool = False,
                      status: str = None,
                      solve_time: Optional[float] = None,
                      dfo_earned: float = 0.0):
        """Update wallet status."""
        with self.lock:
            if wallet_id not in self.wallets:
                self.wallets[wallet_id] = {
                    'id': wallet_id,
                    'address': address,
                    'challenges_solved': [False] * 24,
                    'total_solved': 0,
                    'status': 'Waiting',
                    'solve_time': None,
                    'dfo_earned': 0.0
                }
            
            wallet = self.wallets[wallet_id]
            
            if challenge_idx is not None and solved:
                if 0 <= challenge_idx < 24:
                    wallet['challenges_solved'][challenge_idx] = True
                wallet['total_solved'] += 1
                self.stats['total_solved'] += 1
                
                if solve_time:
                    wallet['solve_time'] = solve_time
            
            if status:
                wallet['status'] = status
            
            if dfo_earned > 0:
                wallet['dfo_earned'] = dfo_earned
        
        self._emit_wallet_update(wallet_id)
    
    def update_stats(self, **kwargs):
        """Update global statistics."""
        with self.lock:
            for key, value in kwargs.items():
                if key in self.stats and value is not None:
                    self.stats[key] = value
        
        self._emit_stats_update()
    
    def _emit_update(self):
        """Emit full update via WebSocket."""
        if self.socketio and self.running:
            with self.lock:
                self.socketio.emit('update', {
                    'wallets': self.wallets,
                    'stats': self.stats,
                    'config': self.config
                })
    
    def _emit_wallet_update(self, wallet_id: int):
        """Emit single wallet update."""
        if self.socketio and self.running:
            with self.lock:
                if wallet_id in self.wallets:
                    self.socketio.emit('wallet_update', self.wallets[wallet_id])
    
    def _emit_stats_update(self):
        """Emit stats update."""
        if self.socketio and self.running:
            with self.lock:
                self.socketio.emit('stats_update', self.stats)
    
    def run(self, threaded: bool = True):
        """Start the web server."""
        if not FLASK_AVAILABLE:
            print("Flask not available. Install with: pip install flask flask-cors")
            return
        
        self.running = True
        
        if threaded:
            thread = threading.Thread(target=self._run_server, daemon=True)
            thread.start()
            print(f"Web dashboard started at http://{self.host}:{self.port}")
            return thread
        else:
            self._run_server()
    
    def _run_server(self):
        """Run the Flask server."""
        if self.socketio:
            self.socketio.run(self.app, host=self.host, port=self.port, 
                            debug=False, use_reloader=False, log_output=False)
        else:
            self.app.run(host=self.host, port=self.port, 
                        debug=False, use_reloader=False)
    
    def stop(self):
        """Stop the web server."""
        self.running = False


# Demo mode
if __name__ == "__main__":
    import random
    
    print("Web Dashboard Demo")
    print("=" * 50)
    print(f"Flask available: {FLASK_AVAILABLE}")
    print(f"SocketIO available: {SOCKETIO_AVAILABLE}")
    
    if not FLASK_AVAILABLE:
        print("\nInstall dependencies:")
        print("  pip install flask flask-cors flask-socketio")
        sys.exit(1)
    
    # Create dashboard
    dashboard = WebDashboard(port=8080)
    dashboard.set_config(num_workers=9, num_cpu=8, num_gpu=1, version="1.0")
    
    # Add sample wallets
    for i in range(200):
        addr = f"addr1q{''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=50))}"
        challenges = [random.random() > 0.3 for _ in range(24)]
        dashboard.wallets[i] = {
            'id': i,
            'address': addr,
            'challenges_solved': challenges,
            'total_solved': sum(challenges),
            'status': random.choice(["Solved", "Waiting", "Mining"]),
            'solve_time': random.uniform(5, 60) if random.random() > 0.2 else None,
            'dfo_earned': random.uniform(0, 10)
        }
    
    dashboard.update_stats(
        hash_rate=246500,
        cpu_usage=99,
        current_task="D09C14",
        difficulty="03FF",
        rate_per_hour=151,
        tokens_earned=1100
    )
    
    print(f"\nStarting web dashboard at http://localhost:8080")
    print("Press Ctrl+C to stop\n")
    
    # Run server (blocking)
    try:
        dashboard.run(threaded=False)
    except KeyboardInterrupt:
        dashboard.stop()
        print("\nDashboard stopped.")
