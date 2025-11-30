#!/usr/bin/env python3
"""
Defensio Miner - Advanced Terminal Dashboard
=============================================
Rich terminal UI with progress bars, stats, and real-time updates.
"""

import os
import sys
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque

# Try to import rich for fancy terminal output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
    from rich.style import Style
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


@dataclass
class WalletStatus:
    """Status of a single wallet."""
    id: int
    address: str
    challenges_solved: List[bool] = field(default_factory=lambda: [False] * 24)
    total_solved: int = 0
    status: str = "Waiting"
    solve_time: Optional[float] = None
    last_updated: datetime = field(default_factory=datetime.now)
    dfo_earned: float = 0.0


@dataclass
class MinerStats:
    """Global miner statistics."""
    total_solved: int = 0
    rate_per_hour: float = 0.0
    hash_rate: float = 0.0
    cpu_usage: float = 0.0
    current_task: str = ""
    difficulty: str = ""
    tokens_earned: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    solve_times: deque = field(default_factory=lambda: deque(maxlen=100))
    
    @property
    def avg_solve_time(self) -> float:
        if not self.solve_times:
            return 0.0
        return sum(self.solve_times) / len(self.solve_times)


class TerminalDashboard:
    """Advanced terminal dashboard using Rich library."""
    
    def __init__(self, title: str = "Defensio Miner"):
        self.title = title
        self.wallets: Dict[int, WalletStatus] = {}
        self.stats = MinerStats()
        self.console = Console() if RICH_AVAILABLE else None
        self.running = False
        self.lock = threading.Lock()
        self.page = 0
        self.wallets_per_page = 50
        self.num_workers = 0
        self.num_cpu = 0
        self.num_gpu = 0
        self.version = "1.0"
        
    def set_config(self, num_workers: int, num_cpu: int, num_gpu: int, version: str = "1.0"):
        """Set miner configuration."""
        self.num_workers = num_workers
        self.num_cpu = num_cpu
        self.num_gpu = num_gpu
        self.version = version
    
    def update_wallet(self, wallet_id: int, address: str, 
                      challenge_idx: Optional[int] = None,
                      solved: bool = False,
                      status: str = None,
                      solve_time: Optional[float] = None,
                      dfo_earned: float = 0.0):
        """Update wallet status."""
        with self.lock:
            if wallet_id not in self.wallets:
                self.wallets[wallet_id] = WalletStatus(
                    id=wallet_id,
                    address=address
                )
            
            wallet = self.wallets[wallet_id]
            wallet.last_updated = datetime.now()
            
            if challenge_idx is not None and solved:
                if 0 <= challenge_idx < 24:
                    wallet.challenges_solved[challenge_idx] = True
                wallet.total_solved += 1
                self.stats.total_solved += 1
                
                if solve_time:
                    wallet.solve_time = solve_time
                    self.stats.solve_times.append(solve_time)
            
            if status:
                wallet.status = status
            
            if dfo_earned > 0:
                wallet.dfo_earned = dfo_earned
    
    def update_stats(self, hash_rate: float = None, cpu_usage: float = None,
                    current_task: str = None, difficulty: str = None,
                    rate_per_hour: float = None, tokens: float = None,
                    total_solved: int = None):
        """Update global statistics."""
        with self.lock:
            if hash_rate is not None:
                self.stats.hash_rate = hash_rate
            if cpu_usage is not None:
                self.stats.cpu_usage = cpu_usage
            if current_task is not None:
                self.stats.current_task = current_task
            if difficulty is not None:
                self.stats.difficulty = difficulty
            if rate_per_hour is not None:
                self.stats.rate_per_hour = rate_per_hour
            if tokens is not None:
                self.stats.tokens_earned = tokens
            if total_solved is not None:
                self.stats.total_solved = total_solved
    
    def _format_address(self, address: str, length: int = 15) -> str:
        """Format address for display."""
        if len(address) <= length:
            return address
        return address[:length-2] + ".." + address[-2:]
    
    def _format_time(self, seconds: float) -> str:
        """Format solve time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    def _create_progress_bar(self, challenges: List[bool], width: int = 24) -> str:
        """Create ASCII progress bar for challenges."""
        bar = ""
        for solved in challenges:
            if solved:
                bar += "█"
            else:
                bar += "░"
        return bar
    
    def _create_rich_progress_bar(self, challenges: List[bool]) -> Text:
        """Create Rich text progress bar for challenges."""
        text = Text()
        for i, solved in enumerate(challenges):
            if solved:
                text.append("██", style="bold green")
            else:
                text.append("░░", style="dim white")
            if (i + 1) % 6 == 0 and i < 23:
                text.append(" ", style="")
        return text
    
    def render_simple(self) -> str:
        """Render simple ASCII dashboard for terminals without Rich."""
        lines = []
        
        # Header
        header = f"Defensio V{self.version} | Hybrid | {self.num_workers} Workers | {self.num_cpu} CPU + {self.num_gpu} GPU | {self.stats.tokens_earned:.1f}K Tokens"
        lines.append("=" * 80)
        lines.append(header.center(80))
        lines.append("=" * 80)
        
        # Wallets table
        lines.append(f"{'#':<4} {'Address':<18} {'Progress':<26} {'Sol':<4} {'Status':<10} {'Time':<6}")
        lines.append("-" * 80)
        
        sorted_wallets = sorted(self.wallets.values(), key=lambda w: w.id)
        start_idx = self.page * self.wallets_per_page
        end_idx = start_idx + self.wallets_per_page
        
        for wallet in sorted_wallets[start_idx:end_idx]:
            addr = self._format_address(wallet.address, 16)
            progress = self._create_progress_bar(wallet.challenges_solved)
            time_str = self._format_time(wallet.solve_time) if wallet.solve_time else ""
            
            lines.append(f"{wallet.id:<4} {addr:<18} {progress:<26} {wallet.total_solved:<4} {wallet.status:<10} {time_str:<6}")
        
        # Pagination
        total_pages = (len(self.wallets) + self.wallets_per_page - 1) // self.wallets_per_page
        remaining = len(self.wallets) - end_idx
        if remaining > 0:
            lines.append(f"↓ {remaining} more below (↓ for next page, b for bottom)")
        
        # Footer stats
        lines.append("=" * 80)
        stats_line = (
            f"Today: {self.stats.total_solved:>4} Solved | "
            f"Rate:{self.stats.rate_per_hour:.0f}/h | "
            f"Hash:{self.stats.hash_rate/1000:.1f}K/s | "
            f"CPU:{self.stats.cpu_usage:.0f}% | "
            f"Task:{self.stats.current_task[:8]} | "
            f"Diff:{self.stats.difficulty}"
        )
        lines.append(stats_line)
        
        return "\n".join(lines)
    
    def render_rich(self) -> Panel:
        """Render rich dashboard with colors and formatting."""
        if not RICH_AVAILABLE:
            return None
        
        # Create main table
        table = Table(
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
            expand=True,
            box=None
        )
        
        table.add_column("", style="dim", width=2)
        table.add_column("#", style="dim", width=4)
        table.add_column("Address", style="white", width=18)
        table.add_column("Progress", width=54)
        table.add_column("Sol", style="cyan", width=4, justify="right")
        table.add_column("Status", width=10)
        table.add_column("Time", style="green", width=6, justify="right")
        
        sorted_wallets = sorted(self.wallets.values(), key=lambda w: w.id)
        start_idx = self.page * self.wallets_per_page
        end_idx = start_idx + self.wallets_per_page
        
        for wallet in sorted_wallets[start_idx:end_idx]:
            # Status indicator
            if wallet.status == "Waiting":
                indicator = "●"
                indicator_style = "yellow"
            elif wallet.status == "Mining":
                indicator = "●"
                indicator_style = "blue"
            elif wallet.status == "Solved":
                indicator = "●"
                indicator_style = "green"
            else:
                indicator = "●"
                indicator_style = "dim"
            
            # Address
            addr = self._format_address(wallet.address, 16)
            
            # Progress bar
            progress = self._create_rich_progress_bar(wallet.challenges_solved)
            
            # Solve time
            time_str = self._format_time(wallet.solve_time) if wallet.solve_time else ""
            
            # Status styling
            status_style = "green" if wallet.status == "Solved" else "yellow" if wallet.status == "Waiting" else "cyan"
            
            table.add_row(
                Text(indicator, style=indicator_style),
                str(wallet.id),
                addr,
                progress,
                str(wallet.total_solved),
                Text(wallet.status, style=status_style),
                time_str
            )
        
        # Pagination info
        remaining = len(self.wallets) - end_idx
        if remaining > 0:
            table.add_row("", "", "", Text(f"↓ {remaining} more below (↓ for next page, b for bottom)", style="dim"), "", "", "")
        
        # Create header
        header_text = Text()
        header_text.append(f"Defensio V{self.version}", style="bold green")
        header_text.append(" | ", style="dim")
        header_text.append("Hybrid", style="cyan")
        header_text.append(" | ", style="dim")
        header_text.append(f"{self.num_workers} Workers", style="white")
        header_text.append(" | ", style="dim")
        header_text.append(f"{self.num_cpu} CPU + {self.num_gpu} GPU", style="yellow")
        header_text.append(" | ", style="dim")
        header_text.append(f"{self.stats.tokens_earned:.1f}K Tokens", style="bold green")
        
        # Create footer
        footer_text = Text()
        footer_text.append("Today: ", style="dim")
        footer_text.append(f"{self.stats.total_solved:>4}", style="bold white")
        footer_text.append(" Solved | ", style="dim")
        footer_text.append("Rate:", style="dim")
        footer_text.append(f"{self.stats.rate_per_hour:.0f}/h", style="cyan")
        footer_text.append(" | ", style="dim")
        footer_text.append("Hash:", style="dim")
        footer_text.append(f"{self.stats.hash_rate/1000:.1f}K/s", style="green")
        footer_text.append(" | ", style="dim")
        footer_text.append("CPU:", style="dim")
        footer_text.append(f"{self.stats.cpu_usage:.0f}%", style="yellow" if self.stats.cpu_usage < 80 else "red")
        footer_text.append(" | ", style="dim")
        footer_text.append("Task:", style="dim")
        footer_text.append(f"{self.stats.current_task[:8]}", style="cyan")
        footer_text.append(" | ", style="dim")
        footer_text.append("Diff:", style="dim")
        footer_text.append(f"{self.stats.difficulty}", style="magenta")
        
        # Combine into layout
        layout = Layout()
        layout.split_column(
            Layout(Panel(header_text, style=""), size=3),
            Layout(table),
            Layout(Panel(footer_text, style=""), size=3)
        )
        
        return Panel(
            layout,
            title="[bold cyan]DEFENSIO MINER[/]",
            border_style="cyan"
        )
    
    def run_live(self, refresh_rate: float = 0.5):
        """Run live updating dashboard."""
        self.running = True
        
        if RICH_AVAILABLE and self.console:
            with Live(self.render_rich(), console=self.console, refresh_per_second=1/refresh_rate) as live:
                while self.running:
                    live.update(self.render_rich())
                    time.sleep(refresh_rate)
        else:
            # Fallback to simple rendering
            while self.running:
                os.system('clear' if os.name == 'posix' else 'cls')
                print(self.render_simple())
                time.sleep(refresh_rate)
    
    def stop(self):
        """Stop the dashboard."""
        self.running = False


class SimpleDashboard:
    """Simple ASCII dashboard without external dependencies."""
    
    # ANSI color codes
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BG_GREEN = "\033[42m"
    
    def __init__(self, title: str = "Defensio Miner"):
        self.title = title
        self.wallets: Dict[int, WalletStatus] = {}
        self.stats = MinerStats()
        self.running = False
        self.lock = threading.Lock()
        self.page = 0
        self.wallets_per_page = 50
        self.num_workers = 0
        self.num_cpu = 0
        self.num_gpu = 0
        self.version = "1.0"
        self.use_colors = sys.stdout.isatty()
    
    def set_config(self, num_workers: int, num_cpu: int, num_gpu: int, version: str = "1.0"):
        """Set miner configuration."""
        self.num_workers = num_workers
        self.num_cpu = num_cpu
        self.num_gpu = num_gpu
        self.version = version
    
    def _c(self, text: str, color: str) -> str:
        """Colorize text if colors are enabled."""
        if self.use_colors:
            return f"{color}{text}{self.RESET}"
        return text
    
    def update_wallet(self, wallet_id: int, address: str, 
                      challenge_idx: Optional[int] = None,
                      solved: bool = False,
                      status: str = None,
                      solve_time: Optional[float] = None,
                      dfo_earned: float = 0.0):
        """Update wallet status."""
        with self.lock:
            if wallet_id not in self.wallets:
                self.wallets[wallet_id] = WalletStatus(
                    id=wallet_id,
                    address=address
                )
            
            wallet = self.wallets[wallet_id]
            wallet.last_updated = datetime.now()
            
            if challenge_idx is not None and solved:
                if 0 <= challenge_idx < 24:
                    wallet.challenges_solved[challenge_idx] = True
                wallet.total_solved += 1
                self.stats.total_solved += 1
                
                if solve_time:
                    wallet.solve_time = solve_time
                    self.stats.solve_times.append(solve_time)
            
            if status:
                wallet.status = status
            
            if dfo_earned > 0:
                wallet.dfo_earned = dfo_earned
    
    def update_stats(self, **kwargs):
        """Update global statistics."""
        with self.lock:
            for key, value in kwargs.items():
                if hasattr(self.stats, key) and value is not None:
                    setattr(self.stats, key, value)
    
    def _format_address(self, address: str) -> str:
        """Format address for display."""
        return address[:8] + "..." + address[-2:]
    
    def _format_time(self, seconds: float) -> str:
        """Format solve time."""
        if seconds is None:
            return ""
        if seconds < 60:
            return f"{seconds:.0f}s"
        return f"{seconds/60:.1f}m"
    
    def _progress_bar(self, challenges: List[bool]) -> str:
        """Create colored progress bar."""
        parts = []
        for i in range(0, 24, 4):
            chunk = challenges[i:i+4]
            for solved in chunk:
                if solved:
                    parts.append(self._c("█", self.GREEN))
                else:
                    parts.append(self._c("░", self.DIM))
            parts.append(" ")
        return "".join(parts)
    
    def render(self) -> str:
        """Render the dashboard."""
        lines = []
        term_width = os.get_terminal_size().columns if hasattr(os, 'get_terminal_size') else 100
        
        # Header
        header = f"Defensio V{self.version} | Hybrid | {self.num_workers} Workers | {self.num_cpu} CPU + {self.num_gpu} GPU | {self.stats.tokens_earned:.1f}K Tokens"
        lines.append(self._c(header, self.BOLD + self.CYAN))
        lines.append("")
        
        # Table header
        hdr = f"{'':2} {'#':>4}  {'Address':<16}  {'Progress':<30}  {'Sol':>3}  {'Status':<8}  {'Time':>5}"
        lines.append(self._c(hdr, self.DIM))
        
        # Wallets
        sorted_wallets = sorted(self.wallets.values(), key=lambda w: w.id)
        start_idx = self.page * self.wallets_per_page
        end_idx = min(start_idx + self.wallets_per_page, len(sorted_wallets))
        
        for wallet in sorted_wallets[start_idx:end_idx]:
            # Indicator
            if wallet.status == "Solved":
                ind = self._c("●", self.GREEN)
            elif wallet.status == "Mining":
                ind = self._c("●", self.CYAN)
            else:
                ind = self._c("●", self.YELLOW)
            
            addr = self._format_address(wallet.address)
            progress = self._progress_bar(wallet.challenges_solved)
            time_str = self._format_time(wallet.solve_time)
            
            # Status color
            if wallet.status == "Solved":
                status = self._c(f"{wallet.status:<8}", self.GREEN)
            elif wallet.status == "Mining":
                status = self._c(f"{wallet.status:<8}", self.CYAN)
            else:
                status = self._c(f"{wallet.status:<8}", self.YELLOW)
            
            line = f"{ind:2} {wallet.id:>4}  {addr:<16}  {progress}  {wallet.total_solved:>3}  {status}  {self._c(time_str, self.GREEN):>5}"
            lines.append(line)
        
        # Pagination
        remaining = len(sorted_wallets) - end_idx
        if remaining > 0:
            lines.append(self._c(f"↓ {remaining} more below (↓ for next page, b for bottom)", self.DIM))
        
        lines.append("")
        
        # Footer stats
        footer = (
            f"Today: {self._c(str(self.stats.total_solved), self.BOLD)} Solved | "
            f"Rate:{self._c(f'{self.stats.rate_per_hour:.0f}/h', self.CYAN)} | "
            f"Hash:{self._c(f'{self.stats.hash_rate/1000:.1f}K/s', self.GREEN)} | "
            f"CPU:{self._c(f'{self.stats.cpu_usage:.0f}%', self.YELLOW)} | "
            f"Task:{self._c(self.stats.current_task[:8], self.CYAN)} | "
            f"Diff:{self._c(self.stats.difficulty, self.CYAN)}"
        )
        lines.append(footer)
        
        return "\n".join(lines)
    
    def run_live(self, refresh_rate: float = 1.0):
        """Run live updating dashboard."""
        self.running = True
        
        # Hide cursor
        if self.use_colors:
            print("\033[?25l", end="")
        
        try:
            while self.running:
                # Clear screen and move to top
                if self.use_colors:
                    print("\033[H\033[J", end="")
                else:
                    os.system('clear' if os.name == 'posix' else 'cls')
                
                print(self.render())
                time.sleep(refresh_rate)
        finally:
            # Show cursor
            if self.use_colors:
                print("\033[?25h", end="")
    
    def stop(self):
        """Stop the dashboard."""
        self.running = False


def get_dashboard(use_rich: bool = True) -> Any:
    """Get appropriate dashboard based on availability."""
    if use_rich and RICH_AVAILABLE:
        return TerminalDashboard()
    return SimpleDashboard()


# Demo mode
if __name__ == "__main__":
    import random
    
    print("Dashboard Demo Mode")
    print("=" * 50)
    print(f"Rich library available: {RICH_AVAILABLE}")
    
    # Create dashboard
    dashboard = get_dashboard(use_rich=True)
    dashboard.set_config(num_workers=9, num_cpu=8, num_gpu=1, version="1.0")
    
    # Add sample wallets
    for i in range(200):
        addr = f"addr1q{''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=50))}"
        challenges = [random.random() > 0.3 for _ in range(24)]
        dashboard.wallets[i] = WalletStatus(
            id=i,
            address=addr,
            challenges_solved=challenges,
            total_solved=sum(challenges),
            status=random.choice(["Solved", "Waiting", "Mining"]),
            solve_time=random.uniform(5, 60) if random.random() > 0.2 else None
        )
    
    dashboard.update_stats(
        hash_rate=246500,
        cpu_usage=99,
        current_task="D09C14",
        difficulty="03FF",
        rate_per_hour=151,
        tokens=1.1
    )
    
    # Run dashboard
    print("\nStarting live dashboard (Ctrl+C to stop)...")
    time.sleep(2)
    
    try:
        dashboard.run_live()
    except KeyboardInterrupt:
        dashboard.stop()
        print("\nDashboard stopped.")
