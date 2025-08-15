"""
Performance metrics tracking for RumiAI v2.
"""
import time
from typing import Dict, Any, Optional
from collections import defaultdict
import psutil
import os


class Metrics:
    """Track performance metrics for the system."""
    
    def __init__(self):
        self.timers = {}
        self.counters = defaultdict(int)
        self.gauges = {}
        self.start_time = time.time()
        self.process = psutil.Process(os.getpid())
    
    def start_timer(self, name: str) -> None:
        """Start a timer."""
        self.timers[name] = time.time()
    
    def stop_timer(self, name: str) -> float:
        """Stop a timer and return elapsed time."""
        if name not in self.timers:
            return 0.0
        
        elapsed = time.time() - self.timers[name]
        del self.timers[name]
        return elapsed
    
    def get_time(self, name: str) -> float:
        """Get elapsed time without stopping timer."""
        if name not in self.timers:
            return 0.0
        return time.time() - self.timers[name]
    
    def increment(self, name: str, value: int = 1) -> None:
        """Increment a counter."""
        self.counters[name] += value
    
    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge value."""
        self.gauges[name] = value
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        memory_info = self.process.memory_info()
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': self.process.memory_percent()
        }
    
    def get_cpu_usage(self) -> float:
        """Get CPU usage percentage."""
        return self.process.cpu_percent(interval=0.1)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all metrics."""
        uptime = time.time() - self.start_time
        
        return {
            'uptime_seconds': uptime,
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'active_timers': list(self.timers.keys()),
            'memory': self.get_memory_usage(),
            'cpu_percent': self.get_cpu_usage()
        }
    
    def log_summary(self, logger) -> None:
        """Log metrics summary."""
        metrics = self.get_all()
        
        logger.info("=== Performance Metrics ===")
        logger.info(f"Uptime: {metrics['uptime_seconds']:.1f}s")
        logger.info(f"Memory: {metrics['memory']['rss_mb']:.1f}MB")
        logger.info(f"CPU: {metrics['cpu_percent']:.1f}%")
        
        if metrics['counters']:
            logger.info("Counters:")
            for name, value in metrics['counters'].items():
                logger.info(f"  {name}: {value}")
        
        if metrics['gauges']:
            logger.info("Gauges:")
            for name, value in metrics['gauges'].items():
                logger.info(f"  {name}: {value}")


class VideoProcessingMetrics:
    """Track metrics specific to video processing."""
    
    def __init__(self):
        self.videos_processed = 0
        self.videos_failed = 0
        self.ml_analysis_times = defaultdict(list)
        self.analysis_times = defaultdict(list)
        # self.prompt_costs = defaultdict(float)  # Removed - Python-only processing
        self.total_cost = 0.0
    
    def record_video(self, success: bool) -> None:
        """Record video processing result."""
        if success:
            self.videos_processed += 1
        else:
            self.videos_failed += 1
    
    def record_ml_time(self, model: str, time_seconds: float) -> None:
        """Record ML analysis time."""
        self.ml_analysis_times[model].append(time_seconds)
    
    def record_analysis_time(self, analysis_type: str, time_seconds: float) -> None:
        """Record analysis processing time."""
        self.analysis_times[analysis_type].append(time_seconds)
    
    # def record_prompt_cost(self, prompt_type: str, cost: float) -> None:
    #     """Record prompt cost."""
    #     self.prompt_costs[prompt_type] += cost
    #     self.total_cost += cost
    
    def get_summary(self) -> Dict[str, Any]:
        """Get processing summary."""
        total_videos = self.videos_processed + self.videos_failed
        
        # Calculate averages
        avg_ml_times = {}
        for model, times in self.ml_analysis_times.items():
            if times:
                avg_ml_times[model] = sum(times) / len(times)
        
        avg_analysis_times = {}
        for analysis, times in self.analysis_times.items():
            if times:
                avg_analysis_times[analysis] = sum(times) / len(times)
        
        return {
            'total_videos': total_videos,
            'successful': self.videos_processed,
            'failed': self.videos_failed,
            'success_rate': self.videos_processed / total_videos if total_videos > 0 else 0,
            'average_ml_times': avg_ml_times,
            'average_analysis_times': avg_analysis_times,
            # 'prompt_costs': dict(self.prompt_costs),  # Removed - Python-only
            # 'total_cost': self.total_cost,  # Removed - Python-only
            # 'cost_per_video': self.total_cost / total_videos if total_videos > 0 else 0  # Removed
        }