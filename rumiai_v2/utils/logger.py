"""
Logging configuration for RumiAI v2.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
import json
from typing import Optional


class JSONFormatter(logging.Formatter):
    """Format logs as JSON for structured logging."""
    
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename',
                          'funcName', 'levelname', 'levelno', 'lineno',
                          'module', 'msecs', 'pathname', 'process',
                          'processName', 'relativeCreated', 'thread',
                          'threadName', 'exc_info', 'exc_text', 'stack_info']:
                log_data[key] = value
        
        return json.dumps(log_data)


class Logger:
    """Configure logging for RumiAI v2."""
    
    @staticmethod
    def setup(name: str = 'rumiai_v2', 
             level: str = 'INFO',
             log_file: Optional[Path] = None,
             json_format: bool = False) -> logging.Logger:
        """
        Set up logger with console and optional file output.
        
        Args:
            name: Logger name
            level: Log level (DEBUG, INFO, WARNING, ERROR)
            log_file: Optional file path for logs
            json_format: Use JSON formatting
            
        Returns:
            Configured logger
        """
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        # Remove existing handlers
        logger.handlers = []
        
        # Console handler (stderr for compatibility with Node.js stdout parsing)
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.INFO)
        
        if json_format:
            console_handler.setFormatter(JSONFormatter())
        else:
            # Human-readable format
            console_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_format)
        
        logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            
            if json_format:
                file_handler.setFormatter(JSONFormatter())
            else:
                file_format = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
                )
                file_handler.setFormatter(file_format)
            
            logger.addHandler(file_handler)
        
        return logger
    
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """Get a logger instance."""
        return logging.getLogger(name)