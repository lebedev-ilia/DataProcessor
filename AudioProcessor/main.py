#!/usr/bin/env python3
"""
Точка входа для AudioProcessor.

Запуск сервера:
    python main.py

Или через uvicorn:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""
import os
import sys
import logging
from pathlib import Path

# Добавляем src в PYTHONPATH
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.utils.silence import silence_all
silence_all()
logger = logging.getLogger(__name__)

def main():
    """Основная функция запуска."""
    try:
        import uvicorn
        from src.api.main import app
        
        # Параметры запуска
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "8000"))
        reload = os.getenv("RELOAD", "false").lower() == "true"
        # Force critical to avoid any server logs
        log_level = "critical"
        
        # All logs silenced
        
        # Запуск сервера
        uvicorn.run(
            "src.api.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level=log_level,
            access_log=False
        )
        
    except KeyboardInterrupt:
        pass
    except Exception as e:
        # Suppressed
        sys.exit(1)

if __name__ == "__main__":
    main()
