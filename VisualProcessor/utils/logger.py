import logging

class ColoredFormatter(logging.Formatter):
    """Formatter with colored level names (INFO=green, WARNING=yellow, ERROR=red)."""
    
    COLORS = {
        "INFO": "\033[92m",    # зеленый
        "WARNING": "\033[93m", # желтый
        "ERROR": "\033[91m",   # красный
    }
    RESET = "\033[0m"

    def format(self, record):
        # сохраняем оригинальный levelname
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        return super().format(record)

def get_logger(name, level = logging.INFO):

    # Настройка логгера
    logger = logging.getLogger(name)
    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = ColoredFormatter("%(asctime)s %(levelname)s: %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    logger.setLevel(level)

    return logger