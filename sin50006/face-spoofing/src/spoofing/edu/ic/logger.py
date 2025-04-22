# spoofing/logger.py
""""Módulo usado para declarar o sistema de Logger para este projeto"""
import logging

def get_logger(name: str) -> logging.Logger:
    """Gerador das logs que será utilizado em todos os modulos deste projeto"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger
