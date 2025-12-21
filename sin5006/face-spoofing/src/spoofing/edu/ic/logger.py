# spoofing/logger.py
""""Módulo usado para declarar o sistema de Logger para este projeto"""
from datetime import datetime
import logging
import psutil
import os

class MemoryFormatter(logging.Formatter):
    def format(self, record):
        # Coleta uso de memória atual (processo)
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        mem_used_mb = mem_info.rss / (1024 * 1024)  # Resident Set Size em MB

        # Adiciona a memória usada ao registro de log
        record.memory_used = f"{mem_used_mb:.2f} MB"
        return super().format(record)

def get_logger_tela(name: str) -> logging.Logger:
    """Gerador das logs que será utilizado em todos os modulos deste projeto"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def get_logger_arquivo(name: str, log_folder:str = "logs", log_file_path_base="app"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Garante que a pasta exista
        os.makedirs(log_folder, exist_ok=True)

        # Gera data atual para o nome do arquivo
        now = datetime.now()
        data_formatada = now.strftime("%Y-%m-%d")
        log_file_name = f"{log_file_path_base}_{data_formatada}.log"
        log_file_path = os.path.join(log_folder, log_file_name)

        # Formato personalizado: data, hora, nível, mensagem e memória
        formatter = MemoryFormatter("%(asctime)s-[%(levelname)s] [%(memory_used)s]-%(funcName)s: %(message)s")

        # Handler para console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Handler para arquivo
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

