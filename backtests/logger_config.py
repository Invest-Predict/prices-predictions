import logging

def setup_logger(name, log_file, level=logging.INFO):
    """Создаёт и настраивает логгер с указанным именем и файлом"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Формат логов
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # Файловый обработчик
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    return logger

# Определяем два логгера
# sandbox_logger = setup_logger("api_logger", "sandbox.log", level=logging.INFO)

