import logging

def setup_custom_logger(handle_name, filename, level):
    # logging.basicConfig(filename=filename, encoding='utf-8', level=level, filemode="w")
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    handler = logging.FileHandler('latest_logs.log', mode="w")
    logging.StreamHandler(stream=None)
    handler.setFormatter(formatter)

    logger = logging.getLogger(handle_name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False
    return logger