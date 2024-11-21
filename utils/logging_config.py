import logging
from colorama import Fore, Style, init

def configure_logging():
    """Configure logging settings with color and formatting."""
    # Initialize colorama
    init(autoreset=True)

    # Custom log level styles
    LOG_LEVEL_STYLES = {
        'INFO': Fore.CYAN + Style.BRIGHT,
        'ERROR': Fore.RED + Style.BRIGHT,
        'WARNING': Fore.YELLOW + Style.BRIGHT,
        'DEBUG': Fore.GREEN + Style.BRIGHT
    }

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            log_fmt = f'{Fore.GREEN}%(asctime)s{Style.RESET_ALL} - {LOG_LEVEL_STYLES.get(record.levelname, "")}%(levelname)s{Style.RESET_ALL} - {Fore.MAGENTA}%(funcName)s{Style.RESET_ALL} - %(message)s'
            formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
            return formatter.format(record)

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    handler.setFormatter(CustomFormatter())
    logger.handlers = [handler]
    return logger