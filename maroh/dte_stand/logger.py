import yaml
import logging
import logging.config


def _disable_loggers(loggers: list[str]) -> None:
    """
    Disable propagation and set warning level for chosen loggers
    """
    for l in loggers:
        try:
            logging.getLogger(l).propagate = False
            logging.getLogger(l).setLevel(logging.WARNING)
        except AttributeError:
            # no such logger exists
            pass


def init_logger(log_path: str, log_level: str, disable_list: list[str]):
    yaml_config = f"""
      version: 1
      disable_existing_loggers: false
      formatters:
        simple:
          format: '%(asctime)s.%(msecs)03d|%(levelname)-8s|%(name)40s:%(lineno)-4s|%(funcName)-35s|%(message)s'
          datefmt: '%Y.%m.%d-%H:%M:%S'
      handlers:
        stdout_console:
          class: logging.StreamHandler
          level: {log_level}
          formatter: simple
          stream: ext://sys.stdout
        file:
          class: logging.FileHandler
          level: {log_level}
          formatter: simple
          filename: {log_path}
          encoding: utf8
      root:
        level: DEBUG
        handlers:
          - stdout_console
          - file
    """
    dict_config = yaml.load(yaml_config, Loader=yaml.Loader)
    logging.captureWarnings(True)
    _disable_loggers(disable_list)
    logging.config.dictConfig(dict_config)
