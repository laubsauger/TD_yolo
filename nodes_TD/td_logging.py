class TDLogger:
    LEVEL_OFF = 0
    LEVEL_ERROR = 1
    LEVEL_WARNING = 2
    LEVEL_INFO = 3
    LEVEL_DEBUG = 4
    
    def __init__(self, name, level=LEVEL_INFO):
        self.name = name
        self.level = level
        
    def set_level(self, level):
        self.level = level
        
    def debug(self, message):
        if self.level >= self.LEVEL_DEBUG:
            print(f"[DEBUG] {message}")
            
    def info(self, message):
        if self.level >= self.LEVEL_INFO:
            print(f"[INFO] {message}")
            
    def warning(self, message):
        if self.level >= self.LEVEL_WARNING:
            print(f"[WARNING] {message}")
            
    def error(self, message):
        if self.level >= self.LEVEL_ERROR:
            print(f"[ERROR] {message}")
            
    def log(self, message):
        if self.level > self.LEVEL_OFF:
            print(message)

def get_logger(script_operator, default_level=TDLogger.LEVEL_INFO):
    """
    Get a logger instance for a TouchDesigner script.
    
    Args:
        script_operator: The TouchDesigner operator (usually 'parent()')
        default_level: Default logging level if no parameter is set
        
    Returns:
        TDLogger instance
    """
    try:
        log_level_param = script_operator.par.Loglevel
        if log_level_param is not None:
            level = int(log_level_param.eval())
        else:
            level = default_level
    except:
        level = default_level
        
    logger = TDLogger(script_operator.name, level)
    return logger