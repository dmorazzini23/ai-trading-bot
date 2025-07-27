import logging
import json

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            'ts': self.formatTime(record),
            'level': record.levelname,
            'name': record.name,
            'message': record.getMessage(),
        }
        if hasattr(record, 'bot_phase'):
            log_record['bot_phase'] = record.bot_phase
        return json.dumps(log_record)

def setup_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])
