"""
=============================================================================
📝 نظام Logging متقدم
=============================================================================
"""

import logging
import logging.handlers
import logging.config
import json
from datetime import datetime
import os

class AppLogger:
    """نظام Logging احترافي"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # إنشاء مجلد السجلات
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        # تهيئة نظام Logging
        self._setup_logging(logs_dir)
        
        # إعدادات خاصة
        self.operation_stack = []
        self.performance_records = {}
        
        self._initialized = True
    
    def _setup_logging(self, logs_dir):
        """إعداد نظام Logging"""
        config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'detailed': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                },
                'simple': {
                    'format': '%(levelname)s: %(message)s'
                }
            },
            'handlers': {
                'file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': os.path.join(logs_dir, 'app.log'),
                    'maxBytes': 10485760,  # 10MB
                    'backupCount': 5,
                    'formatter': 'detailed',
                    'level': 'INFO'
                },
                'console': {
                    'class': 'logging.StreamHandler',
                    'formatter': 'simple',
                    'level': 'WARNING'
                }
            },
            'loggers': {
                'lottery': {
                    'handlers': ['file', 'console'],
                    'level': 'INFO',
                    'propagate': True
                }
            }
        }
        
        logging.config.dictConfig(config)
        self.logger = logging.getLogger('lottery')
        self.logger.info("🚀 تم تهيئة نظام Logging بنجاح")
    
    def start_operation(self, operation_name: str, metadata: dict = None):
        """بدء عملية جديدة مع تتبع"""
        operation_id = f"{operation_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.operation_stack.append({
            'id': operation_id,
            'name': operation_name,
            'start_time': datetime.now(),
            'metadata': metadata or {}
        })
        
        self.logger.info(f"🔧 بدء العملية: {operation_name}", extra={
            'operation_id': operation_id,
            'metadata': metadata
        })
        
        return operation_id
    
    def end_operation(self, operation_id: str, status: str = "completed", metrics: dict = None):
        """إنهاء عملية مع التسجيل"""
        for op in reversed(self.operation_stack):
            if op['id'] == operation_id:
                duration = (datetime.now() - op['start_time']).total_seconds()
                
                log_data = {
                    'operation_id': operation_id,
                    'operation_name': op['name'],
                    'duration_seconds': round(duration, 3),
                    'status': status,
                    'metrics': metrics or {},
                    'metadata': op['metadata']
                }
                
                if status == "completed":
                    self.logger.info(f"✅ اكتملت العملية: {op['name']} ({duration:.2f} ثانية)", extra=log_data)
                elif status == "failed":
                    self.logger.error(f"❌ فشلت العملية: {op['name']}", extra=log_data)
                else:
                    self.logger.warning(f"⚠️ حالة غير معروفة: {op['name']}", extra=log_data)
                
                # حفظ في سجل الأداء
                self.performance_records[operation_id] = log_data
                
                # إزالة من المكدس
                self.operation_stack.remove(op)
                break

# Singleton instance
logger = AppLogger()