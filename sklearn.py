# Mock for sklearn
from types import ModuleType

# Create nested module structure
metrics = ModuleType('metrics')
metrics.accuracy_score = lambda *args, **kwargs: 0.5
metrics.classification_report = lambda *args, **kwargs: ""

model_selection = ModuleType('model_selection')
model_selection.train_test_split = lambda *args, **kwargs: ([], [], [], [])

ensemble = ModuleType('ensemble')
ensemble.RandomForestClassifier = lambda *args, **kwargs: type('Mock', (), {'fit': lambda *a: None, 'predict': lambda *a: []})()

preprocessing = ModuleType('preprocessing')
preprocessing.StandardScaler = lambda *args, **kwargs: type('Mock', (), {'fit_transform': lambda *a: []})()

# Add to globals
globals()['metrics'] = metrics
globals()['model_selection'] = model_selection
globals()['ensemble'] = ensemble
globals()['preprocessing'] = preprocessing

def __getattr__(name):
    return lambda *args, **kwargs: None
