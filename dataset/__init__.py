from .log import Aiops22Log, PlatformLog, GaiaLog
from .metric import Aiops22Metric, PlatformMetric, GaiaMetric
from .trace import Aiops22Trace, PlatformTrace, GaiaTrace

# add your dataset here
LOG_DATASET = {
    "aiops22": Aiops22Log,
    "platform": PlatformLog,
    "gaia": GaiaLog
}
# add your dataset here
METRIC_DATASET = {
    "aiops22": Aiops22Metric,
    "platform": PlatformMetric,
    "gaia": GaiaMetric
}
# add your dataset here
TRACE_DATASET = {
    "aiops22": Aiops22Trace,
    "platform": PlatformTrace,
    "gaia": GaiaTrace
}
