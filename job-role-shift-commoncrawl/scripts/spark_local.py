"""로컬 SparkSession. JAVA_HOME은 환경변수로 (Java 17)"""

import os
import sys

from common import WORK_DIR
from pyspark.sql import SparkSession

# 2코어를 남긴다
DEFAULT_CORES = os.environ.get("SPARK_CORES", str(max(2, (os.cpu_count() or 4) - 2)))


def get_spark(app: str, driver_memory: str = "10g", cores: str = DEFAULT_CORES) -> SparkSession:
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)  # venv를 켜지 않아도 작업자가 같은 Python을 쓰게
    return (
        SparkSession.builder.appName(app)
        .master(f"local[{cores}]")
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.local.dir", str(WORK_DIR / "spark_tmp"))
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
