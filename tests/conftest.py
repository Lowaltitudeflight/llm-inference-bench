"""pytest 全局配置。

压测代码依赖 ``config`` / ``bench`` / ``metrics`` 等顶层包，
这里把项目根加进 sys.path，让测试可以直接按包名 import，
不必安装成 site-package。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
