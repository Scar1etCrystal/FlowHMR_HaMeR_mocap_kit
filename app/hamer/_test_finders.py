import mmcv, mmpose, detectron2
print(f"mmcv {mmcv.__version__}")
print(f"mmpose {mmpose.__version__}")
print(f"detectron2 {detectron2.__version__}")

# Verify the dynamic paths resolved correctly
from __editable___mmcv_1_7_2_finder import MAPPING as m1
from __editable___mmpose_0_24_0_finder import MAPPING as m2
from __editable___detectron2_0_6_finder import MAPPING as m3
import os
print(f"\nmmcv path exists: {os.path.exists(list(m1.values())[0])}")
print(f"mmpose path exists: {os.path.exists(list(m2.values())[0])}")
print(f"detectron2 path exists: {os.path.exists(list(m3.values())[0])}")
