LB_Webb golden: authors' Java reference (https://github.com/GIWebb/DTWBounds, GPL-3; used only to
generate numbers, no code copied). Build:
  git clone --depth 1 https://github.com/GIWebb/DTWBounds
  javac -sourcepath DTWBounds/src -d out DTWBounds/src/distance/Bounds.java DTWBounds/src/distance/elastic/DTW.java golden_lbwebb_Gold.java
  java -cp out Gold   # prints (w, q, t, lbKeogh, lbWebb, DTW) tuples -> WEBB_GOLDEN (w >= 1 rows)
