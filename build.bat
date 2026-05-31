@echo off
python -m nuitka ^
  --zig ^
  --assume-yes-for-downloads ^
  --standalone ^
  --enable-plugin=tk-inter ^
  --include-package=tkinterdnd2 ^
  --include-package-data=tkinterdnd2 ^
  --windows-console-mode=disable ^
  --include-data-dir=assets=assets ^
  --include-data-dir=locales=locales ^
  --windows-icon-from-ico=assets\app_icon.ico ^
  --output-filename=VSM_Analyzer ^
  main.py
