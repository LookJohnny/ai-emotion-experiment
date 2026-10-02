@echo off
chcp 65001 >nul
rem Nous 读心 · Windows 启动（NVIDIA 显卡显存 >= 10 GB；没有显卡会很慢）
cd /d "%~dp0"
set "D=%USERPROFILE%\.nous-lens"
set "VENV=%D%\venv"
set "MODEL=%D%\models\Qwen3-4B-Instruct-2507"
set "MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple"
if not exist "%D%" mkdir "%D%"
if exist "%VENV%\Scripts\python.exe" goto model
echo 首次运行：正在准备运行环境（约 5-10 分钟）...
py -3 -m venv "%VENV%" 2>nul || python -m venv "%VENV%"
if not exist "%VENV%\Scripts\python.exe" (echo 需要 Python 3.10 或更新版本：https://www.python.org/downloads/ & pause & exit /b 1)
nvidia-smi >nul 2>&1 && ("%VENV%\Scripts\pip" install torch --index-url https://download.pytorch.org/whl/cu128 || "%VENV%\Scripts\pip" install torch -f https://mirrors.aliyun.com/pytorch-wheels/cu128/ --no-index)
"%VENV%\Scripts\pip" install -i %MIRROR% -r requirements.txt || "%VENV%\Scripts\pip" install -r requirements.txt || (rmdir /s /q "%VENV%" & echo 安装失败，请检查网络后重试 & pause & exit /b 1)
:model
if exist "%MODEL%\.complete" goto run
echo 首次运行：正在下载本地模型 Qwen3-4B（约 8 GB，只需一次）...
"%VENV%\Scripts\modelscope" download --model Qwen/Qwen3-4B-Instruct-2507 --local_dir "%MODEL%" || (echo 模型下载失败，请重试 & pause & exit /b 1)
type nul > "%MODEL%\.complete"
:run
"%VENV%\Scripts\python" -m nous_lens.server --model "%MODEL%"
pause
