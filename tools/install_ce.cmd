@echo off
rem Cross-encoder deps into .venv311; every cache/temp on D:
set PIP_CACHE_DIR=D:\amazon-ml\.cache\pip
set TMP=D:\amazon-ml\tmp
set TEMP=D:\amazon-ml\tmp
set HF_HOME=D:\amazon-ml\.cache\hf
cd /d D:\amazon-ml
echo [%time%] start torch > runs\pip_install.log
.venv311\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126 >> runs\pip_install.log 2>&1
echo [%time%] torch exit %errorlevel% >> runs\pip_install.log
.venv311\Scripts\python.exe -m pip install "transformers>=4.44" >> runs\pip_install.log 2>&1
echo [%time%] transformers exit %errorlevel% >> runs\pip_install.log
.venv311\Scripts\python.exe -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__)" >> runs\pip_install.log 2>&1
.venv311\Scripts\python.exe -c "from huggingface_hub import snapshot_download as s; print(s('cross-encoder/ms-marco-MiniLM-L6-v2'))" >> runs\pip_install.log 2>&1
echo [%time%] INSTALL DONE >> runs\pip_install.log
