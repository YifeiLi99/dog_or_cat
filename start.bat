@echo off
REM 启动 Gradio 猫狗分类 (conda 虚拟环境)

REM 初始化 conda
rem call C:\Users\你的用户名\anaconda3\Scripts\activate.bat
call conda activate python311

REM 切换到项目目录
cd /d D:\lyf\dog_or_cat\

REM 启动 gradio 服务（同步阻塞）
start python gradio_app.py --server_name 127.0.0.1 --server_port 7860

REM 打开浏览器（可选）
start http://127.0.0.1:7860

