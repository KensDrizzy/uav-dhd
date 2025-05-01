# ------------------------------------------------------------------------
# DHD
# Copyright (c) 2024 Zhechao Wang. All Rights Reserved.
# ------------------------------------------------------------------------

import os
import socket
import time

import pydevd_pycharm
import pytorch_lightning as pl
import torch
from dhd.config import get_cfg, get_parser
from dhd.data import prepare_powerbev_dataloaders
from dhd.trainer import TrainingModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin



# os.environ['MASTER_HOST'] = 'localhost:12345'
#
# # --- 添加以下代码用于远程调试 ---
# # 获取当前进程的 rank
# rank = 0
# if torch.distributed.is_initialized():
#     rank = torch.distributed.get_rank()
#
# # 设置 PyCharm Debug Server 的主机和端口
# # 确保这里的IP是你的本地计算机IP，端口是你PyCharm Debug Server配置的端口
# debug_host = '10.81.129.59' # <-- 替换成你的本地IP
# debug_port = 12345 # <-- 替换成你步骤2设置的端口，例如 12345
#
# # 可选：只调试 rank 0 进程
# if rank == 0:
#     try:
#         print(f"Rank {rank}: Connecting to PyCharm Debug Server at {debug_host}:{debug_port}")
#
#         pydevd_pycharm.settrace(debug_host, port=debug_port, stdoutToServer=True, stderrToServer=True)
#
#         print(f"Rank {rank}: Connected to PyCharm Debug Server.")
#     except ConnectionRefusedError:
#         print(f"Rank {rank}: Connection to PyCharm Debug Server refused. Make sure the server is running in PyCharm.")
#     except Exception as e:
#         print(f"Rank {rank}: An error occurred during debugger connection: {e}")
# # --- 远程调试代码结束 ---

def main():
    args = get_parser().parse_args()
    cfg = get_cfg(args)

    trainloader, valloader = prepare_powerbev_dataloaders(cfg)
    model = TrainingModule(cfg.convert_to_dict())

    if cfg.PRETRAINED.LOAD_WEIGHTS:
        # Load single-image instance segmentation model.
        pretrained_model_weights = torch.load(
            cfg.PRETRAINED.PATH , map_location='cpu'
        )['state_dict']

        model.load_state_dict(pretrained_model_weights, strict=False)
        print(f'Loaded single-image model weights from {cfg.PRETRAINED.PATH}')

    save_dir = os.path.join(
        cfg.LOG_DIR, time.strftime('%d%B%Yat%H:%M:%S%Z') + '_' + socket.gethostname() + '_' + cfg.TAG
    ) 
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=save_dir)
    checkpoint_callback = ModelCheckpoint(monitor='vpq', save_top_k=20, mode='max')
    trainer = pl.Trainer(
        gpus=cfg.GPUS,
        accelerator='ddp',
        precision=cfg.PRECISION,
        # precision=32,
        sync_batchnorm=True,
        gradient_clip_val=cfg.GRAD_NORM_CLIP,
        max_epochs=cfg.EPOCHS,
        weights_summary='full',
        logger=tb_logger,
        log_every_n_steps=cfg.LOGGING_INTERVAL,
        plugins=DDPPlugin(find_unused_parameters=True),
        #fast_dev_run=True,
        profiler='simple',
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, trainloader, valloader)
    print("111")

if __name__ == "__main__":
    main()