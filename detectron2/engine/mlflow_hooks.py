import os
import shutil
from pathlib import Path
from statistics import mean

from detectron2.utils.events import EventStorage, EventWriter, get_event_storage
from .train_loop import HookBase
from .hooks import EvalHook

import mlflow
from mlflow import log_metric, log_param, log_artifact


class MLFlowTrainHook(HookBase):
    """[summary]

    Arguments:
        HookBase {[type]} -- [description]
    """
    def __init__(self, cfg, dataset_name, interval=100):
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.interval = interval

    def before_train(self):
        try:
            log_param("dataset_name", self.dataset_name)
            log_param("gpu_id", self.cfg.MODEL.DEVICE)
            log_param("pre-training", self.cfg.MODEL.WEIGHTS)
            log_param("num_iter", self.cfg.SOLVER.MAX_ITER)
        except KeyError as e:
            print("catch KeyError: ", e)   

    def after_step(self):
        storage = get_event_storage()
        iter_ = self.trainer.iter
        
        try:
            if iter_ % self.interval == 0 and iter_ != 0:
                log_metric("lr", storage.history("lr").avg(self.interval), step=iter_)
                log_metric("total_loss", storage.history("total_loss").avg(self.interval), step=iter_)
                log_metric("loss_cls", storage.history("loss_cls").avg(self.interval), step=iter_)
                log_metric("loss_box_reg", storage.history("loss_box_reg").avg(self.interval), step=iter_)
                log_metric("loss_rpn_cls", storage.history("loss_rpn_cls").avg(self.interval), step=iter_)
                log_metric("loss_rpn_loc", storage.history("loss_rpn_loc").avg(self.interval), step=iter_)
                log_metric("cls_accuracy", storage.history("fast_rcnn/cls_accuracy").avg(self.interval), step=iter_)
                log_metric("fg_cls_accuracy", storage.history("fast_rcnn/fg_cls_accuracy").avg(self.interval), step=iter_)
                log_metric("false_negative", storage.history("fast_rcnn/false_negative").avg(self.interval), step=iter_)
            
        except KeyError as e:
            print("catch KeyError: ", e)

    def after_train(self):
        make_artifact(self.cfg)
        mlflow.log_artifact(os.path.join(self.cfg.OUTPUT_DIR, "model"))
    
class MLFlowEvalHook(EvalHook):
    """[summary]

    Arguments:
        EvalHook {[type]} -- [description]
    """
    def __init__(self, eval_period, eval_function):
        super().__init__(eval_period, eval_function)
        # 必要に応じて修正する
        self.tracking_set = [
            "bbox/AP",
            "bbox/AP50",
            "bbox/AP75",
            "bbox/APs",
            "bbox/APm",
            "bbox/APl"
        ]
    
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_eval()

            storage = get_event_storage()
            iter_ = self.trainer.iter
                            
            try:
                result = avg_multi_set(storage, self.tracking_set)
                for k in result:
                    log_metric(k, result[k], step=iter_)
                
            except KeyError as e:
                print("catch KeyError: ", e)

def avg_multi_set(storage, key_list):
    tmp_result = {k : [] for k in key_list}
    avg_result = {}
    for k, v in storage.histories().items():
        key = os.path.join(Path(k).parent.stem, Path(k).stem)
        if key in key_list:
            tmp_result[key].append(storage.history(k).latest())
    
    for k in tmp_result:
        key = "eval_" + k
        avg_result[key] = mean(tmp_result[k])
        
    return avg_result

def make_artifact(cfg):
    output = os.path.join(cfg.OUTPUT_DIR, "model")
    os.makedirs(output, exist_ok=True)
    
    def file_move(name):
        src_ = os.path.join(cfg.OUTPUT_DIR, name)
        if os.path.isfile(src_):
            tar_ = os.path.join(output, name)
            shutil.copy2(src_, tar_)
    
    # weight    
    model_name = "model_final.pth"
    file_move(model_name)

    # config
    cfg_file = os.path.join(output, "config.yaml")
    with open(cfg_file, mode='w') as f:
        f.write(cfg.dump())
    
    # metric
    metric_name = "metrics.json"
    file_move(metric_name)
    
    # checkpoint
    ckpt_name = "last_checkpoint"
    file_move(ckpt_name)
    
    # inference
    inf_dir = os.path.join(cfg.OUTPUT_DIR, "inference")
    if os.path.isdir(inf_dir):
        shutil.copytree(inf_dir, os.path.join(output, "inference"))