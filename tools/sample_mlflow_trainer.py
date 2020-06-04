import logging
import os
from collections import OrderedDict
import torch
import json
import cv2
import numpy as np
import mlflow
from pathlib import Path
import argparse
import shutil

from detectron2 import model_zoo
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch, mlflow_hooks
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, verify_results
from detectron2.modeling import GeneralizedRCNNWithTTA
from tqdm import tqdm


def get_datasets(ann_root_dir, img_root_dir):
    label_list = []
    
    def get_and_regiser_dataset(ann_root_dir, img_root_dir, mode):
        dataset_list = []
        print("load dataset : ", mode)
        tar_dir = Path(ann_root_dir).glob("{}/*.json".format(mode))
        for p1 in tqdm(list(tar_dir)):
            img_dir = os.path.join(img_root_dir, Path(p1).stem)
            register_coco_instances(
                Path(p1).stem,
                {},
                p1,
                img_dir
            )
            dataset_list.append(Path(p1).stem)
        return dataset_list
    
    train_list =  get_and_regiser_dataset(ann_root_dir, img_root_dir, "train")
    test_list =  get_and_regiser_dataset(ann_root_dir, img_root_dir, "val")
        
    return train_list, test_list

class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop. You can use
    "tools/plain_train_net.py" as an example.
    """
    def __init__(self, cfg, dataset_name):
        self.dataset_name = dataset_name
        super().__init__(cfg)
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        
        evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    def build_hooks(self):
        """
        override.
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN
                
        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            mlflow_hooks.MLFlowTrainHook(cfg, self.dataset_name),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results
        
        ret.append(mlflow_hooks.MLFlowEvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
            
        return ret
    
def main(args):
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.mlflow_name)
    
    # get dataset path
    train_list, test_list = get_datasets(args.annotation_dir, args.image_dir)

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.DATASETS.TRAIN = tuple(x for x in train_list)
    cfg.DATASETS.TEST = tuple(x for x in test_list)
        
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    dataset_name = Path(args.annotation_dir).parent.stem
        
    trainer = Trainer(cfg, dataset_name) 
    trainer.resume_or_load(resume=False)
    trainer.train()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-a", "--annotation_dir", required=True)
    parser.add_argument("-i", "--image_dir", required=True)
    parser.add_argument("-c", "--config_file", required=True)
    parser.add_argument("--mlflow_uri")
    parser.add_argument("--mlflow_name")
    args = parser.parse_args()

    main(args)