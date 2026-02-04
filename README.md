# PMFNet for RGB-T Tracking

## Environment Installation
```
conda create -n tbsi python=3.8
conda activate tbsi
bash install.sh
```

## Project Paths Setup
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Data Preparation
Put the tracking datasets in `./data`. It should look like:
```
${PROJECT_ROOT}
  -- data
      -- lasher
          |-- trainingset
          |-- testingset
          |-- trainingsetList.txt
          |-- testingsetList.txt
          ...
```

## Training
Download [ImageNet or SOT](https://pan.baidu.com/s/1U42J6b3g1htma0OvmXRQCw?pwd=at5b) pretrained weights and put them under `$PROJECT_ROOT$/pretrained_models`.

```
python tracking/train.py --script tbsi_track --config vitb_256_tbsi_32x4_4e4_lasher_p061_in1k --save_dir ./output/vitb_256_tbsi_32x4_4e4_lasher_p061_in1k --mode multiple --nproc_per_node 4
```

Replace `--config` with the desired model config under `experiments/tbsi_track`.

## Evaluation
Put the checkpoint into `$PROJECT_ROOT$/output/config_name/...` or modify the checkpoint path in testing code.

```
python tracking/test.py tbsi_track vitb_256_tbsi_32x4_4e4_lasher_p061_in1k --dataset_name lasher_test --threads 6 --num_gpus 1

python tracking/analysis_results.py --tracker_name tbsi_track --tracker_param vitb_256_tbsi_32x4_4e4_lasher_p061_in1k --dataset_name lasher_test
```

### Results on LasHeR testing set

Model | Backbone | Pretraining | Precision | NormPrec | Success | FPS | Checkpoint | Raw Result
--- |:---:|:---:|:---------:|:---:|:---:|:---:|---:|---:
TBSI | ViT-Base | ImageNet |   64.3    | 60.8 | 51.0 | 36.2 | [download](https://pan.baidu.com/s/18MYRT4jkunIPklD02daFXA?pwd=y2rz) | [download](https://pan.baidu.com/s/1CP07T0VmtxPr6KcWqszY1w?pwd=6v3b)
TBSI | ViT-Base | SOT |   70.5    | 66.5 | 56.5 | 36.2 | [download](https://pan.baidu.com/s/18MYRT4jkunIPklD02daFXA?pwd=y2rz) | [download](https://pan.baidu.com/s/1CP07T0VmtxPr6KcWqszY1w?pwd=6v3b)

## Acknowledgments
Our project is developed upon [OSTrack](https://github.com/botaoye/OSTrack). Thanks for their contributions which help us to quickly implement our ideas.

# PMFNet: 跨模态RGB-T目标跟踪框架
This code is the official implementation of the paper submitted to **The Visual Computer** (《视觉计算机》) titled "PMFNet: Collaborative Prompt Enhancement and Dynamic Fusion for Robust RGB-Thermal Tracking".

## 论文信息
- Journal: The Visual Computer
