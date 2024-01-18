
# LCF3D

 The source code of our work is based on **Cylinder3D** from "Cylindrical and Asymmetrical 3D Convolution Networks for LiDAR Segmentation"

## Installation

### Requirements
- PyTorch >= 1.2 
- yaml
- Cython
- [torch-scatter](https://github.com/rusty1s/pytorch_scatter)
- [nuScenes-devkit](https://github.com/nutonomy/nuscenes-devkit) (optional for nuScenes)
- [spconv](https://github.com/traveller59/spconv) (tested with spconv==1.2.1 and cuda==10.2)

## Data Preparation

### SemanticKITTI
```
./
├── 
├── ...
└── path_to_data_shown_in_config/
    ├──sequences
        ├── 00/           
        │   ├── velodyne/	
        |   |	├── 000000.bin
        |   |	├── 000001.bin
        |   |	└── ...
        │   └── labels/ 
        |       ├── 000000.label
        |       ├── 000001.label
        |       └── ...
        ├── 08/ # for validation
        ├── 11/ # 11-21 for testing
        └── 21/
	    └── ...
```

### nuScenes
```
./
├── 
├── ...
└── path_to_data_shown_in_config/
		├──v1.0-trainval
		├──v1.0-test
		├──samples
		├──sweeps
		├──maps

```

## Training
1. modify the config/semantickitti.yaml with your custom settings. We provide a sample yaml for SemanticKITTI
2. train the network by running "sh train.sh"

### Training for nuScenes
Please refer to [NUSCENES-GUIDE](./NUSCENES-GUIDE.md)

### Pretrained Models
-- We provide a pretrained model for SemanticKITTI [LINK1](https://drive.google.com/file/d/1q4u3LlQXz89LqYW3orXL5oTs_4R2eS8P/view?usp=sharing) or [LINK2](https://pan.baidu.com/s/1c0oIL2QTTcjCo9ZEtvOIvA) (access code: xqmi)

-- For nuScenes dataset, please refer to [NUSCENES-GUIDE](./NUSCENES-GUIDE.md)

## Semantic segmentation demo for a folder of lidar scans
```
python demo_folder.py --demo-folder YOUR_FOLDER --save-folder YOUR_SAVE_FOLDER
```
If you want to validate with your own datasets, you need to provide labels.
--demo-label-folder is optional
```
python demo_folder.py --demo-folder YOUR_FOLDER --save-folder YOUR_SAVE_FOLDER --demo-label-folder YOUR_LABEL_FOLDER
```
```

## Acknowledgments
We thanks for the opensource codebases, [PolarSeg](https://github.com/edwardzhou130/PolarSeg) and [spconv](https://github.com/traveller59/spconv)
