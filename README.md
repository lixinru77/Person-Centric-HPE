# Person-Centric Spatial Normalization for mmWave Radar Human Pose Estimation

This repository contains the official implementation of the paper: **Person-Centric Spatial Normalization for mmWave Radar Human Pose Estimation**.

## 🛠 Preparation

Please install the required environment using Conda:
```
conda env create -f vir_env.yml
```

Datasets We use both a public benchmark and a custom validation dataset: [Public Benchmark Dataset: STC-HSANet GitHub](https://github.com/zylofor/STC-HSANet)  
Our Custom Validation Dataset (Different Scenes): [Baidu Netdisk](https://pan.baidu.com/s/1EpnGoZXsnxbKQx7aMzEovw) 

1. 4D FFT Processing:
First, process the raw .npy radar data using 4D FFT:  

```
python process_iwr1843
```

2. Visual Ground Truth (GT) Generation:
Next, generate the visual labels (GT):
```
python video2frame.py
```

4. Dataset Structure Alignment:
Ensure your dataset directory is organized as follows (refer to the Pictures folder for visual references):  
```
Person-Centric/data/
├── hrnet_annot_test.json
├── hrnet_annot_val.json
├── hrnet_annot_train.json
├── rrv_paors_1/
│   ├── hori/
│   │   ├── 000000000.npy[cite: 1]
│   │   └── ....
│   ├── vert/[cite: 1]
│   │   ├── 000000000.npy[cite: 1]
│   │   └── ....
│   ├── raw_radar/[cite: 1]
│   ├── frame/[cite: 1]
│   ├── gt_labels/[cite: 1]
│   ├── gt_video/[cite: 1]
│   └── visualization/[cite: 1]
├── rrv_paors_2/[cite: 1]
└── ...
```

🚀 Training 
To train the model, run the following command[cite: 1]:
```
python main.py --config config.yaml --dir [Your_Output_Directory_Path]
```

(Please replace [Your_Output_Directory_Path] with your actual output path[cite: 1].)


📈 Evaluation 
To evaluate the trained model, use the following command[cite: 1]:
```
python main.py --dir output --config config.yaml --eval --visDir False --keypoints True
```

Evaluation Notes[cite: 1]:

Visualization: If you need to visualize the results, set --visDir True[cite: 1].

Keypoints Output: If you do not need the output for every single keypoint, set --keypoints False[cite: 1].

Pre-trained Weights: The best model weights are located at ./Person-Centric-HPE/output/model_best.pth[cite: 1].
