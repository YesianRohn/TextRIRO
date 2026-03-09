# Real In, Real Out: Self-Paired Real Data for Realistic Scene Text Editing

## 📢 News




## 📝 TODOs

- [x] Release checkpoints and inference code
- [x] Release tranining pipeline;
- [ ] Provide demo link


## 1 🛠 Installation
### 1.1 Code Preparation 
```bash
# Clone the repo
git clone https://github.com/YesianRohn/TextRIRO.git
cd TextRIRO
# Install required packages
conda create --name textriro python=3.8
conda activate textriro
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirement.txt
```
### 1.2 Checkpoints Preparation
Download the checkpoints from [SD1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) and [SD2-1](https://www.modelscope.cn/models/AI-ModelScope/stable-diffusion-2-1-base) .
## 2 Inference
### 2.1 Data Preparation
The file structure of inference data should be set as the *example/*:  
```bash
TextRIRO/
├── example/
│   ├── i_s/                # source cropped text images
│   └── i_t.txt             # filename and text label of target images
```

### 2.2 Edit Arguments
Edit the arguments in *inference.py*, especially:
```bash
parser.add_argument("--ckpt_path", type=str, default="textriro.pth")
parser.add_argument("--dataset_dir", type=str, default="example/")
parser.add_argument("--output_dir", type=str, default="example_result/")
```

The inference result could be found in *example_result/* after:

### 2.3 Inference Results




## 3 Training
### 3.1 Data Preparation
The training only relies on real data from STR like [Union14M](https://github.com/Mountchicken/Union14M), and we use the [OpenOCR version](https://huggingface.co/datasets/topdu/OpenOCR-Data/tree/main/Union14M-L-LMDB-Filtered).

### 3.1 STR Pretraining
Just train the MAERec model follow this [config](https://github.com/Topdu/OpenOCR/blob/main/configs/rec/maerec/vit_nrtr.yml), and export the [embedding](./model/text_encoder.pth) part.

### 3.2 TextRIRO Training
```bash
cd TextRIRO/
# Modify the path of dir in the config file
cd configs/
vi train.yaml
# Start training
cd ..
python train.py
```

## 4 Evaluation
### 4.1 Data Preparation
Download the ScenePair dataset from [Link](https://drive.google.com/file/d/1m_o2R2kFj_hDXJP5K21aC7lKs-eUky9s/view?usp=sharing) and unzip the files. The structure of each folder is as follows:  
```bash
├── ScenePair/
│   ├── i_s/                # source cropped text images
│   ├── t_f/                # target cropped text images
│   ├── i_full/             # full-size images
│   ├── i_s.txt             # filename and text label of images in i_s/
│   ├── i_t.txt             # filename and text label of images in t_f/
│   ├── i_s_full.txt        # filename, text label, corresponding full-size image name and location information of images in i_s/
│   └── i_t_full.txt        # filename, text label, corresponding full-size image name and location information of images in t_f/
```
### 4.2 Generate Images
Result of some methods on ScenePair dataset are provided here.

### 4.3 Style Fidelity & Text Accuracy
FID, ACC, NED are uesd to evaluate the edited result, with reference to [qqqyd/MOSTEL](https://github.com/qqqyd/MOSTEL).
```bash
cd evaluation/
python evaluation.py --target_path .../result_folder/ --gt_path .../ScenePair/t_f/
python eval_real.py --saved_model models/TPS-ResNet-BiLSTM-Attn.pth --gt_file .../ScenePair/i_t.txt --image_folder .../result_folder/
```

## 5 TextRIRO-3M

Download the generated STR dataset from [HuggingFace]() or [ModelScope]().

## Related Resources

Many thanks to these great projects  [MOSTEL](https://github.com/qqqyd/MOSTEL), [Union14M](https://github.com/Mountchicken/Union14M),  [AnyText](https://github.com/tyxsspa/AnyText), [TextCtrl](https://github.com/weichaozeng/TextCtrl), [OpenOCR](https://github.com/Topdu/OpenOCR), [RS-STE](https://github.com/ZhengyaoFang/RS-STE), [TextSSR](https://github.com/YesianRohn/TextSSR).

## Citation
    TBD
