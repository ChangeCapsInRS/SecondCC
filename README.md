# Robust Change Captioning in Remote Sensing: SECOND-CC Dataset and MModalCC Framework

📢 **This paper is published in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (JSTARS), 2025.**  
🔗 [IEEE Xplore Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11130644)  
📄 DOI: [10.1109/JSTARS.2025.3600613](https://doi.org/10.1109/JSTARS.2025.3600613)
[MOSAIC Research Group Website:](https://avesis.yildiz.edu.tr/arastirma-grubu/mosaic)

## 🔎 Summary
Existing remote sensing change captioning (RSICC) methods struggle under illumination differences, viewpoint changes, blur, resolution mismatch, and registration errors, often leading to inaccurate captions in no-change regions. To overcome these challenges, we introduce **SECOND-CC**, a new dataset with **6,041 bitemporal image pairs** and **30,205 human-annotated sentences**, enriched with semantic segmentation maps and diverse real-world scenarios. We further propose **MModalCC**, a multimodal framework that fuses semantic and visual data via **Cross-Modal Cross Attention** and **Multimodal Gated Cross Attention**

---

<p align="center">
  <img src="fig/blockDiag.jpg" alt="MModalCC Framework" width="850"/>
</p>

[![Visitors](https://visitor-badge.laobi.icu/badge?page_id=MOSAIC-Lab/MModalCC)]()

---

⭐ **Share us a star if this repo helps your research!**  

🔥 Our new work on **change captioning and multimodal reasoning** is continuously updated here. Stay tuned! 🔥

---

## 📘 SECOND-CC Dataset
We introduce **SECOND-CC**, a large-scale remote sensing change captioning dataset.

- [Download Link (Google Drive)](https://drive.google.com/...)  
- Train / Val / Test splits provided  
- Includes paired images + change captions  

Example pair:  

<p align="center">
  <img src="fig/dataset.jpg" alt="SECOND-CC Dataset" width="850"/>
</p>

---

## ⚙️ Installation and Dependencies
```bash
git clone https://github.com/MOSAIC-Lab/MModalCC.git
cd MModalCC
conda create -n mmodalcc_env python=3.6
conda activate mmodalcc_env
pip install -r requirements.txt
```

---

## 📂 Data Preparation

### 1. Download the Dataset
Download **SECOND-CC** dataset from Google Drive:

- [SECOND-CC Dataset (Google Drive)](https://drive.google.com/...)

Place the dataset under:
./SECOND_CC_dataset/

The directory structure should look like:
```
SECOND_CC_dataset/
├── train/
├── val/
├── test/
└── SECOND-CC-AUG.json
```
---

### 2. Preprocess Dataset and Generate HDF5 Files
To convert raw data into HDF5 feature files, run:
```bash
python tools/create_input_files.py --min_word_freq 5 
```

After preprocessing, you will find the generated files in:
./createdFileBlackAUG/

3. Use Preprocessed HDF5 Files (Optional)
We also provide ready-to-use .h5 files:

SECOND-CC HDF5 Files (Google Drive)
Download and extract them into:

./createdFileBlackAUG/



