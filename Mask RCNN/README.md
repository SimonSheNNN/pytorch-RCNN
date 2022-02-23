
Train + Eval 
```bash
main.ipynb
```
Upload Dataset.py, RPN.py, pretrained_models.py, util.py, and backbone.py. To train, run all cells in the "model setup" section. To use trained model, load "modelsimon2.pth"
All models including MaskHead, BoxHead, MaskRCNN are all in the ipynb file. 

All visualization
```bash
main.ipynb
```
To show visualizations, run section "postprocess".
run subsection "box map" for bounding box mAP and pr curve, run subsection "mask map" for mask mAP and pr curve

Dataset:
```bash
python Dataset.py
```

All util functions:
```bash
python util.py
```

Trained model :
```bash
modelsimon2.pth
```
