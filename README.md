Dataset Training andTesting:
<1>sysu:
Train:
nohup python train.py > train.log 2>&1 &
Test:
python test.py --mode indoor --resume 'model_path' --gpu 0 --dataset sysu
python test.py --mode all --resume 'model_path' --gpu 0 --dataset sysu
<2>regdb:
Train:
nohup bash train_regdb.bash > train.log 2>&1 &
Test:
python test.py  --tvsearch True --resume 'model_path' --gpu 0 --dataset regdb
<3>llcm：
Train:
nohup python train.py --dataset llcm --gpu 0 > train.log 2>&1 &
Test:
python test.py --dataset llcm --resume 'model_path' --gpu 0


Our CAFMNet is carried out based on DEEN, We are very grateful for the contribution of the author of DEEN:

@InProceedings{Zhang_2023_CVPR,
    author    = {Zhang, Yukang and Wang, Hanzi},
    title     = {Diverse Embedding Expansion Network and Low-Light Cross-Modality Benchmark for Visible-Infrared Person Re-Identification},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {2153-2162}
}


The final experimental results are as follows:
SYSU-MM01:           All-search            Indoor-search
                                 R=1      mAP            R=1     mAP            
CAFMNet(ours)   77.49     74.19         84.95    87.09

LLCM:
                                   IR-to-VIS                   VIS-to-IR 
                                 R=1      mAP             R=1     mAP            
CAFMNet(ours)    57.58     64.10          69.64   57.16