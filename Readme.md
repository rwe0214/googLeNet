# googLeNet

The simple comparison between naive inception and inception v2 in googLeNet

## Result

```shell
$ python3 main.py -h
usage: main.py [-h] [--lr LR] [--epoch EPOCH]

PyTorch CIFAR100 Training

optional arguments:
  -h, --help            show this help message and exit
  --lr LR               learning rate
  --epoch EPOCH, -e EPOCH
                        epoch numbers
$ python3 main.py -e 5
==> Preparing data..
Files already downloaded and verified
Files already downloaded and verified
==> Building model..

googLeNet_Naive
Epoch: 0
 [===== 391/391 ======>]  Step: 10s487ms | Tot: 10m57s | Loss: 23.933 | Acc: 9.178% (4589/50000)
 [===== 100/100 ======>]  Step: 284ms | Tot: 28s168ms | Loss: 4.384 | Acc: 13.780% (1378/10000)

Epoch: 1
 [===== 391/391 ======>]  Step: 1s451ms | Tot: 10m54s | Loss: 4.290 | Acc: 16.714% (8357/50000)
 [===== 100/100 ======>]  Step: 610ms | Tot: 1m241ms | Loss: 4.194 | Acc: 19.280% (1928/10000)

Epoch: 2
 [===== 391/391 ======>]  Step: 1s482ms | Tot: 13m56s | Loss: 4.062 | Acc: 23.808% (11904/50000)
 [===== 100/100 ======>]  Step: 606ms | Tot: 1m788ms | Loss: 4.015 | Acc: 25.660% (2566/10000)

Epoch: 3
 [===== 391/391 ======>]  Step: 1s492ms | Tot: 14m218ms | Loss: 3.876 | Acc: 29.492% (14746/50000)
 [===== 100/100 ======>]  Step: 290ms | Tot: 1m215ms | Loss: 3.931 | Acc: 29.510% (2951/10000)

Epoch: 4
 [===== 391/391 ======>]  Step: 1s482ms | Tot: 13m46s | Loss: 3.773 | Acc: 32.678% (16339/50000)
 [===== 100/100 ======>]  Step: 613ms | Tot: 1m833ms | Loss: 3.834 | Acc: 33.190% (3319/10000)

googLeNet_BottleLayer
Epoch: 0
 [===== 391/391 ======>]  Step: 6s39ms | Tot: 5m29s | Loss: 3.793 | Acc: 32.954% (16477/50000)
 [===== 100/100 ======>]  Step: 220ms | Tot: 21s706ms | Loss: 3.687 | Acc: 37.960% (3796/10000)

Epoch: 1
 [===== 391/391 ======>]  Step: 572ms | Tot: 5m21s | Loss: 2.942 | Acc: 56.624% (28312/50000)
 [===== 100/100 ======>]  Step: 209ms | Tot: 21s729ms | Loss: 3.231 | Acc: 53.220% (5322/10000)

Epoch: 2
 [===== 391/391 ======>]  Step: 569ms | Tot: 5m37s | Loss: 2.369 | Acc: 70.448% (35224/50000)
 [===== 100/100 ======>]  Step: 224ms | Tot: 21s698ms | Loss: 2.492 | Acc: 68.860% (6886/10000)

Epoch: 3
 [===== 391/391 ======>]  Step: 571ms | Tot: 5m38s | Loss: 2.006 | Acc: 78.070% (39035/50000)
 [===== 100/100 ======>]  Step: 213ms | Tot: 21s745ms | Loss: 2.052 | Acc: 76.910% (7691/10000)

Epoch: 4
 [===== 391/391 ======>]  Step: 573ms | Tot: 5m25s | Loss: 1.785 | Acc: 82.252% (41126/50000)
 [===== 100/100 ======>]  Step: 102ms | Tot: 11s654ms | Loss: 2.331 | Acc: 72.370% (7237/10000)

Net			Top-5 error (train)	Top-5 error (test)	# of param
googLeNet_Naive		77.626%			75.716%			65736148
googLeNet_BottleLayer	35.930%			38.136%			6258500
```
