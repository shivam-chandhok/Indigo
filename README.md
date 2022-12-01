# INDIGO: Intrinsic Multimodality for Domain Generalization (ECCV'22- OOD-CV Workshop)

Puneet Mangla*,
Shivam Chandhok*,
Milan Aggarwal,
Vineeth N Balasubramanian,
Balaji Krishnamurthy (*Equal Contribution)

**[Paper Link](https://arxiv.org/abs/2206.05912)** 


> **Abstract:**
>*For models to generalize under unseen domains (a.k.a do- main generalization), it is crucial to learn feature represen- tations that are domain-agnostic and capture the underlying semantics that makes up an object. Recent advances towards weakly supervised vision-language models (such as CLIP) have shown their ability on object understanding by captur- ing semantic characteristics that generalize under different domains. Hence, it becomes important to investigate how the semantic knowledge present in their representations can be effectively incorporated and utilized for domain generaliza- tion. Motivated from this, we study how semantic informa- tion from existing pre-trained multimodal networks can be leveraged in an ”intrinsic” way to make systems generalize under unseen domains. We propose IntriNsic multimodality for DomaIn GeneralizatiOn (INDIGO), a simple and elegant framework that leverages the intrinsic modality present in pre-trained multimodal networks to enhance generalization to unseen domains at test-time. We experiment on several Do- main Generalization settings (ClosedDG, OpenDG, and Lim- ited sources) and show state-of-the-art generalization perfor- mance on unseen domains. Further, we provide a thorough analysis to develop a holistic understanding of INDIGO.*


<p align="center">
  <img alt="intro_image" src="./main_fig.png" width="1150"/>
</p>



## Usage & Data
Refer to `requirements.txt` for installing all python dependencies.
Install CLIP
```pip install ./CLIP```


### Arguments: 
```--target <Target Domain>```
```--name <Name of experiment>```
```--model <Backbone model>```
```--teacher <Teacher model >```

### Class Token Distillation Training Approach
```CUDA_VISIBLE_DEVICES=2,3 python main.py --dg --target clipart  --config_file configs/zsl+dg/clipart.json --dataset domainnet   --name test  --runs 5 --method class_token_distill --model vit_small_hybrid --teacher clip_vit_b ```

### DeiT Distillation Training Approach
Load teacher model using main.py and then run
```CUDA_VISIBLE_DEVICES=2,3 python main.py --dg --target clipart  --config_file configs/zsl+dg/clipart.json --dataset domainnet   --name test  --runs 5 --method distill --model vit_small_hybrid ```

### Standard Training Approach
```CUDA_VISIBLE_DEVICES=2,3 python main.py --dg --target clipart  --config_file configs/zsl+dg/clipart.json --dataset domainnet   --name test  --runs 5 --method standard --model vit_small_hybrid```
## Results
Results of Class-agnostic Object Detection of MViTS including our proposed Multiscale Attention ViT with Late fusion
(MAVL) model, applications, and exploratory analysis.

<strong>Class-agnostic Object Detection</strong> performance of MViTs in comparison with bottom-up approaches and uni-modal detectors on five natural image OD datasets. MViTs show consistently good results on all datasets.

![Results](./tab1.png)

<hr />

<strong>Generalization to New Domains</strong>: Class-agnostic OD performance of MViTs in comparison with uni-modal detector(RetinaNet) on five out-of-domain OD datasets. MViTs show consistently good results on all datasets.

![Results](./attn.png)

<hr />

<strong> Generalization to Rare/Novel Classes</strong>: MAVL class-agnostic OD performance on rarely and frequently occurring categories in the pretraining captions.
The numbers on top of the bars indicate occurrences of the corresponding category in the training dataset.
The MViT achieves good recall values even for the classes with no or very few occurrences.

![Results](./tab3.png)

<hr />

<strong> Enhanced Interactability</strong>: Effect of using different <strong>intuitive text queries</strong> on the MAVL class-agnostic OD performance.
Combining detections from multiple queries captures varying aspects of objectness.

![Results](./tab2.png)

<hr />

<strong> Language Skeleton/Structure</strong>: Experimental analysis to explore the contribution of language by removing all textual inputs, but maintaining the structure introduced by captions. 
All experiments are performed on Def-DETR. 
In setting 1, annotations corresponding to same images are combined. 
Setting 2 has an additional NMS applied to remove duplicate boxes. 
In setting 3, four to eight boxes are randomly grouped in each iteration. 
The same model is trained longer in setting 4. 
In setting 5, the dataloader structure corresponding to captions is kept intact. 
Results from setting 5 demonstrate the importance of structure introduced by language.

![Results](./ablation.png)

<hr />

<strong> Open-world Object Detection</strong>: Effect of using class-agnostic OD proposals from MAVL for pseudo labelling of unknowns in Open World Detector (ORE).

![Results](./ablation2.png)

<hr />







## Citation

```bibtex
@inproceedings{indigo,
    title={INDIGO: Intrinsic Multimodality for Domain Generalization}, 
    author={P Mangla and S Chandhok and M Aggarwal and VN Balasubramanian and B Krishnamurthy},
    booktitle={ECCV'22- OOD-CV Workshop},
    month = {June},
    year={2022}
  }
```


## Acknowledgements
Our code is based on [DINO](https://github.com/facebookresearch/dino) and [TimeSformer](https://github.com/facebookresearch/TimeSformer) repositories. We thank the authors for releasing their code. If you use our model, please consider citing these works as well.


