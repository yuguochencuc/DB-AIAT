### DB-AIAT: A Dual-branch attention-in-attention transformer for single-channel SE (https://arxiv.org/abs/2110.06467)
This is the repo of the manuscript "Dual-branch Attention-In-Attention Transformer for speech enhancement", which is accepted by ICASSP2022.


Abstract：Curriculum learning begins to thrive in the speech enhancement area, which decouples the original spectrum estimation task into multiple easier sub-tasks to achieve better performance. Motivated by that, we propose a dual-branch attention-in-attention transformer-based module dubbed DB-AIAT to handle both coarse- and fine-grained regions of spectrum in parallel. From a complementary perspective, a magnitude masking branch is proposed to estimate the overall spectral magnitude, while a complex refining branch is designed to compensate for the missing complex spectral details and implicitly derive phase information. Within each branch, we propose a novel attention-in-attention transformer-based module to replace the conventional RNNs and temporal convolutional network for temporal sequence modeling. Specifically, the proposed attention-in-attention transformer consists of adaptive temporal-frequency attention transformer blocks and an adaptive hierarchical attention module, which can capture long-term time-frequency dependencies and further aggregate global hierarchical contextual information. The experimental results on VoiceBank + Demand dataset show that DB-AIAT yields state-of-the-art performance (e.g., 3.31 PESQ, 95.6% STOI and 10.79dB SSNR) over previous advanced systems with a relatively light model size (2.81M).

### Code:
	The code for network architecture is privoided now, while the trainer is released soon. You can use dual_aia_trans_merge_crm() in aia_trans.py for dual-branch SE, while aia_complex_trans_mag() and aia_complex_trans_ri() are single-branch aprroaches.
	The trained weights on VB dataset is also provided. You can directly perform inference or finetune the model by using vb_aia_merge_new.pth.tar. 

### Inference:
	The trained weights vb_aia_merge_new.pth.tar on VB dataset is also provided in BEST_MODEL. 
	you can run python enhance.py to enhance the noisy speech samples.

### requirements:
	
	CUDA 10.1
	torch == 1.8.0
	pesq == 0.0.1
	librosa == 0.7.2
	SoundFile == 0.10.3



### Comparison with SOTA:

![image](https://user-images.githubusercontent.com/51236251/138376964-86f1b0b5-9564-4ca4-a536-5b125e462809.png)



### Citation
If you use our code in your research or wish to refer to the baseline results, please use the following BibTeX entry.

    @article{yu2021dual,
  	title={Dual-branch Attention-In-Attention Transformer for single-channel speech enhancement},
  	author={Yu, Guochen and Li, Andong and Wang, Yutian and Guo, Yinuo and Wang, Hui and Zheng, Chengshi},
  	journal={arXiv preprint arXiv:2110.06467},
  	year={2021}
	}
