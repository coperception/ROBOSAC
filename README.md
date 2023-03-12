# *Among Us*: Adversarially Robust Collaborative Perception by Consensus

[Yiming Li*](https://scholar.google.com/citations?user=i_aajNoAAAAJ), [Qi Fang*](https://scholar.google.com/citations?user=LIuiQlkAAAAJ), [Jiamu Bai](https://github.com/jiamubai), [Siheng Chen](https://scholar.google.com/citations?user=W_Q33RMAAAAJ), [Felix Juefei-Xu](https://scholar.google.com/citations?user=dgN8vtwAAAAJ), [Chen Feng](https://scholar.google.com/citations?user=YeG8ZM0AAAAJ)

**"Simple yet effective sampling mechanisim against malicious attackers in multi-agent collaborative perception settings"**

<p align="center"><img src='figs/teaser.png' align="center" height="350px"> </p>

[**ArXiv: Among Us: Adversarially Robust Collaborative Perception by Consensus**]()        

## Abstract

Multiple robots could perceive a scene (e.g., detect objects) collaboratively better than individuals, although easily suffer from adversarial attacks when using deep learning. This could be addressed by the adversarial defense, but its training requires the often-unknown attacking mechanism.

Differently, we propose **ROBOSAC**, a novel sampling-based defense strategy generalizable to unseen attackers. Our key idea is that collaborative perception should lead to consensus rather than dissensus in results compared to individual perception. This leads to our hypothesize-and-verify framework: perception results with and without collaboration from a random subset of teammates are compared until reaching a consensus. 

In such a framework, more teammates in the sampled subset often entail better perception performance but require longer sampling time to reject potential attackers. Thus, we derive how many sampling trials are needed to ensure the desired size of an attacker-free subset, or equivalently, the maximum size of such a subset that we can successfully sample within a given number of trials. We validate our method on the task of collaborative 3D object detection in autonomous driving scenarios.



## Installation

### Requirements

* Linux (tested on Ubuntu 18.04)
* Python 3.7
* Anaconda
* PyTorch
* CUDA 11.7

### Create Anaconda Environment from yml

in the directory of `AmongUs`:

```bash
cd coperception
conda env create -f environment.yml
conda activate coperception
```

### CUDA

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

### Install CoPerception Library

This installs and links `coperception` library to code in `./coperception` directory.

```bash
pip install -e .
```



## Dataset Preparation

Please download the official [V2X-Sim Dataset](https://ai4ce.github.io/V2X-Sim/download.html)**(<u>V2.0</u>)**

See [Documentation of installing V2X-Sim](https://coperception.readthedocs.io/en/latest/datasets/v2x_sim/) for dataset pre-processing details.



## Run ROBOSAC

in the directory of `AmongUs`:

```bash
cd coperception/tools/det/
```

```bash
python robosac.py [-d DATA] [--log] [--logpath LOGPATH] [--visualization]
                         [--pert_alpha PERT_ALPHA] [--adv_method ADV_METHOD]
                         [--eps EPS] [--unadv_pert_alpha UNADV_PERT_ALPHA]
                         [--scene_id SCENE_ID] [--sample_id SAMPLE_ID]
                         [--iteration_per_sample ITERATION_PER_SAMPLE]
                         [--robosac ROBOSAC] [--ego_agent EGO_AGENT]
                         [--robosac_k ROBOSAC_K] [--ego_loss_only]
                         [--step_budget STEP_BUDGET]
                         [--box_matching_thresh BOX_MATCHING_THRESH]
                         [--adv_iter ADV_ITER]
                         [--number_of_attackers NUMBER_OF_ATTACKERS]
                         [--fix_attackers] [--use_history_frame]
                         [--partial_upperbound]
```

**NOTE: Due to some data syncing issues, ROBOSAC cannot be performed under multi-GPU environment. **

**You may need to specify CUDA_VISIBLE_DEVICES in front of python commands if you are using multiple GPUs:**

```bash
CUDA_VISIBLE_DEVICES=0 python robosac.py [your params]
```



#### Overview of params in ROBOSAC

```python
-d DATA, --data DATA  
			The path to the preprocessed sparse BEV training data, test set.
			(default: *Specify your dataset location here*)

# Adversarial perturbation
--adv_method ADV_METHOD
      pgd/bim/cw-l2 
      (default: pgd)

--eps EPS             	
			epsilon of adv attack. 
			(default: 0.5)

--adv_iter ADV_ITER   
			adv iterations of computing perturbation 
			(default: 15)

    
# Scene and frame settings    
--scene_id SCENE_ID   
			target evaluation scene 
			(default: [8]) 
    	#Scene 8, 96, 97 has 6 agents. This param could not be specify in commandline, you shall change its default value, e.g. [96]

--sample_id SAMPLE_ID
      target evaluation sample 
      (default: None)

    
# ROBOSAC modes and parameters
--amongus ROBOSAC    
			upperbound/lowerbound/no_defense/robosac_validation/robosac_mAP/
  		adaptive/fix_attackers/performance_eval/probing 
    	(default: )

--ego_agent EGO_AGENT
			id of ego agent (default: 1)(agent 0 is RSU/Road Side Unit)

--robosac_k ROBOSAC_K   
			specify consensus set size if needed (default: None)

--ego_loss_only       
			only use ego loss to compute adv perturbation(default: False)

--step_budget STEP_BUDGET
			sampling budget in a single frame (default: 3)

--box_matching_thresh BOX_MATCHING_THRESH
      IoU threshold for validating two detection results
      (default: 0.3)

--number_of_attackers NUMBER_OF_ATTACKERS
      number of malicious attackers in the scene 
      (default:1)

--fix_attackers       
			if true, attackers will not change in different frames
			(default: False)

--use_history_frame   
			use history frame for computing the consensus, reduce 1 step of forward prop. 
			(default: False)

--partial_upperbound  
			use with specifying robosac_k, to perform clean collaboration with a subset of teammates 
			(default: False)
```



### Specifying Dataset

Link the test split of V2X-Sim dataset in the default value of argument "**data**"

```bash
/{Your_localtion}/V2X-Sim/sweeps/test
```

in the `test` folder data are structured like:

```
test
├──agent_0
├──agent_1
├──agent_2
├──agent_3
├──agent_4
├──agent_5
		├──19_0
				├──0.npy		
		...
```



### Specifying Victim Detection Model Checkpoint

Link the checkpoint location in the default value of argument "**resume**"

Pre-trained checkpoint is saved in `AmongUs/coperception/ckpt/meanfusion` folder.

`epoch_49.pth` is the original victim model without adversarial training.

`epoch_advtrain_49.pth` is the PGD-trained model.



### Validation of ROBOSAC Algorithm (Success Rate)

```bash
CUDA_VISIBLE_DEVICES=0 python robosac.py --log --robosac robosac_validation  --adv_iter 15 --number_of_attackers {desired_number_of_attackers}  --robosac_k {desired_number_of_teammates}
```



### Evaluation of Detection Performance (mAP)

#### Upperbound++ (Collaborate with all teammates in the clean environment)

```bash
CUDA_VISIBLE_DEVICES=0 python robosac.py --log --robosac upperbound
```

#### Upperbound (Collaborate with a subset of teammates in the clean environment)

```bash
CUDA_VISIBLE_DEVICES=0 python robosac.py --log --robosac upperbound --robosac_k {desired_number_of_teammates} --partial_upperbound
```

#### Lowerbound (Ego-only predictions)

```bash
CUDA_VISIBLE_DEVICES=0 python robosac.py --log --robosac lowerbound
```

#### No Defense (Collaborate with attackers without robosac or adversarial training)

```bash
CUDA_VISIBLE_DEVICES=0 python robosac.py --log --robosac no_defense --adv_iter 15 --number_of_attackers {desired_number_of_attackers} --ego_agent {desired_idx_of_ego_agent}
```

#### ROBOSAC

```bash
CUDA_VISIBLE_DEVICES=0 python robosac.py --log --robosac robosac_mAP  --adv_iter 15 --number_of_attackers {desired_number_of_attackers} --step_budget {desired_step_budget}
```

### Attacker Ratio Estimation (Aggressive-to-conservative Probing, A2CP)

```bash
CUDA_VISIBLE_DEVICES=0 python robosac.py --robosac probing --adv_iter 15 --number_of_attackers {desired_number_of_attackers} --step_budget {desired_step_budget, in paper: 5}
```





if `--log` specified, logs will be save in the directory of detection model checkpoint.

In default:

```bash
AmongUs/coperception/ckpt/log_epoch{}_scene{}_ego{}_{}attackers_{ROBOSAC_MODE}_{TIME_STR}.txt
```



## Acknowledgment  

*Among Us* is modified from [coperception](https://github.com/coperception/coperception) library.

PGD/BIM/CW attacks are implemented from [adversarial-attacks-pytorch](https://github.com/Harry24k/adversarial-attacks-pytorch) library.

This project is not possible without these great codebases.



## Citation

If you find this project useful in your research, please cite:

```

```