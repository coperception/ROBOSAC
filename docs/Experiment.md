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
