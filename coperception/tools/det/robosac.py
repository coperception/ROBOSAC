import argparse
from itertools import count
import os
from copy import deepcopy

import seaborn as sns
import torch.optim as optim
from torch.utils.data import DataLoader

from coperception.datasets import V2XSimDet
from coperception.configs import Config, ConfigGlobal
from coperception.utils.CoDetModule import *
from coperception.utils.loss import *
from coperception.utils.mean_ap import eval_map
from coperception.models.det import *
from coperception.utils.detection_util import late_fusion
from coperception.utils.data_util import apply_pose_noise
import random
from tqdm import tqdm
from torch.autograd import Variable
from box_matching import associate_2_detections


def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def get_jaccard_index(config, num_agent_list, padded_voxel_point, reg_target, anchors_map, gt_max_iou, result_1, result_2):
    num_sensor = num_agent_list[0][0].numpy()
    det_results_local_1 = [[] for i in range(num_sensor)]
    annotations_local_1 = [[] for i in range(num_sensor)]
    det_results_local_2 = [[] for i in range(num_sensor)]
    annotations_local_2 = [[] for i in range(num_sensor)]
    ego_idx = args.ego_agent
    # for k in range(num_sensor):
    data_agents = {'bev_seq': torch.unsqueeze(padded_voxel_point[ego_idx, :, :, :, :], 1),
                'reg_targets': torch.unsqueeze(reg_target[ego_idx, :, :, :, :, :], 0),
                'anchors': torch.unsqueeze(anchors_map[ego_idx, :, :, :, :], 0)}
    temp = gt_max_iou[ego_idx]
    data_agents['gt_max_iou'] = temp[0]['gt_box'][0, :, :]
    result_temp_1 = result_1[ego_idx]
    result_temp_2 = result_2[ego_idx]
    temp_1 = {'bev_seq': data_agents['bev_seq'][0, -1].cpu().numpy(), 'result': result_temp_1[0][0],
            'reg_targets': data_agents['reg_targets'].cpu().numpy()[0],
            'anchors_map': data_agents['anchors'].cpu().numpy()[0],
            'gt_max_iou': data_agents['gt_max_iou']}
    temp_2 = {'bev_seq': data_agents['bev_seq'][0, -1].cpu().numpy(), 'result': result_temp_2[0][0],
            'reg_targets': data_agents['reg_targets'].cpu().numpy()[0],
            'anchors_map': data_agents['anchors'].cpu().numpy()[0],
            'gt_max_iou': data_agents['gt_max_iou']}
    
    det_results_local_1[ego_idx], annotations_local_1[ego_idx] = cal_local_mAP(config, temp_1, det_results_local_1[ego_idx], annotations_local_1[ego_idx])
    det_results_local_2[ego_idx], annotations_local_2[ego_idx] = cal_local_mAP(config, temp_2, det_results_local_2[ego_idx], annotations_local_2[ego_idx])
    
    print("Calculating in the view of Agent {}:".format(ego_idx))
    # shape of det_results_local_1 [k][0][0] is (N, 9)
    # The final value of the array is confidence. Ignored
    if len(det_results_local_1[ego_idx]) == 0:
        # if ego have no detection, return 0
        return 0 
    det_1 = det_results_local_1[ego_idx][0][0][:,0:8]
    det_2 = det_results_local_2[ego_idx][0][0][:,0:8]
    # jac_index = calculate_jaccard(det_results_local_1[k][0][0], det_results_local_2[k][0][0])
    jac_index = associate_2_detections(det_1, det_2)
    return jac_index

        

def visualize(config, filename0, save_fig_path, fafmodule, data, num_agent_list, padded_voxel_point, gt_max_iou, vis_tag):
    print("Visualizing: {}".format(vis_tag))
    det_results_local = [[] for i in range(6)]
    annotations_local = [[] for i in range(6)]

    padded_voxel_point = data['bev_seq']
    padded_voxel_points_teacher = data['bev_seq_teacher']
    reg_target = data['reg_targets']
    anchors_map = data['anchors']

    loss, cls_loss, loc_loss, result = fafmodule.predict_all(data, 1, num_agent=num_agent_list[0][0])
            
    # local qualitative evaluation
    num_sensor = num_agent_list[0][0].numpy()
    print(f'num_sensor: {num_sensor}')
    for k in range(num_sensor):
        data_agents = {'bev_seq': torch.unsqueeze(padded_voxel_point[k, :, :, :, :], 1),
                    'bev_seq_teacher': torch.unsqueeze(padded_voxel_points_teacher[k, :, :, :, :], 1),
                    'reg_targets': torch.unsqueeze(reg_target[k, :, :, :, :, :], 0),
                    'anchors': torch.unsqueeze(anchors_map[k, :, :, :, :], 0)}
        temp = gt_max_iou[k]
        data_agents['gt_max_iou'] = temp[0]['gt_box'][0, :, :]
        result_temp = result[k]
        
        temp = {'bev_seq': data_agents['bev_seq'][0, -1].cpu().numpy(), 
                'bev_seq_teacher': data_agents['bev_seq_teacher'][0, -1].cpu().numpy(),
                'result': result_temp[0][0],
                'reg_targets': data_agents['reg_targets'].cpu().numpy()[0],
                'anchors_map': data_agents['anchors'].cpu().numpy()[0],
                'gt_max_iou': data_agents['gt_max_iou'],
                'vis_tag': vis_tag}
        
        det_results_local[k], annotations_local[k] = cal_local_mAP(config, temp, det_results_local[k], annotations_local[k])
        print("Agent {}:".format(k))
        filename = str(filename0[0][0])
        cut = filename[filename.rfind('agent') + 7:]
        seq_name = cut[:cut.rfind('_')]
        idx = cut[cut.rfind('_') + 1:cut.rfind('/')]
        seq_save = os.path.join(save_fig_path[k], seq_name)
        check_folder(seq_save)
        idx_save = '{}_{}.png'.format(str(idx), vis_tag)

        if args.visualization:
            visualization(config, temp, None, None, 0, os.path.join(seq_save, idx_save))
    
def time_str():
        t = time.time()- 60*60*24*30
        time_string = time.strftime("%Y_%m_%d_%H:%M:%S", time.localtime(t))
        return time_string

def local_eval(num_agent, padded_voxel_points, reg_target, anchors_map, gt_max_iou, result, config, det_results_local, annotations_local):
    # If has RSU, do not count RSU's output into evaluation
    # eval_start_idx = 0 if args.no_cross_road else 1
    eval_start_idx = 0
    # update global result
    for k in range(eval_start_idx, num_agent):
        data_agents = {
            "bev_seq": torch.unsqueeze(padded_voxel_points[k, :, :, :, :], 1),
            "reg_targets": torch.unsqueeze(reg_target[k, :, :, :, :, :], 0),
            "anchors": torch.unsqueeze(anchors_map[k, :, :, :, :], 0),
        }
        temp = gt_max_iou[k]

        if len(temp[0]["gt_box"]) == 0:
            data_agents["gt_max_iou"] = []
        else:
            data_agents["gt_max_iou"] = temp[0]["gt_box"][0, :, :]


        result_temp = result[k]

        temp = {
            "bev_seq": data_agents["bev_seq"][0, -1].cpu().numpy(),
            "result": [] if len(result_temp) == 0 else result_temp[0][0],
            "reg_targets": data_agents["reg_targets"].cpu().numpy()[0],
            "anchors_map": data_agents["anchors"].cpu().numpy()[0],
            "gt_max_iou": data_agents["gt_max_iou"],
        }
        det_results_local[k], annotations_local[k] = cal_local_mAP(
            config, temp, det_results_local[k], annotations_local[k]
        )
    return det_results_local, annotations_local


def cal_robosac_steps(num_agent, num_consensus, num_attackers):
    # exclude ego agent
    num_agent = num_agent - 1
    eta = num_attackers / num_agent
    # print(f'eta: {eta}')
    # print(f's(num_agent): {num_agent}')
    N = np.ceil(np.log(1 - 0.99) / np.log(1 - np.power(1 - eta, num_consensus))).astype(int)
    return N

def cal_robosac_consensus(num_agent, step_budget, num_attackers):
    num_agent = num_agent - 1
    eta = num_attackers / num_agent
    s = np.floor(np.log(1-np.power(1-0.99, 1/step_budget)) / np.log(1-eta)).astype(int)
    return s


def cw_l2_attack(model, inputs, labels, device, targeted=False, c=1e-4, kappa=0, max_iter=1000, learning_rate=0.01) :
    # Define f-function
    def f(x) :

        outputs = model(x)
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)

        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.byte())
        
        # If targeted, optimize for making the other class most likely 
        if targeted :
            return torch.clamp(i-j, min=-kappa)
        
        # If untargeted, optimize for making the other class most likely 
        else :
            return torch.clamp(j-i, min=-kappa)
    
    w = torch.zeros_like(inputs, requires_grad=True).to(device)

    optimizer = optim.Adam([w], lr=learning_rate)

    prev = 1e10
    
    for step in range(max_iter) :

        a = 1/2*(nn.Tanh()(w) + 1)

        loss1 = nn.MSELoss(reduction='sum')(a, inputs)
        loss2 = torch.sum(c*f(a))

        cost = loss1 + loss2

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # Early Stop when loss does not converge.
        if step % (max_iter//10) == 0 :
            if cost > prev :
                print('Attack Stopped due to CONVERGENCE....')
                return a
            prev = cost
        
        print('- Learning Progress : %2.2f %%        ' %((step+1)/max_iter*100), end='\r')

    attack_inputs = 1/2*(nn.Tanh()(w) + 1)

    return attack_inputs


# @torch.no_grad()
# We cannot use torch.no_grad() since we need to calculate the gradient for perturbation
def main(args):
    config = Config("train", binary=True, only_det=True)
    config_global = ConfigGlobal("train", binary=True, only_det=True)

    need_log = args.log
    num_workers = args.nworker
    apply_late_fusion = args.apply_late_fusion
    pose_noise = args.pose_noise
    compress_level = args.compress_level
    only_v2i = args.only_v2i
    batch_size = args.batch

    # Specify gpu device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = torch.cuda.device_count()
    print("device number", device_num)

    config.inference = args.inference
    if args.bound == "upperbound":
        flag = "upperbound"
    else:
        if args.com == "when2com":
            flag = "when2com"
            if args.inference == "argmax_test":
                flag = "who2com"
            if args.warp_flag:
                flag = flag + "_warp"
        elif args.com in {"v2v", "disco", "sum", "mean", "max", "cat", "agent"}:
            flag = args.com
        else:
            flag = "lowerbound"
            if args.box_com:
                flag += "_box_com"

    print("flag", flag)
    config.flag = flag
    config.split = "test"

    num_agent = args.num_agent
    # agent0 is the cross road
    agent_idx_range = range(1, num_agent) if args.no_cross_road else range(num_agent)
    validation_dataset = V2XSimDet(
        dataset_roots=[f"{args.data}/agent{i}" for i in agent_idx_range],
        config=config,
        config_global=config_global,
        split="val",
        val=True,
        bound=args.bound,
        kd_flag=args.kd_flag,
        no_cross_road=args.no_cross_road,
    )
    validation_data_loader = DataLoader(
        validation_dataset, batch_size=1, shuffle=False, num_workers=num_workers
    )
    print("Validation dataset size:", len(validation_dataset))

    if args.no_cross_road:
        num_agent -= 1

    if flag == "upperbound" or flag.startswith("lowerbound"):
        model = FaFNet(
            config, layer=args.layer, kd_flag=args.kd_flag, num_agent=num_agent
        )
    elif flag.startswith("when2com") or flag.startswith("who2com"):
        # model = PixelwiseWeightedFusionSoftmax(config, layer=args.layer)
        model = When2com(
            config,
            layer=args.layer,
            warp_flag=args.warp_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif args.com == "disco":
        model = DiscoNet(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif args.com == "sum":
        model = SumFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif args.com == "mean":
        model = MeanFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif args.com == "max":
        model = MaxFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif args.com == "cat":
        model = CatFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif args.com == "agent":
        model = AgentWiseWeightedFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    else:
        model = V2VNet(
            config,
            gnn_iter_times=args.gnn_iter_times,
            layer=args.layer,
            layer_channel=256,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )

    model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = {
        "cls": SoftmaxFocalClassificationLoss(),
        "loc": WeightedSmoothL1LocalizationLoss(),
    }

    fafmodule = FaFModule(model, model, config, optimizer, criterion, args.kd_flag)

    model_save_path = args.resume[: args.resume.rfind("/")]

    if args.inference == "argmax_test":
        model_save_path = model_save_path.replace("when2com", "who2com")

    os.makedirs(model_save_path, exist_ok=True)

    checkpoint = torch.load(
        args.resume, map_location="cpu"
    )  # We have low GPU utilization for testing
    start_epoch = checkpoint["epoch"] + 1
    fafmodule.model.load_state_dict(checkpoint["model_state_dict"])
    fafmodule.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    fafmodule.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print("Load model from {}, at epoch {}".format(args.resume, start_epoch - 1))
    
    if args.log:
        log_file_name = os.path.join(model_save_path, "log_epoch{}_scene{}_ego{}_{}attackers_{}_{}.txt".format(checkpoint["epoch"], args.scene_id, args.ego_agent, args.number_of_attackers, args.robosac, time_str()))
        saver = open(log_file_name, "a")
        saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
        saver.flush()

        # Logging the details for this experiment
        saver.write("command line: {}\n".format(" ".join(sys.argv[1:])))
        saver.write(args.__repr__() + "\n\n")
        saver.flush()

    def print_and_write_log(log_str):
        print(log_str)
        if args.log:
            saver.write(log_str + "\n")
            saver.flush()

    #  ===== eval =====
    fafmodule.model.eval()
    save_fig_path = [
        check_folder(os.path.join(model_save_path, f"vis{i}")) for i in agent_idx_range
    ]
    tracking_path = [
        check_folder(os.path.join(model_save_path, f"tracking{i}"))
        for i in agent_idx_range
    ]

    # for local and global mAP evaluation
    det_results_local = [[] for i in agent_idx_range]
    annotations_local = [[] for i in agent_idx_range]
    

    for k, v in fafmodule.model.named_parameters():
        v.requires_grad = False  # fix parameters



    assert args.robosac in ["upperbound", "lowerbound", "no_defense", "robosac_validation", "robosac_mAP", "adaptive", "fix_attackers", "performance_eval", "probing"]



    # NOTE: ONLY SUPPORT SINGLE SCENE BY NOW
    frame_count = 100
    # array for robosac total steps
    steps = np.zeros(frame_count)
    # array for ego prediction count
    ego_steps = np.zeros(frame_count)
    fpss = np.zeros(frame_count)

    # array for consensus set sizes(for adaptive sampling)
    consensus_set_sizes = np.zeros(frame_count)

    # start from select 1 collab agent(for adaptive sampling)
    # keep it out of the loop for not initializing every time
    consensus_set_size = 1
    
    # cnt for adaptive sampling steps from frame 0 
    total_adaptive_steps = 0

    # once failed, need a flag to record(for adaptive sampling)
    failed_once = False
    
    # for probing
    N_th_frame_of_each_estimation = [-1] * 5
    # TODO: set ratios as input
    estimate_attacker_ratio = [0.0, 0.2, 0.4, 0.6, 0.8]
    estimated_attacker_ratio = 1.0
    
    consensus_tries = [5,4,3,2,1]
    consensus_tries_is_needed = [1,1,1,1,1]
    # probing_step_limit_by_attacker_ratio
    NMax = []
    for ratio in estimate_attacker_ratio:
        # TODO: set 5 to a variable
        temp_num_attackers = round(5 * (ratio))
        temp_num_consensus = 5 - temp_num_attackers
        NMax.append(cal_robosac_steps(num_agent, temp_num_consensus, temp_num_attackers))
    
    # Special case when assuming all agents are benign.(i.e. attacker ratio = 1.0)
    # means once if we can't test consensus in 1 try, there's definitely at least 1 attacker.
    NMax[0] = 1
    # print("NMax:", NMax)
    # {5: 1, 4: 9, 3: 19, 2: 27, 1: 21}
    NTry = [0] * len(estimate_attacker_ratio)
    total_sampling_step =0

    # succ count for robosac eval
    succ = 0 
    partial_succ = 0
    fail = 0
    # counters for relative frame in a single scene
    frame_seq = 0

    fix_attackers_generated = False
    fix_attackers_collab_agent_list = []
    fix_attackers_total_step = 0
    
    for cnt, sample in enumerate(tqdm(validation_data_loader)):

        t = time.time()
        (
            padded_voxel_point_list,
            padded_voxel_points_teacher_list,
            label_one_hot_list,
            reg_target_list,
            reg_loss_mask_list,
            anchors_map_list,
            vis_maps_list,
            gt_max_iou,
            filenames,
            target_agent_id_list,
            num_agent_list,
            trans_matrices_list,
        ) = zip(*sample)

        filename0 = filenames[0]
        filename = str(filename0[0][0])
        cut = filename[filename.rfind('agent') + 7:]
        seq_name = cut[:cut.rfind('_')]
        idx = cut[cut.rfind('_') + 1:cut.rfind('/')]
        

        if (int(seq_name) not in args.scene_id):
            continue
        
        if (args.sample_id is not None):
            if (int(idx) < args.sample_id):
                continue

        frame_seq += 1
        print_and_write_log("\nScene {}, Frame {}:".format(seq_name, idx))
        trans_matrices = torch.stack(tuple(trans_matrices_list), 1)
        target_agent_ids = torch.stack(tuple(target_agent_id_list), 1)
        num_all_agents = torch.stack(tuple(num_agent_list), 1)


        if args.no_cross_road:
            num_all_agents -= 1
        padded_voxel_points = torch.cat(tuple(padded_voxel_point_list), 0)
        padded_voxel_points_teacher = torch.cat(tuple(padded_voxel_points_teacher_list), 0)

        label_one_hot = torch.cat(tuple(label_one_hot_list), 0)
        reg_target = torch.cat(tuple(reg_target_list), 0)
        reg_loss_mask = torch.cat(tuple(reg_loss_mask_list), 0)
        anchors_map = torch.cat(tuple(anchors_map_list), 0)
        vis_maps = torch.cat(tuple(vis_maps_list), 0)

        data = {
            "bev_seq": padded_voxel_points.to(device),
            "bev_seq_teacher": padded_voxel_points_teacher.to(device),
            "labels": label_one_hot.to(device),
            "reg_targets": reg_target.to(device),
            "anchors": anchors_map.to(device),
            "vis_maps": vis_maps.to(device),
            "reg_loss_mask": reg_loss_mask.to(device).type(dtype=torch.bool),
            "target_agent_ids": target_agent_ids.to(device),
            "num_agent": num_all_agents.to(device),
            'ego_agent': args.ego_agent,
            'pert': None,
            'no_fuse': False,
            'collab_agent_list': None,
            'trial_agent_id': None,
            'confidence': None,
            'unadv_pert': None,
            'attacker_list' : None,
            'eps': None,
            "trans_matrices": trans_matrices.to(device),
        }

        if args.robosac == "performance_eval":
            fafmodule.cal_forward_time(data, 1)
            continue


        # STEP 1:
        # get original ego agent class prediction of all anchors, without adv pert and fuse, return cls pred of all agents
        cls_result  = fafmodule.cls_predict(data, batch_size, no_fuse=True)
        # change logits to one-hot
        mean = torch.mean(cls_result, dim=2)
        cls_result[:,:,0] = cls_result[:,:,0] > mean
        cls_result[:,:,1] = cls_result[:,:,1] > mean
        pseudo_gt = cls_result.clone().detach()
        # torch.Size([6, 393216, 2])

        if args.visualization:
            # visulize ego only det result, without fusion
            data['no_fuse'] = True
            visualize(config, filename0, save_fig_path, fafmodule, data, num_agent_list, padded_voxel_points, gt_max_iou, vis_tag='ego_only')
            # visulize original fusion result
            data['no_fuse'] = False
            visualize(config, filename0, save_fig_path, fafmodule, data, num_agent_list, padded_voxel_points, gt_max_iou, vis_tag='original_fusion')
            

        if args.robosac == 'upperbound':
            # no attacker is attacking and all agents are in collaboration, everything is just fine
            data['pert'] = None
            if args.partial_upperbound:
                # Sometimes we need to eval partially colloborated agents
                num_sensor = num_agent_list[0][0]
                ego_idx = args.ego_agent
                all_agent_list = [i for i in range(num_sensor)]
                all_agent_list.remove(ego_idx)
                collab_agent_list = random.sample(all_agent_list, k=args.robosac_k)
                data['collab_agent_list'] = collab_agent_list
                print_and_write_log("\nPartial upperbound, collab agent list: {}".format(collab_agent_list))
            else:
                data['collab_agent_list'] = None
            data['trial_agent_id'] = None
            data['no_fuse'] = False
            loss, cls_loss, loc_loss, result = fafmodule.predict_all(data, 1, num_agent=num_agent)
            if args.visualization:
                visualize(config, filename0, save_fig_path, fafmodule, data, num_agent_list, padded_voxel_points, gt_max_iou, vis_tag='upperbound')
            det_results_local, annotations_local = local_eval(num_agent, padded_voxel_points, reg_target, anchors_map, gt_max_iou, result, config, det_results_local, annotations_local)
            continue

        elif args.robosac == 'lowerbound':
            # Suppose all neighboring agents are malicious, and only the ego agent is trusted
            # Each agent only use its own features to perform object detection
            data['pert'] = None
            data['collab_agent_list'] = None
            data['trial_agent_id'] = None
            data['no_fuse'] = True
            loss, cls_loss, loc_loss, result = fafmodule.predict_all(data, 1, num_agent=num_agent)
            if args.visualization:
                # visualize attacked result
                visualize(config, filename0, save_fig_path, fafmodule, data, num_agent_list, padded_voxel_points, gt_max_iou, vis_tag='lowerbound')
            det_results_local, annotations_local = local_eval(num_agent, padded_voxel_points, reg_target, anchors_map, gt_max_iou, result, config, det_results_local, annotations_local)
            continue
        
        else:
            # There are attackers among us: 

            # STEP 2:
            # generate adv perturb
            if args.adv_method == 'pgd':
                # PGD random init   
                pert = torch.randn(6, 256, 32, 32) * 0.1
            elif args.adv_method == 'bim' or args.adv_method == 'cw-l2':
                # BIM/CW-L2 zero init
                pert = torch.zeros(6, 256, 32, 32)
            else:
                raise NotImplementedError

            num_sensor = num_agent_list[0][0]
            ego_idx = args.ego_agent
            all_agent_list = [i for i in range(num_sensor)]
            # We always trust ourself
            all_agent_list.remove(ego_idx)
            # Not including ego agent, since ego agent is always used.
            # Randomly samples neighboring agents as attackers
            # NOTE: 
            if args.robosac == 'fix_attackers':
                # # Agent 2 always attacks if there is only one attacker and random attackers not specified
                # if args.number_of_attackers == 1 :
                #     attacker_list = [0]
                # elif args.number_of_attackers == 2:
                #     # Agent 0,2 always attacks if there is only one attacker and random attackers not specified
                #     attacker_list = [0, 2]
                # elif args.number_of_attackers == 3:
                #     attacker_list = [0, 2, 3]
                # elif args.number_of_attackers == 4:
                #     attacker_list = [0, 2, 3, 4]
                # # ...TBD
                
                # Generate random attackers and keep them always attacking in the scene
                if fix_attackers_generated == False:
                    attacker_list = random.sample(all_agent_list, k=args.number_of_attackers)
                    fix_attackers_generated = True

            else:
                attacker_list = random.sample(all_agent_list, k=args.number_of_attackers)
            data['attacker_list'] = attacker_list
            data['eps'] = args.eps
            data['no_fuse'] = False
            for i in range(args.adv_iter):
                pert.requires_grad = True
                # Introduce adv perturbation
                data['pert'] = pert.to(device)
                        
                # STEP 3: Use inverted classification ground truth, minimze loss wrt inverted gt, to generate adv attacks based on cls(only)
                # NOTE: Actual ground truth is not always available especially in real-world attacks
                # We define the adversarial loss of the perturbed output with respect to an unperturbed output pseudo_gt instead of the ground truth
                cls_loss = fafmodule.cls_step(data, batch_size, ego_loss_only=args.ego_loss_only, ego_agent=args.ego_agent, invert_gt=True, self_result=pseudo_gt, adv_method=args.adv_method)

                pert = pert + args.pert_alpha * pert.grad.sign() * -1
                pert.detach_()
            
            # Detach and clone perturbations from Pytorch computation graph, in case of gradient misuse.
            pert = pert.detach().clone()
            # Apply the final perturbation to attackers' feature maps.
            data['pert'] = pert.to(device)
            print_and_write_log("Perturbation is applied on agent {}".format(attacker_list))
            if args.visualization:
                # visualize attacked result
                visualize(config, filename0, save_fig_path, fafmodule, data, num_agent_list, padded_voxel_points, gt_max_iou, vis_tag='attacked_fusion')

            if args.robosac == 'no_defense':
                # attacker is always attacking and no defense is applied
                data['pert'] = pert.to(device)
                data['no_fuse'] = False
                loss, cls_loss, loc_loss, result = fafmodule.predict_all(data, 1, num_agent=num_agent)
                det_results_local, annotations_local = local_eval(num_agent, padded_voxel_points, reg_target, anchors_map, gt_max_iou, result, config, det_results_local, annotations_local)
                continue
            

            if args.use_history_frame == True:
                # use history frame to save one forward pass
                if int(idx) == 0:
                    # first frame, use current ego only result as reference result
                    print_and_write_log("first frame, use current ego only result as reference result")
                    data['pert'] = None
                    data['collab_agent_list'] = None
                    data['no_fuse'] = True
                    _, _, _, result_reference = fafmodule.predict_all(data, 1, num_agent=num_agent)
                    ego_steps[frame_seq-1] = 1
                # if not first frame, keep no-op since we use history frame and it will be updated at the end of the iteration
            else:
                # if not use history frame, use current frame as reference frame
                # Get the original(ego_only) prediction
                print_and_write_log("performing calculating ego only result...")
                data['pert'] = None
                data['collab_agent_list'] = None
                data['no_fuse'] = True
                _, _, _, result_reference = fafmodule.predict_all(data, 1, num_agent=num_agent)


            if args.robosac == 'fix_attackers':
                # Assume attacker_list is fixed and always attacking in the scene, then after reached consensus, omit sampling process
                num_sensor = num_agent_list[0][0]
                ego_idx = args.ego_agent
                all_agent_list = [i for i in range(num_sensor)]
                # We always trust ourself
                all_agent_list.remove(ego_idx)
                # Not including ego agent, since ego agent is always used.
                
                if fix_attackers_collab_agent_list == []:
                    # if consensus is not reached, keep sampling attackers
                    collab_agent_list = []
                    if args.robosac_k == None:
                        consensus_set_size = cal_robosac_consensus(
                            num_agent, args.step_budget, args.number_of_attackers)

                        print_and_write_log("\nStep Budget {}, Calculated Consensus Set Size {}:".format(
                            args.step_budget, consensus_set_size))

                        if(consensus_set_size < 1):
                            print_and_write_log(
                                'Expected Consensus Agent below 1. Exit.'.format(consensus_set_size))
                            sys.exit()
                    found = False
                    # NOTE: 1~step_budget-1
                    for step in range(1, args.step_budget + 1):
                        # NOTE: random.choices will sample an agent more than once. eg.: [2, 3, 2]
                        # So we should use random.sample(population, k) to avoid this.
                        # collab_agent_list = random.sample(all_agent_list, k=args.robosac_k)
                        fix_attackers_total_step += 1
                        print_and_write_log("\nScene {}, Frame {}, Step {}, Step Budget {}:".format(
                        seq_name, idx, step, args.step_budget))

                        if args.robosac_k == None:
                            collab_agent_list = random.sample(
                                all_agent_list, k=consensus_set_size)
                        else:
                            collab_agent_list = random.sample(
                                all_agent_list, k=args.robosac_k)
                        data['collab_agent_list'] = collab_agent_list
                        data['no_fuse'] = False
                        data['pert'] = pert.to(device)

                        loss, cls_loss, loc_loss, result = fafmodule.predict_all(
                            data, 1, num_agent=num_agent)

                        # We use jaccard index to define the difference between two bbox sets
                        jac_index = get_jaccard_index(
                            config, num_agent_list, padded_voxel_points, reg_target, anchors_map, gt_max_iou, result_reference, result)
                        print_and_write_log(
                            "Jaccard Coefficient: {}".format(jac_index))
                        if jac_index < args.box_matching_thresh:
                            print_and_write_log(
                                'Attacker(s) is(are) among {}'.format(collab_agent_list))
                        else:
                            sus_agent_list = [
                                i for i in all_agent_list if i not in collab_agent_list]
                            print_and_write_log('Achieved consensus at step {}, with agents {}. Attacker(s) is(are) among {}, excluded'.format(
                                step, collab_agent_list, sus_agent_list))
                            print_and_write_log('Now begin to keep collaborating with agents {}'.format(collab_agent_list))
                            
                            found = True
                            # reached consensus, break
                            fix_attackers_collab_agent_list = collab_agent_list
                            steps[frame_seq - 1] = step
                            succ += 1
                            
                            break

                    if not found:
                        print_and_write_log('No consensus!')
                        # Can't achieve consensus, so fall back to original ego only result
                        data['pert'] = None
                        data['collab_agent_list'] = None
                        data['no_fuse'] = True
                        _, _, _, result_self_only = fafmodule.predict_all(
                            data, 1, num_agent=num_agent)
                        result = result_self_only
                        steps[frame_seq - 1] = args.step_budget
                        fail += 1

                    if args.use_history_frame == True:
                        # update reference frame for next iteration
                        print_and_write_log("update frame {} result as reference frame result for the next frame".format(idx))
                        result_reference = result
                    else:
                        ego_steps[frame_seq - 1] = 1
                else: 
                    print_and_write_log("\nfound consensus, use fixed collaborator:{}".format(fix_attackers_collab_agent_list))
                    # found consensus, use fixed collaborator
                    data['collab_agent_list'] = fix_attackers_collab_agent_list
                    data['no_fuse'] = False
                    data['pert'] = pert.to(device)
                    steps[frame_seq - 1] = 1
                    loss, cls_loss, loc_loss, result = fafmodule.predict_all(
                        data, 1, num_agent=num_agent)
                    if args.visualization:
                        # visualize consensus result
                        visualize(config, filename0, save_fig_path, fafmodule, data,
                                num_agent_list, padded_voxel_points, gt_max_iou, vis_tag='consensus')
                    

                # save the step num for current frame, Then calculate mean steps over the scene.

                det_results_local, annotations_local = local_eval(
                    num_agent, padded_voxel_points, reg_target, anchors_map, gt_max_iou, result, config, det_results_local, annotations_local)



            if args.robosac == "probing" :
                
                step = 0
                succ_result = None
                succ_probing_consensus_size = 0

                #TODO: set 5 to a variable
                assert args.step_budget >= 5 #ensuring probing tries will traverse all possible attacker ratios



                while step < args.step_budget and NTry < NMax:
                    # for consensus_set_size in consensus_tries:
                    #     # probe attackers
                    #     temp_num_attackers = (5-consensus_set_size)
                    #     temp_attacker_ratio = temp_num_attackers / 5
                    for i in range(len(estimate_attacker_ratio)):
                        temp_attacker_ratio = estimate_attacker_ratio[i]
                        consensus_set_size = round(5*(1-temp_attacker_ratio))
                        if NTry[i] < NMax[i]:
                            print_and_write_log("Probing {} agents for consensus".format(consensus_set_size))
                            step += 1
                            total_sampling_step += 1
                            # probing_step_tried_by_consensus_set_size[consensus_set_size] += 1
                            # step budget available for probing
                            # try to probe attacker ratio
                            collab_agent_list = random.sample(
                            all_agent_list, k=consensus_set_size)
                            data['collab_agent_list'] = collab_agent_list
                            data['no_fuse'] = False
                            data['pert'] = pert.to(device)

                            loss, cls_loss, loc_loss, result = fafmodule.predict_all(
                                data, 1, num_agent=num_agent)
                            
                            # We use jaccard index to define the difference between two bbox sets
                            jac_index = get_jaccard_index(
                                config, num_agent_list, padded_voxel_points, reg_target, anchors_map, gt_max_iou, result_reference, result)
                            print_and_write_log(
                                "Jaccard Coefficient: {}".format(jac_index))

                            if jac_index < args.box_matching_thresh:
                                # fail to reach consensus
                                print_and_write_log('No consensus reached when probing {} consensus agents. Current step is {} in Frame {}.'.format(consensus_set_size,step,idx))
                                print_and_write_log('Attacker(s) is(are) among {}'.format(collab_agent_list))

                                NTry[i] += 1 
                                
                                # if temp_num_attackers == 0:
                                #     # Assumption of no attackers fails
                                #     consensus_tries_is_needed[i] = 0

                                if NTry[i] == NMax[i]:
                                    print_and_write_log("Probing of {} agents for consensus has reached its sampling limit {} with assumed attacker ratio {} and consensus set size {}.".format(consensus_set_size, NMax[i], temp_attacker_ratio, consensus_set_size))
                                    print_and_write_log("From now on we won't try to probe {} agents consensus since it seems unlikely to reach that.".format(consensus_set_size))
                            else:
                                # succeed to reach consensus
                                sus_agent_list = [
                                    i for i in all_agent_list if i not in collab_agent_list]
                                print_and_write_log('Achieved consensus at step {} in Frame{}, with {} agents: {}. Using the result as temporal final output of this frame, and skipping smaller consensus set tries. \n Attacker(s) is(are) among {}, excluded.'.format(
                                    step, idx, consensus_set_size, collab_agent_list, sus_agent_list))
                                
                                succ_result = result
                                succ_probing_consensus_size = consensus_set_size
                                
                                if temp_attacker_ratio < estimated_attacker_ratio:
                                    print_and_write_log('Larger consensus set ({} agents) probed. We will skip all the smaller consensus set tries. Update attacker ratio estimation to {}'.format(consensus_set_size, temp_attacker_ratio))
                                    estimated_attacker_ratio = temp_attacker_ratio
                                    # Record probing frame
                                    N_th_frame_of_each_estimation[i] = idx
                                    
                                    for j in range(i, len(estimate_attacker_ratio)):
                                        # set all the larger attacker ratio to 0
                                        NTry[j] = NMax[j]

                                    break                                    



            elif args.robosac == 'robosac_mAP': #Needs Evaluation                            
                # Given Step Budget N and Sampling Set Size s, perform predictions

                num_sensor = num_agent_list[0][0]
                ego_idx = args.ego_agent
                all_agent_list = [i for i in range(num_sensor)]
                # We always trust ourself
                all_agent_list.remove(ego_idx)
                # Not including ego agent, since ego agent is always used.
                collab_agent_list = []

                if args.robosac_k == None:
                    consensus_set_size = cal_robosac_consensus(num_agent, args.step_budget, args.number_of_attackers)

                    print_and_write_log("\nStep Budget {}, Calculated Consensus Set Size {}:".format(args.step_budget, consensus_set_size))

                    if(consensus_set_size < 1):
                        print_and_write_log('Expected Consensus Agent below 1. Exit.'.format(consensus_set_size))
                        sys.exit()

                found = False
                # NOTE: 0~step_budget-1
                for step in range(1, args.step_budget + 1):
                    # NOTE: random.choices will sample an agent more than once. eg.: [2, 3, 2]
                    # So we should use random.sample(population, k) to avoid this.
                    # collab_agent_list = random.sample(all_agent_list, k=args.robosac_k)
                    if args.robosac_k == None:
                        collab_agent_list = random.sample(all_agent_list, k=consensus_set_size)
                    else:
                        collab_agent_list = random.sample(all_agent_list, k=args.robosac_k)
                    data['collab_agent_list'] = collab_agent_list
                    data['no_fuse'] = False
                    data['pert'] = pert.to(device)

                    loss, cls_loss, loc_loss, result = fafmodule.predict_all(data, 1, num_agent=num_agent)

                    # We use jaccard index to define the difference between two bbox sets
                    jac_index = get_jaccard_index(config, num_agent_list, padded_voxel_points, reg_target, anchors_map, gt_max_iou, result_reference, result)
                    print_and_write_log("Jaccard Coefficient: {}".format(jac_index))
                    if jac_index < args.box_matching_thresh:
                        print_and_write_log('Attacker(s) is(are) among {}'.format(collab_agent_list))
                    else:
                        sus_agent_list = [i for i in all_agent_list if i not in collab_agent_list]
                        print_and_write_log('Achieved consensus at step {}, with agents {}. Attacker(s) is(are) among {}, excluded'.format(step, collab_agent_list, sus_agent_list))
                        found = True
                        steps[frame_seq-1] = step
                        succ += 1
                        if args.visualization:
                            # visualize consensus result
                            visualize(config, filename0, save_fig_path, fafmodule, data, num_agent_list, padded_voxel_points, gt_max_iou, vis_tag='consensus')
                        break

                if not found:
                    print_and_write_log('No consensus!')
                    # Can't achieve consensus, so fall back to original ego only result
                    data['pert'] = None
                    data['collab_agent_list'] = None
                    data['no_fuse'] = True
                    _, _, _, result_self_only = fafmodule.predict_all(data, 1, num_agent=num_agent)
                    result = result_self_only
                    steps[frame_seq-1] = args.step_budget
                    ego_steps[frame_seq-1] = 1
                    fail += 1
                
                if args.use_history_frame == True:
                    # update reference frame for next iteration
                    print_and_write_log("update frame {} result as reference frame result for the next frame".format(idx))
                    result_reference = result
                else:
                    ego_steps[frame_seq - 1] = 1
                    

                det_results_local, annotations_local = local_eval(num_agent, padded_voxel_points, reg_target, anchors_map, gt_max_iou, result, config, det_results_local, annotations_local)


    print_and_write_log("\n Ego Agent:{}".format(args.ego_agent))

    if args.robosac == 'probing':
        print_and_write_log("Probing: Evaluated on {} frames".format(frame_seq))
        print_and_write_log("Nth frame of each estimation:{}".format(N_th_frame_of_each_estimation))
        print_and_write_log("Final estimation:{}".format(estimated_attacker_ratio))
        print_and_write_log("Ground Truth:{}".format(args.number_of_attackers/(num_agent-1)))
        print_and_write_log("Error of estimation:{}".format(abs(estimated_attacker_ratio - args.number_of_attackers/(num_agent-1))))
        print_and_write_log("Total sampling steps:{}".format(total_sampling_step))
        print_and_write_log("NTry:{}".format(NTry))
        return
        


    if args.robosac == 'adaptive':
        print_and_write_log("Max Consensus set size:{}".format(np.max(consensus_set_sizes)))
        print_and_write_log("Min Consensus set size:{}".format(np.min(consensus_set_sizes)))
        print_and_write_log("Avg Consensus set size:{}".format(np.mean(consensus_set_sizes)))
        # print_and_write_log("Most common Consensus set size:{}".format(np.argmax(np.bincount(consensus_set_sizes))))
        


    if args.robosac == "robosac_validation":
        # validation of robosac theory
        print_and_write_log("robosac VALIDATION: Evaluated on {} frames".format(frame_seq))
        print_and_write_log("Total Neighbor Agents:{}, Sampling Set Size: {}, Number of Attackers: {}".format(num_agent-1, args.robosac_k, args.number_of_attackers))
        print_and_write_log("Expected at least one successful sampling steps at p=0.99: {}".format(cal_robosac_steps(num_agent, args.robosac_k ,args.number_of_attackers)))
        print_and_write_log("Succeeded {}, Total {}, Success Rate: {}".format(succ, frame_seq, succ / frame_seq))
        print_and_write_log("Sampling STEP: MEAN: {}, MAX: {}, MIN:{}".format(np.mean(steps), np.max(steps), np.min(steps)))



    else:
        if args.robosac != 'lowerbound' or args.robosac != 'upperbound':
            print_and_write_log("robosac VALIDATION: Evaluated on {} frames".format(frame_seq))
            print_and_write_log("Total Neighbor Agents:{}, Sampling Set Size: {}, Number of Attackers: {}".format(num_agent-1, args.robosac_k, args.number_of_attackers))
            if args.robosac_k is None:
                consensus_set_size = cal_robosac_consensus(num_agent, args.step_budget, args.number_of_attackers)
                print_and_write_log("Expected guaranteed Consensus Set Size at p=0.99: {}".format(consensus_set_size))
            else:
                print_and_write_log("Expected at least one successful sampling steps at p=0.99: {}".format(cal_robosac_steps(num_agent, args.robosac_k ,args.number_of_attackers)))
            print_and_write_log("Succeeded {}, Total {}, Success Rate: {}".format(succ, frame_seq, succ / frame_seq))
            print_and_write_log("Sampling STEP MEAN: {}, MAX: {}, MIN:{}".format(np.mean(steps), np.max(steps), np.min(steps)))
            total_steps = steps + ego_steps
            print_and_write_log("Total STEP(including ego only step): MEAN: {}, MAX: {}, MIN:{}".format(np.mean(total_steps), np.max(total_steps), np.min(total_steps)))
            fpss = 1000 / (27*steps+17*ego_steps) # forward time: ego only: 17ms; collaborated: 27ms
            print_and_write_log("FPS: MEAN: {}, MAX: {}, MIN:{}".format(np.mean(fpss), np.max(fpss), np.min(fpss)))
            print_and_write_log(
                "Sampling STEP:{}, Ego STEP:{}, Total STEP:{}, FPS:{}".format(steps, ego_steps, total_steps, fpss))
            print_and_write_log("Box set matching threshold: {}".format(args.box_matching_thresh))
            if args.robosac == "fix_attackers":
                print_and_write_log("Fix attackers total step: {}".format(fix_attackers_total_step))
        # mAP evaluation

        # If has RSU, do not count RSU's output into evaluation
        # eval_start_idx = 0 if args.no_cross_road else 1
        eval_start_idx = 0
        # print(len(det_results_local[2][int(idx)][0]), len(annotations_local[2][int(idx)]['bboxes']))
        
        mean_ap_local = []
        # local mAP evaluation
        det_results_all_local = []
        annotations_all_local = []
        for k in range(eval_start_idx, num_agent):
            print_and_write_log("Local mAP@0.5 from agent {}".format(k))
            mean_ap, _ = eval_map(
                det_results_local[k],
                annotations_local[k],
                scale_ranges=None,
                iou_thr=0.5,
                dataset=None,
                logger=None,
            )
            mean_ap_local.append(mean_ap)
            print_and_write_log("Local mAP@0.7 from agent {}".format(k))

            mean_ap, _ = eval_map(
                det_results_local[k],
                annotations_local[k],
                scale_ranges=None,
                iou_thr=0.7,
                dataset=None,
                logger=None,
            )
            mean_ap_local.append(mean_ap)

            det_results_all_local += det_results_local[k]
            annotations_all_local += annotations_local[k]

        # average local mAP evaluation
        print_and_write_log("Average Local mAP@0.5")

        mean_ap_local_average, _ = eval_map(
            det_results_all_local,
            annotations_all_local,
            scale_ranges=None,
            iou_thr=0.5,
            dataset=None,
            logger=None,
        )
        mean_ap_local.append(mean_ap_local_average)

        print_and_write_log("Average Local mAP@0.7")

        mean_ap_local_average, _ = eval_map(
            det_results_all_local,
            annotations_all_local,
            scale_ranges=None,
            iou_thr=0.7,
            dataset=None,
            logger=None,
        )
        mean_ap_local.append(mean_ap_local_average)

        print_and_write_log(
            "Quantitative evaluation results of model from {}, at epoch {}".format(
                args.resume, start_epoch - 1
            )
        )

        for k in range(eval_start_idx, num_agent):
            print_and_write_log(
                "agent{} mAP@0.5 is {} and mAP@0.7 is {}".format(
                    k, mean_ap_local[k * 2], mean_ap_local[(k * 2) + 1]
                )
            )

        print_and_write_log(
            "average local mAP@0.5 is {} and average local mAP@0.7 is {}".format(
                mean_ap_local[-2], mean_ap_local[-1]
            )
        )

        if need_log:
            saver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-d",
        "--data",
        default="{Your_location_to_V2X-Sim}/V2X-Sim/test",
        type=str,
        help="The path to the preprocessed sparse BEV training data",
    )
    parser.add_argument("--batch", default=1, type=int, help="The number of scene")
    parser.add_argument("--nepoch", default=100, type=int, help="Number of epochs")
    parser.add_argument("--nworker", default=4, type=int, help="Number of workers")
    parser.add_argument("--lr", default=0.001, type=float, help="Initial learning rate")
    parser.add_argument("--log", action="store_true", help="Whether to log")
    parser.add_argument("--logpath", default="", help="The path to the output log file")
    parser.add_argument(
        "--resume",
        # default = "../../ckpt/meanfusion/epoch_advtrain_49.pth" #use this adv epoch 49 trained from scratch
          default="../../ckpt/meanfusion/epoch_49.pth",
        type=str,
        help="The path to the saved model that is loaded to resume training",
    )
    parser.add_argument(
        "--resume_teacher",
        default="",
        type=str,
        help="The path to the saved teacher model that is loaded to resume training",
    )
    parser.add_argument(
        "--layer",
        default=3,
        type=int,
        help="Communicate which layer in the single layer com mode",
    )
    parser.add_argument(
        "--warp_flag", action="store_true", help="Whether to use pose info for When2com"
    )
    parser.add_argument(
        "--kd_flag",
        default=0,
        type=int,
        help="Whether to enable distillation (only DiscNet is 1 )",
    )
    parser.add_argument("--kd_weight", default=100000, type=int, help="KD loss weight")
    parser.add_argument(
        "--gnn_iter_times",
        default=3,
        type=int,
        help="Number of message passing for V2VNet",
    )
    parser.add_argument(
        "--visualization", action="store_true", help="Visualize validation result"
    )
    parser.add_argument(
        "--com", default="mean", type=str, help="disco/when2com/v2v/sum/mean/max/cat/agent"
    )
    parser.add_argument(
        "--bound",
        type=str,
        default="both",
        help="The input setting: lowerbound -> single-view or upperbound -> multi-view",
    )
    parser.add_argument("--inference", type=str)
    parser.add_argument("--tracking", action="store_true")
    parser.add_argument("--box_com", action="store_true")
    parser.add_argument(
        "--no_cross_road", action="store_true", help="Do not load data of cross roads"
    )
    # scene_batch => batch size in each scene
    parser.add_argument(
        "--num_agent", default=6, type=int, help="The total number of agents"
    )
    parser.add_argument(
        "--apply_late_fusion",
        default=0,
        type=int,
        help="1: apply late fusion. 0: no late fusion",
    )
    parser.add_argument(
        "--compress_level",
        default=0,
        type=int,
        help="Compress the communication layer channels by 2**x times in encoder",
    )
    parser.add_argument(
        "--pose_noise",
        default=0,
        type=float,
        help="draw noise from normal distribution with given mean (in meters), apply to transformation matrix.",
    )
    parser.add_argument(
        "--only_v2i",
        default=0,
        type=int,
        help="1: only v2i, 0: v2v and v2i",
    )

    # Adversarial perturbation
    parser.add_argument('--pert_alpha', type=float, default=0.1, help='scale of the perturbation')
    parser.add_argument('--adv_method', type=str, default='pgd', help='pgd/bim/cw-l2')
    parser.add_argument('--eps', type=float, default=0.5, help='epsilon of adv attack.')
    parser.add_argument('--adv_iter', type=int, default=15, help='adv iterations of computing perturbation')

    # Scene and frame settings
    parser.add_argument('--scene_id', type=list, default=[8], help='target evaluation scene') #Scene 8, 96, 97 has 6 agents.
    parser.add_argument('--sample_id', type=int, default=None, help='target evaluation sample')
    
    # Among Us modes and parameters
    parser.add_argument('--robosac', type=str, default='', help='upperbound/lowerbound/no_defense/robosac_validation/robosac_mAP/adaptive/fix_attackers/performance_eval/probing')
    parser.add_argument('--ego_agent', type=int, default=1, help='id of ego agent')
    parser.add_argument('--robosac_k', type=int, default=None, help='specify consensus set size if needed')
    parser.add_argument('--ego_loss_only', action="store_true", help='only use ego loss to compute adv perturbation')
    parser.add_argument('--step_budget', type=int, default=3, help='sampling budget in a single frame')
    parser.add_argument('--box_matching_thresh', type=float, default=0.3, help='IoU threshold for validating two detection results')
    parser.add_argument('--number_of_attackers', type=int, default=1, help='number of malicious attackers in the scene')
    parser.add_argument('--fix_attackers', action="store_true", help='if true, attackers will not change in different frames')
    parser.add_argument('--use_history_frame', action="store_true", help='use history frame for computing the consensus, reduce 1 step of forward prop.')
    parser.add_argument('--partial_upperbound', action="store_true", help='use with specifying ransan_k, to perform clean collaboration with a subset of teammates')
    
    torch.multiprocessing.set_sharing_strategy("file_system")
    args = parser.parse_args()
    print(args)
    main(args)
