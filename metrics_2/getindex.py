
import os.path as osp

import numpy as np
from metrics_2.met import mean_iou

def evaluate(results,gt_seg_maps, CLASSES):
    """Evaluate the dataset.

    Args:
        results (list): Testing results of the dataset.
        metric (str | list[str]): Metrics to be evaluated.
        logger (logging.Logger | None | str): Logger used for printing
            related information during evaluation. Default: None.

    Returns:
        dict[str, float]: Default metrics.
    """

    # CLASSES=[]
    # gt_seg_maps=[]
    eval_results = {}
    num_classes = len(CLASSES)
    ignore_index=255

    iou_result = mean_iou(
        results, gt_seg_maps, num_classes, ignore_index)
    all_acc=iou_result["aAcc"]
    iou=iou_result["IoU"]
    acc=iou_result["Acc"]
    #all_acc, TP, FP, TN, FN, acc, iou, dice

    summary_str = ''
    summary_str += 'per class results:\n'
    line_format = '{:<15} {:>10} {:>10}\n'
    summary_str += line_format.format('Class', 'IoU', 'Acc')
    if CLASSES is None:
        class_names = tuple(range(num_classes))
    else:
        class_names = CLASSES
    
    # to record class-wise dice scores
    dice_pos_str=''
    dice_neg_str=''
    pl_dice=0
    bg_dice=0
    
    for i in range(num_classes):
        iou_str = '{:.2f}'.format(iou[i] * 100)
        acc_str = '{:.2f}'.format(acc[i] * 100)
            
        summary_str += line_format.format(class_names[i], iou_str,  acc_str)
    summary_str += 'Summary:\n'
    line_format = '{:<15} {:>10} {:>10} {:>10} \n'
    summary_str += line_format.format('Scope', 'mIoU', 'mAcc',  'aAcc')

    iou_str = '{:.2f}'.format(np.nanmean(iou) * 100)
    acc_str = '{:.2f}'.format(np.nanmean(acc) * 100)
    all_acc_str = '{:.2f}'.format(all_acc * 100)

    # TPR= (TP/(TP+FN)) if (TP+FN) !=0 else 0
    # TNR= (TN/(TN+FP)) if (TN+FP) !=0 else 0
    # FDR= (FP/(FP+TP)) if (FP+TP) !=0 else 0
    # precision= (TP/(TP+FP)) if (TP+FP) !=0 else 0

    # TPR_str = '{:.2f}'.format(TPR * 100)
    # TNR_str = '{:.2f}'.format(TNR * 100)
    # FDR_str = '{:.2f}'.format(FDR * 100)
    # precision_str = '{:.2f}'.format(precision * 100)
    summary_str += line_format.format('global', iou_str,  acc_str, all_acc_str)
    # # print_log(summary_str, logger)

    # eval_results['mIoU'] = np.nanmean(iou)
    # eval_results['mAcc'] = np.nanmean(acc)
    
    # eval_results['plDice'] = pl_dice
    # eval_results['bgDice'] = bg_dice
    # eval_results['aAcc'] = all_acc

    # eval_results['TPR'] = TPR
    # eval_results['TNR'] = TNR
    # eval_results['FDR'] = FDR
    # eval_results['Precision'] = precision
    
    return summary_str,iou_str

