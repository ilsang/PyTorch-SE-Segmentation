
def dice_tensor(output, target):
    smooth = 0.001
    
    oflat, tflat = output.float().view(-1), target.float().view(-1)
    intersection = (oflat * tflat).sum()
    
    return ( 2. * intersection + smooth) / ( oflat.sum() + tflat.sum() + smooth)


def Evaluation(outputs, labels, verbose=False):
    
    smoothing=0.001
    
    rev_pred = np.array([np.flipud(axial) for axial in outputs[::-1]])
    
    gt, pred, rev_pred = np.float32(labels.flatten()), np.float32(outputs.flatten()), np.float32(rev_pred.flatten())
    binary_gt, binary_pred, binary_rev_pred = gt.copy(), pred.copy(), rev_pred.copy()
    
    binary_gt[binary_gt != 0.] = 1.0
    binary_pred[binary_pred != 0.] = 1.0
    binary_rev_pred[binary_rev_pred != 0.] = 1.0
    union_gt, union_pred, union_rev_pred = np.sum(binary_gt), np.sum(binary_pred), np.sum(binary_rev_pred)
    gt_inter_pred = np.sum(binary_gt * binary_pred)
    gt_inter_rev_pred = np.sum(binary_gt * binary_rev_pred)
    
    dice = (2 * gt_inter_pred + smoothing) / (union_gt + union_pred + smoothing)
    rev_dice = (2 * gt_inter_rev_pred + smoothing) / (union_gt + union_rev_pred + smoothing)
    
    jaccard = (gt_inter_pred + smoothing)/ (union_gt + union_pred - gt_inter_pred + smoothing)
    rev_jaccard = (gt_inter_rev_pred + smoothing) / (union_gt + union_rev_pred - gt_inter_rev_pred + smoothing)
    
    if verbose:
        print('%s, Dice : %.6f, Reverse Dice : %.6f, Jaccard : %.6f, Reverse Jaccard : %.6f' % \
              (patient_name, dice, rev_dice, jaccard, rev_jaccard))
    
    return dice, rev_dice, jaccard, rev_jaccard