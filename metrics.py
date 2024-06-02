import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support


class Metric:

    def __call__(self, pred_scores, gt_labels):
        assert np.min(pred_scores) >= 0 and np.max(pred_scores) <= 1
        assert np.all(np.unique(gt_labels) == np.array([0, 1]))
        eps = 1e-12
        micro_ap = average_precision_score(gt_labels, pred_scores, average = 'micro')
        macro_ap = average_precision_score(gt_labels, pred_scores, average = 'macro')
        pred_labels = (pred_scores > 0.5).astype(np.float)
        # added by yk
        ma_prec, ma_recall, _, _ = precision_recall_fscore_support(gt_labels, pred_labels, average='macro')
        ma_f1 = 2 * ma_prec * ma_recall / (ma_prec + ma_recall + eps)
        mi_prec, mi_recall, _, _ = precision_recall_fscore_support(gt_labels, pred_labels, average='micro')
        mi_f1 = 2 * mi_prec * mi_recall / (mi_prec + mi_recall + eps)

        num_images, num_classes = pred_scores.shape

        ma_ap_each = dict()

        res = {
            'micro_ap': round(micro_ap, 4),
            'macro_ap': round(macro_ap, 4),
            'mi_prec': round(mi_prec, 4),
            'ma_prec': round(ma_prec, 4),
            'mi_recall': round(mi_recall, 4),
            'ma_recall': round(ma_recall, 4),
            'mi_f1': round(mi_f1, 4),
            'ma_f1': round(ma_f1, 4)
        }

        res.update(ma_ap_each)
        return res


if __name__ == '__main__':
    pred_scores = np.random.rand(20, 3)
    gt_labels = (np.random.rand(20, 3) > 0.5).astype(np.float)
    metric = Metric()
    print(metric(pred_scores, gt_labels))
