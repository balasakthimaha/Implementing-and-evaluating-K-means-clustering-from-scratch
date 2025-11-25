import numpy as np

def classification_metrics(y_true, y_pred):
    acc=(y_true==y_pred).mean()
    tp=np.sum((y_true==1)&(y_pred==1))
    fp=np.sum((y_true==0)&(y_pred==1))
    fn=np.sum((y_true==1)&(y_pred==0))
    precision=tp/(tp+fp) if tp+fp>0 else 0
    recall=tp/(tp+fn) if tp+fn>0 else 0
    return {"accuracy":acc,"precision":precision,"recall":recall}
