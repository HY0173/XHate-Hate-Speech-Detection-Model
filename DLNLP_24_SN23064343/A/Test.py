from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
# ========================================
#               Testing
# ========================================
def test(model,test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Start Testing...")
    model.eval()
    acc,pre,rec,f1 = [],[],[],[]
    for batch in tqdm(test_dataloader, desc="Evaluating"):
        with torch.no_grad():
            batch = [t.to(device) for t in batch]
            input_ids,token_type_ids, attention_mask, labels = batch
            outputs = model(input_ids=input_ids.long(),
                            token_type_ids=token_type_ids.long(),
                            attention_mask=attention_mask.long(),
                            labels=labels)

            # Free up GPU memory
            del batch,input_ids,token_type_ids, attention_mask,
            torch.cuda.empty_cache()

            y_pred = outputs.argmax(dim=1).cpu()
            labels = labels.cpu()
            acc.append(accuracy_score(labels,y_pred))
            pre.append(precision_score(labels,y_pred))
            rec.append(recall_score(labels,y_pred))
            f1.append(f1_score(labels,y_pred))

    avg_acc,avg_pre,avg_rec,avg_f1 = np.array(acc).mean(),np.array(pre).mean(),np.array(rec).mean(),np.array(f1).mean()

    print(f'Accuracy: {avg_acc:.4f}, Precision: {avg_pre:.4f}, Recall: {avg_rec:.4f}, F1 Score: {avg_f1:.4f}')