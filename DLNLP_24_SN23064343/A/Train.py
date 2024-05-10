
import os,random
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
from transformers import AutoTokenizer,AutoModel
import torch
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from Model import HateDetector
from Dataset import MyDataset


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train(model,train_dataloader,val_dataloader,epochs,lr,criterion,out_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to GPU
    model = model.to(device)

    # Define the Optimizer
    optimizer = AdamW(model.parameters(), lr=lr)

    print("Start Training...")
    # Train model
    train_epoch_loss,val_epoch_loss = [],[]
    train_acc,train_pre,train_rec,train_f1 = [],[],[],[]
    val_acc,val_pre,val_rec,val_f1 = [],[],[],[]

    for epoch in range(epochs):
    # ========================================
    #               Training
    # ========================================
        print(f"\nEpoch: {epoch+1}")
        model.train()
        epoch_loss = []
        epoch_acc,epoch_pre,epoch_rec,epoch_f1 = [],[],[],[]
        step = 0
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            # Push the batch to gpu
            batch = [t.to(device) for t in batch]
            input_ids,token_type_ids, attention_mask, labels = batch
            outputs = model(input_ids=input_ids.long(),
                            token_type_ids=token_type_ids.long(),
                            attention_mask=attention_mask.long(),
                            labels=labels)

            del batch,input_ids,token_type_ids, attention_mask,
            torch.cuda.empty_cache()

            # Calculate Loss
            loss = criterion(outputs, labels)
            # Gradient Calculation
            loss.backward()
            epoch_loss.append(loss.item())

            # Clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update
            optimizer.step()
            step += 1
            if step % 500 == 0:
                print("loss:"+str(np.array(epoch_loss).mean()))

            # Record ACC,PRE,REC,F1
            y_pred = outputs.argmax(dim=1).cpu()
            labels = labels.cpu()
            epoch_acc.append(accuracy_score(labels,y_pred))
            epoch_pre.append(precision_score(labels,y_pred))
            epoch_rec.append(recall_score(labels,y_pred))
            epoch_f1.append(f1_score(labels,y_pred))

            # Free up GPU memory
            del labels, loss,
            torch.cuda.empty_cache()
        
        # Compute Epoch Training Loss
        avg_loss = np.array(epoch_loss).mean()
        train_epoch_loss.append(avg_loss)
        print(f'Training Loss: {avg_loss:.4f}')

        # Compute Epoch ACC,PRE,REC,F1
        avg_acc,avg_pre,avg_rec,avg_f1 = np.array(epoch_acc).mean(),np.array(epoch_pre).mean(),np.array(epoch_rec).mean(),np.array(epoch_f1).mean()
        train_acc.append(avg_acc)
        train_pre.append(avg_pre)
        train_rec.append(avg_rec)
        train_f1.append(avg_f1)
        print(f'Accuracy: {avg_acc:.4f}, Precision: {avg_pre:.4f}, Recall: {avg_rec:.4f}, F1 Score: {avg_f1:.4f}')

        # Save Checkpoints
        if (epoch+1)%2==0:
            save_path = out_path+"/Epoch_"+str(epoch+1)+"_model.pth"
            torch.save(model.state_dict(),save_path)
        
        # ========================================
        #               Validation
        # ========================================
        print('Evaluate on Validation set...')
        model.eval()
        loss_val = []
        epoch_acc,epoch_pre,epoch_rec,epoch_f1 = [],[],[],[]
        for batch in tqdm(val_dataloader, desc="Evaluating"):
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

                # validation loss
                loss = criterion(outputs, labels)
                loss_val.append(loss.item())

                y_pred = outputs.argmax(dim=1).cpu()
                labels = labels.cpu()
                epoch_acc.append(accuracy_score(labels,y_pred))
                epoch_pre.append(precision_score(labels,y_pred))
                epoch_rec.append(recall_score(labels,y_pred))
                epoch_f1.append(f1_score(labels,y_pred))

            del labels, loss,
            torch.cuda.empty_cache()
        val_avg_loss = np.array(loss_val).mean()
        val_epoch_loss.append(val_avg_loss)
        print(f'Validation Loss: {val_avg_loss:.4f}')

        avg_acc,avg_pre,avg_rec,avg_f1 = np.array(epoch_acc).mean(),np.array(epoch_pre).mean(),np.array(epoch_rec).mean(),np.array(epoch_f1).mean()
        val_acc.append(avg_acc)
        val_pre.append(avg_pre)
        val_rec.append(avg_rec)
        val_f1.append(avg_f1)
        print(f'Accuracy: {avg_acc:.4f}, Precision: {avg_pre:.4f}, Recall: {avg_rec:.4f}, F1 Score: {avg_f1:.4f}')






