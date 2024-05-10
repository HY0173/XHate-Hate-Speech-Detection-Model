import torch
import torch.nn as nn
from transformers import AutoModel

encoder = AutoModel.from_pretrained("vinai/bertweet-base")

# Model Architecture
# Classifier_1
class HateDetector(nn.Module):
    def __init__(self,encoder,label_num):
        super(HateDetector,self).__init__()
        self.label_num = label_num
        #1.BERTweet is used as the feature extractor for embedding.
        self.feature = encoder

        #2.Define the Classifier
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(128, self.label_num),
            nn.Softmax()
        )

    def forward(self, input_ids,token_type_ids,attention_mask,labels=None):
        # output shape of BERTweet: [seq_len, 768]
        outputs = self.feature(input_ids=input_ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask)
        
        # [CLS] token
        pooler_output = outputs[1]
        out = self.classifier(pooler_output)

        return out


# Classifier_2
class HateDetector_CNN(nn.Module):
    def __init__(self,encoder,label_num):
        super(HateDetector_CNN,self).__init__()
        self.label_num = label_num
        #1.BERTweet is used as the feature extractor for embedding.
        self.feature = encoder

        #2.Define a CNN classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=13, out_channels=13, kernel_size=(3, 768), padding=True),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Dropout(0.1),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(442, self.label_num),
            nn.Softmax()
        )

    def forward(self, input_ids,token_type_ids,attention_mask,labels=None):
        _, _, all_layers = self.feature(input_ids=input_ids,
                                         token_type_ids=token_type_ids,
                                         attention_mask=attention_mask, 
                                         output_hidden_states=True)
        
        # all_layers  = [13, 32, 64, 768]
        x = torch.transpose(torch.cat(tuple([t.unsqueeze(0) for t in all_layers]), 0), 0, 1)
        del all_layers,
        torch.cuda.empty_cache()

        out = self.classifier(x)
        return out
