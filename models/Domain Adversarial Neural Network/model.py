import torch.nn as nn
from torch.autograd import Function
from transformers import AutoModel, AutoConfig
from config import TOKENIZER_NAME

class GradReverse(Function):
    @staticmethod
    def forward(context, x, lambda_):
        # Save lambda for backpropagation
        context.lambda_ = lambda_
        # Return unchanged unput
        return x.view_as(x)

    @staticmethod
    def backward(context, grad_output):
        # Return reversed gradient for x and None for lambda_
        return grad_output.neg() * context.lambda_, None
    

def grl(x, lambda_):
    return GradReverse.apply(x, lambda_)


class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super().__init__()

        enc_cfg = AutoConfig.from_pretrained(
            TOKENIZER_NAME,
            hidden_dropout_prob=0.2,
            attention_probs_dropout_prob=0.2,
        )

        self.encoder = AutoModel.from_pretrained(TOKENIZER_NAME, config=enc_cfg)

        self.feature = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        pooled = outputs.pooler_output

        features = self.feature(pooled)

        return features
    

class LabelPredictor(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(LabelPredictor, self).__init__()
        # Fully Connected Layer
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, features):
        return self.classifier(features)
    

class DomainClassifier(nn.Module):
    def __init__(self, feature_dim, num_domains):
        super(DomainClassifier, self).__init__()
        self.discriminator = nn.Sequential(
            # Fully Connected Layer
            nn.Linear(feature_dim, 64),
            # Batch Normalization
            nn.BatchNorm1d(64),
            # Activation Function
            nn.ReLU(inplace=True),
            # Dropout Layer: Zeroes x% of activations during training
            nn.Dropout(0.6),
            # Fully Connected Layer
            nn.Linear(64, num_domains)
        )

    def forward(self, features, lambda_):
        # Pass features through Gradient Reversal Layer (grl)
        feat = grl(features, lambda_)
        return self.discriminator(feat)
    

class DANN(nn.Module):
    def __init__(self, input_dim, feature_dim, num_classes, num_domains):
        super(DANN, self).__init__()
        self.feature_extractor = FeatureExtractor(input_dim, feature_dim)
        self.label_predictor  = LabelPredictor(feature_dim, num_classes)
        self.domain_classifier = DomainClassifier(feature_dim, num_domains)

    def forward(self, x, lambda_):
        input_ids, attention_mask = x

        features = self.feature_extractor(input_ids, attention_mask)

        class_logits = self.label_predictor(features)
        domain_logits = self.domain_classifier(features, lambda_)
        
        return class_logits, domain_logits