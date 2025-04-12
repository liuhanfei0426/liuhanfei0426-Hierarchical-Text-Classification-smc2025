import torch
import torch.nn as nn
import pandas as pd
# from torchviz import make_dot
from transformers import BertTokenizer, BertModel

class process_decison_node(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(process_decison_node, self).__init__()
        self.process = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(input_size, input_size),
            nn.ReLU(),
        )
        self.decision = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(input_size, output_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        process = self.process(x)
        decision = self.decision(process)
        return process, decision

class decison_node(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(decison_node, self).__init__()
        self.decision = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(input_size, output_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        decision = self.decision(x)
        return decision

class MeNDT(nn.Module):
    def __init__(self, base_model, tokenizer, input_size, excel_file_path):
        super(MeNDT, self).__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer

        # read Excel file
        df = pd.read_csv(excel_file_path)
        self.tree_dict = df.groupby('Parent')['Child'].apply(list).to_dict()
        # Create regions and nodes as per the tree structure
        for region, nodes in self.tree_dict.items():
            # Add a processing decision node for each region
            self.add_module(region, process_decison_node(input_size))

            # Add a decision node for each child node under current region
            for node in nodes:
                self.add_module(node, decison_node(input_size))

    def forward(self, input_ids, attention_mask):
        # Extract the hidden states from the base BERT model
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        print("outputs: ", outputs)
        x = outputs.last_hidden_state[:, 0, :]  # Use the [CLS] token's hidden state
        print("x: ", x)
        child_decisions = []
        parent_decisions = []

        # Process each region and its nodes
        for region, nodes in self.tree_dict.items():
            region_module = getattr(self, region)
            region_process, parent_decision = region_module(x)
            parent_decisions.append(parent_decision)
            # For each child node of current region
            for node in nodes:
                node_module = getattr(self, node)
                node_decision = node_module(region_process)

                # Calculate final decision for the node
                node_final_decision = node_decision * parent_decision
                child_decisions.append(node_final_decision)

        # Concatenate decisions
        child_decisions = torch.cat(child_decisions, dim=1)
        parent_decisions = torch.cat(parent_decisions, dim=1)
        return child_decisions, parent_decisions

def template_test():
    tokenizer = BertTokenizer.from_pretrained('../Chinese-MentalBERT')
    base_model = BertModel.from_pretrained('../Chinese-MentalBERT')

    # Freeze the base model parameters
    for param in base_model.parameters():
        param.requires_grad = False
    # Initialize the custom decision tree model
    model = MeNDT(base_model, tokenizer, input_size=768, excel_file_path='decision_rules.csv')
    # Tokenize the input text
    texts = ["example text 1", "example text 2"]
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # Forward pass through the model
    decisions = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

    print('Model child output:', decisions[0])
    print('Model parent output:', decisions[1])

    #make_dot(decisions, params=dict(list(model.named_parameters())), show_attrs=False, show_saved=False).render("MeNDT_Table", format="png")

if __name__ == '__main__':
    template_test()
