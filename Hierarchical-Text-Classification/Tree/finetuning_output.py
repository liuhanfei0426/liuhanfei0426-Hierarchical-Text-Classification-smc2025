import warnings
from basic import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, AdamW, BertModel
import numpy as np
import sys
import random
from contextlib import redirect_stdout

from model import MeNDT  # 导入自定义模型

# 保存控制台内容为txt文件
# 计算权重 输入损失函数
# 使用 micro f1 保存最佳模型

warnings.filterwarnings("ignore")


# 设置随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 定义各种训练策略
model_origin_list = ['huggingface', 'pt']
save_base_list = ['best_loss', 'best_f1']
loss_list = ['BCELogits', 'MSE', 'MultiLabelSoftMarginLoss']
learning_rate_list = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]
logit_list = [0.5]

model_origin = model_origin_list[1]
save_base = save_base_list[0]
loss_select = loss_list[1]
backbone_frozen = False     # True
learning_rate = learning_rate_list[2]
logit = logit_list[0]

label_num = 23
num_epochs = 100  # 训练轮数
patience = 20  # 早停策略的耐心值，即多少个 epoch 没有改进后停止
train_batch_size = 32


# 读取训练集和验证集
train_texts, train_labels = read_tsv('train_data.tsv', label_num)
val_texts, val_labels = read_tsv('val_data.tsv', label_num)

# 使用 BERT 分词器
tokenizer = BertTokenizer.from_pretrained('Chinese-MentalBERT')

# 初始化自定义模型
if model_origin == 'huggingface':
    base_model = BertModel.from_pretrained('Chinese-MentalBERT')
elif model_origin == 'pt':
    pt_model_path = 'pt_model.pt'
    base_model = BertModel.from_pretrained('Chinese-MentalBERT')
    # 不能用 因为我的微调模型是用BertForSequenceClassification训练的，比BertModel多一个分类器层，要先过滤。
    # base_model.load_state_dict(torch.load(pt_model_path))

    # state_dict = torch.load(pt_model_path, map_location=torch.device('cpu'))
    state_dict = torch.load(pt_model_path)
    # 过滤掉不需要的键
    def filter_keys(state_dictt, base_modell):
        model_keys = set(base_modell.state_dict().keys())
        # 只保留在base_model中出现的键
        filtered_state_dict = {k: v for k, v in state_dictt.items() if k in model_keys}
        return filtered_state_dict

    state_dict = filter_keys(state_dict, base_model)
    base_model.load_state_dict(state_dict, strict=False)

else:
    raise Exception('Undefined model_origin')

model = MeNDT(base_model, tokenizer, input_size=768, excel_file_path='decision_rules.csv')

# Freeze the base model parameters
for param in base_model.parameters():
    if backbone_frozen:
        param.requires_grad = False
    else:
        param.requires_grad = True


# 创建训练集和验证集的数据集对象
train_dataset = TextDataset(train_texts, train_labels, tokenizer)
val_dataset = TextDataset(val_texts, val_labels, tokenizer)



# 将控制台输出内容保存为txt文件
# best_model_path = f"{save_base}_loss={loss_select}_frozen={backbone_frozen}_logits={logit}_lr={learning_rate}.pt"
best_model_path = f"{model_origin}_{save_base}_loss={loss_select}_frozen={backbone_frozen}_logits={logit}_lr={learning_rate}_patience={patience}.pt"
output_file = f"output_finetuning_{best_model_path}.txt"
with open(output_file, "w") as f:
    with redirect_stdout(f):
        # 设置训练参数
        best_f1 = 0  # 最佳 F1 分数
        no_improvement_count = 0  # 没有改进的 epoch 计数器
        best_val_loss = float('inf')

        # 更新模型名字
        print("*************************************************************************")
        print(f"Setting:\n model_origin: {model_origin}\n Num Epochs: {num_epochs}\n Save_base: {save_base}\n Loss Function: {loss_select}\n "
              f"Backbone Frozen: {backbone_frozen}\n Logits: {logit}\n Learning Rate: {learning_rate}\n Batch Size: {train_batch_size}")
        print("*************************************************************************")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device("mps" if torch.backends.mps.is_available else "cpu")
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=learning_rate)

        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=train_batch_size)

        for epoch in range(num_epochs):
            print("\n--------Epoch-------", epoch+1)
            if no_improvement_count >= patience:
                print("Early stopping triggered. Stopping training.")
                break

            model.train()
            train_loss = 0

            # To store all the labels and predictions for the whole epoch
            all_train_labels = []
            all_train_predictions = []

            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask)
                child_output = outputs[0]
                parent_output = outputs[1]

                probabilities = torch.cat((child_output, parent_output), dim=1)
                binary_outputs = (probabilities > logit).float()

                if loss_select == 'BCELogits': loss_fn = torch.nn.BCEWithLogitsLoss()
                elif loss_select == 'MSE': loss_fn = torch.nn.MSELoss()
                elif loss_select == 'MultiLabelSoftMarginLoss': loss_fn = torch.nn.MultiLabelSoftMarginLoss()
                else: raise Exception('Undefined Loss')
                train_loss_bc = loss_fn(probabilities, labels)
                train_loss_bc.backward()
                optimizer.step()
                train_loss += train_loss_bc.item()

                # Collect labels and predictions for this batch
                all_train_labels.append(labels.cpu().numpy())
                all_train_predictions.append(binary_outputs.cpu().numpy())

            # Calculate metrics for the entire training epoch
            all_train_labels = np.concatenate(all_train_labels, axis=0)
            all_train_predictions = np.concatenate(all_train_predictions, axis=0)
            train_child_labels = all_train_labels[:, :19]
            train_parent_labels = all_train_labels[:, 19:]
            train_child_predictions = all_train_predictions[:, :19]
            train_parent_predictions = all_train_predictions[:, 19:]

            train_child_metrics = calculate_metrics(train_child_labels, train_child_predictions)
            train_parent_metrics = calculate_metrics(train_parent_labels, train_parent_predictions)
            train_metrics = calculate_metrics(all_train_labels, all_train_predictions)

            print(f"[Training] Loss: {train_loss / len(train_loader)}")
            print("Child Micro: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(train_child_metrics['micro']['precision'], train_child_metrics['micro']['recall'], train_child_metrics['micro']['f1']))
            print("Parent Micro: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(train_parent_metrics['micro']['precision'], train_parent_metrics['micro']['recall'], train_parent_metrics['micro']['f1']))
            print("Overall Micro: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(train_metrics['micro']['precision'], train_metrics['micro']['recall'], train_metrics['micro']['f1']))

            model.eval()
            val_loss = 0

            # To store all the labels and predictions for the whole epoch
            all_val_labels = []
            all_val_predictions = []

            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                with torch.no_grad():
                    outputs = model(input_ids, attention_mask=attention_mask)
                    child_output = outputs[0]
                    parent_output = outputs[1]

                    probabilities = torch.cat((child_output, parent_output), dim=1)
                    binary_outputs = (probabilities > logit).float()

                    val_loss_bc = loss_fn(probabilities, labels)
                    val_loss += val_loss_bc.item()

                    # Collect labels and predictions for this batch
                    all_val_labels.append(labels.cpu().numpy())
                    all_val_predictions.append(binary_outputs.cpu().numpy())

            # Calculate metrics for the entire validation epoch
            all_val_labels = np.concatenate(all_val_labels, axis=0)
            all_val_predictions = np.concatenate(all_val_predictions, axis=0)
            val_child_labels = all_val_labels[:, :19]
            val_parent_labels = all_val_labels[:, 19:]
            val_child_predictions = all_val_predictions[:, :19]
            val_parent_predictions = all_val_predictions[:, 19:]
            val_child_metrics = calculate_metrics(val_child_labels, val_child_predictions)
            val_parent_metrics = calculate_metrics(val_parent_labels, val_parent_predictions)
            val_metrics = calculate_metrics(all_val_labels, all_val_predictions)

            val_loss /= len(val_loader)
            print(f"\n[Validation] Loss: {val_loss}")
            print("Child Micro: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(val_child_metrics['micro']['precision'], val_child_metrics['micro']['recall'],val_child_metrics['micro']['f1']))
            print("Parent Micro: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(val_parent_metrics['micro']['precision'], val_parent_metrics['micro']['recall'],val_parent_metrics['micro']['f1']))
            print("Overall Micro: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(val_metrics['micro']['precision'], val_metrics['micro']['recall'], val_metrics['micro']['f1']))


            if save_base == 'best_loss':
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improvement_count = 0
                    # best_model_path = f"{save_base_list[0]}_loss={loss_select}_frozen={backbone_frozen}_logits={logit}_lr={learning_rate}.pt"
                    torch.save(model.state_dict(), best_model_path)
                    print(f"New best model saved at epoch {epoch + 1} with Validation Loss: {val_loss}")
                else:
                    no_improvement_count += 1

            elif save_base == 'best_f1':
                if val_metrics['micro']['f1'] > best_f1:
                    best_f1 = val_metrics['micro']['f1']
                    no_improvement_count = 0
                    # best_model_path = f"{save_base_list[1]}_loss={loss_select}_frozen={backbone_frozen}_logits={logit}_lr={learning_rate}.pt"
                    torch.save(model.state_dict(), best_model_path)
                    print(f"New best model saved at epoch {epoch + 1} with F1_micro: {val_metrics['micro']['f1']}")
                else:
                    no_improvement_count += 1
            else:
                raise Exception('Undefined save_base')


        print("Training Finished!")

print("Training process save to: ", output_file)

