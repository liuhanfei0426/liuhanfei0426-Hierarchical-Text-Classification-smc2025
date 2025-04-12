import warnings
import torch
from transformers import BertTokenizer, BertModel
from model import MeNDT  # 导入自定义模型
from basic import *

warnings.filterwarnings("ignore")

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义各种训练策略
model_origin_list = ['huggingface', 'pt']
save_base_list = ['best_loss', 'best_f1']
loss_list = ['BCELogits', 'MSE']
learning_rate_list = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]
logit_list = [0.3, 0.5]

model_origin = model_origin_list[1]
save_base = save_base_list[0]
loss_select = loss_list[1]
backbone_frozen = False     # True
learning_rate = learning_rate_list[2]
logit = logit_list[1]

label_num = 23
num_epochs = 100  # 训练轮数
patience = 20  # 早停策略的耐心值，即多少个 epoch 没有改进后停止
train_batch_size = 32


# 使用 BERT 分词器
tokenizer = BertTokenizer.from_pretrained('../Chinese-MentalBERT')

# 初始化自定义模型
if model_origin == 'huggingface':
    base_model = BertModel.from_pretrained('../Chinese-MentalBERT')
elif model_origin == 'pt':
    pt_model_path = 'pt_model.pt'
    base_model = BertModel.from_pretrained('../Chinese-MentalBERT')
    # 不能用 因为我的微调模型是用BertForSequenceClassification训练的，比BertModel多一个分类器层，要先过滤。
    # base_model.load_state_dict(torch.load(pt_model_path))

    state_dict = torch.load(pt_model_path, map_location=torch.device('cpu'))
    # state_dict = torch.load(pt_model_path)
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

best_model_path = f"{model_origin}_{save_base}_loss={loss_select}_frozen={backbone_frozen}_logits={logit}_lr={learning_rate}.pt"
model.load_state_dict(torch.load(best_model_path))

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def predict(text):
    # 文本预处理
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    all_test_predictions = []
    # 模型预测
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        child_output = outputs[0]
        parent_output = outputs[1]
        print("child_output: ", child_output)
        print("parent_output: ", parent_output)

        probabilities = torch.cat((child_output, parent_output), dim=1)
        binary_outputs = (probabilities > 0.5).float()

        # Collect labels and predictions for this batch
        # all_test_labels.append(labels.cpu().numpy())
        all_test_predictions.append(binary_outputs.cpu().numpy())

        all_test_predictions = np.concatenate(all_test_predictions, axis=0)

        test_child_predictions = all_test_predictions[:, :19]
        test_parent_predictions = all_test_predictions[:, 19:]

        print("child_predictions: ", test_child_predictions)
        print("parent_predictions: ", test_parent_predictions)

# 示例文本
text = "我有办法，我计划在学校里做，比如在厕所或某个角落走廊的空闲时间。"
print("exp_text: ", text)
# 进行预测
predict(text)