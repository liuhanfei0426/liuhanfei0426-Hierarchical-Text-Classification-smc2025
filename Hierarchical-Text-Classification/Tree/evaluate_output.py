import warnings
import torch
from transformers import BertTokenizer, BertModel
from contextlib import redirect_stdout
from model import MeNDT  # 导入自定义模型
from basic import *

warnings.filterwarnings("ignore")

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


# 读取测试集
test_texts, test_labels = read_tsv('test_data.tsv', label_num)
# 创建测试集的数据集对象
test_dataset = TextDataset(test_texts, test_labels, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=train_batch_size)


all_best_f1_micro = 0

# 将控制台输出内容保存为txt文件
best_model_path = f"{model_origin}_{save_base}_loss={loss_select}_frozen={backbone_frozen}_logits={logit}_lr={learning_rate}_patience={patience}.pt"

output_file = f"output_evaluate_{best_model_path}.txt"
with open(output_file, "w") as f:
    with redirect_stdout(f):
        print("*************************************************************************")
        print(f"当前模型: {best_model_path}")
        print("*************************************************************************")
        model.load_state_dict(torch.load(best_model_path))
        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        all_test_labels = []
        all_test_predictions = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                child_output = outputs[0]
                parent_output = outputs[1]

                probabilities = torch.cat((child_output, parent_output), dim=1)
                binary_outputs = (probabilities > logit).float()

                # Collect labels and predictions for this batch
                all_test_labels.append(labels.cpu().numpy())
                all_test_predictions.append(binary_outputs.cpu().numpy())

        # Calculate metrics for the entire test set
        all_test_labels = np.concatenate(all_test_labels, axis=0)
        all_test_predictions = np.concatenate(all_test_predictions, axis=0)
        test_child_labels = all_test_labels[:, :19]
        test_parent_labels = all_test_labels[:, 19:]
        test_child_predictions = all_test_predictions[:, :19]
        test_parent_predictions = all_test_predictions[:, 19:]

        test_child_metrics = calculate_metrics(test_child_labels, test_child_predictions)
        test_parent_metrics = calculate_metrics(test_parent_labels, test_parent_predictions)
        test_metrics = calculate_metrics(all_test_labels, all_test_predictions)

        if test_metrics['micro']['f1'] > all_best_f1_micro:
            all_best_f1_micro = test_metrics['micro']['f1']
            # 保存预测的结果
            np.savetxt('gt_23.tsv', all_test_labels.astype(int), delimiter='\t', fmt='%d')
            np.savetxt('predict_23.tsv', all_test_predictions.astype(int), delimiter='\t', fmt='%d')
            print(f"{best_model_path} 模型的真实结果gt_23和预测结果predict_23已保存！")

        print("\n统计总体的指标:")
        print("Parent Micro: Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(test_parent_metrics['micro']['precision']*100,
                                                                                   test_parent_metrics['micro']['recall']*100,
                                                                                   test_parent_metrics['micro']['f1']*100))
        print("Child Micro: Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(test_child_metrics['micro']['precision']*100,
                                                                                  test_child_metrics['micro']['recall']*100,
                                                                                  test_child_metrics['micro']['f1']*100))
        print("Overall Micro: Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(test_metrics['micro']['precision']*100,
                                                                                    test_metrics['micro']['recall']*100,
                                                                                    test_metrics['micro']['f1']*100))

        parent_per_metrics = calculate_evaluation_per_class(test_parent_labels, test_parent_predictions)
        print("\n分别统计父类的指标:")
        for key, value in parent_per_metrics.items():
             print(
                f"{key+1}: Precision: {value['precision']*100:.2f}, Recall: {value['recall']*100:.2f}, F1: {value['f1']*100:.2f}")


        child_per_metrics = calculate_evaluation_per_class(test_child_labels, test_child_predictions)
        print("\n分别统计子类的指标:")
        for key, value in child_per_metrics.items():
             print(
                f"{key+1}: Precision: {value['precision']*100:.2f}, Recall: {value['recall']*100:.2f}, F1: {value['f1']*100:.2f}")

print("输出内容已保存为: ", output_file)


