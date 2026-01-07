import os
import glob
import gc
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import PatchTSTConfig, PatchTSTForPrediction
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datasets import Dataset

## custom loss function
class MASE(torch.nn.Module):
    def __init__(self, training_data, period = 1):
        super().__init__()
        ## 원본 코드 구현, 사실상 MAE와 동일, 잘못 짜여진 코드, 일단은 하던대로 할 것.
        self.scale = torch.mean(torch.abs(torch.tensor(training_data[period:] - training_data[:-period])))
    
    def forward(self, yhat, y):
        error = torch.abs(y - yhat)
        return torch.mean(error) / self.scale

def SMAPE(yhat, y):
    numerator = 100*torch.abs(y - yhat)
    denominator = (torch.abs(y) + torch.abs(yhat))/2
    smape = torch.mean(numerator / denominator)
    return smape

def MAPE_pretrained(yhat, y):
    ## M4 데이터셋에는 0이 없음을 확인: 정상적으로 훈련 가능.
    ## 아래의 MAPE와 근본적으로 동일하나, 속도 향상을 위해 따로 처리
    return torch.mean(100*torch.abs((y - yhat) / y))

def MAPE(y_pred, y_true, epsilon=1e-7):
    ## 분모에 0이 들어오는 것을 방지. 문제가 많지만, 케라스 코드를 그대로 이식했음 -> 어차피 중앙값 차원에서 걸러질 듯.
    denominator = torch.clamp(torch.abs(y_true), min=epsilon)
    abs_percent_error = torch.abs((y_true - y_pred) / denominator)

    return torch.mean(100. * abs_percent_error)


def savePredsAndTruth(yhat, y, loss_name, ith):
    """
    Pretrained Model에서 Prediction과 Ground Truth Log 저장 (훈련 후 호출)
    """
    yhat, y = pd.DataFrame(yhat.to("cpu")), pd.DataFrame(y.to("cpu"))   ## 데이터프레임으로 만들거임
    yhat.columns = [f"{i}A" for i in range(yhat.shape[1])]
    y.columns = [f"{i}B" for i in range(y.shape[1])]

    val_result = pd.concat([yhat, y], axis = 1).sort_index(axis = 1)
    val_result.columns = [f"prediction_{(i+1)//2}" if i%2 == 1 else f"ground_truth_{(i+1)//2}" for i in range(1, val_result.shape[1]+1)]
    val_result.to_csv(os.path.join(log_dir, f"prediction_val_results_{loss_name}_model{ith}.csv"), index = False)
    

def scratchTraining(loss_name, ith):
    ## bootstrap
    np.random.seed()
    select = np.random.choice(len(target_X), size=len(target_X), replace=True)
    X_bootstrap = target_X[select]
    y_bootstrap = target_y[select]

    def to_tensor_and_reshape(array):
        result = torch.tensor(array)
        result = result.reshape(-1, result.shape[1], 1)

        return result

    X_train, X_valid = to_tensor_and_reshape(X_bootstrap), to_tensor_and_reshape(target_X_val)
    y_train, y_valid = to_tensor_and_reshape(y_bootstrap), to_tensor_and_reshape(target_y_val)

    ## setting dataloader
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = 8, shuffle = True, num_workers = 2)

    val_dataset = torch.utils.data.TensorDataset(X_valid, y_valid)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = 8, num_workers = 2)

    backbone_model = PatchTSTForPrediction.from_pretrained(os.path.join(output_dir, backbone_name)).to(device)  ## load to gpu

    if loss_name == "mse":
        loss_fn = torch.nn.MSELoss()
    elif loss_name == "mae":
        loss_fn = torch.nn.L1Loss()
    elif loss_name == "SMAPE":
        loss_fn = SMAPE
    elif loss_name == "mape":
        loss_fn = MAPE
    elif loss_name == "MASE":
        loss_fn = MASE(target_y, target_y.shape[1])
    else:
        raise Exception("Your loss name is not valid.")

    optimizer = torch.optim.AdamW(backbone_model.parameters(), lr = learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = num_train_epochs)
    log_data = []

    ## early stopping
    PATIENCE = 10
    best_val_loss = np.inf
    patience_counter = 0

    for epoc in range(num_train_epochs):
        backbone_model.train()

        total_train_loss = 0

        for X, y in train_dataloader:
            X, y = X.to(device), y.to(device)   ## load to gpu

            optimizer.zero_grad()
            yhat = backbone_model(X).prediction_outputs
            loss = loss_fn(yhat, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(backbone_model.parameters(), max_norm = 1.0)
            optimizer.step()

            total_train_loss += loss.item()*X.shape[0]

        avg_train_loss = total_train_loss / len(train_dataloader.dataset)

        backbone_model.eval()

        with torch.no_grad():
            yys = []
            yyhats = []

            for XX, yy in test_dataloader:
                XX = XX.to(device)
                yys.append(yy.to(device))
                yyhats.append(backbone_model(XX).prediction_outputs)

            yyhat = torch.concat(yyhats)
            yy = torch.concat(yys)

            val_loss = loss_fn(yyhat, yy)

        print(f"Epoch {epoc+1}/{num_train_epochs} | Train Loss: {avg_train_loss:.6f}\t\t Val Loss: {val_loss:.6f}")

        log_data.append({"epoch": epoc, "loss": avg_train_loss, "eval_loss": val_loss.item()})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = backbone_model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

        scheduler.step()

    ## save log
    pd.DataFrame(log_data).to_csv(os.path.join(log_dir, f"scratch_{loss_name}_model{ith}.csv"))

    ## load best model
    backbone_model.load_state_dict(best_state_dict)

    with torch.no_grad():
        yyhats = []
        yys = []

        for XX, yy in test_dataloader:
            XX = XX.to(device)
            yys.append(yy.to(device))
            yyhats.append(backbone_model(XX).prediction_outputs)

        yyhat, yy = torch.concat(yyhats).squeeze(), torch.concat(yys).squeeze()
        model_pred_test = yyhat.to("cpu")

        yyhats = []
        yys = []
        
        for XX, yy in val_dataloader:
            XX = XX.to(device)
            yys.append(yy.to(device))
            yyhats.append(backbone_model(XX).prediction_outputs)

        yyhat, yy = torch.concat(yyhats).squeeze(), torch.concat(yys).squeeze()
        model_pred_val = yyhat.to("cpu")

    del backbone_model
    torch.cuda.empty_cache()
    gc.collect()

    return model_pred_val, model_pred_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "data, learning_rate, model_num")

    parser.add_argument("--model_num", type = int, default = 100, help = "Model k per loss")
    parser.add_argument("--data", type = str, default = "coin", help = "target dataset name")
    parser.add_argument("--lr", type = float, default = 1e-4, help = "Scratch model learning rate")
    parser.add_argument("--backbone", type = str, default = "PatchTSTBackbone", help = "backbone model name")
    parser.add_argument("--transfer_loss", type = str, default = "all", help = "transfer loss type")
    parser.add_argument("--device_id", type = int, default = 0, help = "GPU id")

    args = parser.parse_args()

    data = args.data
    backbone_name = args.backbone

    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")

    output_dir = "saved_models"
    log_dir = f'logs/{data}/scratch'
    learning_rate = args.lr
    model_num = args.model_num

    os.makedirs(output_dir, exist_ok = True)
    os.makedirs(log_dir, exist_ok=True)

    num_train_epochs = 2000

    ## target domain
    target_X = pd.read_csv(f"../data/{data}/train_input_7.csv").iloc[:, 1:].values.astype(np.float32)
    target_y = pd.read_csv(f"../data/{data}/train_output_7.csv").iloc[:, 1:].values.astype(np.float32)

    target_X_val = target_X[-round(target_X.shape[0] * 0.2):, :].astype(np.float32)
    target_y_val = target_y[-round(target_y.shape[0] * 0.2):].astype(np.float32)
    target_X = target_X[:-round(target_X.shape[0] * 0.2), :].astype(np.float32)
    target_y = target_y[:-round(target_y.shape[0] * 0.2)].astype(np.float32)

    test_X  = pd.read_csv(f"../data/{data}/val_input_7.csv").iloc[:, 1:].values.astype(np.float32)
    test_y  = pd.read_csv(f"../data/{data}/val_output_7.csv").iloc[:, 1:].values.astype(np.float32)

    def array_to_dataset(X, y):
        X, y = torch.tensor(X), torch.tensor(y)
        X = X.reshape(-1, X.shape[1], 1)
        y = y.reshape(-1, y.shape[1], 1)

        dataset = torch.utils.data.TensorDataset(X, y)

        return dataset
    
    test_dataset = array_to_dataset(test_X, test_y)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 64)

    os.makedirs(f"result/{data}/val", exist_ok = True)
    os.makedirs(f"result/{data}/test", exist_ok = True)


    #### ========== Generate PatchTST(Scratch) Architecture ==========
    if not os.path.isdir(os.path.join(output_dir, backbone_name)):
        TSTconfig = PatchTSTConfig(
            num_input_channels = 1,
            context_length = 168,
            prediction_length = 24,
            patch_length = 24,
            patch_stride = 24,
            d_model = 256,
            num_attention_heads = 8,
            num_hidden_layers = 8,
            ffn_dim = 1024,
            dropout = 0.2,
            head_dropout = 0.2,
            pooling_type = None,
            channel_attention = False,
            scaling = "std",
            pre_norm = True,
            do_mask_input = False
        )

        model = PatchTSTForPrediction(TSTconfig)
        model.save_pretrained(os.path.join(output_dir, backbone_name))

        print("Backbone Architecture is succesfully generated.")

    else:
        print("Backbone Architecture is already generated.")

    
    #### ========== Scratch Training ==========
    val_gts = {}
    val_preds = {}
    test_preds = {}

    ## 변수 이름 설정에 일관성이 없네
    save_name = ["mse", "mae", "mase", "mape", "smape"]

    for i, loss_name in enumerate(["mse", "mae", "MASE", "mape", "SMAPE"]):
        print(f"Start to training with {loss_name}.")

        preds_val = []
        preds_test = []

        for ith in range(1, model_num+1):
            pred_val, pred_test = scratchTraining(loss_name = loss_name, ith = ith)
            preds_val.append(pred_val)
            preds_test.append(pred_test)

            torch.cuda.empty_cache()
            gc.collect()

        ## 최종 결과 저장
        pd.DataFrame(np.array(preds_val).reshape(1, -1)).to_csv(f"result/{data}/val/Scratch_{data}_{save_name[i]}_pred.csv")
        pd.DataFrame(np.array(preds_test).reshape(1, -1)).to_csv(f"result/{data}/test/Scratch_{data}_{save_name[i]}_pred.csv")