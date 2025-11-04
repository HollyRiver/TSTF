import os
import glob
import gc
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import PatchTSTConfig, PatchTSTForPrediction
from datasets import Dataset


## custom loss function
def SMAPE(yhat, y):
    numerator = 100*torch.abs(y - yhat)
    denominator = (torch.abs(y) + torch.abs(yhat))/2
    smape = torch.mean(numerator / denominator)
    return smape

def MAPE(yhat, y):
    return torch.mean(100*torch.abs((y - yhat) / y))

class MASE(torch.nn.Module):
    def __init__(self, training_data, period = 1):
        super().__init__()
        self.scale = torch.mean(torch.abs(training_data[:, period:] - data[:, :-period]))    ## 모든 훈련 데이터에 대한 평균 스케일 계산
    
    def forward(self, yhat, y):
        error = torch.abs(y - yhat)
        return torch.mean(error / self.scale)


def savePredsAndTruth(yhat, y, loss_name, ith):
    """
    Pretrained Model에서 Prediction과 Ground Truth Log 저장 (훈련 후 호출)
    """
    yhat, y = yhat.to("cpu"), y.to("cpu")   ## 데이터프레임으로 만들거임
    yhat.columns = [f"{i}A" for i in range(yhat.shape[1])]
    y.columns = [f"{i}B" for i in range(y.shape[1])]

    val_result = pd.concat([yhat, y], axis = 1).sort_index(axis = 1)
    val_result.columns = [f"prediction_{(i+1)//2}" if i%2 == 1 else f"ground_truth_{(i+1)//2}" for i in range(1, val_result.shape[1]+1)]
    val_result.to_csv(os.path.join(log_dir, f"prediction_val_results_{loss_name}_model{ith}.csv"), index = False)
    

def pretraining(loss_name, ith):
    ## bootstrap
    np.random.seed()
    select = np.random.choice(len(source_X), size=len(source_X), replace=True)
    X_bootstrap = source_X[select]
    y_bootstrap = source_y[select]

    val_split_index = int(len(X_bootstrap) * 0.8)

    def to_tensor_and_reshape(array):
        result = torch.tensor(array)
        result = result.reshape(-1, result.shape[1], 1)

        return result

    X_train, X_valid = to_tensor_and_reshape(X_bootstrap[:val_split_index]), to_tensor_and_reshape(X_bootstrap[val_split_index:])
    y_train, y_valid = to_tensor_and_reshape(y_bootstrap[:val_split_index]), to_tensor_and_reshape(y_bootstrap[val_split_index:])

    ## setting dataloader
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = 256, shuffle = True, num_workers = 16)

    test_dataset = torch.utils.data.TensorDataset(X_valid, y_valid)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 256, num_workers = 16)

    backbone_model = PatchTSTForPrediction.from_pretrained(os.path.join(output_dir, backbone_name)).to(device)  ## load to gpu

    if loss_name == "mse":
        loss_fn = torch.nn.MSELoss()
        lr = pretraining_lr
    else:
        lr = pretraining_lr*2
        if loss_name == "mae":
            loss_fn = torch.nn.L1Loss()
        elif loss_name == "SMAPE":
            loss_fn = SMAPE
        elif loss_name == "mape":
            loss_fn = MAPE
        elif loss_name == "MASE":
            loss_fn = MASE(source_y, 1)
        else:
            raise Exception("Your loss name is not valid.")

    optimizer = torch.optim.AdamW(backbone_model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = num_train_epochs)
    log_data = []

    ## early stopping
    PATIENCE = 15
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
            torch.save(backbone_model.state_dict(), os.path.join(output_dir, f"model_{loss_name}_{ith}.pth"))
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

        scheduler.step()

    ## save log
    pd.DataFrame(log_data).to_csv(os.path.join(log_dir, f"pretrain_{loss_name}_model{ith}.csv"))

    ## load best model
    backbone_model.load_state_dict(torch.load(os.path.join(output_dir, f"model_{loss_name}_{ith}.pth")))

    yyhats = []
    yys = []

    with torch.no_grad():
        for XX, yy in test_dataloader:
            XX = XX.to(device)
            yys.append(yy.to(device))
            yyhats.append(backbone_model(XX).prediction_outputs)

    yyhat, yy = torch.concat(yyhats).squeeze(), torch.concat(yys).squeeze()

    savePredsAndTruth(yyhat, yy, loss_name, ith)    ## f"prediction_val_results_{loss_name}_model{ith}.csv"


def transfer_FC(model_num, loss_name, batch_size):
    for i in range(1, model_num + 1):
        current_path = os.path.join(output_dir, f"model_{loss_name}_{i}.pth")

        backbone_model = PatchTSTForPrediction.from_pretrained(os.path.join(output_dir, "PatchTSTBackbone")).to(device)
        backbone_model.load_state_dict(torch.load(current_path))    ## 구조 변경 없이 그대로 로드

        optimizer = torch.optim.AdamW(backbone_model.parameters(), lr = learning_rate)
        log_data = []

        if loss_name == "mse":
            loss_fn = torch.nn.MSELoss()
            lr = learning_rate
        else:
            lr = learning_rate*2
            if loss_name == "mae":
                loss_fn = torch.nn.L1Loss()
            elif loss_name == "SMAPE":
                loss_fn = SMAPE
            elif loss_name == "mape":
                loss_fn = MAPE
            elif loss_name == "MASE":
                loss_fn = MASE(target_y, 1)
            else:
                raise Exception("Your loss name is not valid.")

        ## early stopping
        PATIENCE = 10
        best_val_loss = np.inf
        patience_counter = 0

        for epoc in range(num_train_epochs):
            backbone_model.train()

            total_train_loss = 0

            for X, y in train_dataloader:
                X, y = X.to(device), y.to(device)

                optimizer.zero_grad()
                yhat = backbone_model(X).prediction_outputs
                loss = loss_fn(yhat, y)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()*X.shape[0]

            avg_train_loss = total_train_loss/len(train_dataloader.dataset)

            backbone_model.eval()

            with torch.no_grad():
                yys = []
                yyhats = []

                for XX, yy in val_dataloader:
                    XX = XX.to(device)
                    yys.append(yy.to(device))
                    yyhats.append(backbone_model(XX).prediction_outputs)

                yyhat = torch.concat(yyhats)
                yy = torch.concat(yys)

                val_loss = loss_fn(yyhat, yy).item()

            print(f"Epoch {epoc+1}/{num_train_epochs} | Train Loss: {avg_train_loss:.6f}\t\t Val Loss: {val_loss:.6f}")

            log_data.append({"epoch": epoc, "loss": avg_train_loss, "eval_loss": val_loss})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = backbone_model.state_dict()   ## 저장 없이 결과물만 산출...
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                break

        backbone_model.load_state_dict(best_state_dict)

        pd.DataFrame(log_data).to_csv(os.path.join(log_dir, f"transfer_{loss_name}_lr{learning_rate}_run{1}.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "data, learning_rate, model_num")

    parser.add_argument("--model_num", type = int, default = 100, help = "Model k per loss")
    parser.add_argument("--data", type = str, default = "coin", help = "target dataset name")
    parser.add_argument("--lr", type = float, default = 1e-6, help = "transfer learning rate")
    parser.add_argument("--backbone", type = str, default = "PatchTSTBackbone", help = "backbone model name")

    args = parser.parse_args()

    data = args.data
    backbone_name = args.backbone
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = "saved_models"
    log_dir = os.path.join('logstf', data)
    learning_rate = args.lr
    pretraining_lr = 5e-5
    model_num = args.model_num

    os.makedirs(output_dir, exist_ok = True)
    os.makedirs(log_dir, exist_ok=True)

    num_train_epochs = 400

    ## target domain
    target_X = pd.read_csv(f"../data/{data}/train_input_7.csv").iloc[:, 1:].values.astype(np.float32)
    target_y = pd.read_csv(f"../data/{data}/train_output_7.csv").iloc[:, 1:].values.astype(np.float32)

    target_X_val = target_X[-round(target_X.shape[0] * 0.2):, :].astype(np.float32)
    target_y_val = target_y[-round(target_y.shape[0] * 0.2):].astype(np.float32)
    target_X = target_X[:-round(target_X.shape[0] * 0.2), :].astype(np.float32)
    target_y = target_y[:-round(target_y.shape[0] * 0.2)].astype(np.float32)

    test_X  = pd.read_csv(f"../data/{data}/val_input_7.csv").iloc[:, 1:].values.astype(np.float32)
    test_y  = pd.read_csv(f"../data/{data}/val_output_7.csv").iloc[:, 1:].values.astype(np.float32)

    ## source domain
    np.random.seed(2)
    random_indices1 = np.random.choice(pd.read_csv("../data/M4_train.csv").iloc[:, (1):].index,
                                    size=target_X.shape[0] * 20, replace=True)
    source_X = pd.read_csv("../data/M4_train.csv").iloc[:, 1 + (24 * 0):].loc[random_indices1].values.astype(np.float32)
    source_y = pd.read_csv("../data/M4_test.csv").iloc[:, 1:].loc[random_indices1].values.astype(np.float32)


    #### ========== Generate Backbone Architecture ==========
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

    
    #### ========== Pretraining ==========
    for loss_name in ["mse", "mae", "MASE", "mape", "SMAPE"]:
        ## 훈련되지 않은 모델만 훈련
        if not os.path.isfile(os.path.join(output_dir, f"model_{loss_name}_1.pth")):
            print(f"Start to pretraining with {loss_name}.")

            for ith in range(1, model_num+1):
                ## 사전학습, 손실 로그, val_results, state_dict
                pretraining(loss_name = loss_name, ith = ith)

                torch.cuda.empty_cache()
                gc.collect()
        else :
            print(f"Model {loss_name} is Already pretrained.")

    #### ========== Transfer ===========
    def array_to_dataset(X, y):
        X, y = torch.tensor(X), torch.tensor(y)
        X = X.reshape(-1, X.shape[1], 1)
        y = y.reshape(-1, y.shape[1], 1)

        dataset = torch.utils.data.TensorDataset(X, y)

        return dataset

    train_dataset = array_to_dataset(target_X, target_y)
    val_dataset = array_to_dataset(target_X_val, target_y_val)
    test_dataset = array_to_dataset(test_X, test_y)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = 8, shuffle = True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = 64)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 64)