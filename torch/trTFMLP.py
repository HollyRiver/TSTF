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

    
class TransferModel(torch.nn.Module):
    """
    Backbone 전용 아키텍쳐.
    backbone_model.head.projection 자리에 삽입
    backbone_model.head.dropout 자리에 torch.nn.Identity() 삽입 (드롭아웃 레이어를 제거하는 방법을 찾는 것이 더 깔끔하지만, 임시로)
    """
    def __init__(self, t_out):
        super().__init__()
        self.t_out = t_out  ## 24. target_y.shape[1]

        self.adapter = torch.nn.Linear(1792, self.t_out * 128)   ## Dense(128)

        self.head = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 1)
        )

    def forward(self, x):
        self.adapted_feat = self.adapter(x)
        head_input = self.adapted_feat.view(-1, self.t_out, 128)  ## (B, 24, 128)
        output = self.head(head_input)

        return output

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
    elif loss_name == "mae":
        loss_fn = torch.nn.L1Loss() ## 2배면 잘 작동
        lr = pretraining_lr * 2
    elif loss_name == "SMAPE":
        loss_fn = SMAPE             ## 4배면 잘 작동
        lr = pretraining_lr * 4
    elif loss_name == "mape":
        loss_fn = MAPE_pretrained   ## 2배면 잘 작동
        lr = pretraining_lr * 2
    elif loss_name == "MASE":
        loss_fn = MASE(source_y, source_y.shape[1])
        lr = pretraining_lr * 14  ## 학습률 정상화... 그래도 잘 안됨
    else:
        raise Exception("Your loss name is not valid.")

    optimizer = torch.optim.AdamW(backbone_model.parameters(), lr = lr)
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

    del backbone_model
    torch.cuda.empty_cache()
    gc.collect()


def transfer_FC(model_num, loss_name):
    model_pred_val, model_pred_test = [], []

    T_OUT = target_y.shape[1]

    for i in range(1, model_num + 1):
        current_path = os.path.join(output_dir, f"model_{loss_name}_{i}.pth")

        model_instance = PatchTSTForPrediction.from_pretrained(os.path.join(output_dir, "PatchTSTBackbone"))
        model_instance.load_state_dict(torch.load(current_path))

        ## output layer 제거, MLP 헤드 부착
        model_instance.head.projection = TransferModel(T_OUT)
        model_instance.head.dropout = torch.nn.Identity()
        model_instance.to(device)

        optimizer = torch.optim.Adam(model_instance.parameters(), lr = learning_rate)
        log_data = []

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

        ## early stopping
        PATIENCE = 10
        best_val_loss = np.inf
        patience_counter = 0

        for epoc in range(num_train_epochs):
            model_instance.train()

            total_train_loss = 0

            for X, y in train_dataloader:
                X, y = X.to(device), y.to(device)

                optimizer.zero_grad()
                yhat = model_instance(X).prediction_outputs
                loss = loss_fn(yhat, y)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()*X.shape[0]

            avg_train_loss = total_train_loss/len(train_dataloader.dataset)

            model_instance.eval()

            with torch.no_grad():
                yys = []
                yyhats = []

                for XX, yy in val_dataloader:
                    XX = XX.to(device)
                    yys.append(yy.to(device))
                    yyhats.append(model_instance(XX).prediction_outputs)

                yyhat = torch.concat(yyhats)
                yy = torch.concat(yys)

                val_loss = loss_fn(yyhat, yy).item()

            print(f"Epoch {epoc+1}/{num_train_epochs} | Train Loss: {avg_train_loss:.6f}\t\t Val Loss: {val_loss:.6f}")

            log_data.append({"epoch": epoc, "loss": avg_train_loss, "eval_loss": val_loss})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = model_instance.state_dict()   ## 저장 없이 결과물만 산출...
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                break

        model_instance.load_state_dict(best_state_dict)

        pd.DataFrame(log_data).to_csv(os.path.join(log_dir, f"transfer_{loss_name}_lr{learning_rate}_run{1}.csv"))

        with torch.no_grad():
            yys = []
            yyhats = []

            for XX, yy in test_dataloader:
                XX = XX.to(device)
                yys.append(yy.to(device))
                yyhats.append(model_instance(XX))

            yyhat = torch.concat(yyhats)
            yy = torch.concat(yys)

            model_pred_test.append(yyhat.squeeze().to("cpu"))

            yys = []
            yyhats = []

            for XX, yy in val_dataloader:
                XX = XX.to(device)
                yys.append(yy.to(device))
                yyhats.append(model_instance(XX))

            yyhat = torch.concat(yyhats)
            yy = torch.concat(yys)

            model_pred_val.append(yyhat.squeeze().to("cpu"))

        del model_instance
        torch.cuda.empty_cache()
        gc.collect()

    return model_pred_val, model_pred_test


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


    os.makedirs("resulttf/val", exist_ok = True)
    os.makedirs("resulttf/test", exist_ok = True)

    val_preds = {}
    test_preds = {}

    ## 변수 이름 설정에 일관성이 없네
    save_name = ["mse", "mae", "mase", "mape", "smape"]

    for i, loss_name in enumerate(["mse", "mae", "MASE", "mape", "SMAPE"]):
        print(f"Start to Transfer Learning with {loss_name}.")

        pred_val, pred_test = transfer_FC(model_num, loss_name = loss_name)

        ## 예측 결과 저장
        pd.DataFrame(np.array(pred_val).reshape(1, -1)).to_csv(f"resulttf/val/trTFMLP_{data}_{save_name[i]}_pred.csv")
        pd.DataFrame(np.array(pred_test).reshape(1, -1)).to_csv(f"resulttf/test/trTFMLP_{data}_{save_name[i]}_pred.csv")

        val_preds[loss_name] = pred_val
        test_preds[loss_name] = pred_test

    ## ========== 전체/부분 앙상블 RMSE 출력 ==========
    concat_G = np.concatenate(list(val_preds.values()))
    fin_pred_G = np.median(concat_G, axis = 0)
    print("all (RMSE):", np.sqrt(mean_squared_error(target_y_val.flatten(), fin_pred_G.flatten())).round(5))
    print(f"all (MAE):  {mean_absolute_error(target_y_val.flatten(), fin_pred_G.flatten()):.5f}")

    ## 어떤 점에서 best?
    concat_G = np.concatenate([val_preds[loss] for loss in ["mae", "MASE", "mse"]])
    fin_pred_G = np.median(concat_G, axis = 0)
    print("best (RMSE):", np.sqrt(mean_squared_error(target_y_val.flatten(), fin_pred_G.flatten())).round(5))
    print(f"best (MAE):  {mean_absolute_error(target_y_val.flatten(), fin_pred_G.flatten()):.5f}")

    ## 변수별 앙상블 결과
    for name in ["mse", "mae", "MASE", "mape", "SMAPE"]:
        concat_G = np.concatenate([np.nan_to_num(np.array(val_preds[name]), nan=0)])
        fin_pred_G = np.median(concat_G, axis=0)
        print(name, "(RMSE):", np.sqrt(mean_squared_error(target_y_val.flatten(), fin_pred_G.flatten())).round(5))
        print(name, f"(MAE):  {mean_absolute_error(target_y_val.flatten(), fin_pred_G.flatten()):.5f}")

    ## ========== 지표별 단독 RMSE 저장 (어차피 안쓰는 파일인데?) ==========
    rmse = []
    mae = []

    for name in ["MASE", "mape", "SMAPE", "mae", "mse"]:
        concat_G = np.concatenate([np.nan_to_num(np.array(val_preds[name]), nan = 0)])
        fin_pred = np.median(concat_G, axis = 0)
        rmse.append(np.sqrt(mean_squared_error(target_y_val.flatten(), fin_pred.flatten())).round(5))
        mae.append(round(mean_absolute_error(target_y_val.flatten(), fin_pred.flatten()), 5))

    performance = np.array(rmse)
    os.makedirs("resulttf", exist_ok = True)
    pd.DataFrame(performance).to_csv(f"resulttf/trTFMLP_{data}_weight.csv")
    pd.DataFrame(np.array(mae)).to_csv(f"resulttf/trTFMLP_{data}_weight_mae.csv")