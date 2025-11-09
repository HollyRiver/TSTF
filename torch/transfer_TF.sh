nohup python trTFTF.py --model_num=100 --data="coin" --lr=1e-6 --backbone="PatchTSTBackbone" --transfer_loss="mse" &
nohup python trTFTF.py --model_num=100 --data="coin" --lr=1e-6 --backbone="PatchTSTBackbone" --transfer_loss="mae" &
nohup python trTFTF.py --model_num=100 --data="coin" --lr=1e-6 --backbone="PatchTSTBackbone" --transfer_loss="MASE" &
nohup python trTFTF.py --model_num=100 --data="coin" --lr=1e-6 --backbone="PatchTSTBackbone" --transfer_loss="mape" &
nohup python trTFTF.py --model_num=100 --data="coin" --lr=1e-6 --backbone="PatchTSTBackbone" --transfer_loss="SMAPE" &