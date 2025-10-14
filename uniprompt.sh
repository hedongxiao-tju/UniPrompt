d_wd=(0.00005)
epochs=(1000)
hid_dims=(256)
down_epochs=(2000)
trails=(10)
seeds=(42)
prompt='UniPrompt'
# DGI
python main.py --model 'DGI' --shot 1 --dataset 'Cora' --lr 0.001 --down_lr 0.05 --k 50 --tau 0.99999 --hid_dim $hid_dims --epochs $down_epochs --down_wd $d_wd --trails $trails --seed $seeds
python main.py --model 'DGI' --shot 1 --dataset 'CiteSeer' --lr 0.0005 --down_lr 0.05 --k 50 --tau 0.9999 --hid_dim $hid_dims --epochs $down_epochs --down_wd $d_wd --trails $trails --seed $seeds
python main.py --model 'DGI' --shot 1 --dataset 'PubMed' --lr 0.0005 --down_lr 0.001 --k 1 --tau 0.9999 --hid_dim $hid_dims --epochs $down_epochs --down_wd $d_wd --trails $trails --seed $seeds
python main.py --model 'DGI' --shot 1 --dataset 'Cornell' --lr 0.001 --down_lr 0.0005 --k 50 --tau 0.99 --hid_dim $hid_dims --epochs $down_epochs --down_wd $d_wd --trails $trails --seed $seeds
python main.py --model 'DGI' --shot 1 --dataset 'Texas' --lr 0.00001 --down_lr 0.0001 --k 50 --tau 0.999 --hid_dim $hid_dims --epochs $down_epochs --down_wd $d_wd --trails $trails --seed $seeds
python main.py --model 'DGI' --shot 1 --dataset 'Wisconsin' --lr 0.0001 --down_lr 0.001 --k 50 --tau 0.999 --hid_dim $hid_dims --epochs $down_epochs --down_wd $d_wd --trails $trails --seed $seeds 
python main.py --model 'DGI' --shot 1 --dataset 'Chameleon' --lr 0.00005 --down_lr 0.001 --k 10 --tau 0.9999 --hid_dim $hid_dims --epochs $down_epochs --down_wd $d_wd --trails $trails --seed $seeds
python main.py --model 'DGI' --shot 1 --dataset 'Actor' --lr 0.001 --down_lr 0.01 --k 50 --tau 0.999 --hid_dim $hid_dims --epochs $down_epochs --down_wd $d_wd --trails $trails --seed $seeds
python main.py --model 'DGI' --shot 1 --dataset 'Squirrel' --lr 0.00005 --down_lr 0.005 --k 50 --tau 0.99999 --hid_dim $hid_dims --epochs $down_epochs --down_wd $d_wd --trails $trails --seed $seeds
# GRACE
python main.py --model 'GRACE' --shot 1 --dataset 'Cora' --lr 0.001 --down_lr 0.005 --k 50 --tau 0.9999 --hid_dim $hid_dims --epochs $down_epochs --down_wd $d_wd --trails $trails --seed $seeds
python main.py --model 'GRACE' --shot 1 --dataset 'CiteSeer' --lr 0.005 --down_lr 0.001 --k 50 --tau 0.9999 --hid_dim $hid_dims --epochs $down_epochs --down_wd $d_wd --trails $trails --seed $seeds
python main.py --model 'GRACE' --shot 1 --dataset 'PubMed' --lr 0.01 --down_lr 0.05 --k 1 --tau 0.9999 --hid_dim $hid_dims --epochs $down_epochs --down_wd $d_wd --trails $trails --seed $seeds
python main.py --model 'GRACE' --shot 1 --dataset 'Cornell' --lr 0.0001 --down_lr 0.0005 --k 50 --tau 0.99 --hid_dim $hid_dims --epochs $down_epochs --down_wd $d_wd --trails $trails --seed $seeds
python main.py --model 'GRACE' --shot 1 --dataset 'Texas' --lr 0.0001 --down_lr 0.00005 --k 50 --tau 0.9999 --hid_dim $hid_dims --epochs $down_epochs --down_wd $d_wd --trails $trails --seed $seeds 
python main.py --model 'GRACE' --shot 1 --dataset 'Wisconsin' --lr 0.0001 --down_lr 0.01 --k 50 --tau 0.999 --hid_dim $hid_dims --epochs $down_epochs --down_wd $d_wd --trails $trails --seed $seeds
python main.py --model 'GRACE' --shot 1 --dataset 'Chameleon' --lr 0.005 --down_lr 0.001 --k 1 --tau 0.99999 --hid_dim $hid_dims --epochs $down_epochs --down_wd $d_wd --trails $trails --seed $seeds
python main.py --model 'GRACE' --shot 1 --dataset 'Actor' --lr 0.0005 --down_lr 0.01 --k 50 --tau 0.9999 --hid_dim $hid_dims --epochs $down_epochs --down_wd $d_wd --trails $trails --seed $seeds 
python main.py --model 'GRACE' --shot 1 --dataset 'Squirrel' --lr 0.01 --down_lr 0.05 --k 50 --tau 0.9999 --hid_dim $hid_dims --epochs $down_epochs --down_wd $d_wd --trails $trails --seed $seeds
# GraphMAE
python main.py --model 'GraphMAE' --shot 1 --dataset 'Cora' --lr 0.0005 --down_lr 0.0005 --k 50 --tau 0.9999 --hid_dim $hid_dims --epochs $down_epochs --down_wd $d_wd --trails $trails --seed $seeds
python main.py --model 'GraphMAE' --shot 1 --dataset 'CiteSeer' --lr 0.001 --down_lr 0.0001 --k 1 --tau 0.99999 --hid_dim $hid_dims --epochs $down_epochs --down_wd $d_wd --trails $trails --seed $seeds
python main.py --model 'GraphMAE' --shot 1 --dataset 'PubMed' --lr 0.005 --down_lr 0.01 --k 1 --tau 0.999 --hid_dim $hid_dims --epochs $down_epochs --down_wd $d_wd --trails $trails --seed $seeds
python main.py --model 'GraphMAE' --shot 1 --dataset 'Cornell' --lr 0.00005 --down_lr 0.05 --k 50 --tau 0.9999 --hid_dim $hid_dims --epochs $down_epochs --down_wd $d_wd --trails $trails --seed $seeds
python main.py --model 'GraphMAE' --shot 1 --dataset 'Texas' --lr 0.00001 --down_lr 0.0005 --k 50 --tau 0.9999 --hid_dim $hid_dims --epochs $down_epochs --down_wd $d_wd --trails $trails --seed $seeds
python main.py --model 'GraphMAE' --shot 1 --dataset 'Wisconsin' --lr 0.00005 --down_lr 0.01 --k 50 --tau 0.9999 --hid_dim $hid_dims --epochs $down_epochs --down_wd $d_wd --trails $trails --seed $seeds
python main.py --model 'GraphMAE' --shot 1 --dataset 'Chameleon' --lr 0.00001 --down_lr 0.005 --k 50 --tau 0.99999 --hid_dim $hid_dims --epochs $down_epochs --down_wd $d_wd --trails $trails --seed $seeds 
python main.py --model 'GraphMAE' --shot 1 --dataset 'Actor' --lr 0.005 --down_lr 0.05 --k 50 --tau 0.9999 --hid_dim $hid_dims --epochs $down_epochs --down_wd $d_wd --trails $trails --seed $seeds 
python main.py --model 'GraphMAE' --shot 1 --dataset 'Squirrel' --lr 0.005 --down_lr 0.05 --k 50 --tau 0.9999 --hid_dim $hid_dims --epochs $down_epochs --down_wd $d_wd --trails $trails --seed $seeds
