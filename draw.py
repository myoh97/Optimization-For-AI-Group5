import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import glob

dataset = ['market', 'duke', 'cuhk']
optimizers = ['adadelta', 'adagrad', 'adam', 'sgd', 'rmsprop', 'adamw', 'asgd']
batch = ['64', '128', '256']
epochs = ['60', '70', '80']

for epoch in epochs:
    files = [f'logs_{d}/{optim}_b{b}_epoch{epoch}/loss.pkl' for d in dataset for optim in optimizers for b in batch]
    # 데이터를 저장할 딕셔너리를 생성합니다.
    data = {}

    # 각 pickle 파일에서 데이터를 읽어옵니다.
    for file in files:
        dname = file.split('/')[0].split('_')[-1]
        if dname == 'cuhk':
            dname = 'CUHK-SYSU'
        elif dname == 'market':
            dname = 'market1501'
        elif dname == 'duke':
            dname = 'DukeMTMC'
            
        oname = file.split('/')[1].split('_')[0]
        bname = file.split('/')[1].split('_')[1][1:]
        with open(file, 'rb') as f:
            loss_values = pickle.load(f)
            if dname not in data:
                data[dname] = {}
            if bname not in data[dname]:
                data[dname][bname] = {}
            data[dname][bname][oname] = loss_values

    # subplot을 생성합니다.
    fig, axes = plt.subplots(3, 3, figsize=(30, 15))
    axes = axes.ravel()

    # 각 subplot에 데이터를 플로팅합니다.
    for i, (key, val) in enumerate(data.items()):
        for j, (bb, optimizers_data) in enumerate(val.items()):
            ax = axes[i * 3 + j]
            for optimizer, loss_values in optimizers_data.items():
                ax.plot(loss_values, label=optimizer, alpha=0.3)
            ax.set_title(f'Dataset: {key}, Batch: {bb}')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss')
            ax.legend(loc='upper right')
            ax.set_ylim(1, 12)

    plt.tight_layout()
    plt.savefig(f"loss_epoch{epoch}.png")
