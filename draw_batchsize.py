import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import glob


dataset = ['market', 'duke', 'cuhk']
optimizers = ['adadelta', 'adagrad', 'adam', 'sgd', 'rmsprop', 'adamw', 'asgd']
# optimizers = ['adam','rmsprop', 'adamw']
batch = ['64', '128', '256']
epochs = ['60', '70', '80']

for optim in optimizers:
    files = [f'logs_{d}/{optim}_b{b}_epoch{epoch}/loss.pkl' for d in dataset for b in batch for epoch in epochs]
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
        ename = file.split('/')[1].split('_')[2][5:]
        
        with open(file, 'rb') as f:
            loss_values = pickle.load(f)
            if dname not in data:
                data[dname] = {}
            if ename not in data[dname]:
                data[dname][ename] = {}
            data[dname][ename][bname] = loss_values

    # subplot을 생성합니다.
    fig, axes = plt.subplots(3, 3, figsize=(30,15))
    axes = axes.ravel()

    # 각 subplot에 데이터를 플로팅합니다.
    for i, (key, val) in enumerate(data.items()):
        for j, (ee, bsize) in enumerate(val.items()):
            ax = axes[i*3 + j]
            for bb, loss_values in bsize.items():
                ax.plot(loss_values, label=f"batch{bb}", alpha=0.3)
            ax.set_title(f'Dataset: {key}, Epoch: {ee}')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss')
            ax.legend(loc='upper right')
            ax.set_ylim(1, 12)

    plt.tight_layout()
    plt.savefig(f"loss_optim{optim}.png")
