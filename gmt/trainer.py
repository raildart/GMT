# gmt/trainer.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from .variants import MODEL_NAME

def train(ds, model, variant='medium', epochs=30, batch_size=64, lr=5e-5, num_workers=4):
    device = torch.device('mps') if torch.backends.mps.is_available() \
              else torch.device('cuda') if torch.cuda.is_available() \
              else torch.device('cpu')
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        pin_memory=True, num_workers=num_workers)
    model.to(device)
    opt    = optim.Adam(model.parameters(), lr=lr)
    crit   = torch.nn.MSELoss()
    history=[]
    for e in range(1,epochs+1):
        model.train(); total=0
        for tgt,base,lab, *_ in loader:
            tgt,base,lab = tgt.to(device), base.to(device), lab.to(device)
            pred = model(tgt,base)
            loss = crit(pred, lab)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()*tgt.size(0)
        avg = total / len(ds); history.append(avg)
        print(f"Epoch {e}/{epochs} — MSE: {avg:.4f}")
        if device.type=='cuda': torch.cuda.empty_cache()
    plt.plot(history,'-o'); plt.title(f"{MODEL_NAME}-{variant} Loss"); plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.show()
    torch.save(model.state_dict(), f"{MODEL_NAME}_{variant}.pth")
