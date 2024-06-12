import re, torch, chess
import time
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from peewee import *
from random import randrange
from torch import nn
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader

db = SqliteDatabase('5m.db')
LABEL_COUNT = 5_000_000

class Evaluations(Model):
  id = IntegerField()
  fen = TextField()
  binary = BlobField()
  eval = FloatField()

  class Meta:
    database = db

class EvaluationDataset(Dataset):
  def __init__(self, count):
    self.count = count

  def __len__(self):
    return self.count
  
  def __getitem__(self, idx):
    eval = Evaluations.get(Evaluations.id == idx)
    bin = np.fromiter(map(int, eval.binary), dtype=np.single)
    if type(eval.eval) == str:
        eval.eval = eval.eval.removeprefix('#')
        eval.eval = re.sub(r'\d+', '15', eval.eval) 
    else:
        eval.eval = max(eval.eval, -15)
        eval.eval = min(eval.eval, 15)
    ev = np.array([eval.eval], dtype=np.single)
    return {'binary': bin, 'eval': ev}    

class EvaluationModel(pl.LightningModule):
  def __init__(self,learning_rate=1e-3, batch_size=1024, layer_count=10):
    super().__init__()
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    layers = []
    # TODO this can be changed
    # for now its just n 804x804-layers and finally a reduction with an 804x1 layer
    # but maybe something better is to reduce slowly like 804x804, 804x402, 402x32, 32x32, 32x1
    # there are infinite possibilities to try out
    for i in range(layer_count-1):
      layers.append((f"linear-{i}", nn.Linear(804, 804)))
      layers.append((f"relu-{i}", nn.ReLU()))
    layers.append((f"linear-{layer_count-1}", nn.Linear(804, 1)))
    self.save_hyperparameters()
    self.seq = nn.Sequential(OrderedDict(layers))

  def forward(self, x):
    return self.seq(x)

  def training_step(self, batch, batch_idx):
    x, y = batch['binary'], batch['eval']
    y_pred = self(x)
    loss = F.l1_loss(y_pred, y)
    self.log("train_loss", loss)
    return loss

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

  def train_dataloader(self):
    dataset = EvaluationDataset(count=LABEL_COUNT)
    return DataLoader(dataset, batch_size=self.batch_size, num_workers=8, pin_memory=True)

MODES = [TRAINING, TESTING] = [True, False]

mode = TRAINING

if mode == TRAINING:
    if __name__ == '__main__':
            dataset = EvaluationDataset(count=LABEL_COUNT)

            # TODO depending on how drastically the constructor is changed this has to be changed a lot as well
            # for the original, layers just has to be >= 2 and batchsize can be changed as well, for fewer layers maybe try 128, 512, ...
            config = {"layer_count": 6, "batch_size": 1024}

            version_name = f'{int(time.time())}-batch_size-{config["batch_size"]}-layer_count-{config["layer_count"]}'
            logger = pl.loggers.TensorBoardLogger("lightning_logs", name="chessml", version=version_name)
            trainer = pl.Trainer(precision=32, max_epochs=1, logger=logger)
            model = EvaluationModel(layer_count=config["layer_count"], batch_size=config["batch_size"], learning_rate=1e-3)
            trainer.fit(model)
else:
    model = EvaluationModel.load_from_checkpoint(checkpoint_path=r'C:\Stuff\4210-Project\lightning_logs\chessml\1682643292-batch_size-1024-layer_count-6\checkpoints\checkpoint.ckpt')

    MATERIAL_LOOKUP = {chess.KING:0,chess.QUEEN:9,chess.ROOK:5,chess.BISHOP:3,chess.KNIGHT:3,chess.PAWN:1}

    def avg(lst):
        return sum(lst) / len(lst)

    def material_balance(board):
        white = board.occupied_co[chess.WHITE]
        black = board.occupied_co[chess.BLACK]
        value = (
            chess.popcount(white & board.pawns) - chess.popcount(black & board.pawns) +
            3 * (chess.popcount(white & board.knights) - chess.popcount(black & board.knights)) +
            3 * (chess.popcount(white & board.bishops) - chess.popcount(black & board.bishops)) +
            5 * (chess.popcount(white & board.rooks) - chess.popcount(black & board.rooks)) +
            9 * (chess.popcount(white & board.queens) - chess.popcount(black & board.queens))
        )
        return min(max(value, -15), 15)

    
    def guess_zero_loss(idx):
        batch = dataset[idx]
        y = torch.tensor(batch['eval'])
        y_pred = torch.zeros_like(y)
        loss = F.l1_loss(y_pred, y)
        return loss

    def guess_material_loss(idx):
        batch = dataset[idx]
        ev = Evaluations.get(Evaluations.id == idx)
        board = chess.Board(ev.fen)
        y = torch.tensor(batch['eval'])
        y_pred = torch.tensor([material_balance(board)])
        loss = F.l1_loss(y_pred, y)
        return loss

    def guess_model_loss(idx):
        batch = dataset[idx]
        x, y = torch.tensor(batch['binary']), torch.tensor(batch['eval'])
        y_pred = model(x)
        loss = F.l1_loss(y_pred, y)
        return loss

    zero_losses = []
    mat_losses = []
    model_losses = []
    for i in range(1000):
        idx = randrange(LABEL_COUNT)
        zero_losses.append(guess_zero_loss(idx))
        mat_losses.append(guess_material_loss(idx))
        model_losses.append(guess_model_loss(idx))
    print(f'Guess Zero Avg Loss {avg(zero_losses)}')
    print(f'Guess Material Avg Loss {avg(mat_losses)}')
    print(f'Guess Model Avg Loss {avg(model_losses)}')
        