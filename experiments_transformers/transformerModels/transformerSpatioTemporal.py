import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformerModels.spatial_temporal_tensor_attention import SpatialTemporalTensorAttention

from transformerModels.utils import PositionalEncodingSpatioTemporal, generate_square_subsequent_mask, _get_subsequent_mask

class TransformerEncoderLayerSpatioTemporal(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",batch_first=True):
        super().__init__(d_model=d_model, nhead=nhead,dim_feedforward=dim_feedforward, dropout=dropout, activation=activation,batch_first=batch_first)
        self.self_attn = SpatialTemporalTensorAttention(d_model, nhead, dropout)

class TransformerSpatioTemporalModel(pl.LightningModule):
    '''
    Autoregresive Decoder-Only TransformerSpatioTemporal. Variant number 1.
    '''
    def __init__(self,input_size,output_size, n_features, d_model=256,nhead=8, num_layers=3, dropout=0.1):
        super(TransformerSpatioTemporalModel, self).__init__()

        self.d_model = d_model
        self.criterion = nn.L1Loss()
        self.warmup_steps = 4000

        self.output_size = output_size
        #self.n_features = n_features



        self.encoder = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncodingSpatioTemporal(d_model, dropout)

        self.decoder = nn.Linear(1, d_model)
        self.pos_decoder = PositionalEncodingSpatioTemporal(d_model, dropout) 

        decoder_layer = TransformerEncoderLayerSpatioTemporal(d_model=d_model, nhead=nhead,dim_feedforward=d_model*4, dropout=dropout, activation='relu',batch_first=True)
        self.transformer_decoder  = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 1)

        self.src_mask = None
        self.trg_mask = None
        self.memory_mask = None
        
    def forward(self, src,trg):

        if self.trg_mask is None or self.trg_mask.size(1) != trg.shape[1]:
            self.trg_mask = _get_subsequent_mask(trg).to(trg.device)



        src = self.encoder(src)
        src = self.pos_encoder(src)

        trg = self.decoder(trg)
        trg = self.pos_decoder(trg)


        output = self.transformer_decoder(trg,src, tgt_mask=self.trg_mask, memory_mask=self.src_mask)
        output = self.fc_out(output)

        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = torch.cat((x[:,-1,:].unsqueeze(1),y ),1)
        
        y_hat = self(x,y[:, :-1])
        loss = self.criterion(y_hat, y[:, 1:])
        self.log("train_loss",loss,on_epoch=True,logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        decoderInput = x[:, -1].unsqueeze(-1)
        for i in range(0,self.output_size):
            out = self(x,decoderInput)
            decoderInput = torch.cat((decoderInput,out[:,-1].unsqueeze(-1).detach()),1) 
            
        y_hat = decoderInput[:, 1:]
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss,on_epoch=True,prog_bar=True,logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-9)
        return optimizer
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
        on_tpu=False, using_native_amp=False, using_lbfgs=False):

        d_model, steps, warmup_steps = self.d_model, self.global_step + 1, self.warmup_steps
        rate = (d_model ** (- 0.5)) * min(steps ** (- 0.5), steps * warmup_steps ** (- 1.5))

        for pg in optimizer.param_groups:
            pg['lr'] = rate

        optimizer.step(closure=optimizer_closure)