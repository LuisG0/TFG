import torch
from torch.autograd import Variable
import numpy as np
from early_stopping import EarlyStopping
from metrics import METRICS, evaluate

def loss_backprop(criterion, out, targets):
    """
    Memory optmization. Compute each timestep separately and sum grads.
    """
    assert out.size(1) == targets.size(1)
    total = 0.0
    out_grad = []

    for i in range(out.size(1)):

        out_column = Variable(out[:, i].data, requires_grad=True)
        loss = criterion(out_column, targets[:, i])
        total += float(loss.data.item())
        loss.backward()
        out_grad.append(out_column.grad.data.clone())

    out_grad = torch.stack(out_grad, dim=1)
    out.backward(gradient=out_grad)
    return total/out.size(1)

def loss_validation(generator, criterion, out, targets):

    assert out.size(1) == targets.size(1)
    total = 0.0
  

    for i in range(out.size(1)):

        out_column = Variable(out[:, i].data, requires_grad=True)
        gen = generator(out_column)
        loss = criterion(gen, targets[:, i])
        total += float(loss.data.item())

    return total/out.size(1)


def train_epoch(train_iter, model, criterion, opt,steps_per_epoch):
    
    model.train()
    totalLoss = 0
    for i, batch in enumerate(train_iter):
        
        src, trg = \
              batch.src, batch.trg
        
        if i > steps_per_epoch:
            break
 
        out = model.forward(src, trg[:, :-1])
     
        loss = loss_backprop(criterion, out, trg[:, 1:])  #model.module necesario para acceder al generador a traves del wraper nn.DataParallel
       
        totalLoss += float(loss)
       
        opt.step()
        if hasattr(opt, "optimizer"):
            opt.optimizer.zero_grad()
        else:
            opt.zero_grad()
        #if i % 5 == 1:
        #print(i,"Batch loss", loss)
    print("Train mean loss",totalLoss/(i+1))
        
    return totalLoss/(i+1)

def singleStepvalidEpoch(valid_iter, model, criterion):
    model.eval()
    totalLoss = 0
    with torch.no_grad():
      for i, batch in enumerate(valid_iter):
          src, trg, trg_mask = \
              batch.src, batch.trg, batch.trg_mask
          out = model.forward(src, trg[:, :-1], None, trg_mask[:, :-1, :-1])
          loss = loss_validation(model.generator, criterion, out, trg[:, 1:]) 
          totalLoss += loss
          #print(i,"Batch validation loss", loss)
    #print("Single Step Validation loss:" , totalLoss/(i+1))



def multiStepValidEpoch(valid_iter, model, criterion):
    model.eval()
    totalLoss = 0
    with torch.no_grad():
      for i, batch in enumerate(valid_iter):
          
          src, trg = \
              batch.src, batch.trg
          nSteps = trg.shape[1] - 1
          decoderInput = trg[:, 0]
          decoderInput = decoderInput.unsqueeze(-2)
          
          for _ in range(nSteps): # We append the prediction on step 0 to the original input to get the input for step 1 and so on.
    
            out = model.forward(src,decoderInput)  # We don't need mask for evaluation, we are not giving the model any future input.
           
            gen = out[:,-1,:].detach() #detach() is quite important, otherwise we will keep the variable "gen" in memory and cause an out of memory error.
            
            gen = gen.unsqueeze(-2)
         
            decoderInput = torch.cat((decoderInput,gen),1) 
      

          output = decoderInput[:, 1:]

          loss = criterion(output,trg[:, 1:])

          totalLoss += float(loss) #float() is quite important, otherwise we will keep "loss" and its gradient in memory and cause an out of memory error.
          #print(i,"Batch validation loss", loss)
    print("Multi Step Validation Total loss:" , totalLoss)
    print("Multi Step Validation loss:" , totalLoss/(i+1))
    return totalLoss/(i+1)

class Batch:
    def __init__(self, src, trg, seqLen):
        self.src = src
        self.trg = trg
        self.seqLen = seqLen
def getBatches(x,y,batchSize,solarPanel):

    lastElement = torch.unsqueeze(x[:,-1,:],1)

    if solarPanel != "ALL":
        lastElement = torch.unsqueeze(lastElement[:,:,solarPanel], 2)

    y = torch.cat((lastElement,y ),1) # We put the last element of the encoder input as the first element of the decoder input.
                                        # This mirror the function of the start token in nlp 
    permutation = torch.randperm(x.shape[0])

    for i in range(0, len(x), batchSize):
      seqLen = min(batchSize, len(x) - i)   
      indices = permutation[i:i+seqLen]
      src = Variable(x[indices], requires_grad = False)
      tgt = Variable(y[indices], requires_grad = False)
    
      yield Batch(src, tgt, seqLen) 
def getBatchesEval(x,batchSize):


  for i in range(0, len(x), batchSize):
    seqLen = min(batchSize, len(x) - i)

    src = Variable(x[i:i+seqLen], requires_grad = False)
    
    yield Batch(src,None, seqLen)




def train(model,x_train,y_train,x_val,y_val,solarPanel,epochs,steps_per_epoch,criterion,model_opt,batchSize):
    
    es = EarlyStopping(patience=3)

    bestValLoss = np.inf

    for e in range(epochs):
        print("Epoch: " , e)
        trainLoss = train_epoch(getBatches(x_train,y_train,batchSize,solarPanel), model, criterion, model_opt,steps_per_epoch)
        
        valLoss = multiStepValidEpoch(getBatches(x_val,y_val,batchSize,solarPanel), model, criterion)
        if valLoss < bestValLoss:
            bestValLoss = valLoss
            bestState = model.state_dict()
        if es.step(valLoss):
            break  # early stop criterion is met, we can stop now
    #torch.save(bestState, '/usr/desarrollo/tsf/solar-irradiance-forecasting-transformers/src/experiments/models/lastmodel.pt')
    model.load_state_dict(bestState) # Get the best model
    print("Best Validation Loss: ")
    print(bestValLoss)
    valLoss = bestValLoss

    return trainLoss,valLoss, e+1


def predictMultiStepBatching(x_test, model, nSteps,solarPanel):
    model.eval()
    with torch.no_grad():
      valid_iter = getBatchesEval(x_test,256)

      pred = []
      for i, batch in enumerate(valid_iter):
        x = batch.src
        encoderInpot = x
        decoderInput = x[:, -1,:]
        decoderInput = decoderInput.unsqueeze(-2)
        if solarPanel != "ALL":
            decoderInput = torch.unsqueeze(decoderInput[:,:,solarPanel], 2)

        for i in range(nSteps): # We append the prediction on step 0 to the original input to get the input for step 1 and so on.
            out = model.forward(encoderInpot,decoderInput)  # We don't need mask for evaluation, we are not giving the model any future input.
            lastPrediction = out[:,-1].detach() #detach() is quite important, otherwise we will keep the variable "gen" in memory and cause an out of memory error.
            lastPrediction = lastPrediction.unsqueeze(-2)
            decoderInput = torch.cat((decoderInput,lastPrediction),1) 

        batchPrediction = decoderInput.squeeze(-1).to("cpu")
        batchPrediction = batchPrediction[:,1:]
        pred.append(batchPrediction)
    totalPred = pred[0]
    for i in range(1,len(pred)):
        totalPred = torch.cat((totalPred,pred[i]),0)

    return totalPred

def predictMultiStep(x_test, model, nSteps):
    model.eval()
    with torch.no_grad():
        encoderInpot = x_test
        decoderInput = x_test[:, -1,:]
        decoderInput = decoderInput.unsqueeze(-1)
        for i in range(nSteps): # We append the prediction on step 0 to the original input to get the input for step 1 and so on.
            out = model.forward(encoderInpot,decoderInput)  # We don't need mask for evaluation, we are not giving the model any future input.
            lastPrediction = out[:,-1].detach() #detach() is quite important, otherwise we will keep the variable "gen" in memory and cause an out of memory error.
            lastPrediction = lastPrediction.unsqueeze(-1)
            decoderInput = torch.cat((decoderInput,lastPrediction),1) 

        decoderInput = decoderInput.squeeze(-1).to("cpu")
    return decoderInput[:,1:]



