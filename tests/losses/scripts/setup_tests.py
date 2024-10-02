
import torch

def save_jit_module(module, fname):
  try:
    module.to("cuda")
  except:
    print("PyTorch does not have CUDA support. Saving on CPU.")
  module_jit = torch.jit.script(module)

  module_jit.save(fname)

# Create simple models that just return input for testing
class Net1(torch.nn.Module):
  def __init__(self):
    super(Net1, self).__init__()
    self.layer = torch.nn.Linear(10, 10)

  def forward(self, input1):
    x = self.layer(input1)
    return input1 + 0.0 * x

class Net2(torch.nn.Module):
  def __init__(self):
    super(Net2, self).__init__()
    self.layer = torch.nn.Linear(10, 10)

  def forward(self, input1, input2):
    x = self.layer(input1)
    return input1 + 0.0 * x, input2 + 0.0 * x


# Create loss functions with various argument combinations
class Loss1(torch.nn.Module):
  def __init__(self):
    super(Loss1, self).__init__()

  def forward(self, prediction, label):
    return (torch.sum(prediction) + torch.sum(label)) / (2 * prediction.numel())

class Loss2(torch.nn.Module):
  def __init__(self):
    super(Loss2, self).__init__()

  def forward(self, prediction1, prediction2, label1, label2):
    return (torch.sum(prediction1) + torch.sum(prediction2) + torch.sum(label1) + torch.sum(label2)) / (4 * prediction1.numel())

class Loss2Aux(torch.nn.Module):
  def __init__(self):
    super(Loss2Aux, self).__init__()

  def forward(self, prediction1, prediction2, label1, label2, aux_data1, aux_data2):
    return (torch.sum(prediction1) + torch.sum(prediction2) + torch.sum(label1) + torch.sum(label2) +
            torch.sum(aux_data1) + torch.sum(aux_data2)) / (6 * prediction1.numel())

class Loss3(torch.nn.Module):
  def __init__(self):
    super(Loss3, self).__init__()

  def forward(self, prediction, label):
    return torch.sum(prediction), torch.sum(label)

def main():
  model1 = Net1()
  model2 = Net2()
  loss1 = Loss1()
  loss2 = Loss2()
  loss2_aux = Loss2Aux()
  loss3 = Loss3()

  save_jit_module(model1, "model.pt")
  save_jit_module(model2, "model_multiarg.pt")
  save_jit_module(loss1, "loss.pt")
  save_jit_module(loss2, "loss_multiarg.pt")
  save_jit_module(loss2_aux, "loss_multiarg_aux.pt")
  save_jit_module(loss3, "loss_multiout.pt")

if __name__ == "__main__":
  main()
