import torch
import torch.nn as nn
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self,*dims):
        super().__init__()
        self.fc1 = nn.Linear(dims[0],dims[1])
        self.fc2 = nn.Linear(dims[1],dims[2])
        self.fc3 = nn.Linear(dims[2],1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)

        return x 


model = MLP(2,4,2)
crit = nn.BCELoss()
optimizer = optim.SGD(model.parameters(),lr = 0.1)

##training dataset
x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])

try:
    # training
    print(x) 
    for epoch in range(100):
        optimizer.zero_grad()

        #forward pass: 1. Compute predictions and loss

        #below model instance now calls the forward function
        outputs = model(x)
        loss = crit(outputs,y)

        print('Output for the epoch',outputs.round())

        # 2. Backpropagation — compute gradients
        loss.backward()

        # 3. Optimizer step — update weights
        optimizer.step()


        if (epoch+1) % 20 == 0:
            print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")

    # Test output
    # print("Final Predictions:\n", model(x).detach().round())

except Exception as e:
    print("error is ", e)
