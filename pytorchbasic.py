import torch 

torch.tensor #pytorch tensor data structure
#from python data structure
matrix = [[]]
torch.tensor(matrix)
#changing pytorch tensor to numpy array
tensor.numpy()
#creating tensor from numpy
torch.from_numpy(np.array)
torch.rand(shape) #[0,1] random generation
torch.randn(shape) #[0,1] random generation ~ N
torch.randint(a, b, shape) #random integer from a to b
torch.ones(shape) #matrix with all entries being 1
torch.zeros(shape) #matrix with all entries being 0
torch.arange(a, b) #vector with range a to b
#attributes
torch.shape #tuple of its shape
torch.dtype #data type of entries
torch.device #shows where it is uploaded
#types
torch.FloatTensor and torch.cuda.FloatTensor #entries are float32 #cuda tensor for gpu upload 
torch.finfo() #getting the pytorch data type
torch.finfn().bits #getting number of bits used by the type
torch.finfo().max #getting the max value of the type
torch.finfo().min #getting the min value 
torch.finfo().tiny #getting the least positive
torch.DoubleTensor #float64
torch.HalfTensor #float16
torch.ByteTensor #boolean
torch.CharTensor #int8
torch.ShortTensor #int16
torch.IntTensor #int32
torch.LongTensor #int64
#operations
tensor.long() #changes to int
tensor.float() #changes to float
torch.cat([tensor1, tensor2]) #appending
torch.matmul(), tensor1@tensor2 #matrix multiplication
tensor1 * tensor2 #element-wise multiplication
tensor.sum() #summing all entries in 1 x 1 tensor
tensor.sum(dim = n) #sums n-axiswise
tensor.item() #getting python data type from the element
tesnor.mean() #find mean of the entries
tensor.max() #find the max of the tensor
tensor.max(dim = n) #finds max n-axiswise
tensor.argmax() #find the index of the max
tensor.argmax(dim = n) #finds argmax n-axiswise
#in-place operations usually ends with _
tensor.add_(number) #in-place element-wise addtion
tensor.t_() #transposing
tensor.squeeze() #changes (n x 1) to (n, ) #during numpy backpropagation this was one of the tasks
tensor.unsqueeze() #changes (n, ) to (n, 1)
tensor.expand() #expands 1 dimension to n #(1, 3) -> (n, 3) #use (n, -1) to copy the dimension
tensor.view() #reshape #dimension change is also possible
tensor.repeat(d3, d2, d1) #repeating along x axis d1 time, d2 time along y axis and also d3 along z axis
tensor.masked_fill_(mask, value) #fills tensor with value where mask's index is True
#einstein summation
a = torch.randint(1, 10, (2,2))
torch.einsum('ij -> ji', a) #transpose
torch.einsum('ij ->', a)
torch.einsum('ij -> j', a) #column-wise addition
torch.einsum('ij -> i', a) #row-wise addtion
b = torch.randint(1, 10, (3,))
torch.einsum('ij,j -> i', a, b) #matrix multiplication
b = torch.randint(1, 10, (2, 4))
torch.einsum('ij, jk -> ik', a, b) 
a = torch.arange(1,4) #dot product
b = torch.arange(4,7)
torch.einsum('i,i ->', a, b)
torch.einsum('i,j ->ij', a, b) #outer product
a = torch.arange(6).resize(2,3) #hadamard product
b = torch.arange(6).resize(2,3)
torch.einsum('ij, ij -> ij')
i,j,k,l = 2,1,2,3 #batch multipliaction
a = torch.randint(0,10,(i,j,k))
b = torch.randint(0,10,(i,k,l))
torch.einsum('ijk, ikl -> ijl', a, b) #i number of outcome from jk@kl
i, j, k, l = 2,3,2,2 #bilinear transformation 
a = torch.randint(0,10, (i,k))
x = torch.randint(0,10, (j,k,l))
b = torch.randint(0,10, (i,l))
torch.einsum('ik,jkl,il -> ij', a, x, b)
#defining d[i,j] = sum of a[i,j]*b[j,k,l]*c[i,l] regrading indices without i and j 
#device
device = torch.device() #creates a class that connects python-pytorch object to the hardware device
object.to.device(device) #sending the object to the device
torch.device('cpu')
#multi process service
torch.backends.mps.is_available() #mac m1 gpu check
torch.backends.mps.is_built() #mac os version check
torch.device('mps')
#cuda 
torch.cuda.is_available() #nvidia cuda check
torch.device('cuda')
torch.cuda.device_count() #how many gpus are cuda using
torch.cuda.current_device() #which device is allocated for the instance
torch.device('cuda:#') #this is for allocating certain gpu for the program
torch.cuda.get_device_name(device) #checking the objects cuda allocation
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  #gpu devices arrangement 
os.environ["CUDA_VISIBLE_DEVICES"]=1,2 #allocating certain gpus #if not defined, cuda allocates all visible gpus for the following parallel processings
nn.DataParallel(model) #wraps the model and send it to multiple gpus #merges activation results of each gpu then allocate backpropagation
nn.parallel.DistributedDataParallel(model) #backpropagation starts at each gpus #better 
#changing pytorch tensor to numpy array
torch.Tensor.numpy()
#dataset
torch.utils.data.Dataset 
#define these
__init__() #to define the followings
__getitem__() #returns a tensor, args of tensors (usually tensor of data and tensor of label), or kwargs of tensors
__len__() #define the total data number 
ToTensor() #changes numpy or PIL(Python Image Library) image to floattensor #this can go into __init__() 
MyToTensor(lambda y: torch.zeros(10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1)) #function can be defined in this format #this is one of the task of numpy mnist training
#dataloader 
#after dataset is turned into a pytorch dataset, dataloader instance can be defined from this class
#shuffles and create batches for neural network
torch.utils.data.DataLoader(dataset=dataset, batch_size = batch_size, shuffle = True) #num_workers -> faster data loading #samplers to control, shuffles and data order
#return index, (input_batch, label_batch)
#neural network 
torch.nn #package of layers, activation function, and loss functions #also includes containers that addresses the whole network
torch.nn.Module #seperated to provide basic structure of neural layers
torch.nn.Squential() #seqeuntial container
torch.nn.Flatten() #(n, m) input -> (1, mn) output
torch.nn.Linear(m, n) #matrix weight layer
#example: basic model structure using nn.Module
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(256, 64)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(64, 10)

    def forward(self, x): #model needs a forward pass function for nn.Module inheritance
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        return x

model = MyModel()
model.forward(x)
#example: using both
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        x = self.layers(x)
        return x

model = MyModel()
model.forward(x)
#model setting
model.train() #updates gradient
model.eval() #does not update gradient
#example: using a container
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)
model(x)
model.forward(x) #this is built in nn.Sequential, also train() and eval() works, too
#autograd
torch.autograd #helps to calculate gradient of tensor operation 
#this feature implements nn.layers gradient update
tensor = torch.ones(m, n, requires_grad = True) #if requires_grad is set True, this is now consider as a weight matrix and gradient of the function respect to this tensor can be returned
tensor.requires_grad_(True) #this will change the autograd setting of a tensor
#example: simple forward and backward pass without activation function
import torch.nn as nn
loss = nn.MSELoss()
input = torch.rand(3, 5) 
weight = torch.rand(5, 5, requires_grad = True)
weight2 = torch.rand(5,5, requires_grad = True)
result = input@weight1@weight2
target = torch.rand(3, 5)
output = loss(result, target)
output.backward() #starts derivatation and backward pass #does chain backward pass automatically
print(weight1.grad, weight2.grad) #returns the gradients
#if the output is not a number, then the output must go into the function respect to any input
weight2.backward(output)
print(weight1.grad)
#no_grad -> does not use autograd
#different from eval() -> this calculates the gradient but does not update the weights
#no_grad() -> does not calculate the gradient, so memory usage is less
@torch.no_grad()
def add(x, y):
    return x + y #this function will turn off autograd of two tensors and calculate the result
with torch.no_grad():
    for i in range(100):
        model(i) #also code can be written in this format
#optimizer 
#updates gradient and add hyperparameters for update
torch.optim
torch.optim.SGD(model.parameter(), lr = learning_rate) #model.parameter() connects the weights with the optmizier 
torch.optim.SGD().zero_grad() #sets gradient to zero for new forwarding and backwarding
torch.optim.SGD().step() #updates the parameters 
#description about methods and hyperparameters for optimizer is in pytorch homepage
#example: training loop 
#set hyperparameters
learning_rate = 1e-3
batch_size = 64
epochs = 5
#define loss function
loss_fn = nn.CrossEntropyLoss()
#define optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#set training loop
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for idx, (X, y) in enumerate(dataloader):
        #prediction and loss finding
        pred = model(X)
        loss = loss_fn(pred, y)

        #backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (idx + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)

print("Done!")
#saving weights 
torch.save(model.state_dict(), 'path') #parameters are saved in state_dict()
#uploading weights 
model.load_state_dict(torch.load('path')) 
#saving models 
torch.save(model, 'path')
#loading models
model = torch.load('path')
#onnx -> makes machine learning frameworks compatible
torch.onnx 
#exporting torch model
torch.onnx.export(model, data, "/path/output.onnx") #export model as an onnx file 
#data can be a sinlge tensor dummy data
#can identify input and output names
torch.onnx.export(model, dummy_data, "output.onnx",  input_names = ['input'], output_names = ['output'])
#multiple names
torch.onnx.export(model, dummy_data, "output.onnx",  input_names = ['input'], output_names = ['cls_score','bbox_pred'])
#adding shape information to onnx file
onnx.save(onnx.shape_inference.infer_shapes(onnx.load(path)), path)
#importing onnx 
model = onnx.load('onnx_path')
