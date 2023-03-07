import torch
import time
import numpy as np
import torch.backends.cudnn as cudnn




cudnn.benchmark = True # Enables cuDNN autotuner

def benchmark(model, input_shape=(1024, 3, 512, 512), dtype='fp32', nwarmup=50, nruns=1000):
    input_data = torch.randn(input_shape)
    input_data = input_data.to("cuda")
    if dtype=='fp16':
        input_data = input_data.half()
        
    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)
    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns+1):
            start_time = time.time()
            pred_loc  = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i%10==0:
                print('Iteration %d/%d, avg batch time %.2f ms'%(i, nruns, np.mean(timings)*1000))

    print("Input shape:", input_data.size())
    print('Average throughput: %.2f images/second'%(input_shape[0]/np.mean(timings)))


mobilenetv3_large = "/home/andreasgp/MEGAsync/DTU/9. Semester/Deep Learning/object-tracking-project/02456-project/models/mobilenetv3_large_25_epochs.pth"
model_mobilenetv3_large = torch.load(mobilenetv3_large).to("cuda")

mobilenetv3_large_320 = "/home/andreasgp/MEGAsync/DTU/9. Semester/Deep Learning/object-tracking-project/02456-project/models/mobilenetv3_large_320_15epochs_entire_dataset.pth"
model_mobilenetv3_large_320 = torch.load(mobilenetv3_large_320).to("cuda")

#print("model_mobilenetv3_large:")
#benchmark(model_mobilenetv3_large, input_shape=(1, 3, 480, 320), nruns=100)

#print("model_mobilenetv3_large_320:")
#benchmark(model_mobilenetv3_large_320, input_shape=(1, 3, 480, 320), nruns=100)



# Set param.grad to 0 for entire model
#for param in model_mobilenetv3_large_320.parameters():
#    param.grad = None

modemodel_mobilenetv3_large_320_mem = model_mobilenetv3_large_320.to(memory_format=torch.channels_last)
print("model_mobilenetv3_large_320:")
benchmark(modemodel_mobilenetv3_large_320_mem, input_shape=(1, 3, 480, 320), nruns=100)