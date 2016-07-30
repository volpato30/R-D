from PIL import Image
import numpy as np
im = Image.open("lena512.bmp")

def action1(start_point, width=72, height=72, stride = 9, steps = 20):
    data = np.zeros((steps, width, height))
    data_out = np.zeros((steps, width, height))
    box = (start_point[0], start_point[1], start_point[0]+width, start_point[1]+height)
    region = im.crop(box)
    data[0] = np.asarray(region,dtype = np.float32)
    for i in np.arange(1,steps+1,1):
        box = (start_point[0]+i*stride, start_point[1], start_point[0]+i*stride+width, start_point[1]+height)
        region = im.crop(box)
        temp = np.asarray(region,dtype = np.float32)
        data_out[i-1] = temp
        if i != steps:
            data[i] = temp
    end_point = (box[0], box[1])
    return data, data_out, end_point


def action2(start_point, width=72, height=72, stride = 9, steps = 20):
    data = np.zeros((steps, width, height))
    data_out = np.zeros((steps, width, height))
    box = (start_point[0], start_point[1], start_point[0]+width, start_point[1]+height)
    region = im.crop(box)
    data[0] = np.asarray(region, dtype = np.float32)
    for i in np.arange(1,steps+1,1):
        box = (start_point[0]-i*stride, start_point[1], start_point[0]-i*stride+width, start_point[1]+height)
        region = im.crop(box)
        temp = np.asarray(region,dtype = np.float32)
        data_out[i-1] = temp
        if i != steps:
            data[i] = temp
    end_point = (box[0], box[1])
    return data, data_out, end_point

def get_trajectory(num_tj=100, forward_steps=40, backward_steps=40, width=72, height=72, stride=9):
    X_forward = np.zeros((num_tj,forward_steps,1,width,height), dtype=np.float32)
    X_forward_out = np.zeros((num_tj,forward_steps,width*height), dtype=np.float32)
    X_backward = np.zeros((num_tj,backward_steps,1,width,height), dtype=np.float32)
    X_backward_out = np.zeros((num_tj,backward_steps,width*height), dtype=np.float32)
    for i in range(num_tj):
        h = np.random.randint(0,512-height)
        start_point = (50,h)
        temp_data, temp_out, end_point = action1(start_point, steps=forward_steps)
        temp_data = np.reshape(temp_data, (temp_data.shape[0], 1, temp_data.shape[1], temp_data.shape[2]))/255.0
        temp_out = np.reshape(temp_out, (temp_out.shape[0],-1))/255.0
        X_forward[i] = temp_data
        X_forward_out[i] = temp_out
        temp_data, temp_out, end_point = action2(end_point, steps=backward_steps)
        temp_data = np.reshape(temp_data, (temp_data.shape[0], 1, temp_data.shape[1], temp_data.shape[2]))/255.0
        temp_out = np.reshape(temp_out, (temp_out.shape[0],-1))/255.0
        X_backward[i] = temp_data
        X_backward_out[i] = temp_out
    return (X_forward, X_forward_out, X_backward, X_backward_out)

data = get_trajectory()
np.savez('./data/lena_data.npz', *data)
