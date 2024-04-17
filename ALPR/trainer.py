import torch
from torch.utils.data import DataLoader
from model import LPImageDataset, load_model, training_loop, custom_collate_fn
from preprocess import to_csv

# # Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
torch.cuda.empty_cache() if device == "cuda" else None

# # Load data
path = ".\\processed_images\\data\\"
# path = ".\\benchmarks\\endtoend\\us\\"
# path = '.\\processed_images\\License_Plate_tensorflow\\test\\'
# to_csv(path)
data = LPImageDataset(path + "annotations.csv", path, device=device)
train_loader = DataLoader(data, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)

# # Load model   
model = load_model(data.num_classes)   
model.load_state_dict(torch.load(".\\model.pth"))
model.to(device)

# # Train model
# training_loop(model, train_loader, device=device, epochs=150, lr=0.001)

# # Save model
# torch.save(model.state_dict(), ".\\model.pth")

# # Evaluate model
model.eval()

# # Predict
# print('=='*50)
# wrong =[]
# is_true = 0
# for i in range(len(data)):
#     sample = data[i]
#     sample_LP = data.key_by_label[sample[1]['labels'].item()]
#     pred = model([sample[0]])
#     pred_list = pred[0]['labels'].tolist()
#     pred_LP = [data.key_by_label[item] for item in pred_list]
#     # print(f"Model Prediction: {pred_LP}")
#     # print('--'*50)
#     # print(f"Actual: {sample_LP}")
#     # print(f'Is Actual in Predicted Candidates? {sample_LP in pred_LP}')
#     # print('=='*50)
#     if sample_LP in pred_LP:
#         is_true+=1 
#     else:
#         wrong.append((i, sample_LP))
# print(is_true)
# print(wrong)


''' During inference, the model requires only the input tensors, and returns the post-processed 
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows, where N is the number of detections:
        - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, 
                                        with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
        - labels (Int64Tensor[N]): the predicted labels for each detection
        - scores (Tensor[N]): the scores of each detection
'''
