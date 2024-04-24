import torch
from torch.utils.data import DataLoader, Dataset
from model import LPImageDataset, load_model, training_loop, custom_collate_fn
from preprocess import to_csv

# # Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
torch.cuda.empty_cache() if device == "cuda" else None

# # Load data
# path = ".\\processed_images\\data\\"

path = ".\\processed_images\\dheeraj.v17i\\train\\"
data = LPImageDataset(path, path + "annotations.csv", device=device)
train_loader = DataLoader(
    data, batch_size=8, shuffle=True, collate_fn=custom_collate_fn
)
# test_data = (test_path)
# test_loader = DataLoader(test_data, batch_size=2, shuffle=True)

# # Load model
model = load_model(data.num_classes)
# model = torch.load(".\\model_3.0")
# model.load_state_dict(torch.load(".\\model_2.0.pth"))
model.to(device)

# # Train model
training_loop(model, train_loader, device=device, epochs=50, lr=0.0001)

# # Save model
torch.save(model, ".\\model_3.0")

# # Evaluate model
# model.eval()

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

""" The model returns a Dict[Tensor] during training, containing the classification and regression 
    losses for both the RPN and the R-CNN.
    Format:
    dict = {
        'loss_classifier': tensor(5.3523, device='cuda:0', grad_fn=<NllLossBackward0>), 
        'loss_box_reg': tensor(0.0001, device='cuda:0', grad_fn=<DivBackward0>), 
        'loss_objectness': tensor(0.6812, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward0>), 
        'loss_rpn_box_reg': tensor(0.0018, device='cuda:0', grad_fn=<DivBackward0>)
    }
"""
""" During inference, the model requires only the input tensors, and returns the post-processed 
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows, where N is the number of detections:
        - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, 
                                        with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
        - labels (Int64Tensor[N]): the predicted labels for each detection
        - scores (Tensor[N]): the scores of each detection
"""
