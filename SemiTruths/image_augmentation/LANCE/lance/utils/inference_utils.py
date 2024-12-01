from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean, std),
    ]
)

def predict(model, image_path, device, idx_to_class):
    img = Image.open(image_path)
    model.to(device)
    # get normalized image
    img_normalized = preprocess(img).float()
    img_normalized = img_normalized.unsqueeze_(0)
    img_normalized = img_normalized.to(device)
    with torch.no_grad():
        output = F.softmax(model(img_normalized), dim=1)
        scores, indices = output.topk(5)
        scores = scores.squeeze()
        indices = indices.squeeze()
        preds = "<style='text-align:center;'>"
        for ix, index in enumerate(indices):
            preds += "{} ({:.2f}%)<br/>".format(
                idx_to_class[index.item()].split(",")[0], 100.0 * scores[ix].item()
            )
        preds += "</style>"
    return preds, indices.cpu(), scores.cpu() * 100.0