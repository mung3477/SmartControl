import torch
import clip
import ImageReward as RM
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Resize
from evaluate_vit import VitExtractor

def self_similarity_score(device):
    model_name = 'dino_vitb8'
    # dino_global_patch_size = 224
    
    extractor = VitExtractor(model_name=model_name, device=device)
    imagenet_norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    global_resize_transform = Resize((256,256))
    global_transform = transforms.Compose([transforms.ToTensor(), global_resize_transform, imagenet_norm])
    
    def _fn(source_image_path:str, target_image_path:str):
        source_image = Image.open(source_image_path).convert('RGB')
        target_image = Image.open(target_image_path).convert('RGB')
                
        source_image = global_transform(source_image).unsqueeze(0).to(device)
        target_image = global_transform(target_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            source_sim = extractor.get_keys_self_sim_from_input(source_image, layer_num=11)
            target_sim = extractor.get_keys_self_sim_from_input(target_image, layer_num=11)
            
        score = torch.nn.functional.mse_loss(source_sim, target_sim).item()
        return score
    
    return _fn

def image_reward_score(device):
    model = RM.load("ImageReward-v1.0").to(device)
    
    def _fn(image_path:str, prompt:str):
        return model.score(prompt, [image_path])
    
    return _fn

def clip_score(device):
    model, preprocess = clip.load('ViT-B/32')
    model.eval()
    model.to(device)
    cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    
    def _fn(image_path:str, prompt:str):
        img = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        text = clip.tokenize([prompt]).to(device)
        
        with torch.no_grad():
            image_feature = model.encode_image(img)
            text_embedding = model.encode_text(text)
            score = cos_sim(text_embedding, image_feature).item()
        return score
    
    return _fn


device = "cuda"
########### self_similarity ###########
self_similarity_fn = self_similarity_score(device)

image_path1 = 'assets/images/Cheer.jpg'
image_path2 = 'assets/images/Gesture.jpg'

score = self_similarity_fn(image_path1, image_path2)
print(f"Self Similarity Score: {score}")

########### image_reward ###########
image_reward_fn = image_reward_score(device)

image_path = 'assets/images/Guitar.png'
prompt = "a man playing a guitar"

score = image_reward_fn(image_path, prompt)
print(f"Image Reward Score: {score}")

########### clip ###########
clip_fn = clip_score(device)

image_path = 'assets/images/Guitar.png'
prompt = "a man playing a guitar"

score = clip_fn(image_path, prompt)
print(f"CLIP Score: {score}")

