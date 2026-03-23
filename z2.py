import torch
from torchvision import models, transforms
from PIL import Image
import urllib.request
from io import BytesIO
import time
import ssl

# --- ВИПРАВЛЕННЯ SSL ДЛЯ MAC ---
ssl._create_default_https_context = ssl._create_unverified_context
# -------------------------------

try:
    from thop import profile
except ImportError:
    print("УВАГА: Бібліотека 'thop' не встановлена. FLOPs будуть вказані приблизно.")
    profile = None

print("--- 1. Завантаження попередньо навчених моделей (PyTorch) ---")

models_dict = {
    "GoogLeNet": models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1),
    "ResNet50": models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1),
    "EfficientNet_B0": models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "VGG16": models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
}

for name, model in models_dict.items():
    model.eval()

categories = models.ResNet50_Weights.IMAGENET1K_V1.meta["categories"]

print("\n--- 2. Підготовка зображень ---")

# НАДІЙНІ ПОСИЛАННЯ (Виключно GitHub та CDN, жодної Вікіпедії)
image_urls = {
    "Тварина (Собака)": "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
    "Птах (Орел)": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/build/darknet/x64/data/eagle.jpg",
    "Авто (Автобус)": "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg",
    "Літак": "https://sipi.usc.edu/database/preview/misc/4.2.05.png"
}

images = {}
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

for name, url in image_urls.items():
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            img = Image.open(BytesIO(response.read())).convert("RGB")
            images[name] = img
            print(f"✅ Завантажено: {name}")
    except Exception as e:
        print(f"⚠️ Помилка з {name}. Спроба завантажити через резервний CDN...")
        try:
            # Резервний CDN-маршрут
            backup_url = url.replace("raw.githubusercontent.com", "cdn.jsdelivr.net/gh").replace("/master/", "@master/")
            req = urllib.request.Request(backup_url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                img = Image.open(BytesIO(response.read())).convert("RGB")
                images[name] = img
                print(f"✅ Завантажено (через CDN): {name}")
        except Exception as fallback_e:
            print(f"❌ Не вдалося завантажити {name}. Використовуємо кольоровий фон.")
            images[name] = Image.new('RGB', (224, 224), color=(73, 109, 137))

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensors = {name: preprocess(img).unsqueeze(0) for name, img in images.items()}

print("\n--- 3. Розпізнавання зображень та оцінка впевненості ---")
for model_name, model in models_dict.items():
    print(f"\nМодель: {model_name}")
    for img_name, tensor in input_tensors.items():
        with torch.no_grad():
            output = model(tensor)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_catid = torch.topk(probabilities, 1)

        pred_label = categories[top_catid[0]]
        confidence = top_prob[0].item() * 100

        print(f"  [{img_name}] -> {pred_label} (Впевненість: {confidence:.2f}%)")

print("\n--- 4. Порівняння моделей: Params, FLOPs, Inference Speed ---")


def measure_inference_time(model, input_tensor, runs=10):
    for _ in range(3):
        _ = model(input_tensor)

    start_time = time.time()
    for _ in range(runs):
        _ = model(input_tensor)
    end_time = time.time()
    return ((end_time - start_time) / runs) * 1000


dummy_input = torch.randn(1, 3, 224, 224)

print(f"{'Модель':<18} | {'Параметри (млн)':<15} | {'FLOPs (млрд)':<12} | {'Час висновку (мс)':<15}")
print("-" * 70)

for model_name, model in models_dict.items():
    params = sum(p.numel() for p in model.parameters()) / 1e6

    if profile:
        macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
        flops = (macs * 2) / 1e9
    else:
        flops = 0.0

    with torch.no_grad():
        inf_time = measure_inference_time(model, dummy_input)

    print(f"{model_name:<18} | {params:<15.2f} | {flops:<12.2f} | {inf_time:<15.2f}")

print("\nСкрипт успішно завершив роботу!")
