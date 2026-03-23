import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from scipy.io import wavfile
from PIL import Image
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# 10 класів згідно з датасетом Urban Sound 8K
CLASSES = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark",
    "drilling", "engine_idling", "gun_shot", "jackhammer",
    "siren", "street_music"
]

print("--- 1. Підготовка реальних даних (Спектрограми) ---")
DATA_DIR = "./urbanSound8KImageDataset"

if not os.path.exists(DATA_DIR):
    print(f"❌ ПОМИЛКА: Папку '{DATA_DIR}' не знайдено!")
    exit()


# СТВОРЮЄМО ВЛАСНИЙ ЗАВАНТАЖУВАЧ, ЩОБ ІГНОРУВАТИ ПАПКИ FOLD І ЧИТАТИ КЛАСИ З НАЗВ ФАЙЛІВ
class UrbanSoundDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = CLASSES
        # Шукаємо всі .png файли у всіх підпапках (fold1, fold2 і т.д.)
        self.image_paths = glob.glob(os.path.join(root_dir, '**', '*.png'), recursive=True)
        if len(self.image_paths) == 0:
            print("❌ ПОМИЛКА: Не знайдено жодного .png файлу! Перевір структуру папок.")
            exit()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Назва файлу виглядає так: 100263-2-0-114.png
        # Друге число (індекс 1 після розділення по '-') це ідентифікатор класу (0-9)
        filename = os.path.basename(img_path)
        try:
            class_id = int(filename.split('-')[1])
        except (IndexError, ValueError):
            class_id = 0  # Заглушка на випадок неправильного імені файлу

        return image, class_id


# Трансформації для зображень
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Використовуємо наш новий завантажувач
dataset = UrbanSoundDataset(root_dir=DATA_DIR, transform=transform)

# Оскільки датасет великий (~8732 фото), розбиваємо 80% на 20%
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Додано num_workers=0 для стабільності на Mac
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

print(f"Дані завантажено: {len(train_dataset)} тренувальних, {len(test_dataset)} тестових.")
print(f"Класи налаштовано правильно: {dataset.classes}")

print("\n--- 2. Створення архітектури ЗНМ (CNN) ---")


class AudioCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu4(self.fc1(x)))
        x = self.fc2(x)
        return x


model = AudioCNN(num_classes=len(dataset.classes))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\n--- 3. Навчання мережі ---")
# 5 епох буде достатньо
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # Додав вивід прогресу, щоб ти бачила, що процес іде і не завис!
        if i % 50 == 49:
            print(f"  [Епоха {epoch + 1}, Батч {i + 1}] Втрати: {running_loss / 50:.4f}")
            running_loss = 0.0
    print(f"✅ Епоха {epoch + 1} завершена.")

print("\n--- 4. Оцінка моделі та Матриця плутанини ---")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

print("\nМетрики точності для кожного класу:")
report = classification_report(all_labels, all_preds, target_names=dataset.classes, zero_division=0)
print(report)

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))  # Збільшив графік, щоб назви влазили
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=dataset.classes, yticklabels=dataset.classes)
plt.title("Матриця плутанини (Справжній Urban Sound 8K)")
plt.xlabel("Передбачений клас")
plt.ylabel("Справжній клас")
plt.xticks(rotation=45, ha='right')  # Нахилив текст для краси
plt.tight_layout()
plt.show()

print("\n--- 5. Робота з власним аудіофайлом ---")
CUSTOM_AUDIO = "my_city_sound.wav"

if not os.path.exists(CUSTOM_AUDIO):
    print(f"⚠️ Файл '{CUSTOM_AUDIO}' не знайдено. Створюю тестовий міський шум (сирена)...")
    sample_rate = 22050
    t = np.linspace(0, 2, sample_rate * 2)
    audio_data = np.sin(2 * np.pi * (800 + 400 * np.sin(2 * np.pi * 2 * t)) * t)
    wavfile.write(CUSTOM_AUDIO, sample_rate, (audio_data * 32767).astype(np.int16))


def predict_custom_audio(audio_path, model, classes):
    print(f"Обробка файлу: {audio_path}")
    sr, y = wavfile.read(audio_path)
    if len(y.shape) > 1:
        y = y.mean(axis=1)
    y = y.astype(float)

    temp_img_path = "temp_melspec.png"
    plt.figure(figsize=(4, 4))
    plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    plt.specgram(y, Fs=sr, cmap='viridis', NFFT=1024, noverlap=512)
    plt.savefig(temp_img_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    img = Image.open(temp_img_path).convert('RGB')
    tensor = transform(img).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_catid = torch.topk(probabilities, 1)

    print(f"\n✅ РЕЗУЛЬТАТ РОЗПІЗНАВАННЯ ВЛАСНОГО ЗВУКУ:")
    print(f"Передбачений клас: {classes[top_catid[0].item()]}")
    print(f"Ймовірність: {top_prob[0].item() * 100:.2f}%")


predict_custom_audio(CUSTOM_AUDIO, model, dataset.classes)
