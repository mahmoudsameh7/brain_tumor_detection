from PIL import Image
import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import normalize, to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout


IMAGE_DIR = Path(r"C:/Users/hp/OneDrive/سطح المكتب/data/archive/dataset")



NO_DIR  = IMAGE_DIR / "no"
YES_DIR = IMAGE_DIR / "yes"

assert NO_DIR.exists(),  f"Folder not found: {NO_DIR}"
assert YES_DIR.exists(), f"Folder not found: {YES_DIR}"


def imread_unicode(p: Path):
    try:
        data = np.fromfile(str(p), dtype=np.uint8)   
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)  
        return img
    except Exception:
        return None


INPUT_SIZE = 64
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

dataset = []
label   = []

def load_folder(folder: Path, y_value: int):
    skipped = 0
    for name in os.listdir(folder):
        p = folder / name
        if not p.is_file():
            continue
        if p.suffix.lower() not in VALID_EXTS:
            continue

        img = imread_unicode(p)   
        if img is None:
            skipped += 1
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)         
        img = Image.fromarray(img).resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(img))
        label.append(y_value)
    return skipped

bad_no  = load_folder(NO_DIR,  0)
bad_yes = load_folder(YES_DIR, 1)

print(f"Loaded: {len(dataset)} images  | Skipped unreadable: no={bad_no}, yes={bad_yes}")

dataset = np.array(dataset, dtype=np.float32)
label   = np.array(label,   dtype=np.int64)


x_train, x_test, y_train, y_test = train_test_split(
    dataset, label, test_size=0.2, random_state=0, stratify=label
)


x_train = normalize(x_train, axis=1)
x_test  = normalize(x_test,  axis=1)

y_train = to_categorical(y_train, num_classes=2)
y_test  = to_categorical(y_test,  num_classes=2)


model = Sequential()
model.add(Conv2D(32,(3,3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), kernel_initializer="he_uniform"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3), kernel_initializer="he_uniform"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(
    x_train, y_train,
    batch_size=16,
    verbose=1,
    epochs=10,
    validation_data=(x_test, y_test),
    shuffle=False
)

model.save("braintumor10Epoccategorical.h5")
print("✅ Saved: braintumor10Epoccategorical.h5")
